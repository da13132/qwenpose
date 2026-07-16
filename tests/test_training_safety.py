from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from qwenpose.spatial_features import MultiScaleSpatialFeatureBatch, SpatialFeatureBatch

from qwenpose.eagle_lora import (
    EagleFeatureExtractor,
    PrunedEagleLMHead,
    build_eagle_lm_inputs,
    compute_eagle_pbd_grounding_losses,
    eagle_generation_is_pruned,
    extract_eagle_pbd_blocks_from_lm_logits,
    prune_eagle_generation_components,
)
from qwenpose.train_pose import (
    iter_named_floating_tensors,
    load_deepspeed_config,
    synchronized_finite_check,
)


class _DummyTokenizer:
    def __init__(self) -> None:
        self.vocab: dict[str, int] = {"<pad>": 0}
        self.inverse: dict[int, str] = {0: "<pad>"}

    def _encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for token in str(text).split():
            if token not in self.vocab:
                index = len(self.vocab)
                self.vocab[token] = index
                self.inverse[index] = token
            ids.append(self.vocab[token])
        return ids

    def __call__(self, texts, **_: object) -> dict[str, object]:
        if isinstance(texts, str):
            return {"input_ids": self._encode(texts)}
        rows = [self._encode(text) for text in texts]
        return {
            "input_ids": rows,
            "attention_mask": [[1] * len(row) for row in rows],
        }

    def decode(self, ids, **_: object) -> str:
        return " ".join(self.inverse[int(index)] for index in ids)


class _DummyProcessor:
    def __init__(self) -> None:
        self.tokenizer = _DummyTokenizer()
        self.image_processor = SimpleNamespace(in_token_limit=None)

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        del tokenize
        user_text = messages[0]["content"][1]["text"]
        text = f"USER IMAGE {user_text} ASSISTANT"
        if len(messages) > 1:
            text += f" {messages[1]['content']} END"
        elif not add_generation_prompt:
            text += " END"
        return text

    def __call__(self, *, text, images, padding: bool, return_tensors: str):
        del images, padding, return_tensors
        rows = [self.tokenizer._encode(item) for item in text]
        width = max(len(row) for row in rows)
        input_ids = torch.zeros(len(rows), width, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for row_idx, row in enumerate(rows):
            start = width - len(row)
            input_ids[row_idx, start:] = torch.tensor(row, dtype=torch.long)
            attention_mask[row_idx, start:] = 1
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": torch.zeros(len(rows), 3, 2, 2),
            "image_grid_hws": torch.ones(len(rows), 2, dtype=torch.long),
        }


class _DummyEagle(torch.nn.Module):
    def __init__(self, hidden_size: int = 4) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(hidden_size=hidden_size)
        )


class _DummyDecoder(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(use_cache=True)
        self._qwenpose_safe_image_processing = True
        self.last_kwargs: dict[str, object] = {}
        self.hidden_size = hidden_size

    def forward(self, input_ids: torch.Tensor, **kwargs: object) -> SimpleNamespace:
        self.last_kwargs = kwargs
        hidden = torch.nn.functional.one_hot(
            input_ids.remainder(self.hidden_size),
            num_classes=self.hidden_size,
        ).float()
        return SimpleNamespace(last_hidden_state=hidden)


class _DummyLanguageModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 7, hidden_size: int = 4) -> None:
        super().__init__()
        self.config = SimpleNamespace(use_cache=True)
        self.generation_config = SimpleNamespace(use_cache=True)
        self.model = _DummyDecoder(hidden_size)
        self.embed_tokens = torch.nn.Embedding(vocab_size, hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.embed_tokens

    def forward(self, *_: object, **__: object) -> None:
        raise AssertionError("feature extraction must bypass the causal-LM wrapper/lm_head")


class _DummyMultimodalEagle(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            use_cache=True,
            text_config=SimpleNamespace(hidden_size=4),
        )
        self.language_model = _DummyLanguageModel()
        self.image_token_index = 6


def test_eagle_lm_inputs_supervise_only_complete_assistant_response() -> None:
    processor = _DummyProcessor()
    inputs = build_eagle_lm_inputs(
        processor,
        ["a.jpg", "b.jpg"],
        ["short prompt", "a much longer prompt"],
        ["<box><100><200><300><400></box>", "None"],
        torch.device("cpu"),
        image_tensors=[
            torch.zeros(3, 4, 4, dtype=torch.uint8),
            torch.zeros(3, 4, 4, dtype=torch.uint8),
        ],
    )

    supervised = []
    for row in range(2):
        ids = inputs["input_ids"][row][inputs["labels"][row].ne(-100)]
        supervised.append(processor.tokenizer.decode(ids))

    assert supervised == [
        "<box><100><200><300><400></box> END",
        "None END",
    ]


def test_pbd_grounding_loss_backpropagates_to_coordinate_logits() -> None:
    model = _DummyEagle(hidden_size=4)
    model.token_ids = {
        "box_start_token_id": 8,
        "box_end_token_id": 9,
        "coord_start_token_id": 10,
        "coord_end_token_id": 14,
    }
    extractor = EagleFeatureExtractor(
        model,
        refiner_layers=0,
        feature_source="raw_visual",
    )
    logits = torch.randn(1, 6, 20, requires_grad=True)
    target_ids = torch.tensor([[8, 10, 11, 12, 13, 9]])
    gt_boxes = torch.tensor([[0.0, 0.25, 0.5, 0.75]])

    losses, soft_boxes = compute_eagle_pbd_grounding_losses(
        extractor,
        logits,
        target_ids,
        gt_boxes,
        temperature=1.0,
    )
    total = sum(losses.values()) + soft_boxes.sum() * 0.0
    total.backward()

    assert soft_boxes.shape == (1, 4)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad[:, 1:5].abs().sum()) > 0.0


def test_extracts_all_batched_teacher_forced_pbd_blocks_in_response_order() -> None:
    model = _DummyEagle(hidden_size=4)
    model.token_ids = {
        "box_start_token_id": 8,
        "box_end_token_id": 9,
        "coord_start_token_id": 10,
        "coord_end_token_id": 14,
    }
    extractor = EagleFeatureExtractor(model, refiner_layers=0, feature_source="raw_visual")
    shift_labels = torch.tensor(
        [
            [8, 10, 11, 12, 13, 9, 3, -100, -100, -100, -100, -100, -100],
            [8, 14, 13, 12, 11, 9, 8, 10, 10, 14, 14, 9, 3],
        ]
    )
    valid = shift_labels.ne(-100)
    logits = torch.randn(int(valid.sum()), 20, requires_grad=True)

    blocks, targets = extract_eagle_pbd_blocks_from_lm_logits(
        extractor,
        logits,
        shift_labels,
        valid,
        torch.tensor([1, 2]),
    )

    assert blocks.shape == (3, 6, 20)
    assert targets.tolist() == [
        [8, 10, 11, 12, 13, 9],
        [8, 14, 13, 12, 11, 9],
        [8, 10, 10, 14, 14, 9],
    ]
    blocks.sum().backward()
    assert logits.grad is not None
    assert int(torch.count_nonzero(logits.grad)) == 3 * 6 * 20


def test_raw_visual_extractor_has_no_lm_image_fusion_parameters() -> None:
    extractor = EagleFeatureExtractor(
        _DummyEagle(),
        refiner_layers=0,
        feature_source="raw_visual",
    )
    raw_maps = torch.randn(2, 4, 2, 2)
    refined = extractor.feature_refiner(raw_maps)

    assert refined.shape == raw_maps.shape
    assert torch.equal(refined, raw_maps)
    assert not hasattr(extractor, "raw_feature_norm")
    assert not hasattr(extractor, "dual_feature_fuse")
    assert not hasattr(extractor, "lm_feature_norm")


def test_raw_visual_only_runs_language_path_when_text_is_required() -> None:
    extractor = EagleFeatureExtractor(
        _DummyEagle(),
        refiner_layers=0,
        feature_source="raw_visual",
    )
    raw_maps = SpatialFeatureBatch.from_maps([torch.randn(4, 2, 2)])
    raw_levels = MultiScaleSpatialFeatureBatch((raw_maps, raw_maps))
    expected_text = torch.randn(1, 4)
    calls = {"vision": 0, "language": 0}

    def vision_only(*args, **kwargs):
        del args, kwargs
        calls["vision"] += 1
        return raw_levels, torch.zeros_like(expected_text)

    def with_language(*args, **kwargs):
        del args, kwargs
        calls["language"] += 1
        return raw_levels, expected_text

    extractor._extract_eagle_vision_features = vision_only  # type: ignore[method-assign]
    extractor._extract_eagle_feature_maps = with_language  # type: ignore[method-assign]

    _, no_text = extractor({}, require_text=False)
    _, text = extractor({}, require_text=True)

    assert calls == {"vision": 1, "language": 1}
    torch.testing.assert_close(no_text, torch.zeros_like(expected_text))
    torch.testing.assert_close(text, expected_text)


def test_pruned_locate_generation_keeps_input_embeddings_and_disables_cache() -> None:
    model = _DummyMultimodalEagle()
    embedding_weight = model.language_model.get_input_embeddings().weight

    stats = prune_eagle_generation_components(model)

    assert stats == {"lm_head_numel": 28, "lm_head_tied": True}
    assert eagle_generation_is_pruned(model)
    assert isinstance(model.language_model.lm_head, PrunedEagleLMHead)
    assert model.language_model.get_input_embeddings().weight is embedding_weight
    assert model.config.use_cache is False
    assert model.language_model.config.use_cache is False
    assert model.language_model.model.config.use_cache is False
    assert model.language_model.generation_config.use_cache is False


def test_pruned_locate_text_features_bypass_lm_head_and_do_not_build_cache() -> None:
    model = _DummyMultimodalEagle()
    prune_eagle_generation_components(model)
    extractor = EagleFeatureExtractor(
        model,
        refiner_layers=0,
        feature_source="raw_visual",
    )
    input_ids = torch.tensor([[1, 2, 3]])
    hidden = extractor.run_language_hidden(
        input_ids,
        torch.ones_like(input_ids),
        torch.randn(1, 4),
    )

    assert hidden.shape == (1, 3, 4)
    assert model.language_model.model.last_kwargs["use_cache"] is False
    assert model.language_model.model.last_kwargs["output_hidden_states"] is False
    try:
        extractor.run_language_prefill(
            input_ids,
            torch.ones_like(input_ids),
            torch.randn(1, 4),
        )
    except RuntimeError as exc:
        assert "pruned" in str(exc)
    else:
        raise AssertionError("pruned generation must reject KV-cache prefill")


def test_synchronized_finite_check_reports_nested_nonfinite_tensor() -> None:
    outputs = {
        "keypoints": torch.ones(2, 3),
        "refine": [torch.zeros(1), torch.tensor([float("nan")])],
    }

    finite, bad_names = synchronized_finite_check(
        iter_named_floating_tensors(outputs),
        torch.device("cpu"),
    )

    assert not finite
    assert bad_names == ["refine[1]"]


def test_synchronized_finite_check_accepts_finite_tensors() -> None:
    finite, bad_names = synchronized_finite_check(
        iter_named_floating_tensors({"x": torch.ones(2)}),
        torch.device("cpu"),
    )

    assert finite
    assert bad_names == []


def test_bf16_deepspeed_config_checks_partitioned_gradient_overflow(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "zero2.json"
    config_path.write_text(
        json.dumps(
            {
                "bf16": {"enabled": True},
                "fp16": {"enabled": False},
                "zero_optimization": {"stage": 2},
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        batch_size=12,
        grad_accum_steps=1,
        grad_clip=1.0,
        backbone="eagle",
        eagle_dtype="bfloat16",
    )

    config = load_deepspeed_config(config_path, args, world_size=4)

    assert config["train_micro_batch_size_per_gpu"] == 12
    assert config["train_batch_size"] == 48
    assert config["bf16"] == {
        "enabled": True,
        "check_grad_overflow": True,
    }


def test_repo_zero2_reduces_bf16_gradients_in_fp32() -> None:
    config_path = Path(__file__).resolve().parents[1] / "scripts" / "zero2.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert config["bf16"]["enabled"] is True
    assert config["bf16"]["check_grad_overflow"] is True
    assert config["communication_data_type"] == "fp32"
