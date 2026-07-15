from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from qwenpose.eagle_lora import (
    EagleFeatureExtractor,
    PrunedEagleLMHead,
    eagle_generation_is_pruned,
    prune_eagle_generation_components,
)
from qwenpose.train_pose import (
    iter_named_floating_tensors,
    load_deepspeed_config,
    synchronized_finite_check,
)


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


def test_raw_visual_extractor_has_no_lm_image_fusion_parameters() -> None:
    extractor = EagleFeatureExtractor(
        _DummyEagle(),
        output_size=2,
        refiner_layers=0,
        feature_source="raw_visual",
    )
    raw_maps = torch.randn(2, 4, 2, 2)
    normalized = extractor.normalize_raw_feature_maps(raw_maps)

    assert normalized.shape == raw_maps.shape
    assert not hasattr(extractor, "dual_feature_fuse")
    assert not hasattr(extractor, "lm_feature_norm")


def test_raw_visual_only_runs_language_path_when_text_is_required() -> None:
    extractor = EagleFeatureExtractor(
        _DummyEagle(),
        output_size=2,
        refiner_layers=0,
        feature_source="raw_visual",
    )
    raw_maps = torch.randn(1, 4, 2, 2)
    expected_text = torch.randn(1, 4)
    calls = {"vision": 0, "language": 0}

    def vision_only(*args, **kwargs):
        del args, kwargs
        calls["vision"] += 1
        return raw_maps, torch.zeros_like(expected_text)

    def with_language(*args, **kwargs):
        del args, kwargs
        calls["language"] += 1
        return raw_maps, torch.randn_like(raw_maps), expected_text

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
        output_size=2,
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
