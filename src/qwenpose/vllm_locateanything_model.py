from __future__ import annotations

import importlib.util
import json
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import BatchFeature

from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    init_vllm_registered_model,
    maybe_prefix,
    merge_multimodal_embeddings,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalInputs,
    MultiModalKwargsItems,
    MultiModalUUIDDict,
)
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors


_IMAGE_TOKEN_LIMIT_KEYS = (
    "locate_image_token_limit",
    "eagle_image_token_limit",
    "image_token_limit",
    "in_token_limit",
)
_VISION_LORA_ENV = "QWENPOSE_VLLM_LOCATE_VISION_LORA_ADAPTER"


def _extract_image_token_limit(kwargs: Mapping[str, object] | None) -> int | None:
    if not kwargs:
        return None
    for key in _IMAGE_TOKEN_LIMIT_KEYS:
        value = kwargs.get(key)
        if value is not None and int(value) > 0:
            return int(value)
    images_kwargs = kwargs.get("images_kwargs")
    if isinstance(images_kwargs, Mapping):
        for key in _IMAGE_TOKEN_LIMIT_KEYS:
            value = images_kwargs.get(key)
            if value is not None and int(value) > 0:
                return int(value)
    return None


def _strip_image_token_limit_kwargs(kwargs: Mapping[str, object]) -> dict[str, object]:
    cleaned = {k: v for k, v in dict(kwargs).items() if k not in _IMAGE_TOKEN_LIMIT_KEYS}
    images_kwargs = cleaned.get("images_kwargs")
    if isinstance(images_kwargs, Mapping):
        cleaned["images_kwargs"] = {
            k: v
            for k, v in dict(images_kwargs).items()
            if k not in _IMAGE_TOKEN_LIMIT_KEYS
        }
    return cleaned


def _load_locateanything_module(model_path: str | Path, module_name: str):
    model_path = Path(model_path).expanduser().resolve()
    module_path = model_path / f"{module_name}.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"LocateAnything module not found: {module_path}")
    package_name = f"_qwenpose_vllm_locateanything_{model_path.name}"
    if package_name not in sys.modules:
        import types

        package = types.ModuleType(package_name)
        package.__path__ = [str(model_path)]
        sys.modules[package_name] = package
    full_name = f"{package_name}.{module_name}"
    if full_name in sys.modules:
        return sys.modules[full_name]
    spec = importlib.util.spec_from_file_location(full_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import LocateAnything module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = package_name
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


def _normalize_lora_module_name(name: str) -> str:
    if name.startswith("base_model.model."):
        name = name[len("base_model.model.") :]
    for suffix in (".lora_A.weight", ".lora_B.weight"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _resolve_lora_scale(config: Mapping[str, object], module_name: str, rank: int) -> float:
    default_alpha = float(config.get("lora_alpha", rank) or rank)
    alpha_pattern = config.get("alpha_pattern") or {}
    if isinstance(alpha_pattern, Mapping):
        alpha = float(alpha_pattern.get(module_name, default_alpha) or default_alpha)
    else:
        alpha = default_alpha
    use_rslora = bool(config.get("use_rslora", False))
    denom = rank**0.5 if use_rslora else rank
    return float(alpha / max(float(denom), 1.0))


class LocateAnythingProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}

    def get_num_image_tokens(self, image_grid_hw: Sequence[int]) -> int:
        processor = self.get_hf_processor()
        merge_kernel = getattr(processor.image_processor, "merge_kernel_size", (2, 2))
        h, w = int(image_grid_hw[0]), int(image_grid_hw[1])
        return max((h * w) // (int(merge_kernel[0]) * int(merge_kernel[1])), 1)

    def get_image_token_limit(self, overrides: Mapping[str, object] | None = None) -> int:
        mm_config = self.ctx.model_config.get_multimodal_config()
        token_limit = _extract_image_token_limit(overrides)
        if token_limit is None:
            token_limit = _extract_image_token_limit(mm_config.mm_processor_kwargs)
        if token_limit is not None:
            return int(token_limit)
        processor = self.get_hf_processor()
        return int(getattr(processor.image_processor, "in_token_limit", 25600))

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Optional[Mapping[str, int]]:
        processor = self.get_hf_processor()
        image_processor = processor.image_processor
        in_token_limit = self.get_image_token_limit()
        merge_kernel = getattr(image_processor, "merge_kernel_size", (2, 2))
        max_tokens = max(
            in_token_limit // (int(merge_kernel[0]) * int(merge_kernel[1])),
            1,
        )
        return {"image": max_tokens}


class LocateAnythingDummyInputsBuilder(
    BaseDummyInputsBuilder[LocateAnythingProcessingInfo]
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = int(mm_counts.get("image", 0))
        return "<|im_start|>user\n" + "".join(
            f"<image-{idx + 1}>" for idx in range(num_images)
        ) + "<|im_end|>\n<|im_start|>assistant\n"

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        num_images = int(mm_counts.get("image", 0))
        processor = self.info.get_hf_processor()
        image_processor = processor.image_processor
        patch_size = int(getattr(image_processor, "patch_size", 14))
        in_token_limit = self.info.get_image_token_limit()
        side_patches = max(int(in_token_limit**0.5), 1)
        side = max(side_patches * patch_size, patch_size)
        return {
            "image": self._get_dummy_images(
                width=side,
                height=side,
                num_images=num_images,
            )
        }


class LocateAnythingMultiModalProcessor(
    BaseMultiModalProcessor[LocateAnythingProcessingInfo]
):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        clean_mm_kwargs = _strip_image_token_limit_kwargs(mm_kwargs)
        token_limit = self.info.get_image_token_limit(mm_kwargs)
        hf_processor = self.info.get_hf_processor()
        image_processor = getattr(hf_processor, "image_processor", None)
        old_token_limit = (
            getattr(image_processor, "in_token_limit", None)
            if image_processor is not None
            else None
        )
        if image_processor is not None and token_limit is not None:
            image_processor.in_token_limit = int(token_limit)
        if not mm_data:
            try:
                tokenizer = self.info.get_tokenizer()
                prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
                return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")
            finally:
                if image_processor is not None and old_token_limit is not None:
                    image_processor.in_token_limit = old_token_limit
        try:
            processed = hf_processor(
                text=prompt,
                **mm_data,
                **clean_mm_kwargs,
                **tok_kwargs,
                return_tensors="pt",
            )
        finally:
            if image_processor is not None and old_token_limit is not None:
                image_processor.in_token_limit = old_token_limit
        model_dtype = self.info.ctx.model_config.dtype
        for key, value in list(processed.items()):
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                processed[key] = value.to(dtype=model_dtype)
        return processed

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        image_grid_hws = hf_inputs.get("image_grid_hws")
        if image_grid_hws is None:
            return {}
        if isinstance(image_grid_hws, np.ndarray):
            image_grid_hws = torch.from_numpy(image_grid_hws)
        image_grid_hws = image_grid_hws.to(dtype=torch.int32)
        raw_patch_counts = image_grid_hws[:, 0] * image_grid_hws[:, 1]
        num_images = int(image_grid_hws.shape[0])
        return {
            "pixel_values": MultiModalFieldConfig.flat_from_sizes(
                "image", raw_patch_counts
            ),
            "image_grid_hws": MultiModalFieldConfig.batched("image"),
        }

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_token = getattr(processor, "image_token", "<IMG_CONTEXT>")
        image_token_id = int(getattr(processor, "image_token_id"))
        image_start = getattr(processor, "image_start_token", "<img>")
        image_end = getattr(processor, "image_end_token", "</img>")

        out_mm_data = out_mm_kwargs.get_data()
        grid_hws = out_mm_data.get("image_grid_hws")
        if isinstance(grid_hws, torch.Tensor):
            grid_hws_list = grid_hws.reshape(-1, 2).tolist()
        elif grid_hws is None:
            grid_hws_list = []
        else:
            grid_hws_list = np.asarray(grid_hws).reshape(-1, 2).tolist()

        def get_replacement(item_idx: int) -> PromptUpdateDetails[str]:
            num_tokens = self.info.get_num_image_tokens(grid_hws_list[item_idx])
            full = (
                f"<image {item_idx + 1}>"
                f"{image_start}{image_token * num_tokens}{image_end}"
            )
            return PromptUpdateDetails.select_token_id(full, image_token_id)

        return [
            PromptReplacement(
                modality="image",
                target=lambda item_idx: f"<image-{item_idx + 1}>",
                replacement=get_replacement,
            )
        ]

    def apply(
        self,
        prompt: str | list[int],
        mm_data: MultiModalDataDict,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Optional[Mapping[str, object]] = None,
        *,
        mm_uuids: Optional[MultiModalUUIDDict] = None,
    ) -> MultiModalInputs:
        if tokenization_kwargs is None:
            tokenization_kwargs = {"padding": False}
        return super().apply(
            prompt,
            mm_data,
            hf_processor_mm_kwargs,
            tokenization_kwargs,
            mm_uuids=mm_uuids,
        )


@MULTIMODAL_REGISTRY.register_processor(
    LocateAnythingMultiModalProcessor,
    info=LocateAnythingProcessingInfo,
    dummy_inputs=LocateAnythingDummyInputsBuilder,
)
class LocateAnythingVLLMForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsPP, SupportsLoRA
):
    hf_to_vllm_mapper = None
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality.startswith("image"):
            return f"<image-{i + 1}>"
        raise ValueError("Only image modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.image_token_index = int(config.image_token_index)
        self.dtype = vllm_config.model_config.dtype
        self._qwenpose_vision_lora_merged = False
        self._qwenpose_capture_features = False
        self._qwenpose_pending_visuals: list[dict[str, torch.Tensor]] = []
        self._qwenpose_feature_cache: list[dict[str, torch.Tensor]] = []
        self._qwenpose_last_input_ids: torch.Tensor | None = None
        self._qwenpose_feature_adapter_loaded = False
        self._qwenpose_pose_loaded = False

        model_path = vllm_config.model_config.model
        modeling_vit = _load_locateanything_module(model_path, "modeling_vit")
        self.vision_model = modeling_vit.MoonVitPretrainedModel(config.vision_config)

        vit_hidden_size = int(config.vision_config.hidden_size)
        llm_hidden_size = int(config.text_config.hidden_size)
        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * 4),
            nn.Linear(vit_hidden_size * 4, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

        config.text_config.architectures = ["Qwen2ForCausalLM"]
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

        from qwenpose.qwen_lora import QwenFeatureRefiner

        feature_size = int(os.environ.get("QWENPOSE_VLLM_FEATURE_SIZE", "64"))
        refiner_layers = int(os.environ.get("QWENPOSE_VLLM_FEATURE_REFINER_LAYERS", "2"))
        refiner_bottleneck_dim = int(os.environ.get("QWENPOSE_VLLM_FEATURE_REFINER_BOTTLENECK_DIM", "256"))
        refiner_init_scale = float(os.environ.get("QWENPOSE_VLLM_FEATURE_REFINER_INIT_SCALE", "0.1"))
        self.qwenpose_feature_size = feature_size
        self.raw_feature_norm = nn.LayerNorm(llm_hidden_size)
        self.lm_feature_norm = nn.LayerNorm(llm_hidden_size)
        self.dual_feature_fuse = nn.Sequential(
            nn.Conv2d(llm_hidden_size * 2, llm_hidden_size, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(llm_hidden_size, llm_hidden_size, kernel_size=3, padding=1, groups=llm_hidden_size),
            nn.Conv2d(llm_hidden_size, llm_hidden_size, kernel_size=1),
        )
        self.dual_feature_gate = nn.Parameter(torch.tensor(0.0))
        self.feature_refiner = QwenFeatureRefiner(
            llm_hidden_size,
            num_layers=refiner_layers,
            bottleneck_dim=refiner_bottleneck_dim,
            init_scale=refiner_init_scale,
        )
        self.qwenpose_pose_model: nn.Module | None = None

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def set_qwenpose_feature_capture(self, enabled: bool) -> None:
        self._qwenpose_capture_features = bool(enabled)

    def reset_qwenpose_feature_cache(self) -> None:
        self._qwenpose_pending_visuals.clear()
        self._qwenpose_feature_cache.clear()
        self._qwenpose_last_input_ids = None

    def pop_qwenpose_feature_cache(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._qwenpose_feature_cache:
            raise RuntimeError("vLLM LocateAnything produced no cached qwenpose features.")
        feature_map = torch.cat([item["feature_map"] for item in self._qwenpose_feature_cache], dim=0)
        text_embed = torch.cat([item["text_embed"] for item in self._qwenpose_feature_cache], dim=0)
        self._qwenpose_feature_cache.clear()
        return feature_map, text_embed

    def load_qwenpose_checkpoint(self, checkpoint_path: str | Path) -> None:
        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
        extractor_state = checkpoint.get("backbone_extractor", checkpoint.get("qwen_extractor"))
        if isinstance(extractor_state, Mapping):
            adapter_prefixes = (
                "raw_feature_norm.",
                "lm_feature_norm.",
                "dual_feature_fuse.",
                "dual_feature_gate",
                "feature_refiner.",
            )
            adapter_state = {
                key: value
                for key, value in extractor_state.items()
                if key == "dual_feature_gate" or key.startswith(adapter_prefixes)
            }
            self.load_state_dict(adapter_state, strict=False)
            self._qwenpose_feature_adapter_loaded = True
            print(
                f"[qwenpose vLLM] loaded feature adapter tensors={len(adapter_state)} from {checkpoint_path}",
                flush=True,
            )

        if "model" not in checkpoint:
            raise KeyError(f"qwenpose checkpoint has no PoseHead model state: {checkpoint_path}")
        from qwenpose.model import QwenPoseConfig, QwenPoseModel

        hidden_size = int(self.config.text_config.hidden_size)
        saved_pose_config = checkpoint.get("pose_config") or {}
        pose_config_kwargs = {
            key: saved_pose_config[key]
            for key in QwenPoseConfig.__dataclass_fields__
            if key in saved_pose_config
        }
        pose_config_kwargs["external_dim"] = hidden_size
        pose_model = QwenPoseModel(QwenPoseConfig(**pose_config_kwargs))
        pose_model.load_state_dict(checkpoint["model"], strict=True)
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        pose_model.to(device=device, dtype=dtype)
        pose_model.eval()
        self.qwenpose_pose_model = pose_model
        self._qwenpose_pose_loaded = True
        print(f"[qwenpose vLLM] loaded PoseHead from {checkpoint_path}", flush=True)

    def _fuse_qwenpose_feature_maps(self, raw_maps: torch.Tensor, lm_maps: torch.Tensor) -> torch.Tensor:
        adapter_param = self.raw_feature_norm.weight
        adapter_device = adapter_param.device
        adapter_dtype = adapter_param.dtype
        raw_maps = raw_maps.to(device=adapter_device, dtype=adapter_dtype)
        lm_maps = lm_maps.to(device=adapter_device, dtype=adapter_dtype)
        b, c, h, w = raw_maps.shape
        raw_tokens = raw_maps.permute(0, 2, 3, 1).reshape(b, h * w, c)
        lm_tokens = lm_maps.permute(0, 2, 3, 1).reshape(b, h * w, c)
        raw_maps = self.raw_feature_norm(raw_tokens).view(b, h, w, c).permute(0, 3, 1, 2)
        lm_maps = self.lm_feature_norm(lm_tokens).view(b, h, w, c).permute(0, 3, 1, 2)
        fuse_dtype = next(self.dual_feature_fuse.parameters()).dtype
        if raw_maps.dtype != fuse_dtype:
            raw_maps = raw_maps.to(dtype=fuse_dtype)
            lm_maps = lm_maps.to(dtype=fuse_dtype)
        fused_delta = self.dual_feature_fuse(torch.cat([raw_maps, lm_maps], dim=1))
        gate = torch.sigmoid(self.dual_feature_gate).to(device=fused_delta.device, dtype=fused_delta.dtype)
        return lm_maps + gate * fused_delta

    def _cache_qwenpose_prefill_features(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> None:
        if not self._qwenpose_capture_features or not self._qwenpose_pending_visuals:
            return
        if input_ids is None or hidden_states is None:
            return
        ids = input_ids.reshape(-1).to(device=hidden_states.device)
        if positions.ndim > 1:
            pos = positions[0].reshape(-1).to(device=hidden_states.device)
        else:
            pos = positions.reshape(-1).to(device=hidden_states.device)
        n = min(int(ids.numel()), int(pos.numel()), int(hidden_states.shape[0]))
        if n <= 0:
            return
        ids = ids[:n]
        pos = pos[:n]
        hidden = hidden_states[:n]
        image_mask_all = ids == self.image_token_index
        if not bool(image_mask_all.any().item()):
            return

        starts = torch.nonzero(pos == 0, as_tuple=False).flatten().tolist()
        if not starts or starts[0] != 0:
            starts = [0] + starts
        boundaries = starts + [n]
        merge_kernel = getattr(self.vision_model, "merge_kernel_size", (2, 2))
        merge_h = max(int(merge_kernel[0]), 1) if isinstance(merge_kernel, (tuple, list)) else max(int(merge_kernel), 1)
        merge_w = max(int(merge_kernel[1]), 1) if isinstance(merge_kernel, (tuple, list)) else max(int(merge_kernel), 1)

        for left, right in zip(boundaries[:-1], boundaries[1:]):
            if right <= left:
                continue
            seq_ids = ids[left:right]
            image_mask = seq_ids == self.image_token_index
            if not bool(image_mask.any().item()):
                continue
            if not self._qwenpose_pending_visuals:
                raise RuntimeError("qwenpose vLLM feature cache lost the matching projected visual tokens.")
            visual = self._qwenpose_pending_visuals.pop(0)
            raw_tokens = visual["projected"].to(device=hidden_states.device)
            lm_tokens = hidden[left:right][image_mask]
            grid_hw = visual["grid_hw"].detach().cpu().tolist()
            raw_h = max(int(grid_hw[0]), 1)
            raw_w = max(int(grid_hw[1]), 1)
            h = max(raw_h // merge_h, 1)
            w = max(raw_w // merge_w, 1)
            expected = h * w
            if int(lm_tokens.shape[0]) != expected or int(raw_tokens.shape[0]) != expected:
                raise ValueError(
                    "qwenpose vLLM visual token/grid mismatch: "
                    f"lm_tokens={int(lm_tokens.shape[0])}, raw_tokens={int(raw_tokens.shape[0])}, "
                    f"merged_grid={h}x{w}, raw_grid={grid_hw}, merge_kernel={merge_h}x{merge_w}."
                )
            raw_map = raw_tokens.float().view(h, w, -1).permute(2, 0, 1).unsqueeze(0)
            lm_map = lm_tokens.float().view(h, w, -1).permute(2, 0, 1).unsqueeze(0)
            raw_map = F.interpolate(raw_map, size=(self.qwenpose_feature_size, self.qwenpose_feature_size), mode="bilinear", align_corners=False)
            lm_map = F.interpolate(lm_map, size=(self.qwenpose_feature_size, self.qwenpose_feature_size), mode="bilinear", align_corners=False)
            fused = self._fuse_qwenpose_feature_maps(raw_map, lm_map)
            feature_map = self.feature_refiner(fused)
            non_image = ~image_mask
            text_mask = non_image.float().to(device=hidden_states.device).unsqueeze(-1)
            seq_hidden = hidden[left:right]
            text_embed = (seq_hidden * text_mask).sum(dim=0, keepdim=True) / text_mask.sum(dim=0, keepdim=True).clamp(min=1.0)
            self._qwenpose_feature_cache.append(
                {
                    "feature_map": feature_map.detach(),
                    "text_embed": text_embed.detach(),
                }
            )

    @torch.inference_mode()
    def run_qwenpose_pose(
        self,
        *,
        schema_ids: torch.Tensor,
        task_ids: torch.Tensor,
        target_boxes: torch.Tensor,
        target_box_mask: torch.Tensor,
        images: torch.Tensor | None,
        external_feature_map: torch.Tensor,
        external_text_embed: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.qwenpose_pose_model is None:
            raise RuntimeError("qwenpose PoseHead is not loaded in the vLLM model.")
        device = next(self.qwenpose_pose_model.parameters()).device
        schema_ids = schema_ids.to(device)
        task_ids = task_ids.to(device)
        target_boxes = target_boxes.to(device=device)
        target_box_mask = target_box_mask.to(device=device)
        external_feature_map = external_feature_map.to(device=device)
        external_text_embed = external_text_embed.to(device=device)
        if images is not None:
            images = images.to(device=device)
        return self.qwenpose_pose_model(
            schema_ids=schema_ids,
            task_ids=task_ids,
            external_feature_map=external_feature_map,
            external_text_embed=external_text_embed,
            target_boxes=target_boxes,
            target_box_mask=target_box_mask,
            images=images,
        )

    def _parse_image_inputs(
        self,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_grid_hws = kwargs.pop("image_grid_hws", None)
        image_embeds = kwargs.pop("image_embeds", None)
        if image_embeds is not None:
            return image_embeds, torch.empty(0)
        if pixel_values is None:
            return None
        if image_grid_hws is None:
            raise ValueError("LocateAnything vLLM inputs require image_grid_hws.")
        if isinstance(pixel_values, list):
            pixel_values = torch.cat(pixel_values, dim=0)
        if isinstance(image_grid_hws, list):
            if image_grid_hws and isinstance(image_grid_hws[0], torch.Tensor):
                image_grid_hws = torch.stack(image_grid_hws, dim=0)
            else:
                image_grid_hws = torch.as_tensor(np.asarray(image_grid_hws))
        if isinstance(image_grid_hws, np.ndarray):
            image_grid_hws = torch.from_numpy(image_grid_hws)
        assert isinstance(pixel_values, torch.Tensor)
        assert isinstance(image_grid_hws, torch.Tensor)
        if pixel_values.ndim > 4:
            pixel_values = pixel_values.flatten(0, pixel_values.ndim - 4)
        image_grid_hws = image_grid_hws.reshape(-1, 2)
        return pixel_values, image_grid_hws

    def get_multimodal_embeddings(self, **kwargs: object) -> MultiModalEmbeddings:
        parsed = self._parse_image_inputs(**kwargs)
        if parsed is None:
            return []
        pixel_values, image_grid_hws = parsed
        if image_grid_hws.numel() == 0:
            return pixel_values
        device = next(self.vision_model.parameters()).device
        pixel_values = pixel_values.to(device=device, dtype=self.dtype)
        image_grid_hws = image_grid_hws.to(device=device, dtype=torch.int32)
        image_features = self.vision_model(
            pixel_values=pixel_values,
            grid_hws=image_grid_hws,
        )
        projected = self.mlp1(torch.cat(image_features, dim=0))
        lengths = [int(feature.shape[0]) for feature in image_features]
        projected_list = tuple(projected.split(lengths, dim=0))
        if self._qwenpose_capture_features:
            for item, grid_hw in zip(projected_list, image_grid_hws):
                self._qwenpose_pending_visuals.append(
                    {
                        "projected": item.detach(),
                        "grid_hw": grid_hw.detach().clone(),
                    }
                )
        return projected_list

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        if self._qwenpose_capture_features:
            self._qwenpose_last_input_ids = input_ids.detach().clone()
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None and len(multimodal_embeddings) != 0:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                multimodal_embeddings,
                self.image_token_index,
            )
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None
        elif inputs_embeds is None:
            multimodal_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids, multimodal_embeddings)
            input_ids = None
        cache_input_ids = input_ids if input_ids is not None else self._qwenpose_last_input_ids
        hidden_states = self.language_model.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        if isinstance(hidden_states, torch.Tensor) and cache_input_ids is not None:
            self._cache_qwenpose_prefill_features(cache_input_ids, positions, hidden_states)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states)

    def _merge_qwenpose_vision_lora(self) -> None:
        if self._qwenpose_vision_lora_merged:
            return
        self._qwenpose_vision_lora_merged = True
        adapter_path = os.environ.get(_VISION_LORA_ENV, "").strip()
        if not adapter_path:
            return
        adapter_dir = Path(adapter_path).expanduser()
        config_path = adapter_dir / "adapter_config.json"
        tensors_path = adapter_dir / "adapter_model.safetensors"
        if not config_path.is_file() or not tensors_path.is_file():
            print(
                f"[qwenpose vLLM] vision LoRA adapter is incomplete: {adapter_dir}",
                flush=True,
            )
            return
        try:
            from safetensors.torch import load_file

            config = json.loads(config_path.read_text(encoding="utf-8"))
            tensors = load_file(str(tensors_path), device="cpu")
        except Exception as exc:
            print(
                f"[qwenpose vLLM] failed to load vision LoRA adapter {adapter_dir}: {exc}",
                flush=True,
            )
            return

        merged = 0
        with torch.no_grad():
            for key, lora_a in tensors.items():
                if "vision_model." not in key or not key.endswith(".lora_A.weight"):
                    continue
                key_b = key[: -len(".lora_A.weight")] + ".lora_B.weight"
                lora_b = tensors.get(key_b)
                if lora_b is None:
                    continue
                module_name = _normalize_lora_module_name(key)
                module = self.get_submodule(module_name)
                if not isinstance(module, nn.Linear):
                    raise TypeError(
                        f"Vision LoRA target is not nn.Linear: {module_name}"
                    )
                rank = int(lora_a.shape[0])
                scale = _resolve_lora_scale(config, module_name, rank)
                delta = torch.matmul(lora_b.float(), lora_a.float()).mul_(scale)
                if tuple(delta.shape) != tuple(module.weight.shape):
                    raise ValueError(
                        f"Vision LoRA delta shape mismatch for {module_name}: "
                        f"{tuple(delta.shape)} vs {tuple(module.weight.shape)}"
                    )
                module.weight.add_(delta.to(device=module.weight.device, dtype=module.weight.dtype))
                merged += 1
        if merged:
            print(
                f"[qwenpose vLLM] merged {merged} vision LoRA modules from {adapter_dir}",
                flush=True,
            )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(weights)
        self._merge_qwenpose_vision_lora()
        qwenpose_prefixes = (
            "raw_feature_norm.",
            "lm_feature_norm.",
            "dual_feature_fuse.",
            "dual_feature_gate",
            "feature_refiner.",
        )
        for name, _ in self.named_parameters():
            if name == "dual_feature_gate" or name.startswith(qwenpose_prefixes):
                loaded.add(name)
        return loaded
