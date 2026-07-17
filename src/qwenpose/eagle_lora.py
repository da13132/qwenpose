"""Eagle/Embodied (LocateAnything-3B) backbone loading, LoRA, and feature extraction.

LocateAnything-3B architecture:
  Vision: MoonViT-SO-400M (hidden=1152, patch=14, merge=2x2) -> List[Tensor] per image
  Projector: MLP (4608 -> 2048 -> 2048)
  LLM: Qwen2.5-3B-Instruct (hidden=2048, 36 layers)
  Output feature dim: 2048

Key interface details from modeling_locateanything.py:
  - extract_feature(pixel_values, image_grid_hws) -> List[Tensor], each (num_merged_tokens, 4608)
  - mlp1(vit_embeds) -> (num_merged_tokens, 2048)
  - image_token_index = 151665
  - forward() replaces image token embeddings with projected vision features
  - Multimodal processor returns pixel_values, image_grid_hws, input_ids, attention_mask
  - Vision-only Stage 1 uses AutoImageProcessor and returns only pixel_values/image_grid_hws
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import tempfile
import types
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from PIL import Image

from .qwen_lora import QwenFeatureRefiner, _dtype_from_name
from .spatial_features import MultiScaleSpatialFeatureBatch, SpatialFeatureBatch


@dataclass
class EagleLoRAConfig:
    model_path: str
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    vision_lora_r: int = 16
    vision_lora_alpha: int = 32
    vision_lora_dropout: float = 0.05
    dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"
    gradient_checkpointing: bool = False
    prune_generation: bool = False


class PrunedEagleLMHead(nn.Module):
    """Zero-parameter sentinel for LocateAnything's unused vocabulary head."""

    def forward(self, *_: object, **__: object) -> torch.Tensor:
        raise RuntimeError(
            "LocateAnything autoregressive generation was pruned for this LocatePose run. "
            "Use the person-query detection/ref/pose heads, or reload with "
            "prune_generation=False when token generation is explicitly required."
        )


def eagle_generation_is_pruned(model: nn.Module) -> bool:
    return bool(getattr(get_eagle_base_model(model), "generation_components_pruned", False))


def prune_eagle_generation_components(model: nn.Module) -> dict[str, int | bool]:
    """Remove the generation-only output projection and disable KV caching.

    LocateAnything ties ``lm_head.weight`` to the input token embedding after
    loading, so deleting the module does not delete the embedding needed by the
    RefHuman text path.  The important runtime change is that feature extraction
    calls the decoder directly and never materializes vocabulary logits.
    """
    base = get_eagle_base_model(model)
    language_model = getattr(base, "language_model", None)
    if language_model is None:
        return {"lm_head_numel": 0, "lm_head_tied": False}
    if bool(getattr(base, "generation_components_pruned", False)):
        return dict(getattr(base, "generation_prune_stats", {}))

    lm_head = getattr(language_model, "lm_head", None)
    lm_head_weight = getattr(lm_head, "weight", None)
    input_embeddings = language_model.get_input_embeddings()
    input_weight = getattr(input_embeddings, "weight", None)
    tied = bool(
        isinstance(lm_head_weight, torch.Tensor)
        and isinstance(input_weight, torch.Tensor)
        and lm_head_weight.data_ptr() == input_weight.data_ptr()
    )
    stats: dict[str, int | bool] = {
        "lm_head_numel": int(lm_head_weight.numel()) if isinstance(lm_head_weight, torch.Tensor) else 0,
        "lm_head_tied": tied,
    }
    language_model.lm_head = PrunedEagleLMHead()

    for candidate in (
        base,
        language_model,
        getattr(language_model, "model", None),
    ):
        cfg = getattr(candidate, "config", None)
        if cfg is not None and hasattr(cfg, "use_cache"):
            cfg.use_cache = False
        generation_cfg = getattr(candidate, "generation_config", None)
        if generation_cfg is not None and hasattr(generation_cfg, "use_cache"):
            generation_cfg.use_cache = False
    base.generation_components_pruned = True
    base.generation_prune_stats = stats
    language_model.generation_components_pruned = True
    return stats


def _load_eagle_without_generation_head(
    model_path: str,
    eagle_config,
    *,
    dtype: torch.dtype | None,
    text_attn_impl: str,
) -> nn.Module:
    """Load a local sharded checkpoint without reading duplicate lm_head data.

    The official checkpoint stores ``lm_head.weight`` even though it is tied to
    ``embed_tokens.weight``.  A temporary filtered index lets Transformers skip
    that 152681x2048 tensor while preserving the original checkpoint directory.
    Unsupported/non-local checkpoint layouts fall back to normal loading and
    are still pruned immediately afterwards.
    """
    from transformers import AutoModel

    root = Path(model_path).expanduser().resolve()
    index_path = root / "model.safetensors.index.json"
    auto_map = getattr(eagle_config, "auto_map", None) or {}
    model_ref = auto_map.get("AutoModel") or auto_map.get("AutoModelForCausalLM")
    if not index_path.is_file() or not model_ref:
        warnings.warn(
            "LocateAnything generation pruning could not filter the checkpoint index; "
            "loading the full checkpoint once before removing lm_head.",
            RuntimeWarning,
        )
        return AutoModel.from_pretrained(
            model_path,
            config=eagle_config,
            trust_remote_code=True,
            torch_dtype=dtype,
            attn_implementation=text_attn_impl,
        )

    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    with index_path.open("r", encoding="utf-8") as handle:
        checkpoint_index = json.load(handle)
    weight_map = checkpoint_index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise TypeError(f"Invalid weight_map in {index_path}")
    filtered_weight_map = {
        str(name): str(shard)
        for name, shard in weight_map.items()
        if str(name) != "language_model.lm_head.weight"
    }
    if len(filtered_weight_map) == len(weight_map):
        warnings.warn(
            f"No language_model.lm_head.weight entry found in {index_path}; loading normally.",
            RuntimeWarning,
        )
        return AutoModel.from_pretrained(
            model_path,
            config=eagle_config,
            trust_remote_code=True,
            torch_dtype=dtype,
            attn_implementation=text_attn_impl,
        )

    model_class = get_class_from_dynamic_module(str(model_ref), str(root))
    with tempfile.TemporaryDirectory(prefix="qwenpose-locate-no-generation-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        filtered_index = dict(checkpoint_index)
        filtered_index["weight_map"] = filtered_weight_map
        with (temp_dir / index_path.name).open("w", encoding="utf-8") as handle:
            json.dump(filtered_index, handle)
        for shard_name in sorted(set(filtered_weight_map.values())):
            shard_path = root / shard_name
            if not shard_path.is_file():
                raise FileNotFoundError(f"Missing LocateAnything weight shard: {shard_path}")
            (temp_dir / shard_name).symlink_to(shard_path)
        model = model_class.from_pretrained(
            str(temp_dir),
            config=eagle_config,
            local_files_only=True,
            torch_dtype=dtype,
            attn_implementation=text_attn_impl,
        )
    # Do not leak the deleted temporary directory into PEFT adapter metadata.
    model.config._name_or_path = str(root)
    model.name_or_path = str(root)
    return model


def find_eagle_lora_targets(model: nn.Module) -> tuple[list[str], dict[str, int], dict[str, int]]:
    """Find LLM and vision-tower modules for PEFT LoRA.

    LocateAnything-3B structure (after PEFT wrapping the top-level model):
      - base_model.model.vision_model.* : MoonViT encoder
      - base_model.model.language_model.* : Qwen2.5 LLM
      - base_model.model.mlp1.* : MLP projector (not targeted)
    """
    llm_suffixes = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    # LocateAnything ships a local MoonViT implementation whose attention uses
    # fused ``wqkv``/``wo`` projections and whose MLP uses ``fc0``/``fc1``.
    # These names are verified against model.safetensors.index.json.
    vision_suffixes = ("wqkv", "wo", "fc0", "fc1")
    targets: list[str] = []
    rank_pattern: dict[str, int] = {}
    alpha_pattern: dict[str, int] = {}

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        # LLM targets
        if "language_model" in name and name.endswith(llm_suffixes):
            targets.append(name)
        # Vision targets
        if "vision_model" in name and name.endswith(vision_suffixes):
            targets.append(name)
            rank_pattern[name] = -1
            alpha_pattern[name] = -1

    seen = set()
    unique = []
    for name in targets:
        if name not in seen:
            seen.add(name)
            unique.append(name)
    return unique, rank_pattern, alpha_pattern


def _set_eagle_vision_attention(model: nn.Module, attn_impl: str) -> None:
    base = get_eagle_base_model(model)
    vision_model = getattr(base, "vision_model", None)
    if vision_model is None:
        return
    vision_cfg = getattr(vision_model, "config", None)
    if vision_cfg is not None and hasattr(vision_cfg, "_attn_implementation"):
        vision_cfg._attn_implementation = attn_impl
    for module in vision_model.modules():
        if hasattr(module, "attn_implementation"):
            module.attn_implementation = attn_impl


class EagleVisionOnlyBackbone(nn.Module):
    """Minimal LocateAnything Stage-1 backbone without a language model."""

    def __init__(self, config, vision_model: nn.Module, mlp1: nn.Module) -> None:
        super().__init__()
        self.config = config
        self.vision_model = vision_model
        self.mlp1 = mlp1
        self.is_vision_only_backbone = True

    def extract_feature(
        self,
        pixel_values: torch.Tensor,
        image_grid_hws: torch.Tensor | np.ndarray | None,
    ) -> list[torch.Tensor] | torch.Tensor:
        return self.vision_model(pixel_values=pixel_values, grid_hws=image_grid_hws)

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_hws: torch.Tensor | np.ndarray | None = None,
        **_: object,
    ) -> list[torch.Tensor] | torch.Tensor:
        return self.extract_feature(pixel_values, image_grid_hws)


def _load_eagle_vision_projector_weights(
    model_path: str | Path,
    vision_model: nn.Module,
    mlp1: nn.Module,
) -> None:
    """Load only ``vision_model.*`` and ``mlp1.*`` tensors from sharded weights."""
    from safetensors import safe_open

    root = Path(model_path).expanduser()
    index_path = root / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(
            "Vision-only Locate loading requires a local sharded safetensors checkpoint with "
            f"model.safetensors.index.json, but it was not found under {root}."
        )
    with index_path.open("r", encoding="utf-8") as f:
        weight_map = json.load(f).get("weight_map", {})
    if not isinstance(weight_map, dict):
        raise TypeError(f"Invalid weight_map in {index_path}")

    module_specs = {
        "vision_model.": vision_model,
        "mlp1.": mlp1,
    }
    for prefix, module in module_specs.items():
        selected = {
            str(key): str(shard)
            for key, shard in weight_map.items()
            if str(key).startswith(prefix)
        }
        if not selected:
            raise KeyError(f"No {prefix} tensors were found in {index_path}")
        by_shard: dict[str, list[str]] = {}
        for key, shard in selected.items():
            by_shard.setdefault(shard, []).append(key)
        state: dict[str, torch.Tensor] = {}
        for shard, keys in by_shard.items():
            shard_path = root / shard
            if not shard_path.is_file():
                raise FileNotFoundError(f"Missing LocateAnything weight shard: {shard_path}")
            with safe_open(shard_path, framework="pt", device="cpu") as tensors:
                for key in keys:
                    state[key[len(prefix):]] = tensors.get_tensor(key)
        incompatible = module.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                f"Failed to load {prefix} selectively: missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )


def load_eagle_vision_only_with_lora(config: EagleLoRAConfig):
    """Load only MoonViT + ``mlp1`` and attach vision LoRA for Stage 1.

    The 3.4B-parameter language model is never instantiated and its checkpoint
    tensors are never read. Adapter parameter names intentionally match the
    full LocateAnything PEFT model so Stage-1 vision LoRA can be injected into
    Stage 2 with a normal ``load_state_dict(..., strict=False)`` call.
    """
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoConfig, AutoImageProcessor
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    model_path = str(Path(config.model_path).expanduser())
    requested_attn_impl = str(config.attn_implementation or "flash_attention_2")
    if requested_attn_impl not in ("flash_attention_2", "sdpa", "eager"):
        requested_attn_impl = "flash_attention_2"

    eagle_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    eagle_config.vision_config._attn_implementation = requested_attn_impl
    vision_cls = get_class_from_dynamic_module(
        "modeling_vit.MoonVitPretrainedModel",
        model_path,
    )
    dtype = _dtype_from_name(config.dtype)
    vision_model = vision_cls(eagle_config.vision_config).to(dtype=dtype)
    vision_hidden = int(eagle_config.vision_config.hidden_size)
    output_hidden = int(eagle_config.text_config.hidden_size)
    mlp1 = nn.Sequential(
        nn.LayerNorm(vision_hidden * 4),
        nn.Linear(vision_hidden * 4, output_hidden),
        nn.GELU(),
        nn.Linear(output_hidden, output_hidden),
    ).to(dtype=dtype)
    _load_eagle_vision_projector_weights(model_path, vision_model, mlp1)

    wrapper = EagleVisionOnlyBackbone(eagle_config, vision_model, mlp1)
    _set_eagle_vision_attention(wrapper, requested_attn_impl)
    for param in wrapper.parameters():
        param.requires_grad = False

    targets, _, _ = find_eagle_lora_targets(wrapper)
    targets = [name for name in targets if "vision_model" in name]
    if not targets:
        raise RuntimeError("No MoonViT LoRA target modules were found in the vision-only backbone.")
    lora_config = LoraConfig(
        r=config.vision_lora_r,
        lora_alpha=config.vision_lora_alpha,
        lora_dropout=config.vision_lora_dropout,
        target_modules=targets,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    model = get_peft_model(wrapper, lora_config)
    _set_eagle_vision_attention(model, requested_attn_impl)
    processor = AutoImageProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False,
    )
    setattr(processor, "_qwenpose_vision_only", True)
    if config.gradient_checkpointing:
        _enable_moonvit_gradient_checkpointing(model)
    return model, processor


def load_eagle_with_lora(config: EagleLoRAConfig):
    """Load LocateAnything-3B with LoRA on LLM and vision encoder.

    Returns (model, processor) where model has frozen base weights + trainable LoRA.
    """
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoConfig, AutoModel, AutoProcessor

    model_path = str(Path(config.model_path).expanduser())

    # LocateAnything's Qwen2 decoder has custom block-mask preparation that is
    # stable with sdpa here, while MoonViT can use flash_attention_2 for packed
    # variable-length visual tokens. This mirrors the Qwen3VL-style training
    # shape: batched vision forward without dense sdpa masks over summed tokens.
    requested_attn_impl = str(config.attn_implementation or "flash_attention_2")
    if requested_attn_impl not in ("flash_attention_2", "sdpa", "eager"):
        requested_attn_impl = "flash_attention_2"
    text_attn_impl = "sdpa"
    vision_attn_impl = requested_attn_impl

    # Eagle's _autoset_attn_implementation intercepts attn_implementation when
    # config._attn_implementation is already 'magi' (from config.json). We must
    # explicitly override it before loading the model.
    eagle_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    eagle_config._attn_implementation = text_attn_impl
    # Also propagate to sub-configs
    if hasattr(eagle_config, "text_config"):
        eagle_config.text_config._attn_implementation = text_attn_impl
    if hasattr(eagle_config, "vision_config"):
        eagle_config.vision_config._attn_implementation = vision_attn_impl

    model_dtype = _dtype_from_name(config.dtype)
    if config.prune_generation:
        model = _load_eagle_without_generation_head(
            model_path,
            eagle_config,
            dtype=model_dtype,
            text_attn_impl=text_attn_impl,
        )
        prune_eagle_generation_components(model)
    else:
        model = AutoModel.from_pretrained(
            model_path,
            config=eagle_config,
            trust_remote_code=True,
            torch_dtype=model_dtype,
            attn_implementation=text_attn_impl,
        )
    _set_eagle_vision_attention(model, vision_attn_impl)
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False,
        fix_mistral_regex=True,
    )

    # Freeze all base parameters
    for param in model.parameters():
        param.requires_grad = False

    targets, rank_pattern, alpha_pattern = find_eagle_lora_targets(model)
    if not targets:
        raise RuntimeError("No Eagle LoRA target modules were found.")
    for name in list(rank_pattern):
        rank_pattern[name] = config.vision_lora_r
        alpha_pattern[name] = config.vision_lora_alpha

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=targets,
        rank_pattern=rank_pattern or None,
        alpha_pattern=alpha_pattern or None,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    _set_eagle_vision_attention(model, vision_attn_impl)

    # PEFT does not need the vocabulary projection for feature-extraction LoRA,
    # but keep the invariant explicit in case a future PEFT version reties it.
    if config.prune_generation:
        prune_eagle_generation_components(model)

    if config.gradient_checkpointing:
        _enable_gradient_checkpointing(model)

    return model, processor


def _enable_moonvit_gradient_checkpointing(model: nn.Module) -> None:
    """Enable project-local block checkpointing for MoonViT.

    LocateAnything's remote-code MoonVitPretrainedModel does not declare the
    Transformers gradient-checkpointing contract. The extractor therefore
    checkpoints encoder blocks explicitly instead of calling the unsupported
    ``gradient_checkpointing_enable`` method.
    """
    base = get_eagle_base_model(model)
    vision_model = getattr(base, "vision_model", None)
    if vision_model is not None:
        setattr(vision_model, "_qwenpose_gradient_checkpointing", True)


def _enable_gradient_checkpointing(model: nn.Module) -> None:
    base = get_eagle_base_model(model)
    _enable_moonvit_gradient_checkpointing(model)
    for candidate in (model, base, getattr(base, "language_model", None)):
        if candidate is None:
            continue
        cfg = getattr(candidate, "config", None)
        if cfg is not None and hasattr(cfg, "use_cache"):
            cfg.use_cache = False
    language_model = getattr(base, "language_model", None)
    enable_fn = getattr(language_model, "gradient_checkpointing_enable", None)
    if enable_fn is not None:
        try:
            enable_fn(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            enable_fn()
    input_grad_fn = getattr(model, "enable_input_require_grads", None)
    if input_grad_fn is None:
        input_grad_fn = getattr(base, "enable_input_require_grads", None)
    if input_grad_fn is not None:
        input_grad_fn()


def get_eagle_base_model(model: nn.Module) -> nn.Module:
    """Unwrap PEFT to get the base LocateAnything model."""
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def eagle_hidden_size(model: nn.Module) -> int:
    """Return LLM hidden size (2048 for Qwen2.5-3B)."""
    base = get_eagle_base_model(model)
    cfg = getattr(base, "config", None)
    if cfg is not None and hasattr(cfg, "text_config"):
        return int(cfg.text_config.hidden_size)
    return 2048


def count_eagle_lora_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def _locate_image_token_limit(
    image_token_limit: int | None = None,
) -> int | None:
    if image_token_limit is not None and int(image_token_limit) > 0:
        return int(image_token_limit)
    return None


def _load_eagle_images(
    image_paths: list[str],
    image_tensors: list[torch.Tensor] | None,
) -> list[Image.Image]:
    images: list[Image.Image] = []
    if image_tensors is not None:
        if len(image_tensors) != len(image_paths):
            raise ValueError(
                f"image_tensors length {len(image_tensors)} does not match image_paths {len(image_paths)}"
            )
        for tensor in image_tensors:
            if tensor.ndim != 3 or int(tensor.shape[0]) != 3:
                raise ValueError(f"Expected CHW vision image tensor, got {tuple(tensor.shape)}")
            array = tensor.detach().cpu()
            if array.dtype != torch.uint8:
                array = array.clamp(0, 255).to(torch.uint8)
            images.append(Image.fromarray(array.permute(1, 2, 0).contiguous().numpy(), mode="RGB"))
        return images
    for image_path in image_paths:
        with Image.open(image_path) as image:
            images.append(image.convert("RGB").copy())
    return images


def _process_eagle_texts(
    processor,
    images: list[Image.Image],
    texts: list[str] | None,
    device: torch.device,
    *,
    image_token_limit: int | None,
    batch_token_limit: int | None = None,
) -> dict[str, torch.Tensor]:
    is_vision_only = bool(getattr(processor, "_qwenpose_vision_only", False))
    image_processor = getattr(processor, "image_processor", processor)
    token_limit = _locate_image_token_limit(image_token_limit=image_token_limit)
    if batch_token_limit is not None and int(batch_token_limit) > 0 and images:
        per_image_batch_limit = max(int(int(batch_token_limit) / len(images) * 0.875), 64)
        token_limit = per_image_batch_limit if token_limit is None else min(int(token_limit), per_image_batch_limit)
    old_token_limit = getattr(image_processor, "in_token_limit", None) if image_processor is not None else None
    if image_processor is not None and token_limit is not None:
        image_processor.in_token_limit = int(token_limit)
    try:
        if is_vision_only:
            inputs = image_processor(images=images, return_tensors="pt")
        else:
            if texts is None:
                raise ValueError("Multimodal Locate input construction requires text templates.")
            inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
    finally:
        if image_processor is not None and old_token_limit is not None:
            image_processor.in_token_limit = old_token_limit
    result = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.to(device)
        elif isinstance(value, np.ndarray):
            result[key] = torch.from_numpy(value).to(device)
        else:
            result[key] = value
    return result


def _eagle_user_messages(prompt: str) -> list[dict[str, object]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": str(prompt)},
            ],
        }
    ]


def _prefix_mask_from_attention(
    attention_mask: torch.Tensor,
    prefix_lengths: torch.Tensor,
) -> torch.Tensor:
    active = attention_mask.bool()
    if active.numel() == 0:
        return torch.zeros_like(active)
    ranks = torch.cumsum(active.to(dtype=torch.long), dim=1)
    lengths = prefix_lengths.to(device=attention_mask.device, dtype=torch.long).clamp(min=0)[:, None]
    return active & ranks.le(lengths)


def build_eagle_inputs(
    processor,
    image_paths: list[str],
    prompts: list[str] | None,
    device: torch.device,
    image_token_limit: int | None = None,
    batch_token_limit: int | None = None,
    image_tensors: list[torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Build LocateAnything processor inputs from image paths and task prompts."""
    images = _load_eagle_images(image_paths, image_tensors)
    texts = None
    if not bool(getattr(processor, "_qwenpose_vision_only", False)):
        if prompts is None:
            raise ValueError("Multimodal Locate input construction requires prompts.")
        texts = [
            processor.apply_chat_template(
                _eagle_user_messages(prompt),
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]
    return _process_eagle_texts(
        processor,
        images,
        texts,
        device,
        image_token_limit=image_token_limit,
        batch_token_limit=batch_token_limit,
    )


def build_eagle_lm_inputs(
    processor,
    image_paths,
    prompts,
    responses,
    device,
    image_token_limit=None,
    batch_token_limit=None,
    image_tensors=None,
):
    """Build Locate teacher-forcing inputs with only assistant answer tokens supervised."""
    if bool(getattr(processor, "_qwenpose_vision_only", False)):
        raise RuntimeError(
            "Locate grounding LM supervision requires the full LocateAnything "
            "model; vision-only loading is not supported."
        )
    if not (len(image_paths) == len(prompts) == len(responses)):
        raise ValueError("image_paths, prompts, and responses must have identical lengths.")
    images = _load_eagle_images(list(image_paths), image_tensors)
    prompt_messages = [_eagle_user_messages(str(prompt)) for prompt in prompts]
    full_messages = [
        [
            user_messages[0],
            {"role": "assistant", "content": str(response)},
        ]
        for user_messages, response in zip(prompt_messages, responses)
    ]
    prompt_texts = [
        processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        for messages in prompt_messages
    ]
    full_texts = [
        processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        for messages in full_messages
    ]
    prompt_inputs = _process_eagle_texts(
        processor,
        images,
        prompt_texts,
        device,
        image_token_limit=image_token_limit,
        batch_token_limit=batch_token_limit,
    )
    full_inputs = _process_eagle_texts(
        processor,
        images,
        full_texts,
        device,
        image_token_limit=image_token_limit,
        batch_token_limit=batch_token_limit,
    )
    prompt_attention = prompt_inputs.get("attention_mask")
    if prompt_attention is None:
        prompt_attention = torch.ones_like(prompt_inputs["input_ids"], dtype=torch.long)
    full_attention = full_inputs.get("attention_mask")
    if full_attention is None:
        full_attention = torch.ones_like(full_inputs["input_ids"], dtype=torch.long)
        full_inputs["attention_mask"] = full_attention
    prompt_lengths = prompt_attention.sum(dim=1).to(dtype=torch.long)
    prefix_mask = _prefix_mask_from_attention(full_attention, prompt_lengths)
    labels = full_inputs["input_ids"].clone()
    labels = labels.masked_fill(~full_attention.bool(), -100)
    labels = labels.masked_fill(prefix_mask, -100)

    for row in range(labels.shape[0]):
        prompt_ids = prompt_inputs["input_ids"][row][prompt_attention[row].bool()]
        full_ids = full_inputs["input_ids"][row][full_attention[row].bool()]
        if int(full_ids.numel()) < int(prompt_ids.numel()) or not torch.equal(
            full_ids[: int(prompt_ids.numel())], prompt_ids
        ):
            raise RuntimeError(
                "Locate chat-template prefix mismatch: assistant response cannot be masked safely."
            )
        if not labels[row].ne(-100).any():
            raise RuntimeError("Locate grounding response produced no supervised assistant tokens.")
    full_inputs["labels"] = labels
    return full_inputs


class EagleFeatureExtractor(nn.Module):
    """Extract dense visual features and pooled text features from LocateAnything-3B.

    Replicates the model's forward pass to get intermediate features:
    1. Run vision encoder (extract_feature) -> List[Tensor] per image
    2. Project via mlp1 -> (num_tokens, 2048)
    3. Replace image tokens in input embeddings
    4. Run LLM -> last_hidden_state
    5. Extract image token hidden states -> spatial feature map
    6. Mean-pool non-image tokens -> text embedding
    """

    def __init__(
        self,
        eagle_model: nn.Module,
        refiner_layers: int = 0,
        refiner_bottleneck_dim: int = 256,
        refiner_init_scale: float = 0.1,
        feature_source: str = "raw_visual",
    ) -> None:
        super().__init__()
        self.eagle_model = eagle_model
        self.feature_source = str(feature_source)
        if self.feature_source not in {"vision_only", "raw_visual"}:
            raise ValueError(
                f"Unsupported Locate feature source {self.feature_source!r}; "
                "expected 'vision_only' or 'raw_visual'."
            )
        hidden_size = eagle_hidden_size(eagle_model)
        self.feature_refiner = QwenFeatureRefiner(
            hidden_size,
            num_layers=refiner_layers,
            bottleneck_dim=refiner_bottleneck_dim,
            init_scale=refiner_init_scale,
        )

    def forward(
        self,
        eagle_inputs: dict[str, torch.Tensor],
        freeze_eagle: bool = False,
        require_text: bool = True,
    ) -> tuple[MultiScaleSpatialFeatureBatch, torch.Tensor]:
        # P2 is the true MoonViT pre-merger grid. P3 is the existing projected
        # post-merger Locate grid. Only P3 keeps the legacy feature refiner;
        # PoseHead owns the independent P2/P3 channel projections.
        if self.feature_source == "vision_only" or not bool(require_text):
            raw_levels, text_embed = self._extract_eagle_vision_features(
                eagle_inputs,
                freeze_backbone=bool(freeze_eagle),
            )
        else:
            raw_levels, text_embed = self._extract_eagle_feature_maps(
                eagle_inputs,
                freeze_backbone=bool(freeze_eagle),
            )
        p2, p3 = raw_levels.levels
        p3 = p3.map_samples(self.feature_refiner)
        return MultiScaleSpatialFeatureBatch((p2, p3)), text_embed

    def forward_with_vision_cache(
        self,
        eagle_inputs: dict[str, torch.Tensor],
        freeze_eagle: bool = False,
        require_text: bool = True,
    ) -> tuple[MultiScaleSpatialFeatureBatch, torch.Tensor, list[torch.Tensor]]:
        """Extract P2/P3 pose features and retain projected P3 tokens for the LM."""
        _, input_ids, attention_mask, pixel_values, image_grid_hws = self._prepare_locate_inputs(
            eagle_inputs
        )

        def extract() -> tuple[MultiScaleSpatialFeatureBatch, torch.Tensor, list[torch.Tensor]]:
            premerge_list, _, projected_vit_list, projected_vit = (
                self.run_multiscale_vision_tokens(pixel_values, image_grid_hws)
            )
            p2_maps = self.build_premerge_feature_maps(image_grid_hws, premerge_list)
            if self.feature_source == "vision_only" or not bool(require_text):
                p3_maps = self.build_raw_feature_maps(image_grid_hws, projected_vit_list)
                hidden_size = int(p3_maps.shape[1])
                text_embed = p3_maps.new_zeros((len(projected_vit_list), hidden_size))
            else:
                if input_ids is None:
                    raise ValueError("Multimodal Locate feature extraction requires input_ids.")
                hidden = self.run_language_hidden(
                    input_ids,
                    attention_mask,
                    projected_vit,
                )
                p3_maps, text_embed = self.build_feature_maps(
                    input_ids,
                    attention_mask,
                    image_grid_hws,
                    projected_vit_list,
                    hidden,
                )
            return MultiScaleSpatialFeatureBatch((p2_maps, p3_maps)), text_embed, projected_vit_list

        if freeze_eagle:
            with torch.no_grad():
                raw_levels, text_embed, projected_vit_list = extract()
            raw_levels = raw_levels.detach()
            text_embed = text_embed.detach()
            projected_vit_list = [tokens.detach() for tokens in projected_vit_list]
        else:
            raw_levels, text_embed, projected_vit_list = extract()
        p2, p3 = raw_levels.levels
        p3 = p3.map_samples(self.feature_refiner)
        return MultiScaleSpatialFeatureBatch((p2, p3)), text_embed, projected_vit_list

    def _prepare_locate_inputs(
        self,
        eagle_inputs: dict[str, torch.Tensor],
    ) -> tuple[nn.Module, torch.Tensor | None, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        base = get_eagle_base_model(self.eagle_model)
        input_ids = eagle_inputs.get("input_ids")
        attention_mask = eagle_inputs.get("attention_mask")
        pixel_values = eagle_inputs["pixel_values"]
        image_grid_hws = eagle_inputs.get("image_grid_hws")

        model_dtype = next(base.parameters()).dtype
        if pixel_values.dtype != model_dtype:
            pixel_values = pixel_values.to(dtype=model_dtype)
        if image_grid_hws is not None and isinstance(image_grid_hws, torch.Tensor):
            image_grid_hws = image_grid_hws.to(device=pixel_values.device, dtype=torch.long)
        return base, input_ids, attention_mask, pixel_values, image_grid_hws

    def run_multiscale_vision_tokens(
        self,
        pixel_values: torch.Tensor,
        image_grid_hws: torch.Tensor | None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        """Run MoonViT once and expose true pre-merger P2 plus legacy merged P3."""
        if image_grid_hws is None:
            raise ValueError("MoonViT multi-scale extraction requires image_grid_hws.")
        base = get_eagle_base_model(self.eagle_model)
        vision_model = base.vision_model
        grid_rows = image_grid_hws.detach().cpu().tolist()
        merge_kernel = getattr(vision_model, "merge_kernel_size", (2, 2))
        merge_h = max(int(merge_kernel[0]), 1)
        merge_w = max(int(merge_kernel[1]), 1)
        premerge_list: list[torch.Tensor] = []
        merged_list: list[torch.Tensor] = []

        if hasattr(vision_model, "patch_embed") and hasattr(vision_model, "encoder"):
            hidden_states = vision_model.patch_embed(pixel_values, image_grid_hws)
            encoder = vision_model.encoder
            use_checkpointing = bool(
                getattr(vision_model, "_qwenpose_gradient_checkpointing", False)
                and vision_model.training
                and torch.is_grad_enabled()
                and hasattr(encoder, "blocks")
                and hasattr(encoder, "rope_2d")
                and hasattr(encoder, "final_layernorm")
            )
            if use_checkpointing:
                rope_freqs_cis = encoder.rope_2d.get_freqs_cis(
                    grid_hws=image_grid_hws
                )
                lengths = torch.cat(
                    (
                        torch.zeros(
                            1,
                            device=hidden_states.device,
                            dtype=image_grid_hws.dtype,
                        ),
                        image_grid_hws[:, 0] * image_grid_hws[:, 1],
                    )
                )
                cu_seqlens = lengths.cumsum(dim=0, dtype=torch.int32)
                for block in encoder.blocks:
                    def run_block(states: torch.Tensor, block=block) -> torch.Tensor:
                        return block(
                            states,
                            cu_seqlens,
                            rope_freqs_cis=rope_freqs_cis,
                        )

                    hidden_states = torch_checkpoint(
                        run_block,
                        hidden_states,
                        use_reentrant=False,
                    )
                hidden_states = encoder.final_layernorm(hidden_states)
            else:
                hidden_states = encoder(hidden_states, image_grid_hws)
            offset = 0
            for raw_h, raw_w in grid_rows:
                raw_h = int(raw_h)
                raw_w = int(raw_w)
                count = raw_h * raw_w
                tokens = hidden_states[offset : offset + count]
                if int(tokens.shape[0]) != count:
                    raise ValueError(
                        "MoonViT pre-merger token/grid mismatch: "
                        f"tokens={int(tokens.shape[0])}, grid={raw_h}x{raw_w}."
                    )
                if raw_h % merge_h != 0 or raw_w % merge_w != 0:
                    raise ValueError(
                        "MoonViT grid must be divisible by its spatial merger: "
                        f"grid={raw_h}x{raw_w}, merger={merge_h}x{merge_w}."
                    )
                premerge_list.append(tokens)
                merged = tokens.view(
                    raw_h // merge_h,
                    merge_h,
                    raw_w // merge_w,
                    merge_w,
                    int(tokens.shape[-1]),
                )
                merged = merged.permute(0, 2, 1, 3, 4).contiguous().view(
                    (raw_h // merge_h) * (raw_w // merge_w), -1
                )
                merged_list.append(merged)
                offset += count
            if offset != int(hidden_states.shape[0]):
                raise ValueError(
                    "MoonViT batch token/grid mismatch after splitting: "
                    f"consumed={offset}, total={int(hidden_states.shape[0])}."
                )
        else:
            # Some lightweight wrappers expose only extract_feature(). The
            # Locate merger is a lossless 2x2 channel concatenation, so invert
            # it exactly to recover the pre-merger grid instead of upsampling.
            merged_output = base.extract_feature(pixel_values, image_grid_hws)
            merged_list = (
                list(merged_output)
                if isinstance(merged_output, (list, tuple))
                else [merged_output]
            )
            if len(merged_list) != len(grid_rows):
                raise ValueError("MoonViT merged feature batch/grid count mismatch.")
            for merged, (raw_h, raw_w) in zip(merged_list, grid_rows):
                raw_h = int(raw_h)
                raw_w = int(raw_w)
                merged_h = raw_h // merge_h
                merged_w = raw_w // merge_w
                channels = int(merged.shape[-1])
                kernel_area = merge_h * merge_w
                if channels % kernel_area != 0:
                    raise ValueError("Cannot invert MoonViT spatial merger channel layout.")
                hidden_dim = channels // kernel_area
                premerge = merged.view(
                    merged_h, merged_w, merge_h, merge_w, hidden_dim
                ).permute(0, 2, 1, 3, 4).contiguous().view(
                    raw_h * raw_w, hidden_dim
                )
                premerge_list.append(premerge)
        projected_vit_list = [base.mlp1(tokens) for tokens in merged_list]
        projected_vit = torch.cat(projected_vit_list, dim=0)
        return premerge_list, merged_list, projected_vit_list, projected_vit

    def run_vision_tokens(
        self,
        pixel_values: torch.Tensor,
        image_grid_hws: torch.Tensor | None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        _, merged_list, projected_vit_list, projected_vit = (
            self.run_multiscale_vision_tokens(pixel_values, image_grid_hws)
        )
        return merged_list, projected_vit_list, projected_vit

    def _ensure_safe_image_processing(self, lm: nn.Module) -> None:
        qwen_model = getattr(lm, "model", None)
        if qwen_model is None or getattr(qwen_model, "_qwenpose_safe_image_processing", False):
            return

        def safe_image_processing(model_self, input_ids_arg, visual_features_arg, image_token_index_arg):
            input_embeds = model_self.get_input_embeddings()(input_ids_arg)
            if visual_features_arg is None:
                return input_embeds
            batch_size, seq_len, hidden_dim = input_embeds.shape
            flat_embeds = input_embeds.reshape(batch_size * seq_len, hidden_dim).clone()
            flat_ids = input_ids_arg.reshape(batch_size * seq_len)
            selected = flat_ids == int(image_token_index_arg)
            selected_count = int(selected.sum().item())
            visual_flat = visual_features_arg.reshape(-1, hidden_dim).to(
                device=flat_embeds.device,
                dtype=flat_embeds.dtype,
            )
            if int(visual_flat.shape[0]) != selected_count:
                raise ValueError(
                    "LocateAnything image token count mismatch before Qwen2 injection: "
                    f"image_tokens={selected_count}, visual_features={int(visual_flat.shape[0])}. "
                    "The visual feature order is preserved as row-major input_ids order; "
                    "this error means the processor placeholders and MoonViT outputs disagree."
                )
            flat_embeds[selected] = visual_flat
            return flat_embeds.reshape(batch_size, seq_len, hidden_dim)

        qwen_model.image_processing = types.MethodType(safe_image_processing, qwen_model)
        qwen_model._qwenpose_safe_image_processing = True

    def run_language_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        projected_visual_tokens: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        base = get_eagle_base_model(self.eagle_model)
        image_token_id = int(base.image_token_index)
        lm = base.language_model
        self._ensure_safe_image_processing(lm)
        was_training = lm.training
        lm.eval()
        decoder = getattr(lm, "model", None)
        if decoder is None:
            raise RuntimeError("LocateAnything language_model has no decoder core.")
        lm_outputs = decoder(
            input_ids=input_ids,
            visual_features=projected_visual_tokens,
            image_token_index=image_token_id,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        if was_training:
            lm.train()
        return lm_outputs.last_hidden_state

    def run_language_prefill(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        projected_visual_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, object]:
        base = get_eagle_base_model(self.eagle_model)
        if eagle_generation_is_pruned(self.eagle_model):
            raise RuntimeError(
                "LocateAnything KV-cache prefill is unavailable because generation components were pruned."
            )
        image_token_id = int(base.image_token_index)
        lm = base.language_model
        self._ensure_safe_image_processing(lm)
        was_training = lm.training
        lm.eval()
        lm_outputs = lm(
            input_ids=input_ids,
            visual_features=projected_visual_tokens,
            image_token_index=image_token_id,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
        )
        if was_training:
            lm.train()
        past_key_values = lm_outputs.past_key_values
        if hasattr(past_key_values, "to_legacy_cache"):
            past_key_values = past_key_values.to_legacy_cache()
        return lm_outputs.hidden_states[-1], past_key_values

    def build_premerge_feature_maps(
        self,
        image_grid_hws: torch.Tensor | np.ndarray | None,
        premerge_vit_list: list[torch.Tensor],
    ) -> SpatialFeatureBatch:
        """Restore the true MoonViT encoder grid before the 2x2 spatial merger."""
        if isinstance(image_grid_hws, torch.Tensor):
            grid_hws_np = image_grid_hws.detach().cpu().numpy()
        elif isinstance(image_grid_hws, np.ndarray):
            grid_hws_np = image_grid_hws
        else:
            grid_hws_np = None
        if grid_hws_np is None:
            raise ValueError("Pre-merger feature maps require image_grid_hws.")
        maps: list[torch.Tensor] = []
        for batch_idx, tokens in enumerate(premerge_vit_list):
            if batch_idx >= len(grid_hws_np):
                raise ValueError("Missing MoonViT grid shape for a pre-merger feature sample.")
            raw_h = max(int(grid_hws_np[batch_idx][0]), 1)
            raw_w = max(int(grid_hws_np[batch_idx][1]), 1)
            expected = raw_h * raw_w
            if int(tokens.shape[0]) != expected:
                raise ValueError(
                    "MoonViT pre-merger token/grid mismatch: "
                    f"sample={batch_idx}, tokens={int(tokens.shape[0])}, "
                    f"grid={raw_h}x{raw_w}."
                )
            maps.append(tokens.float().view(raw_h, raw_w, -1).permute(2, 0, 1))
        return SpatialFeatureBatch.from_maps(maps)

    def build_raw_feature_maps(
        self,
        image_grid_hws: torch.Tensor | np.ndarray | None,
        projected_vit_list: list[torch.Tensor],
    ) -> SpatialFeatureBatch:
        base = get_eagle_base_model(self.eagle_model)
        if isinstance(image_grid_hws, torch.Tensor):
            grid_hws_np = image_grid_hws.detach().cpu().numpy()
        elif isinstance(image_grid_hws, np.ndarray):
            grid_hws_np = image_grid_hws
        else:
            grid_hws_np = None
        merge_kernel = getattr(getattr(base, "vision_model", None), "merge_kernel_size", (2, 2))
        merge_h = max(int(merge_kernel[0]), 1) if isinstance(merge_kernel, (tuple, list)) else max(int(merge_kernel), 1)
        merge_w = max(int(merge_kernel[1]), 1) if isinstance(merge_kernel, (tuple, list)) else max(int(merge_kernel), 1)
        raw_maps: list[torch.Tensor] = []
        for batch_idx, raw_tokens in enumerate(projected_vit_list):
            if grid_hws_np is not None and batch_idx < len(grid_hws_np):
                raw_h = max(int(grid_hws_np[batch_idx][0]), 1)
                raw_w = max(int(grid_hws_np[batch_idx][1]), 1)
                h = max(raw_h // merge_h, 1)
                w = max(raw_w // merge_w, 1)
            else:
                n = int(raw_tokens.shape[0])
                h = max(int(round(n ** 0.5)), 1)
                w = max(n // h, 1)
            expected = h * w
            if int(raw_tokens.shape[0]) != expected:
                raise ValueError(
                    "LocateAnything raw visual token/grid mismatch: "
                    f"sample={batch_idx}, raw_tokens={int(raw_tokens.shape[0])}, "
                    f"merged_grid={h}x{w}, "
                    f"raw_grid={None if grid_hws_np is None else grid_hws_np[batch_idx].tolist()}, "
                    f"merge_kernel={merge_h}x{merge_w}."
                )
            raw_maps.append(raw_tokens.float().view(h, w, -1).permute(2, 0, 1))
        return SpatialFeatureBatch.from_maps(raw_maps)

    def build_feature_maps(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        image_grid_hws: torch.Tensor | np.ndarray | None,
        projected_vit_list: list[torch.Tensor],
        hidden: torch.Tensor,
    ) -> tuple[SpatialFeatureBatch, torch.Tensor]:
        base = get_eagle_base_model(self.eagle_model)
        image_token_id = int(base.image_token_index)
        image_mask = input_ids == image_token_id
        batch_size = int(input_ids.shape[0])
        if len(projected_vit_list) != batch_size:
            raise ValueError(
                "LocateAnything batch/image count mismatch: "
                f"projected_images={len(projected_vit_list)}, batch_size={batch_size}."
            )

        if isinstance(image_grid_hws, torch.Tensor):
            grid_hws_np = image_grid_hws.detach().cpu().numpy()
        elif isinstance(image_grid_hws, np.ndarray):
            grid_hws_np = image_grid_hws
        else:
            grid_hws_np = None
        merge_kernel = getattr(getattr(base, "vision_model", None), "merge_kernel_size", (2, 2))
        merge_h = max(int(merge_kernel[0]), 1) if isinstance(merge_kernel, (tuple, list)) else max(int(merge_kernel), 1)
        merge_w = max(int(merge_kernel[1]), 1) if isinstance(merge_kernel, (tuple, list)) else max(int(merge_kernel), 1)

        visual_maps = []
        for batch_idx in range(batch_size):
            lm_tokens = hidden[batch_idx][image_mask[batch_idx]]
            raw_tokens = projected_vit_list[batch_idx]
            if grid_hws_np is not None and batch_idx < len(grid_hws_np):
                raw_h = max(int(grid_hws_np[batch_idx][0]), 1)
                raw_w = max(int(grid_hws_np[batch_idx][1]), 1)
                h = max(raw_h // merge_h, 1)
                w = max(raw_w // merge_w, 1)
            else:
                n = int(min(lm_tokens.shape[0], raw_tokens.shape[0]))
                h = w = max(int(n ** 0.5), 1)
            expected = h * w
            if int(lm_tokens.shape[0]) != expected or int(raw_tokens.shape[0]) != expected:
                raise ValueError(
                    "LocateAnything visual token/grid mismatch: "
                    f"sample={batch_idx}, lm_tokens={int(lm_tokens.shape[0])}, "
                    f"raw_tokens={int(raw_tokens.shape[0])}, merged_grid={h}x{w}, "
                    f"raw_grid={None if grid_hws_np is None else grid_hws_np[batch_idx].tolist()}, "
                    f"merge_kernel={merge_h}x{merge_w}."
                )

            raw_map = raw_tokens.float().view(h, w, -1).permute(2, 0, 1)
            visual_maps.append(raw_map)
        raw_maps = SpatialFeatureBatch.from_maps(visual_maps)

        non_image = ~image_mask
        if attention_mask is not None:
            non_image = non_image & attention_mask.bool()
        text_mask = non_image.float().unsqueeze(-1)
        text_embed = (hidden * text_mask).sum(dim=1) / text_mask.sum(dim=1).clamp(min=1.0)
        return raw_maps, text_embed

    @staticmethod
    def _cache_seq_len(past_key_values: object) -> int:
        if hasattr(past_key_values, "get_seq_length"):
            return int(past_key_values.get_seq_length())
        return int(past_key_values[0][0].size(2))  # type: ignore[index]

    @staticmethod
    def _truncate_legacy_cache(past_key_values: object, length: int) -> object:
        if hasattr(past_key_values, "to_legacy_cache"):
            past_key_values = past_key_values.to_legacy_cache()
        return tuple(
            (kv[0][:, :, :length, :], kv[1][:, :, :length, :])
            for kv in past_key_values  # type: ignore[union-attr]
        )

    def generate_response_with_cached_features(
        self,
        eagle_inputs: dict[str, torch.Tensor],
        tokenizer,
        *,
        max_new_tokens: int,
        generation_mode: str = "hybrid",
        n_future_tokens: int = 6,
        **generate_kwargs,
    ) -> tuple[str, SpatialFeatureBatch, torch.Tensor]:
        """Generate LocateAnything text while reusing the prompt prefill features.

        This mirrors LocateAnythingForConditionalGeneration.generate(), but the
        prompt prefill is run once with output_hidden_states=True. The same hidden
        states become the PoseHead visual/text features, and the returned KV cache
        is used to continue box generation without a second LocateAnything pass.
        """
        if eagle_generation_is_pruned(self.eagle_model):
            raise RuntimeError(
                "LocateAnything cached autoregressive generation is unavailable because "
                "generation components were pruned."
            )
        base = get_eagle_base_model(self.eagle_model)
        token_ids = getattr(base, "token_ids", None)
        generate_globals = getattr(base.generate, "__globals__", {})
        sample_tokens = generate_globals.get("sample_tokens")
        handle_pattern = generate_globals.get("handle_pattern")
        get_token_ids = generate_globals.get("get_token_ids_from_config")
        module_name = getattr(type(base), "__module__", "")
        if sample_tokens is None or handle_pattern is None or get_token_ids is None:
            try:
                modeling_module = importlib.import_module(module_name)
                sample_tokens = sample_tokens or getattr(modeling_module, "sample_tokens", None)
                handle_pattern = handle_pattern or getattr(modeling_module, "handle_pattern", None)
                get_token_ids = get_token_ids or getattr(modeling_module, "get_token_ids_from_config", None)
            except Exception:
                pass
        if (sample_tokens is None or handle_pattern is None or get_token_ids is None) and "." in module_name:
            try:
                generate_utils_module = importlib.import_module(module_name.rsplit(".", 1)[0] + ".generate_utils")
                sample_tokens = sample_tokens or getattr(generate_utils_module, "sample_tokens", None)
                handle_pattern = handle_pattern or getattr(generate_utils_module, "handle_pattern", None)
                get_token_ids = get_token_ids or getattr(generate_utils_module, "get_token_ids_from_config", None)
            except Exception:
                pass
        if token_ids is None:
            if get_token_ids is None:
                raise RuntimeError("LocateAnything token_ids are unavailable for cached generation.")
            token_ids = get_token_ids(base.config)
        if sample_tokens is None or handle_pattern is None:
            raise RuntimeError("LocateAnything generation helpers are unavailable for cached generation.")

        _, input_ids, attention_mask, pixel_values, image_grid_hws = self._prepare_locate_inputs(eagle_inputs)
        batch_size, seq_len = input_ids.shape
        if batch_size != 1:
            raise ValueError("cached LocateAnything generation currently expects batch size 1.")
        if tokenizer is None:
            raise ValueError("LocateAnything tokenizer is required for cached generation.")
        generation_mode = str(generation_mode or "hybrid")
        if generation_mode not in {"fast", "slow", "hybrid"}:
            raise ValueError(f"Unsupported generation_mode={generation_mode!r}.")

        with torch.inference_mode():
            _, projected_vit_list, projected_vit = self.run_vision_tokens(pixel_values, image_grid_hws)
            hidden, past_key_values = self.run_language_prefill(input_ids, attention_mask, projected_vit)
            raw_maps, text_embed = self.build_feature_maps(
                input_ids,
                attention_mask,
                image_grid_hws,
                projected_vit_list,
                hidden,
            )
            feature_map = raw_maps.map_samples(self.feature_refiner)

            generated = input_ids.clone()
            tokenizer_max_length = int(getattr(tokenizer, "model_max_length", seq_len + int(max_new_tokens)))
            total_gen_length = min(tokenizer_max_length, seq_len + max(1, int(max_new_tokens)))
            use_mtp = generation_mode in {"fast", "hybrid"}
            default_mask_token_id = int(token_ids["default_mask_token_id"])
            n_future_tokens = max(int(n_future_tokens), 1)
            pre_mask_tokens = torch.full(
                (batch_size, max(n_future_tokens - 1, 0)),
                default_mask_token_id,
                dtype=generated.dtype,
                device=generated.device,
            )
            max_possible_len = total_gen_length + n_future_tokens + 1
            full_position_ids = torch.arange(0, max_possible_len, device=generated.device).unsqueeze(0)
            lm = base.language_model
            im_end_token_id = int(token_ids["im_end_token_id"])
            box_end_token_id = int(token_ids["box_end_token_id"])
            coord_start_token_id = int(token_ids["coord_start_token_id"])
            coord_end_token_id = int(token_ids["coord_end_token_id"])
            none_token_id = int(token_ids["none_token_id"])

            while generated.size(1) < total_gen_length:
                if use_mtp:
                    generated_with_mask = torch.cat(
                        (generated, generated[:, -1].unsqueeze(1), pre_mask_tokens),
                        dim=1,
                    )
                    start_idx = self._cache_seq_len(past_key_values)
                    position_ids = full_position_ids[:, start_idx : generated_with_mask.size(1)].clone()
                    position_ids[0, -n_future_tokens:] -= 1
                    prepare_inputs = lm.prepare_inputs_for_generation(
                        generated_with_mask,
                        past_key_values,
                        None,
                        inputs_embeds=None,
                        use_cache=True,
                        position_ids=position_ids,
                    )
                else:
                    start_idx = self._cache_seq_len(past_key_values)
                    position_ids = full_position_ids[:, start_idx : generated.size(1)]
                    prepare_inputs = lm.prepare_inputs_for_generation(
                        generated,
                        past_key_values,
                        None,
                        inputs_embeds=None,
                        use_cache=True,
                        position_ids=position_ids,
                    )

                outputs = lm(**prepare_inputs)
                next_cache = outputs.past_key_values
                if hasattr(next_cache, "to_legacy_cache"):
                    next_cache = next_cache.to_legacy_cache()
                past_key_values = self._truncate_legacy_cache(next_cache, int(generated.shape[1]))

                if use_mtp:
                    next_token_logits = outputs.logits[:, -n_future_tokens:, :]
                    _, _, x0, box_avg = sample_tokens(
                        next_token_logits,
                        generated,
                        token_ids,
                        keep_k=5,
                        generation_mode=generation_mode,
                        **generate_kwargs,
                    )
                    is_box_empty = box_avg is None or (box_avg[0] == 0).all()
                    new_tokens = x0[0] if is_box_empty else box_avg[0]
                    out_pattern = handle_pattern(new_tokens, token_ids, generation_mode)
                    out_type = out_pattern["type"]
                    out_token = torch.tensor(out_pattern["tokens"], dtype=x0.dtype, device=x0.device)
                else:
                    next_token_logits = outputs.logits[:, -1:, :]
                    _, _, x0, _ = sample_tokens(
                        next_token_logits,
                        generated,
                        token_ids,
                        generation_mode=generation_mode,
                        **generate_kwargs,
                    )
                    out_token = x0[0]
                    out_type = "continue_ar"
                    token_val = int(out_token[0].item())
                    if generation_mode == "hybrid":
                        if token_val == box_end_token_id:
                            out_type = "box_end_ar"
                        elif coord_start_token_id <= token_val <= coord_end_token_id or token_val == none_token_id:
                            out_type = "coord_ar"
                        else:
                            out_type = "im_end"
                    elif token_val == im_end_token_id:
                        out_type = "im_end"

                generated = torch.cat([generated, out_token.unsqueeze(0)], dim=1)
                if out_type == "im_end":
                    break
                if generation_mode == "hybrid":
                    if out_type == "error_box":
                        use_mtp = False
                    elif out_type == "box_end_ar":
                        use_mtp = True

            generated_ids = generated[:, seq_len:]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
        return str(response).strip(), feature_map, text_embed

    def _extract_eagle_vision_features(
        self,
        eagle_inputs: dict[str, torch.Tensor],
        freeze_backbone: bool = False,
    ) -> tuple[MultiScaleSpatialFeatureBatch, torch.Tensor]:
        _, _, _, pixel_values, image_grid_hws = self._prepare_locate_inputs(eagle_inputs)

        def extract() -> tuple[MultiScaleSpatialFeatureBatch, torch.Tensor]:
            premerge_list, _, projected_vit_list, _ = self.run_multiscale_vision_tokens(
                pixel_values, image_grid_hws
            )
            p2_maps = self.build_premerge_feature_maps(image_grid_hws, premerge_list)
            p3_maps = self.build_raw_feature_maps(image_grid_hws, projected_vit_list)
            hidden_size = int(p3_maps.shape[1])
            batch_size = len(projected_vit_list)
            text_embed = p3_maps.new_zeros((batch_size, hidden_size))
            return MultiScaleSpatialFeatureBatch((p2_maps, p3_maps)), text_embed

        if freeze_backbone:
            with torch.no_grad():
                raw_levels, text_embed = extract()
            return raw_levels.detach(), text_embed.detach()
        return extract()

    def _extract_eagle_feature_maps(
        self,
        eagle_inputs: dict[str, torch.Tensor],
        freeze_backbone: bool = False,
    ) -> tuple[MultiScaleSpatialFeatureBatch, torch.Tensor]:
        _, input_ids, attention_mask, pixel_values, image_grid_hws = self._prepare_locate_inputs(eagle_inputs)
        if input_ids is None:
            raise ValueError("Multimodal Locate feature extraction requires input_ids.")

        def extract() -> tuple[MultiScaleSpatialFeatureBatch, torch.Tensor]:
            premerge_list, _, projected_vit_list, projected_vit = (
                self.run_multiscale_vision_tokens(pixel_values, image_grid_hws)
            )
            hidden = self.run_language_hidden(input_ids, attention_mask, projected_vit)
            p2_maps = self.build_premerge_feature_maps(image_grid_hws, premerge_list)
            p3_maps, text_embed = self.build_feature_maps(
                input_ids,
                attention_mask,
                image_grid_hws,
                projected_vit_list,
                hidden,
            )
            return MultiScaleSpatialFeatureBatch((p2_maps, p3_maps)), text_embed

        if freeze_backbone:
            with torch.no_grad():
                raw_levels, text_embed = extract()
            return raw_levels.detach(), text_embed.detach()
        return extract()

    def _extract_eagle_features(
        self, eagle_inputs: dict[str, torch.Tensor]
    ) -> tuple[MultiScaleSpatialFeatureBatch, torch.Tensor]:
        return self._extract_eagle_feature_maps(eagle_inputs, freeze_backbone=False)
