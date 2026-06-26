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
  - Processor returns: pixel_values, image_grid_hws (numpy), input_ids, attention_mask
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .qwen_lora import QwenFeatureRefiner, _dtype_from_name


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


def find_eagle_lora_targets(model: nn.Module) -> tuple[list[str], dict[str, int], dict[str, int]]:
    """Find LLM and vision-tower modules for PEFT LoRA.

    LocateAnything-3B structure (after PEFT wrapping the top-level model):
      - base_model.model.vision_model.* : MoonViT encoder
      - base_model.model.language_model.* : Qwen2.5 LLM
      - base_model.model.mlp1.* : MLP projector (not targeted)
    """
    llm_suffixes = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    vision_suffixes = ("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2")
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


def load_eagle_with_lora(config: EagleLoRAConfig):
    """Load LocateAnything-3B with LoRA on LLM and vision encoder.

    Returns (model, processor) where model has frozen base weights + trainable LoRA.
    """
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoConfig, AutoModel, AutoProcessor

    model_path = str(Path(config.model_path))

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

    model = AutoModel.from_pretrained(
        model_path,
        config=eagle_config,
        trust_remote_code=True,
        torch_dtype=_dtype_from_name(config.dtype),
        attn_implementation=text_attn_impl,
    )
    _set_eagle_vision_attention(model, vision_attn_impl)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

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

    if config.gradient_checkpointing:
        _enable_gradient_checkpointing(model)

    return model, processor


def _enable_gradient_checkpointing(model: nn.Module) -> None:
    base = get_eagle_base_model(model)
    for candidate in (model, base):
        cfg = getattr(candidate, "config", None)
        if cfg is not None and hasattr(cfg, "use_cache"):
            cfg.use_cache = False
    enable_fn = getattr(model, "gradient_checkpointing_enable", None)
    if enable_fn is None:
        enable_fn = getattr(base, "gradient_checkpointing_enable", None)
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
    processor,
    max_pixels: int | None = None,
    image_token_limit: int | None = None,
) -> int | None:
    if image_token_limit is not None and int(image_token_limit) > 0:
        return int(image_token_limit)
    image_processor = getattr(processor, "image_processor", None)
    patch_size = int(getattr(image_processor, "patch_size", 14)) if image_processor is not None else 14
    if max_pixels is not None and int(max_pixels) > 0:
        return max(int(max_pixels) // max(patch_size * patch_size, 1), 1)
    return None


def build_eagle_inputs(
    processor,
    image_paths: list[str],
    prompts: list[str],
    device: torch.device,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
    image_token_limit: int | None = None,
) -> dict[str, torch.Tensor]:
    """Build LocateAnything processor inputs from image paths and task prompts.

    Returns dict with: pixel_values, image_grid_hws, input_ids, attention_mask.
    The Eagle processor handles dynamic resolution internally.

    Args:
        min_pixels: kept only for API compatibility; LocateAnythingProcessor
            does not consume Qwen-style min_pixels.
        max_pixels: optional pixel budget converted into LocateAnything's raw
            MoonViT patch-token limit when image_token_limit is not set.
        image_token_limit: LocateAnything native raw MoonViT patch-token budget
            per image. This controls processor.image_processor.in_token_limit.
    """
    images = []
    texts = []
    for image_path, prompt in zip(image_paths, prompts):
        with Image.open(image_path) as image:
            images.append(image.convert("RGB").copy())
        # Build chat-style text with image placeholder
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        texts.append(
            processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    image_processor = getattr(processor, "image_processor", None)
    token_limit = _locate_image_token_limit(processor, max_pixels=max_pixels, image_token_limit=image_token_limit)
    old_token_limit = getattr(image_processor, "in_token_limit", None) if image_processor is not None else None
    if image_processor is not None and token_limit is not None:
        image_processor.in_token_limit = int(token_limit)
    try:
        # LocateAnythingProcessor does not accept Qwen-style min_pixels/max_pixels
        # kwargs. Its real control knob is image_processor.in_token_limit, which
        # limits raw MoonViT patch tokens before the 2x2 merger.
        inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
    finally:
        if image_processor is not None and old_token_limit is not None:
            image_processor.in_token_limit = old_token_limit
    result = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.to(device)
        elif isinstance(v, np.ndarray):
            # image_grid_hws: convert to tensor for vision model compatibility
            result[k] = torch.from_numpy(v).to(device)
        else:
            result[k] = v
    return result


def build_eagle_lm_inputs(
    processor,
    image_paths,
    prompts,
    responses,
    device,
    min_pixels=None,
    max_pixels=None,
    image_token_limit=None,
):
    mixed_prompts = [p + " " + str(r) for p, r in zip(prompts, responses)]
    inputs = build_eagle_inputs(
        processor,
        image_paths,
        mixed_prompts,
        device,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        image_token_limit=image_token_limit,
    )
    prompt_inputs = build_eagle_inputs(
        processor,
        image_paths,
        prompts,
        device,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        image_token_limit=image_token_limit,
    )
    labels = inputs["input_ids"].clone()
    prompt_mask = prompt_inputs.get("attention_mask")
    for row in range(labels.shape[0]):
        prompt_len = int(prompt_mask[row].sum().item()) if prompt_mask is not None else int(prompt_inputs["input_ids"].shape[1])
        labels[row, : min(prompt_len, labels.shape[1])] = -100
    if "attention_mask" in inputs:
        labels = labels.masked_fill(inputs["attention_mask"].eq(0), -100)
    inputs["labels"] = labels
    return inputs


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
        output_size: int = 32,
        refiner_layers: int = 0,
        refiner_bottleneck_dim: int = 256,
        refiner_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.eagle_model = eagle_model
        self.output_size = output_size
        hidden_size = eagle_hidden_size(eagle_model)
        self.raw_feature_norm = nn.LayerNorm(hidden_size)
        self.lm_feature_norm = nn.LayerNorm(hidden_size)
        self.dual_feature_fuse = nn.Sequential(
            nn.Conv2d(hidden_size * 2, hidden_size, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1, groups=hidden_size),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=1),
        )
        self.dual_feature_gate = nn.Parameter(torch.tensor(0.0))
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Match Qwen3VL Stage-1 behavior: when LocateAnything is frozen, run the
        # heavy MoonViT+Qwen2.5 backbone under no_grad to avoid storing LLM
        # activations for batch16. The extractor-owned fusion/refiner layers are
        # applied outside no_grad and remain trainable.
        raw_maps, lm_maps, text_embed = self._extract_eagle_feature_maps(
            eagle_inputs,
            freeze_backbone=bool(freeze_eagle),
        )
        visual_map = self.fuse_feature_maps(raw_maps, lm_maps)
        visual_map = self.feature_refiner(visual_map)
        return visual_map, text_embed

    def _prepare_locate_inputs(
        self,
        eagle_inputs: dict[str, torch.Tensor],
    ) -> tuple[nn.Module, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        base = get_eagle_base_model(self.eagle_model)
        input_ids = eagle_inputs["input_ids"]
        attention_mask = eagle_inputs.get("attention_mask")
        pixel_values = eagle_inputs["pixel_values"]
        image_grid_hws = eagle_inputs.get("image_grid_hws")

        model_dtype = next(base.parameters()).dtype
        if pixel_values.dtype != model_dtype:
            pixel_values = pixel_values.to(dtype=model_dtype)
        if image_grid_hws is not None and isinstance(image_grid_hws, torch.Tensor):
            image_grid_hws = image_grid_hws.to(device=pixel_values.device, dtype=torch.long)
        return base, input_ids, attention_mask, pixel_values, image_grid_hws

    def run_vision_tokens(
        self,
        pixel_values: torch.Tensor,
        image_grid_hws: torch.Tensor | None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        base = get_eagle_base_model(self.eagle_model)
        vit_embeds = base.extract_feature(pixel_values, image_grid_hws)
        vit_embeds_list = vit_embeds if isinstance(vit_embeds, list) else [vit_embeds]
        projected_vit_list = [base.mlp1(vit_embeds) for vit_embeds in vit_embeds_list]
        projected_vit = torch.cat(projected_vit_list, dim=0)
        return vit_embeds_list, projected_vit_list, projected_vit

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
    ) -> torch.Tensor:
        base = get_eagle_base_model(self.eagle_model)
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
            use_cache=False,
            output_hidden_states=True,
        )
        if was_training:
            lm.train()
        return lm_outputs.hidden_states[-1]

    def build_feature_maps(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        image_grid_hws: torch.Tensor | np.ndarray | None,
        projected_vit_list: list[torch.Tensor],
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

            raw_map = raw_tokens.float().view(h, w, -1).permute(2, 0, 1).unsqueeze(0)
            lm_map = lm_tokens.float().view(h, w, -1).permute(2, 0, 1).unsqueeze(0)
            raw_map = F.interpolate(raw_map, size=(self.output_size, self.output_size), mode="bilinear", align_corners=False)
            lm_map = F.interpolate(lm_map, size=(self.output_size, self.output_size), mode="bilinear", align_corners=False)
            visual_maps.append((raw_map.squeeze(0), lm_map.squeeze(0)))
        raw_maps = torch.stack([item[0] for item in visual_maps], dim=0)
        lm_maps = torch.stack([item[1] for item in visual_maps], dim=0)

        non_image = ~image_mask
        if attention_mask is not None:
            non_image = non_image & attention_mask.bool()
        text_mask = non_image.float().unsqueeze(-1)
        text_embed = (hidden * text_mask).sum(dim=1) / text_mask.sum(dim=1).clamp(min=1.0)
        return raw_maps, lm_maps, text_embed

    def fuse_feature_maps(self, raw_maps: torch.Tensor, lm_maps: torch.Tensor) -> torch.Tensor:
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

    def _extract_eagle_feature_maps(
        self,
        eagle_inputs: dict[str, torch.Tensor],
        freeze_backbone: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, input_ids, attention_mask, pixel_values, image_grid_hws = self._prepare_locate_inputs(eagle_inputs)
        if freeze_backbone:
            with torch.no_grad():
                _, projected_vit_list, projected_vit = self.run_vision_tokens(pixel_values, image_grid_hws)
                hidden = self.run_language_hidden(input_ids, attention_mask, projected_vit)
                raw_maps, lm_maps, text_embed = self.build_feature_maps(
                    input_ids,
                    attention_mask,
                    image_grid_hws,
                    projected_vit_list,
                    hidden,
                )
            return raw_maps.detach(), lm_maps.detach(), text_embed.detach()
        _, projected_vit_list, projected_vit = self.run_vision_tokens(pixel_values, image_grid_hws)
        hidden = self.run_language_hidden(input_ids, attention_mask, projected_vit)
        return self.build_feature_maps(input_ids, attention_mask, image_grid_hws, projected_vit_list, hidden)

    def _extract_eagle_features(self, eagle_inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        raw_maps, lm_maps, text_embed = self._extract_eagle_feature_maps(eagle_inputs, freeze_backbone=False)
        return self.fuse_feature_maps(raw_maps, lm_maps), text_embed
