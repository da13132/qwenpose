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


def load_eagle_with_lora(config: EagleLoRAConfig):
    """Load LocateAnything-3B with LoRA on LLM and vision encoder.

    Returns (model, processor) where model has frozen base weights + trainable LoRA.
    """
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoConfig, AutoModel, AutoProcessor

    model_path = str(Path(config.model_path))

    # Determine attn_implementation: Eagle uses 'magi' by default which requires
    # Hopper/Blackwell GPUs. For RTX 4090 (Ada), fall back to sdpa.
    # NOTE: Eagle's custom Qwen2 forward only supports 'magi' and 'sdpa' for
    # attention mask preparation — NOT 'flash_attention_2'.
    attn_impl = config.attn_implementation
    if attn_impl not in ("magi", "sdpa"):
        attn_impl = "sdpa"

    # Eagle's _autoset_attn_implementation intercepts attn_implementation when
    # config._attn_implementation is already 'magi' (from config.json). We must
    # explicitly override it before loading the model.
    eagle_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    eagle_config._attn_implementation = attn_impl
    # Also propagate to sub-configs
    if hasattr(eagle_config, "text_config"):
        eagle_config.text_config._attn_implementation = attn_impl
    if hasattr(eagle_config, "vision_config"):
        vision_attn = attn_impl if attn_impl != "sdpa" else "sdpa"
        eagle_config.vision_config._attn_implementation = vision_attn

    model = AutoModel.from_pretrained(
        model_path,
        config=eagle_config,
        trust_remote_code=True,
        torch_dtype=_dtype_from_name(config.dtype),
        attn_implementation=attn_impl,
    )
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


def build_eagle_inputs(
    processor,
    image_paths: list[str],
    prompts: list[str],
    device: torch.device,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
) -> dict[str, torch.Tensor]:
    """Build LocateAnything processor inputs from image paths and task prompts.

    Returns dict with: pixel_values, image_grid_hws, input_ids, attention_mask.
    The Eagle processor handles dynamic resolution internally.

    Args:
        min_pixels: optional lower bound on resized image area (in pixels). The
            processor's image preprocessor uses it to decide the minimum number
            of image tokens. Effective when > 0.
        max_pixels: optional upper bound on resized image area (in pixels).
            Limits the number of image tokens for memory control. Effective
            when > 0.

    Both bounds follow the Qwen3-VL "28*28*patch_count" convention (MoonViT
    also uses a 28x28 patch grid, so the same pixel budgets translate to the
    same number of image tokens).
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

    image_kwargs: dict[str, int] = {}
    if min_pixels is not None and int(min_pixels) > 0:
        image_kwargs["min_pixels"] = int(min_pixels)
    if max_pixels is not None and int(max_pixels) > 0:
        image_kwargs["max_pixels"] = int(max_pixels)
    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt", **image_kwargs)
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
        if freeze_eagle:
            with torch.no_grad():
                visual_map, text_embed = self._extract_eagle_features(eagle_inputs)
        else:
            visual_map, text_embed = self._extract_eagle_features(eagle_inputs)
        visual_map = self.feature_refiner(visual_map)
        return visual_map, text_embed

    def _extract_eagle_features(self, eagle_inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        base = get_eagle_base_model(self.eagle_model)

        input_ids = eagle_inputs["input_ids"]
        attention_mask = eagle_inputs.get("attention_mask")
        pixel_values = eagle_inputs["pixel_values"]
        image_grid_hws = eagle_inputs.get("image_grid_hws")
        image_flags = eagle_inputs.get("image_flags")

        # Cast pixel_values to model dtype (processor returns float32, model may be bfloat16)
        model_dtype = next(base.parameters()).dtype
        if pixel_values.dtype != model_dtype:
            pixel_values = pixel_values.to(dtype=model_dtype)

        # Ensure image_grid_hws is long tensor on correct device
        if image_grid_hws is not None and isinstance(image_grid_hws, torch.Tensor):
            image_grid_hws = image_grid_hws.to(device=pixel_values.device, dtype=torch.long)

        # Step 1: Run vision encoder -> List[Tensor], each (num_merged_tokens, 4608)
        vit_embeds_list = base.extract_feature(pixel_values, image_grid_hws)

        # Step 2: Filter valid images if image_flags provided
        if image_flags is not None and isinstance(vit_embeds_list, list):
            valid_indices = torch.where(image_flags.view(-1) != 0)[0]
            if valid_indices.numel() > 0:
                vit_embeds_list = [vit_embeds_list[i] for i in valid_indices.tolist()]

        # Step 3: Concatenate and project through MLP
        if isinstance(vit_embeds_list, list):
            vit_embeds = torch.cat(vit_embeds_list, dim=0)  # [total_tokens, 4608]
        else:
            vit_embeds = vit_embeds_list
        vit_embeds = base.mlp1(vit_embeds)  # [total_tokens, 2048]

        # Step 4: Run LLM.
        # Eagle's Qwen2ForCausalLM has an image_processing() method that
        # replaces image tokens with visual_features inside input_embeds.
        # Pass input_ids + visual_features (NOT inputs_embeds) so the model's
        # custom attention mask code can access input_ids for block masking.
        # IMPORTANT: Run the LLM in eval() mode to avoid Eagle's custom
        # _prepare_block_mask_for_training which creates masks with batch_size=1
        # (incompatible with our batched training).
        B = input_ids.shape[0]
        image_token_id = int(base.image_token_index)
        lm = base.language_model
        was_training = lm.training
        lm.eval()
        lm_outputs = lm(
            input_ids=input_ids,
            visual_features=vit_embeds,
            image_token_index=image_token_id,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
        )
        if was_training:
            lm.train()
        # CausalLMOutputWithPast: hidden_states is a tuple of all layer outputs
        hidden = lm_outputs.hidden_states[-1]  # [B, seq_len, 2048]

        # Step 5: Extract visual feature map from image token positions
        image_mask = (input_ids == image_token_id)
        visual_maps = []

        # Parse grid_hws — may be tensor or numpy
        if isinstance(image_grid_hws, torch.Tensor):
            grid_hws_np = image_grid_hws.detach().cpu().numpy()
        elif isinstance(image_grid_hws, np.ndarray):
            grid_hws_np = image_grid_hws
        else:
            grid_hws_np = None

        for batch_idx in range(B):
            tokens = hidden[batch_idx][image_mask[batch_idx]]
            if grid_hws_np is not None and batch_idx < len(grid_hws_np):
                h = max(int(grid_hws_np[batch_idx][0]), 1)
                w = max(int(grid_hws_np[batch_idx][1]), 1)
            else:
                n = tokens.shape[0]
                h = w = max(int(n ** 0.5), 1)
            expected = h * w
            if tokens.shape[0] < expected:
                pad = tokens.new_zeros(expected - tokens.shape[0], tokens.shape[-1])
                tokens = torch.cat([tokens, pad], dim=0)
            elif tokens.shape[0] > expected:
                tokens = tokens[:expected]
            fmap = tokens.view(h, w, -1).permute(2, 0, 1).unsqueeze(0)
            fmap = F.interpolate(fmap, size=(self.output_size, self.output_size), mode="bilinear", align_corners=False)
            visual_maps.append(fmap.squeeze(0))
        visual_map = torch.stack(visual_maps, dim=0)

        # Step 6: Mean-pool non-image tokens for text embedding
        non_image = ~image_mask
        if attention_mask is not None:
            non_image = non_image & attention_mask.bool()
        text_mask = non_image.float().unsqueeze(-1)
        text_embed = (hidden * text_mask).sum(dim=1) / text_mask.sum(dim=1).clamp(min=1.0)

        return visual_map, text_embed
