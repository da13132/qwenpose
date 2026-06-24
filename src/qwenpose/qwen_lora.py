from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

QWEN_FORWARD_INPUT_KEYS = (
    "input_ids",
    "attention_mask",
    "pixel_values",
    "image_grid_thw",
    "pixel_values_videos",
    "video_grid_thw",
)


def _group_count(hidden_dim: int, max_groups: int = 32) -> int:
    for groups in range(min(max_groups, hidden_dim), 0, -1):
        if hidden_dim % groups == 0:
            return groups
    return 1


@dataclass
class QwenLoRAConfig:
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


def _dtype_from_name(name: str) -> torch.dtype | None:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    if name in ("none", "auto"):
        return None
    raise ValueError(f"Unsupported dtype {name!r}")


def find_qwen_lora_targets(model: nn.Module) -> tuple[list[str], dict[str, int], dict[str, int]]:
    """Find LLM attention and vision-tower Linear modules for PEFT LoRA.

    This keeps qwenpose independent from any external SFT project tree while
    still applying LoRA to both the LLM attention modules and visual tower.
    """
    llm_suffixes = ("q_proj", "k_proj", "v_proj", "o_proj")
    targets: list[str] = []
    rank_pattern: dict[str, int] = {}
    alpha_pattern: dict[str, int] = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name.endswith(llm_suffixes):
            targets.append(name)
        if (name.startswith("visual.") or name.startswith("model.visual.")) and "merger" not in name:
            targets.append(name)
            rank_pattern[name] = -1
            alpha_pattern[name] = -1
    # Preserve order and remove duplicates.
    seen = set()
    unique = []
    for name in targets:
        if name in seen:
            continue
        seen.add(name)
        unique.append(name)
    return unique, rank_pattern, alpha_pattern


def load_qwen_model(
    model_path: str,
    *,
    dtype: str = "bfloat16",
    attn_implementation: str = "flash_attention_2",
):
    from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration

    resolved_model_path = str(Path(model_path))
    auto_config = AutoConfig.from_pretrained(resolved_model_path)
    architectures = [str(name) for name in getattr(auto_config, "architectures", []) or []]
    model_type = str(getattr(auto_config, "model_type", ""))
    model_cls = (
        Qwen3VLMoeForConditionalGeneration
        if model_type.endswith("_moe") or any("Moe" in name for name in architectures)
        else Qwen3VLForConditionalGeneration
    )
    model = model_cls.from_pretrained(
        resolved_model_path,
        dtype=_dtype_from_name(dtype),
        attn_implementation=attn_implementation,
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(resolved_model_path, fix_mistral_regex=True)
    return model, processor


def load_qwen_with_lora(config: QwenLoRAConfig):
    """Load Qwen3-VL with LoRA on LLM attention and visual tower.

    The returned model has frozen base weights and trainable LoRA parameters.
    The pose decoder/head modules should be optimized normally alongside these
    adapters in full Qwen-backed training.
    """
    from peft import LoraConfig, TaskType, get_peft_model

    model, processor = load_qwen_model(
        config.model_path,
        dtype=config.dtype,
        attn_implementation=config.attn_implementation,
    )
    for param in model.parameters():
        param.requires_grad = False

    targets, rank_pattern, alpha_pattern = find_qwen_lora_targets(model)
    if not targets:
        raise RuntimeError("No Qwen LoRA target modules were found.")
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


def load_qwen_with_existing_lora(
    *,
    base_model_path: str,
    adapter_path: str,
    dtype: str = "bfloat16",
    attn_implementation: str = "flash_attention_2",
    gradient_checkpointing: bool = False,
):
    from peft import PeftModel
    from transformers import AutoProcessor

    model, processor = load_qwen_model(
        base_model_path,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )
    for param in model.parameters():
        param.requires_grad = False

    model = PeftModel.from_pretrained(
        model,
        str(adapter_path),
        is_trainable=True,
    )
    try:
        processor = AutoProcessor.from_pretrained(str(adapter_path), fix_mistral_regex=True)
    except Exception:
        pass
    if gradient_checkpointing:
        _enable_gradient_checkpointing(model)
    return model, processor


def _enable_gradient_checkpointing(model: nn.Module) -> None:
    base = get_qwen_base_model(model)
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


def count_qwen_lora_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def get_qwen_base_model(model: nn.Module) -> nn.Module:
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def qwen_hidden_size(model: nn.Module) -> int:
    base = get_qwen_base_model(model)
    return int(base.config.text_config.hidden_size)


def qwen_forward_kwargs(qwen_inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value for key, value in qwen_inputs.items() if key in QWEN_FORWARD_INPUT_KEYS}


def _processor_image_kwargs(
    min_pixels: int | None = None,
    max_pixels: int | None = None,
) -> dict[str, int]:
    image_kwargs: dict[str, int] = {}
    if min_pixels is not None:
        image_kwargs["min_pixels"] = int(min_pixels)
    if max_pixels is not None:
        image_kwargs["max_pixels"] = int(max_pixels)
    return image_kwargs


def _move_processor_tensors_to_device(
    inputs: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}


def _load_rgb_images(image_paths: list[str]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            images.append(image.convert("RGB").copy())
    return images


def _build_user_messages(image_path: str, prompt: str) -> list[dict[str, object]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _apply_chat_template_batch(
    processor,
    conversations: list[list[dict[str, object]]],
    *,
    add_generation_prompt: bool,
) -> list[str]:
    texts = processor.apply_chat_template(
        conversations,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    if isinstance(texts, str):
        return [texts]
    return list(texts)


def _tokenized_lengths(tokenizer, texts: list[str]) -> torch.Tensor:
    encoded = tokenizer(texts, add_special_tokens=False, return_attention_mask=True)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        return torch.as_tensor(attention_mask, dtype=torch.long).sum(dim=1)
    input_ids = encoded.get("input_ids", [])
    return torch.tensor([len(ids) for ids in input_ids], dtype=torch.long)


def _build_prefix_mask(attention_mask: torch.Tensor, prefix_lengths: torch.Tensor) -> torch.Tensor:
    if attention_mask.numel() == 0:
        return torch.zeros_like(attention_mask, dtype=torch.bool)
    active = attention_mask.bool()
    prefix_lengths = prefix_lengths.to(device=attention_mask.device, dtype=torch.long).clamp(min=0).unsqueeze(1)
    active_ranks = torch.cumsum(active.to(dtype=torch.long), dim=1)
    return active & (active_ranks <= prefix_lengths)


def _pack_image_tokens(
    hidden: torch.Tensor,
    image_mask: torch.Tensor,
    expected_tokens: torch.Tensor,
) -> torch.Tensor:
    batch_size, _, hidden_dim = hidden.shape
    if batch_size == 0:
        return hidden.new_zeros(0, 0, hidden_dim)
    max_expected = int(expected_tokens.max().item()) if expected_tokens.numel() > 0 else 0
    if max_expected <= 0:
        return hidden.new_zeros(batch_size, 0, hidden_dim)
    token_ranks = torch.cumsum(image_mask.to(dtype=torch.long), dim=1) - 1
    keep_mask = image_mask & (token_ranks >= 0) & (token_ranks < expected_tokens[:, None])
    packed = hidden.new_zeros(batch_size, max_expected, hidden_dim)
    batch_indices = torch.arange(batch_size, device=hidden.device)[:, None].expand_as(token_ranks)
    packed[batch_indices[keep_mask], token_ranks[keep_mask]] = hidden[keep_mask]
    return packed


def _resize_packed_visual_tokens(
    packed_tokens: torch.Tensor,
    spatial_hw: torch.Tensor,
    output_size: int,
) -> torch.Tensor:
    batch_size, max_tokens, hidden_dim = packed_tokens.shape
    if batch_size == 0:
        return packed_tokens.new_zeros(0, hidden_dim, output_size, output_size)
    if max_tokens == 0:
        return packed_tokens.new_zeros(batch_size, hidden_dim, output_size, output_size)

    spatial_hw = spatial_hw.to(device=packed_tokens.device, dtype=torch.long).clamp(min=1)
    heights = spatial_hw[:, 0]
    widths = spatial_hw[:, 1]
    max_h = int(heights.max().item()) if heights.numel() > 0 else 0
    max_w = int(widths.max().item()) if widths.numel() > 0 else 0
    if max_h <= 0 or max_w <= 0:
        return packed_tokens.new_zeros(batch_size, hidden_dim, output_size, output_size)

    token_ids = torch.arange(max_tokens, device=packed_tokens.device, dtype=torch.long).unsqueeze(0)
    valid = token_ids < (heights * widths).unsqueeze(1)
    row_idx = torch.div(token_ids, widths.unsqueeze(1), rounding_mode="floor")
    col_idx = torch.remainder(token_ids, widths.unsqueeze(1))
    batch_indices = (
        torch.arange(batch_size, device=packed_tokens.device, dtype=torch.long)
        .unsqueeze(1)
        .expand(batch_size, max_tokens)
    )

    spatial = packed_tokens.new_zeros(batch_size, max_h, max_w, hidden_dim)
    spatial[batch_indices[valid], row_idx[valid], col_idx[valid]] = packed_tokens[valid]
    spatial = spatial.permute(0, 3, 1, 2)

    dtype = packed_tokens.dtype
    heights_f = heights.to(dtype=dtype).view(batch_size, 1, 1)
    widths_f = widths.to(dtype=dtype).view(batch_size, 1, 1)
    output_scale = float(output_size)
    out_x = torch.arange(output_size, device=packed_tokens.device, dtype=dtype).view(1, 1, output_size)
    out_y = torch.arange(output_size, device=packed_tokens.device, dtype=dtype).view(1, output_size, 1)
    sample_x = ((out_x + 0.5) * widths_f / output_scale) - 0.5
    sample_y = ((out_y + 0.5) * heights_f / output_scale) - 0.5
    sample_x = torch.minimum(sample_x.clamp(min=0.0), widths_f - 1.0)
    sample_y = torch.minimum(sample_y.clamp(min=0.0), heights_f - 1.0)
    grid_x = ((sample_x + 0.5) * (2.0 / float(max_w))) - 1.0
    grid_y = ((sample_y + 0.5) * (2.0 / float(max_h))) - 1.0
    grid = torch.stack(
        [
            grid_x.expand(batch_size, output_size, output_size),
            grid_y.expand(batch_size, output_size, output_size),
        ],
        dim=-1,
    )
    return F.grid_sample(
        spatial,
        grid,
        mode="bilinear",
        align_corners=False,
        padding_mode="border",
    )


def build_qwen_inputs(
    processor,
    image_paths: list[str],
    prompts: list[str],
    device: torch.device,
    add_generation_prompt: bool = False,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
) -> dict[str, torch.Tensor]:
    """Build Qwen3-VL processor inputs from image paths and task prompts."""
    images = _load_rgb_images(image_paths)
    messages_batch = [_build_user_messages(image_path, prompt) for image_path, prompt in zip(image_paths, prompts)]
    texts = _apply_chat_template_batch(
        processor,
        messages_batch,
        add_generation_prompt=add_generation_prompt,
    )
    inputs = processor(
        text=texts,
        images=images,
        padding=True,
        return_tensors="pt",
        **_processor_image_kwargs(min_pixels=min_pixels, max_pixels=max_pixels),
    )
    return _move_processor_tensors_to_device(inputs, device)


def build_qwen_lm_inputs(
    processor,
    image_paths: list[str],
    prompts: list[str],
    responses: list[str],
    device: torch.device,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
) -> dict[str, torch.Tensor]:
    """Build supervised multimodal LM inputs with labels masked to answer tokens."""
    images = _load_rgb_images(image_paths)
    prompt_messages_batch = [_build_user_messages(image_path, prompt) for image_path, prompt in zip(image_paths, prompts)]
    full_messages_batch = [
        [
            prompt_messages[0],
            {"role": "assistant", "content": response},
        ]
        for prompt_messages, response in zip(prompt_messages_batch, responses)
    ]
    user_texts = _apply_chat_template_batch(
        processor,
        prompt_messages_batch,
        add_generation_prompt=False,
    )
    prompt_texts = _apply_chat_template_batch(
        processor,
        prompt_messages_batch,
        add_generation_prompt=True,
    )
    full_texts = _apply_chat_template_batch(
        processor,
        full_messages_batch,
        add_generation_prompt=False,
    )

    image_kwargs = _processor_image_kwargs(min_pixels=min_pixels, max_pixels=max_pixels)
    prompt_inputs = processor(text=prompt_texts, images=images, padding=True, return_tensors="pt", **image_kwargs)
    full_inputs = processor(text=full_texts, images=images, padding=True, return_tensors="pt", **image_kwargs)
    prompt_attention = prompt_inputs.get("attention_mask")
    if prompt_attention is None:
        prompt_attention = torch.ones_like(prompt_inputs["input_ids"], dtype=torch.long)
        prompt_inputs["attention_mask"] = prompt_attention
    full_attention = full_inputs.get("attention_mask")
    if full_attention is None:
        full_attention = torch.ones_like(full_inputs["input_ids"], dtype=torch.long)
        full_inputs["attention_mask"] = full_attention

    prompt_lengths = prompt_attention.sum(dim=1).to(dtype=torch.long)
    prompt_template_lengths = _tokenized_lengths(processor.tokenizer, prompt_texts)
    user_template_lengths = _tokenized_lengths(processor.tokenizer, user_texts)
    generation_suffix_lengths = (prompt_template_lengths - user_template_lengths).clamp(min=0)
    pose_prompt_lengths = (prompt_lengths - generation_suffix_lengths.to(prompt_lengths.device)).clamp(min=0)

    prompt_mask = _build_prefix_mask(full_attention, prompt_lengths)
    pose_text_mask = _build_prefix_mask(full_attention, pose_prompt_lengths)
    labels = full_inputs["input_ids"].clone()
    labels = labels.masked_fill(~full_attention.bool(), -100)
    labels = labels.masked_fill(prompt_mask, -100)
    full_inputs["labels"] = labels
    full_inputs["pose_text_mask"] = pose_text_mask
    return _move_processor_tensors_to_device(full_inputs, device)


class QwenFeatureRefinerBlock(nn.Module):
    """Bottleneck residual conv block for the resized Qwen visual grid."""

    def __init__(self, dim: int, bottleneck_dim: int, init_scale: float = 0.1) -> None:
        super().__init__()
        bottleneck_dim = max(int(bottleneck_dim), 1)
        self.net = nn.Sequential(
            nn.Conv2d(dim, bottleneck_dim, 1),
            nn.GroupNorm(_group_count(bottleneck_dim), bottleneck_dim),
            nn.GELU(),
            nn.Conv2d(bottleneck_dim, bottleneck_dim, 3, padding=1),
            nn.GroupNorm(_group_count(bottleneck_dim), bottleneck_dim),
            nn.GELU(),
            nn.Conv2d(bottleneck_dim, dim, 1),
        )
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale.to(dtype=x.dtype) * self.net(x)


class QwenFeatureRefiner(nn.Module):
    """Learn task-specific spatial patterns after dynamic-token interpolation."""

    def __init__(
        self,
        dim: int,
        num_layers: int = 0,
        bottleneck_dim: int = 256,
        init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_layers = max(int(num_layers), 0)
        self.bottleneck_dim = int(bottleneck_dim)
        self.init_scale = float(init_scale)
        self.blocks = nn.Sequential(
            *[
                QwenFeatureRefinerBlock(dim, bottleneck_dim=bottleneck_dim, init_scale=init_scale)
                for _ in range(self.num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_layers <= 0:
            return x
        param_dtype = next(self.parameters()).dtype
        x = x.to(dtype=param_dtype)
        return self.blocks(x)


class QwenFeatureExtractor(nn.Module):
    """Return dense visual features and pooled multimodal text features from Qwen3-VL.

    The forward pass runs the real Qwen3-VL model, so LoRA parameters in the
    visual tower and LLM receive gradients from the pose losses.
    """

    def __init__(
        self,
        qwen_model: nn.Module,
        output_size: int = 32,
        refiner_layers: int = 0,
        refiner_bottleneck_dim: int = 256,
        refiner_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.qwen_model = qwen_model
        self.output_size = output_size
        hidden_size = qwen_hidden_size(qwen_model)
        self.feature_refiner = QwenFeatureRefiner(
            hidden_size,
            num_layers=refiner_layers,
            bottleneck_dim=refiner_bottleneck_dim,
            init_scale=refiner_init_scale,
        )

    def forward(
        self,
        qwen_inputs: dict[str, torch.Tensor],
        freeze_qwen: bool = False,
        text_keep_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.run_backbone_hidden(qwen_inputs, freeze_qwen=freeze_qwen)
        return self.project_hidden_state(qwen_inputs, hidden, text_keep_mask=text_keep_mask)

    def run_backbone_hidden(
        self,
        qwen_inputs: dict[str, torch.Tensor],
        freeze_qwen: bool = False,
    ) -> torch.Tensor:
        if freeze_qwen:
            with torch.no_grad():
                return self._run_qwen_backbone(qwen_inputs)
        return self._run_qwen_backbone(qwen_inputs)

    def project_hidden_state(
        self,
        qwen_inputs: dict[str, torch.Tensor],
        hidden: torch.Tensor,
        text_keep_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        visual_map, text_embed = self._project_hidden_state(
            qwen_inputs,
            hidden,
            text_keep_mask=text_keep_mask,
        )
        visual_map = self.feature_refiner(visual_map)
        return visual_map, text_embed

    def _run_qwen_backbone(self, qwen_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        base = get_qwen_base_model(self.qwen_model)
        qwen_core = base.model
        outputs = qwen_core(use_cache=False, **qwen_forward_kwargs(qwen_inputs))
        return outputs.last_hidden_state

    def _project_hidden_state(
        self,
        qwen_inputs: dict[str, torch.Tensor],
        hidden: torch.Tensor,
        text_keep_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base = get_qwen_base_model(self.qwen_model)
        qwen_core = base.model
        input_ids = qwen_inputs["input_ids"]
        attention_mask = qwen_inputs.get("attention_mask")
        image_token_id = int(base.config.image_token_id)
        image_mask = input_ids == image_token_id

        grid_thw = qwen_inputs["image_grid_thw"]
        merge = int(qwen_core.visual.spatial_merge_size)
        spatial_hw = torch.stack(
            [
                (grid_thw[:, 1].to(dtype=torch.long) // merge).clamp(min=1),
                (grid_thw[:, 2].to(dtype=torch.long) // merge).clamp(min=1),
            ],
            dim=1,
        )
        expected_tokens = spatial_hw[:, 0] * spatial_hw[:, 1]
        packed_tokens = _pack_image_tokens(hidden, image_mask, expected_tokens)
        visual_map = _resize_packed_visual_tokens(
            packed_tokens,
            spatial_hw,
            output_size=self.output_size,
        )

        text_mask_bool = ~image_mask
        if text_keep_mask is not None:
            text_mask_bool = text_mask_bool & text_keep_mask.to(device=hidden.device).bool()
        if attention_mask is not None:
            text_mask_bool = text_mask_bool & attention_mask.bool()
        text_mask = text_mask_bool.float().unsqueeze(-1)
        text_embed = (hidden * text_mask).sum(dim=1) / text_mask.sum(dim=1).clamp(min=1.0)
        return visual_map, text_embed
