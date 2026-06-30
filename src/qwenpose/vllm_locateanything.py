from __future__ import annotations

from pathlib import Path
from typing import Any


LOCATEANYTHING_ARCH = "LocateAnythingForConditionalGeneration"
LOCATEANYTHING_VLLM_BACKEND = (
    "qwenpose.vllm_locateanything_model:"
    "LocateAnythingVLLMForConditionalGeneration"
)


def enable_locateanything_vllm_transformers_backend(model_path: str | Path) -> dict[str, Any]:
    """Register the local LocateAnything custom vLLM model.

    vLLM 0.11 does not know the LocateAnything remote-code architecture.  The
    generic Transformers multimodal backend is not enough because the model has
    a custom processor and a custom Qwen2 implementation.  The project wrapper
    below keeps MoonViT+MLP from LocateAnything and uses vLLM's native Qwen2
    executor for the language model.
    """
    from transformers import AutoConfig
    from vllm.model_executor.models.registry import ModelRegistry
    from vllm.transformers_utils.dynamic_module import try_get_class_from_dynamic_module

    model_path = str(model_path)
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    architectures = list(getattr(cfg, "architectures", []) or [])
    auto_map = getattr(cfg, "auto_map", None) or {}
    patched_class = None
    if LOCATEANYTHING_ARCH in architectures:
        model_ref = auto_map.get("AutoModel") or auto_map.get("AutoModelForCausalLM")
        if model_ref:
            patched_class = try_get_class_from_dynamic_module(
                model_ref,
                model_path,
                revision=None,
                warn_on_fail=True,
            )
            if patched_class is not None:
                setattr(patched_class, "_supports_attention_backend", True)
                setattr(patched_class, "supports_multimodal", True)

        ModelRegistry.register_model(LOCATEANYTHING_ARCH, LOCATEANYTHING_VLLM_BACKEND)

    return {
        "architectures": architectures,
        "registered": LOCATEANYTHING_ARCH in architectures,
        "patched_class": getattr(patched_class, "__name__", None),
        "backend": LOCATEANYTHING_VLLM_BACKEND,
    }
