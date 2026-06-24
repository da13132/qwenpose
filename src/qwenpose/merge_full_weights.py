from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

from qwenpose.qwen_lora import load_qwen_model
from qwenpose.train_pose import CHECKPOINT_PAYLOAD_NAME, checkpoint_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a QwenPose LoRA checkpoint into full deployable weights.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="checkpoint dir, qwenpose_checkpoint.pt, or run dir")
    parser.add_argument("--base_model_path", type=Path, required=True, help="base Qwen model path")
    parser.add_argument("--output_dir", type=Path, required=True, help="output dir for merged full weights")
    parser.add_argument("--qwen_dtype", type=str, default="bfloat16")
    parser.add_argument("--qwen_attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_checkpoint(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {path}")
    if (path / CHECKPOINT_PAYLOAD_NAME).is_file():
        return path

    checkpoints: list[tuple[int, Path]] = []
    for child in list(path.glob("checkpoint-*")) + list(path.glob("checkpoint_step_*.pt")):
        step = checkpoint_step(child)
        if step is None:
            continue
        payload_path = child / CHECKPOINT_PAYLOAD_NAME if child.is_dir() else child
        if payload_path.is_file():
            checkpoints.append((step, child))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint-* or checkpoint_step_*.pt found under: {path}")
    return sorted(checkpoints)[-1][1]


def load_payload(path: Path) -> tuple[Path, dict]:
    resolved = resolve_checkpoint(path)
    payload_path = resolved / CHECKPOINT_PAYLOAD_NAME if resolved.is_dir() else resolved
    try:
        payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(payload_path, map_location="cpu")
    return resolved, payload


def locate_adapter_dir(resolved_checkpoint: Path) -> Path:
    candidates: list[Path] = []
    if resolved_checkpoint.is_dir():
        candidates.append(resolved_checkpoint / "qwen_lora_adapter")
        candidates.append(resolved_checkpoint.parent / "qwen_lora_adapter")
    else:
        candidates.append(resolved_checkpoint.parent / "qwen_lora_adapter")
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "adapter_config.json").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find qwen_lora_adapter beside the checkpoint. "
        f"Checked: {', '.join(str(path) for path in candidates)}"
    )


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            existing = list(path.iterdir())
            if existing:
                raise FileExistsError(f"Output dir already exists and is not empty: {path}")
        else:
            import shutil

            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def build_merged_payload(source_payload: dict, resolved_checkpoint: Path, base_model_path: Path) -> dict:
    merged_payload = {
        key: value
        for key, value in source_payload.items()
        if key not in {"optimizer", "scaler", "rng_state", "training_state", "qwen_trainable"}
    }
    merged_payload["backbone_name"] = str(source_payload.get("backbone_name", "qwen3vl"))
    merged_payload["backbone_merged"] = True
    merged_payload["checkpoint_format"] = "qwenpose-merged-v1"
    merged_payload["merge_metadata"] = {
        "source_checkpoint": str(resolved_checkpoint),
        "base_model_path": str(base_model_path),
        "merged_at": datetime.now().isoformat(timespec="seconds"),
    }
    return merged_payload


def save_metadata(output_dir: Path, payload: dict, resolved_checkpoint: Path, adapter_dir: Path) -> None:
    metadata = {
        "checkpoint_file": CHECKPOINT_PAYLOAD_NAME,
        "source_checkpoint": str(resolved_checkpoint),
        "source_adapter_dir": str(adapter_dir),
        "step": int(payload.get("step", 0)),
        "backbone_name": str(payload.get("backbone_name", "qwen3vl")),
        "backbone_merged": bool(payload.get("backbone_merged", False)),
        "pose_config": payload.get("pose_config"),
        "feature_config": payload.get("qwen_feature_config"),
        "feature_refiner_config": payload.get("qwen_feature_refiner_config"),
    }
    with (output_dir / "qwenpose_merged_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    with (output_dir / "qwenpose_state.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "step": int(payload.get("step", 0)),
                "checkpoint": str(output_dir),
                "payload": CHECKPOINT_PAYLOAD_NAME,
                "deepspeed_tag": None,
                "training_state": None,
                "backbone_merged": True,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.write("\n")


def merge_qwen_lora(
    *,
    resolved_checkpoint: Path,
    base_model_path: Path,
    output_dir: Path,
    qwen_dtype: str,
    qwen_attn_implementation: str,
    source_payload: dict,
) -> None:
    from peft import PeftModel

    adapter_dir = locate_adapter_dir(resolved_checkpoint)
    base_model, processor = load_qwen_model(
        str(base_model_path),
        dtype=qwen_dtype,
        attn_implementation=qwen_attn_implementation,
    )
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir), is_trainable=False)
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(output_dir, safe_serialization=True)
    processor.save_pretrained(output_dir)

    merged_payload = build_merged_payload(source_payload, resolved_checkpoint, base_model_path)
    torch.save(merged_payload, output_dir / CHECKPOINT_PAYLOAD_NAME)
    save_metadata(output_dir, merged_payload, resolved_checkpoint, adapter_dir)


def main() -> None:
    args = parse_args()
    resolved_checkpoint, source_payload = load_payload(args.checkpoint)
    backbone_name = str(source_payload.get("backbone_name", "qwen3vl"))
    if backbone_name != "qwen3vl":
        raise NotImplementedError(
            f"Only qwen3vl merge is currently supported, but checkpoint backbone_name={backbone_name!r}."
        )
    ensure_output_dir(args.output_dir, overwrite=args.overwrite)
    merge_qwen_lora(
        resolved_checkpoint=resolved_checkpoint,
        base_model_path=args.base_model_path,
        output_dir=args.output_dir,
        qwen_dtype=args.qwen_dtype,
        qwen_attn_implementation=args.qwen_attn_implementation,
        source_payload=source_payload,
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
