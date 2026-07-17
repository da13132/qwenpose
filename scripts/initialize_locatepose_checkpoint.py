#!/usr/bin/env python3
"""Create a step-0 LocatePose checkpoint after an architecture adjustment."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import torch

from qwenpose.model import QwenPoseConfig, QwenPoseModel


PAYLOAD_NAME = "qwenpose_checkpoint.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260717)
    return parser.parse_args()


def resolve_payload(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    direct = path / PAYLOAD_NAME
    if direct.is_file():
        return direct
    candidates: list[tuple[int, Path]] = []
    for checkpoint in path.glob("checkpoint-*"):
        try:
            step = int(checkpoint.name.rsplit("-", 1)[-1])
        except ValueError:
            continue
        payload = checkpoint / PAYLOAD_NAME
        if payload.is_file():
            candidates.append((step, payload))
    if not candidates:
        raise FileNotFoundError(f"No {PAYLOAD_NAME} found under {path}")
    return sorted(candidates)[-1][1]


def load_payload(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise TypeError(f"Invalid LocatePose checkpoint payload: {path}")
    return payload


def build_config(payload: dict[str, Any]) -> QwenPoseConfig:
    source = dict(payload.get("pose_config") or {})
    source["box_condition_scale"] = 1.15
    source["use_native_spatial_features"] = True
    source["use_detrpose_architecture"] = True
    source["legacy_checkpoint_compat"] = False
    source["person_confidence_rescue"] = False
    valid_fields = {field.name for field in fields(QwenPoseConfig)}
    kwargs = {key: value for key, value in source.items() if key in valid_fields}
    return QwenPoseConfig(**kwargs)


def migrate_external_multiscale_projection(
    source: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    num_levels: int,
) -> list[str]:
    migrated: list[str] = []
    for suffix in ("weight", "bias"):
        old_key = f"external_box_token_proj.0.{suffix}"
        if old_key not in source or old_key not in target:
            continue
        old = source[old_key]
        expected = target[old_key]
        repeated = old.repeat(num_levels)
        if repeated.shape == expected.shape:
            target[old_key] = repeated.to(dtype=expected.dtype)
            migrated.append(old_key)

    linear_key = "external_box_token_proj.1.weight"
    if linear_key in source and linear_key in target:
        old = source[linear_key]
        expected = target[linear_key]
        repeated = old.repeat(1, num_levels) / float(max(num_levels, 1))
        if repeated.shape == expected.shape:
            target[linear_key] = repeated.to(dtype=expected.dtype)
            migrated.append(linear_key)
    return migrated


def migrate_pre_pose_hidden_layers(
    source: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
) -> list[str]:
    migrated: list[str] = []
    for layer_index in (0, 2):
        for suffix in ("weight", "bias"):
            source_key = f"external_box_refine_head.net.{layer_index}.{suffix}"
            target_key = f"pre_pose_box_refine_head.net.{layer_index}.{suffix}"
            if source_key not in source or target_key not in target:
                continue
            if source[source_key].shape != target[target_key].shape:
                continue
            target[target_key] = source[source_key].to(dtype=target[target_key].dtype)
            migrated.append(target_key)
    return migrated


def main() -> None:
    args = parse_args()
    source_path = resolve_payload(args.source)
    payload = load_payload(source_path)
    source_state: dict[str, torch.Tensor] = payload["model"]

    torch.manual_seed(int(args.seed))
    config = build_config(payload)
    model = QwenPoseModel(config).cpu()
    initialized_state = model.state_dict()

    loaded: list[str] = []
    skipped_shape: list[str] = []
    skipped_missing: list[str] = []
    for key, value in source_state.items():
        if key not in initialized_state:
            skipped_missing.append(key)
            continue
        if value.shape != initialized_state[key].shape:
            skipped_shape.append(key)
            continue
        initialized_state[key] = value.to(dtype=initialized_state[key].dtype)
        loaded.append(key)

    special_migrations = migrate_external_multiscale_projection(
        source_state, initialized_state, model.num_feature_levels
    )
    special_migrations.extend(
        migrate_pre_pose_hidden_layers(source_state, initialized_state)
    )
    model.load_state_dict(initialized_state, strict=True)

    for key in ("optimizer", "scaler", "training_state", "rng_state"):
        payload.pop(key, None)
    payload["model"] = model.state_dict()
    payload["pose_config"] = asdict(config)
    payload["step"] = 0
    payload["deepspeed_managed"] = False
    payload["allow_partial_model_init"] = False
    payload["weight_only_init_from"] = str(source_path)
    payload["architecture_initialization"] = {
        "seed": int(args.seed),
        "loaded_tensors": len(loaded),
        "source_tensors": len(source_state),
        "new_model_tensors": len(initialized_state),
        "special_migrations": special_migrations,
        "skipped_shape": skipped_shape,
        "skipped_removed": skipped_missing,
    }

    checkpoint_dir = args.output.expanduser().resolve() / "checkpoint-0"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    destination = checkpoint_dir / PAYLOAD_NAME
    torch.save(payload, destination)
    state = {
        "step": 0,
        "checkpoint": str(checkpoint_dir),
        "payload": PAYLOAD_NAME,
        "initialized_from": str(source_path),
        "loaded_tensors": len(loaded),
        "special_migrations": special_migrations,
        "skipped_shape": skipped_shape,
        "skipped_removed_count": len(skipped_missing),
    }
    (checkpoint_dir / "qwenpose_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
