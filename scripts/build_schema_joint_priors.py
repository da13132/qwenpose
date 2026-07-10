#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from qwenpose.data import build_datasets
from qwenpose.model import canonical_joint_priors
from qwenpose.schemas import SCHEMA_KEYPOINTS, SCHEMA_TO_ID, UNION_KEYPOINTS, UNION_TO_ID


DATASET_TO_SCHEMA = {
    "coco": "COCO17",
    "mpii": "MPII16",
    "crowdpose": "CrowdPose14",
    "aic": "AIC14",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate per-schema box-relative joint priors.")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/schema_joint_priors.json"),
    )
    parser.add_argument(
        "--max-samples-per-dataset",
        type=int,
        default=None,
        help="Optional record cap for a fast approximate prior build.",
    )
    parser.add_argument("--record-cache-dir", type=Path, default=Path(".cache/qwenpose_records"))
    parser.add_argument("--disable-record-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_names = list(DATASET_TO_SCHEMA)
    mixed = build_datasets(
        dataset_root=args.dataset_root,
        names=dataset_names,
        max_instances=80,
        image_size=1,
        load_image_tensors=False,
        split="train",
        max_samples_per_dataset=args.max_samples_per_dataset,
        mixing_strategy="interleave",
        dataset_mix_weights="auto",
        record_cache_dir=args.record_cache_dir,
        disable_record_cache=args.disable_record_cache,
        show_progress=True,
    )

    if not hasattr(mixed, "names") or not hasattr(mixed, "datasets"):
        raise RuntimeError("Expected an interleaved multi-dataset object.")

    fallback = canonical_joint_priors()
    output: dict[str, dict[str, list[float]]] = {}
    output_counts: dict[str, dict[str, int]] = {}
    for dataset_name, dataset in zip(mixed.names, mixed.datasets):
        schema_name = DATASET_TO_SCHEMA[dataset_name]
        values: dict[int, list[torch.Tensor]] = defaultdict(list)
        for record in dataset.records:
            boxes = record.boxes_xyxy
            scales = record.box_context_scale[:, None].clamp(min=1e-4)
            center = (boxes[:, :2] + boxes[:, 2:]) * 0.5
            wh = (boxes[:, 2:] - boxes[:, :2]).clamp(min=1e-6) * scales
            condition_boxes = torch.cat(
                [center - wh * 0.5, center + wh * 0.5], dim=-1
            ).clamp(0.0, 1.0)
            condition_wh = (
                condition_boxes[:, 2:] - condition_boxes[:, :2]
            ).clamp(min=1e-6)
            relative_xy = (
                record.keypoints[..., :2] - condition_boxes[:, None, :2]
            ) / condition_wh[:, None, :]
            for union_idx in range(len(UNION_KEYPOINTS)):
                mask = record.keypoint_valid[:, union_idx]
                if mask.any():
                    values[union_idx].append(relative_xy[mask, union_idx].cpu())

        schema_payload: dict[str, list[float]] = {}
        schema_counts: dict[str, int] = {}
        for joint_name in SCHEMA_KEYPOINTS[schema_name]:
            union_idx = UNION_TO_ID[joint_name]
            if values[union_idx]:
                stacked = torch.cat(values[union_idx], dim=0)
                xy = stacked.median(dim=0).values.clamp(0.02, 0.98)
                schema_counts[joint_name] = int(stacked.shape[0])
            else:
                xy = fallback[union_idx]
                schema_counts[joint_name] = 0
            schema_payload[joint_name] = [round(float(xy[0]), 6), round(float(xy[1]), 6)]
        output[schema_name] = schema_payload
        output_counts[schema_name] = schema_counts

    # Structural checks catch left/right label or prior inversions before training.
    for schema_name, payload in output.items():
        for suffix in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle"):
            left = f"left_{suffix}"
            right = f"right_{suffix}"
            counts = output_counts[schema_name]
            enough_samples = min(counts.get(left, 0), counts.get(right, 0)) >= 100
            if (
                enough_samples
                and left in payload
                and right in payload
                and payload[left][0] <= payload[right][0]
            ):
                raise RuntimeError(
                    f"Unexpected left/right prior ordering for {schema_name}: "
                    f"{left}={payload[left]}, {right}={payload[right]}, "
                    f"counts=({counts[left]}, {counts[right]})"
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Saved schema joint priors to {args.output}")


if __name__ == "__main__":
    main()
