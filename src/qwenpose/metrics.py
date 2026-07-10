from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image

from qwenpose.schemas import SCHEMA_INDICES, SCHEMA_KEYPOINTS, UNION_SIGMAS


AP_DATASETS = {"coco", "crowdpose", "refhuman", "aic"}
PCKH_DATASETS = {"mpii"}
OKS_THRESHOLDS = [round(float(v), 2) for v in np.arange(0.5, 0.96, 0.05)]


@dataclass
class GTInstance:
    dataset: str
    image_id: str
    schema: str
    width: float
    height: float
    bbox_xyxy: list[float]
    keypoints: np.ndarray
    valid: np.ndarray
    area: float
    difficulty: str = ""
    head_size: float = 0.0


@dataclass
class PredInstance:
    dataset: str
    image_id: str
    schema: str
    keypoints: np.ndarray
    scores: np.ndarray
    score: float
    bbox_xyxy: list[float]


def load_prediction_rows(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        return payload["rows"]
    raise ValueError(f"Unsupported prediction file format: {path}")


def normalize_image_id(dataset: str, image_id: object) -> str:
    value = str(image_id)
    if dataset.lower() in AP_DATASETS and value.isdigit():
        return str(int(value))
    return value


def prediction_rows_to_instances(rows: Iterable[dict[str, Any]]) -> list[PredInstance]:
    preds: list[PredInstance] = []
    for row in rows:
        dataset = str(row.get("dataset") or row.get("format") or "").lower()
        image_id = normalize_image_id(dataset, row.get("image_id"))
        schema = str(row.get("schema") or "")
        if schema not in SCHEMA_INDICES:
            continue
        schema_indices = [int(v) for v in SCHEMA_INDICES[schema].tolist()]
        for pred in row.get("predictions", []):
            kpts = _prediction_schema_keypoints(pred, schema, schema_indices)
            if kpts is None:
                continue
            coords = kpts[:, :2].astype(np.float64)
            scores = kpts[:, 2].astype(np.float64)
            person_score = float(pred.get("person_score", pred.get("score", 0.0)))
            mean_kpt_score = float(np.mean(np.clip(scores, 0.0, 1.0))) if scores.size else 0.0
            score = person_score * max(mean_kpt_score, 1e-3)
            preds.append(
                PredInstance(
                    dataset=dataset,
                    image_id=image_id,
                    schema=schema,
                    keypoints=coords,
                    scores=scores,
                    score=score,
                    bbox_xyxy=[float(v) for v in pred.get("bbox_2d", pred.get("bbox_xyxy", []))[:4]],
                )
            )
    return preds


def _prediction_schema_keypoints(
    pred: dict[str, Any],
    schema: str,
    schema_indices: list[int],
) -> np.ndarray | None:
    flat = pred.get("schema_keypoints_flat") or pred.get("keypoints_flat")
    if flat:
        arr = np.asarray(flat, dtype=np.float64).reshape(-1, 3)
        return arr[: len(schema_indices)] if arr.shape[0] >= len(schema_indices) else None
    schema_kpts = pred.get("schema_keypoints")
    if schema_kpts:
        arr = np.asarray(
            [[kp.get("x", 0.0), kp.get("y", 0.0), kp.get("score", 0.0)] for kp in schema_kpts],
            dtype=np.float64,
        )
        return arr[: len(schema_indices)] if arr.shape[0] >= len(schema_indices) else None
    union_kpts = pred.get("keypoints")
    if union_kpts:
        arr = np.asarray(union_kpts, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[0] > max(schema_indices) and arr.shape[1] >= 3:
            return arr[schema_indices, :3]
    return None


def targets_to_gt_instances(
    targets: Iterable[dict[str, Any]],
    *,
    crowd_index_by_image: dict[str, float] | None = None,
) -> list[GTInstance]:
    instances: list[GTInstance] = []
    for target in targets:
        dataset = str(target.get("dataset", "")).lower()
        image_id = normalize_image_id(dataset, target.get("image_id"))
        schema = str(target.get("schema") or "")
        if schema not in SCHEMA_INDICES:
            continue
        width = float(target.get("width", 0.0))
        height = float(target.get("height", 0.0))
        boxes = _as_numpy(target.get("boxes"), dtype=np.float64)
        loss_areas_value = target.get("loss_areas")
        loss_areas = (
            _as_numpy(loss_areas_value, dtype=np.float64)
            if loss_areas_value is not None
            else None
        )
        keypoints = _as_numpy(target.get("keypoints"), dtype=np.float64)
        valid = _as_numpy(target.get("keypoint_valid"), dtype=bool)
        if boxes.ndim != 2 or keypoints.ndim != 3 or valid.ndim != 2:
            continue
        schema_indices = [int(v) for v in SCHEMA_INDICES[schema].tolist()]
        for inst_idx in range(min(boxes.shape[0], keypoints.shape[0], valid.shape[0])):
            valid_schema = valid[inst_idx, schema_indices].astype(bool)
            if not bool(valid_schema.any()):
                continue
            box_norm = boxes[inst_idx].astype(np.float64)
            bbox_xyxy = [
                float(box_norm[0] * width),
                float(box_norm[1] * height),
                float(box_norm[2] * width),
                float(box_norm[3] * height),
            ]
            kpts = keypoints[inst_idx, schema_indices, :3].astype(np.float64)
            kpts[:, 0] *= width
            kpts[:, 1] *= height
            if loss_areas is not None and inst_idx < int(loss_areas.shape[0]):
                area = max(float(loss_areas[inst_idx]) * width * height, 1.0)
            else:
                area = max(
                    (bbox_xyxy[2] - bbox_xyxy[0])
                    * (bbox_xyxy[3] - bbox_xyxy[1]),
                    1.0,
                )
            crowd_index = (crowd_index_by_image or {}).get(image_id)
            instances.append(
                GTInstance(
                    dataset=dataset,
                    image_id=image_id,
                    schema=schema,
                    width=width,
                    height=height,
                    bbox_xyxy=bbox_xyxy,
                    keypoints=kpts[:, :2],
                    valid=valid_schema,
                    area=area,
                    difficulty=_crowdpose_difficulty(crowd_index) if dataset == "crowdpose" else "",
                    head_size=_infer_head_size(schema, kpts[:, :2], valid_schema, target.get("scale")),
                )
            )
    return instances


def _as_numpy(value: Any, *, dtype: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(dtype)
    return np.asarray(value, dtype=dtype)


def _crowdpose_difficulty(crowd_index: float | None) -> str:
    if crowd_index is None:
        return ""
    value = float(crowd_index)
    if value < 0.1:
        return "easy"
    if value < 0.8:
        return "medium"
    return "hard"


def _schema_sigmas(schema: str) -> np.ndarray:
    indices = [int(v) for v in SCHEMA_INDICES[schema].tolist()]
    return UNION_SIGMAS[indices].detach().cpu().numpy().astype(np.float64)


def _compute_oks(gt: GTInstance, pred: PredInstance) -> float:
    if gt.schema != pred.schema or gt.keypoints.shape[0] != pred.keypoints.shape[0]:
        return 0.0
    valid = gt.valid.astype(bool)
    if not bool(valid.any()):
        return 0.0
    sigmas = _schema_sigmas(gt.schema)[valid]
    variances = (sigmas * 2.0) ** 2
    d2 = np.sum((pred.keypoints[valid] - gt.keypoints[valid]) ** 2, axis=1)
    e = d2 / np.maximum(variances, 1e-12) / max(float(gt.area), 1.0) / 2.0
    return float(np.mean(np.exp(-e)))


def compute_oks_ap(
    gt_instances: list[GTInstance],
    pred_instances: list[PredInstance],
) -> dict[str, float]:
    total_gt = len(gt_instances)
    if total_gt == 0:
        return {"num_gt": 0, "num_predictions": len(pred_instances), "AP": 0.0, "AP50": 0.0, "AP75": 0.0, "AR": 0.0}
    preds = sorted(pred_instances, key=lambda item: item.score, reverse=True)
    gt_by_image: dict[str, list[GTInstance]] = defaultdict(list)
    for gt in gt_instances:
        gt_by_image[gt.image_id].append(gt)

    ap_by_threshold: dict[float, float] = {}
    recall_by_threshold: dict[float, float] = {}
    for threshold in OKS_THRESHOLDS:
        matched: set[tuple[str, int]] = set()
        tp: list[float] = []
        fp: list[float] = []
        for pred in preds:
            candidates = gt_by_image.get(pred.image_id, [])
            best_idx = -1
            best_oks = threshold
            for gt_idx, gt in enumerate(candidates):
                key = (pred.image_id, gt_idx)
                if key in matched:
                    continue
                oks = _compute_oks(gt, pred)
                if oks >= best_oks:
                    best_oks = oks
                    best_idx = gt_idx
            if best_idx >= 0:
                matched.add((pred.image_id, best_idx))
                tp.append(1.0)
                fp.append(0.0)
            else:
                tp.append(0.0)
                fp.append(1.0)
        if not tp:
            ap_by_threshold[threshold] = 0.0
            recall_by_threshold[threshold] = 0.0
            continue
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recalls = tp_cum / max(total_gt, 1)
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
        ap_by_threshold[threshold] = _average_precision(recalls, precisions)
        recall_by_threshold[threshold] = float(recalls[-1])

    metrics = {
        "num_gt": float(total_gt),
        "num_predictions": float(len(preds)),
        "AP": float(np.mean(list(ap_by_threshold.values()))),
        "AP50": ap_by_threshold.get(0.5, 0.0),
        "AP75": ap_by_threshold.get(0.75, 0.0),
        "AR": float(np.mean(list(recall_by_threshold.values()))),
        "AR50": recall_by_threshold.get(0.5, 0.0),
        "AR75": recall_by_threshold.get(0.75, 0.0),
    }
    return metrics


def _average_precision(recalls: np.ndarray, precisions: np.ndarray) -> float:
    if recalls.size == 0:
        return 0.0
    recall_grid = np.linspace(0.0, 1.0, 101)
    values = []
    for recall in recall_grid:
        mask = recalls >= recall
        values.append(float(np.max(precisions[mask])) if bool(mask.any()) else 0.0)
    return float(np.mean(values))


def _infer_head_size(schema: str, keypoints: np.ndarray, valid: np.ndarray, scale: Any = None) -> float:
    names = SCHEMA_KEYPOINTS.get(schema, [])
    try:
        top_idx = names.index("head_top")
        neck_idx = names.index("upper_neck") if "upper_neck" in names else names.index("neck")
        if valid[top_idx] and valid[neck_idx]:
            dist = float(np.linalg.norm(keypoints[top_idx] - keypoints[neck_idx]))
            if dist > 1.0:
                return dist
    except ValueError:
        pass
    try:
        value = float(scale)
        if value > 0:
            return value * 200.0 * 0.6
    except Exception:
        pass
    xs = keypoints[valid, 0]
    ys = keypoints[valid, 1]
    if xs.size:
        return max(float(max(xs.max() - xs.min(), ys.max() - ys.min()) * 0.2), 1.0)
    return 1.0


def compute_mpii_pckh(
    gt_instances: list[GTInstance],
    pred_instances: list[PredInstance],
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    if not gt_instances:
        return {"num_gt": 0, "num_predictions": len(pred_instances), "PCKh@0.5": 0.0}
    pred_by_image: dict[str, list[PredInstance]] = defaultdict(list)
    for pred in sorted(pred_instances, key=lambda item: item.score, reverse=True):
        pred_by_image[pred.image_id].append(pred)

    schema = gt_instances[0].schema
    names = SCHEMA_KEYPOINTS.get(schema, [])
    correct = np.zeros(len(names), dtype=np.float64)
    counts = np.zeros(len(names), dtype=np.float64)
    used_by_image: dict[str, set[int]] = defaultdict(set)

    for gt in gt_instances:
        preds = pred_by_image.get(gt.image_id, [])
        best_idx = -1
        best_oks = -1.0
        for pred_idx, pred in enumerate(preds):
            if pred_idx in used_by_image[gt.image_id]:
                continue
            oks = _compute_oks(gt, pred)
            if oks > best_oks:
                best_oks = oks
                best_idx = pred_idx
        if best_idx < 0:
            counts += gt.valid.astype(np.float64)
            continue
        used_by_image[gt.image_id].add(best_idx)
        pred = preds[best_idx]
        distances = np.linalg.norm(pred.keypoints - gt.keypoints, axis=1)
        normalizer = max(float(gt.head_size), 1.0)
        joint_hits = distances <= float(threshold) * normalizer
        counts += gt.valid.astype(np.float64)
        correct += (joint_hits & gt.valid).astype(np.float64)

    per_joint = {
        name: (float(correct[idx] / counts[idx]) if counts[idx] > 0 else 0.0)
        for idx, name in enumerate(names)
    }
    total_correct = float(correct.sum())
    total_count = float(counts.sum())
    return {
        "num_gt": float(len(gt_instances)),
        "num_predictions": float(len(pred_instances)),
        "PCKh@0.5": total_correct / total_count if total_count > 0 else 0.0,
        "per_joint": per_joint,
    }


def load_crowdpose_crowd_index(dataset_root: Path, split: str) -> dict[str, float]:
    ann_path = Path(dataset_root) / "crowdpose" / "annotations" / f"mmpose_crowdpose_{split}.json"
    if not ann_path.is_file():
        return {}
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    return {str(img["id"]): float(img.get("crowdIndex", 0.0)) for img in data.get("images", [])}


def load_annotation_gt_instances(
    dataset_root: Path,
    datasets: Iterable[str],
    split: str,
    *,
    image_ids_by_dataset: dict[str, set[str]] | None = None,
) -> list[GTInstance]:
    instances: list[GTInstance] = []
    dataset_root = Path(dataset_root)
    for dataset in datasets:
        name = dataset.lower()
        wanted = (image_ids_by_dataset or {}).get(name)
        if name == "coco":
            coco_split = "val2017" if split == "val" else "train2017"
            instances.extend(_load_coco_like_gt(dataset_root / "coco" / "annotations" / f"person_keypoints_{coco_split}.json", name, "COCO17", wanted))
        elif name == "crowdpose":
            instances.extend(_load_coco_like_gt(dataset_root / "crowdpose" / "annotations" / f"mmpose_crowdpose_{split}.json", name, "CrowdPose14", wanted))
        elif name == "refhuman":
            instances.extend(_load_coco_like_gt(dataset_root / "refhuman" / f"RefHuman_{split}.json", name, "COCO17", wanted, refhuman=True))
        elif name == "mpii":
            instances.extend(_load_mpii_gt(dataset_root / "mpii" / "annotations" / f"mpii_{split}.json", wanted))
        elif name == "aic":
            instances.extend(_load_aic_gt(dataset_root / "aic", split, wanted))
    return instances


def _load_coco_like_gt(
    ann_path: Path,
    dataset: str,
    schema: str,
    wanted_image_ids: set[str] | None,
    *,
    refhuman: bool = False,
) -> list[GTInstance]:
    if not ann_path.is_file():
        return []
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    images = {str(img["id"]): img for img in data.get("images", [])}
    crowd_index = {str(img["id"]): float(img.get("crowdIndex", 0.0)) for img in data.get("images", [])}
    instances: list[GTInstance] = []
    for ann in data.get("annotations", []):
        image_id = normalize_image_id(dataset, ann.get("image_id"))
        if wanted_image_ids is not None and image_id not in wanted_image_ids:
            continue
        if ann.get("iscrowd", 0) or ann.get("num_keypoints", 0) <= 0:
            continue
        img = images.get(image_id)
        if img is None:
            continue
        width = float(img.get("width", 0.0))
        height = float(img.get("height", 0.0))
        kpts, valid = _flat_to_schema_keypoints(ann.get("keypoints", []), schema)
        if not bool(valid.any()):
            continue
        bbox = ann.get("bbox", [0.0, 0.0, 0.0, 0.0])
        bbox_xyxy = [float(bbox[0]), float(bbox[1]), float(bbox[0]) + max(float(bbox[2]), 0.0), float(bbox[1]) + max(float(bbox[3]), 0.0)]
        area = float(ann.get("area", max((bbox_xyxy[2] - bbox_xyxy[0]) * (bbox_xyxy[3] - bbox_xyxy[1]), 1.0)))
        instances.append(
            GTInstance(
                dataset=dataset,
                image_id=normalize_image_id(dataset, img.get("id") if refhuman else image_id),
                schema=schema,
                width=width,
                height=height,
                bbox_xyxy=bbox_xyxy,
                keypoints=kpts,
                valid=valid,
                area=max(area, 1.0),
                difficulty=_crowdpose_difficulty(crowd_index.get(image_id)) if dataset == "crowdpose" else "",
                head_size=_infer_head_size(schema, kpts, valid),
            )
        )
    return instances


def _flat_to_schema_keypoints(flat: list[float] | list[int], schema: str) -> tuple[np.ndarray, np.ndarray]:
    count = len(SCHEMA_KEYPOINTS[schema])
    arr = np.asarray(flat, dtype=np.float64).reshape(-1, 3)
    arr = arr[:count]
    if arr.shape[0] < count:
        padded = np.zeros((count, 3), dtype=np.float64)
        padded[: arr.shape[0]] = arr
        arr = padded
    valid = arr[:, 2] > 0
    return arr[:, :2], valid


def _mpii_bbox_from_center_scale(
    center: Any,
    scale: Any,
    *,
    scale_multiplier: float = 1.25,
) -> list[float] | None:
    """MMPose-compatible MPII padded bbox for metric instance matching."""
    if center is None or len(center) < 2:
        return None
    try:
        scale_value = float(scale)
        cx = float(center[0]) - 1.0
        cy = float(center[1]) - 1.0 + 15.0 * scale_value
        side = scale_value * 200.0 * float(scale_multiplier)
    except Exception:
        return None
    if not math.isfinite(cx) or not math.isfinite(cy) or not math.isfinite(side) or side <= 1.0:
        return None
    half = side * 0.5
    return [cx - half, cy - half, cx + half, cy + half]


def _load_mpii_gt(ann_path: Path, wanted_image_ids: set[str] | None) -> list[GTInstance]:
    if not ann_path.is_file():
        return []
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    instances: list[GTInstance] = []
    for ann in data:
        image_id = str(ann.get("image"))
        if wanted_image_ids is not None and image_id not in wanted_image_ids:
            continue
        joints = np.asarray(ann.get("joints", []), dtype=np.float64)
        valid = np.asarray(ann.get("joints_vis", []), dtype=np.float64) > 0
        if joints.ndim != 2 or joints.shape[0] < 16:
            continue
        joints = joints[:16, :2]
        valid = valid[:16]
        if not bool(valid.any()):
            continue
        bbox = _mpii_bbox_from_center_scale(ann.get("center"), ann.get("scale"))
        if bbox is None:
            xs = joints[valid, 0]
            ys = joints[valid, 1]
            bbox = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
        try:
            base_side = float(ann.get("scale")) * 200.0
            area = max(base_side * base_side * 0.53, 1.0)
        except Exception:
            area = max((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), 1.0)
        instances.append(
            GTInstance(
                dataset="mpii",
                image_id=image_id,
                schema="MPII16",
                width=0.0,
                height=0.0,
                bbox_xyxy=bbox,
                keypoints=joints,
                valid=valid,
                area=area,
                head_size=_infer_head_size("MPII16", joints, valid, ann.get("scale")),
            )
        )
    return instances


def _load_aic_gt(root: Path, split: str, wanted_image_ids: set[str] | None) -> list[GTInstance]:
    try:
        from qwenpose.data import resolve_aic_annotation_path, resolve_aic_image_root
    except Exception:
        return []
    try:
        ann_path = resolve_aic_annotation_path(root, split)
        image_root = resolve_aic_image_root(root, split)
    except FileNotFoundError:
        return []
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    instances: list[GTInstance] = []
    for item in data:
        image_id = normalize_image_id("aic", item.get("image_id"))
        if wanted_image_ids is not None and image_id not in wanted_image_ids:
            continue
        image_path = image_root / f"{image_id}.jpg"
        if image_path.is_file():
            with Image.open(image_path) as img:
                width, height = [float(v) for v in img.size]
        else:
            width = height = 0.0
        for human_id, box in item.get("human_annotations", {}).items():
            flat = item.get("keypoint_annotations", {}).get(human_id)
            if flat is None:
                continue
            kpts, valid = _flat_to_aic_keypoints(flat)
            if not bool(valid.any()):
                continue
            bbox = [float(v) for v in box[:4]]
            area = max((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), 1.0)
            instances.append(
                GTInstance(
                    dataset="aic",
                    image_id=normalize_image_id("aic", image_id),
                    schema="AIC14",
                    width=width,
                    height=height,
                    bbox_xyxy=bbox,
                    keypoints=kpts,
                    valid=valid,
                    area=area,
                    head_size=_infer_head_size("AIC14", kpts, valid),
                )
            )
    return instances


def _flat_to_aic_keypoints(flat: list[float] | list[int]) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(flat, dtype=np.float64).reshape(-1, 3)[:14]
    valid = (arr[:, 2] > 0) & (arr[:, 2] < 3)
    return arr[:, :2], valid


def compute_pose_metrics(
    prediction_rows: list[dict[str, Any]],
    *,
    gt_instances: list[GTInstance] | None = None,
    dataset_root: Path | None = None,
    split: str = "val",
) -> dict[str, Any]:
    pred_instances = prediction_rows_to_instances(prediction_rows)
    datasets = sorted({str(row.get("dataset") or row.get("format") or "").lower() for row in prediction_rows if row.get("dataset") or row.get("format")})
    if gt_instances is None:
        if dataset_root is None:
            raise ValueError("dataset_root is required when gt_instances is not provided.")
        image_ids_by_dataset: dict[str, set[str]] = defaultdict(set)
        for row in prediction_rows:
            dataset = str(row.get("dataset") or row.get("format") or "").lower()
            if dataset:
                image_ids_by_dataset[dataset].add(normalize_image_id(dataset, row.get("image_id")))
        gt_instances = load_annotation_gt_instances(dataset_root, datasets, split, image_ids_by_dataset=image_ids_by_dataset)

    gt_by_dataset: dict[str, list[GTInstance]] = defaultdict(list)
    pred_by_dataset: dict[str, list[PredInstance]] = defaultdict(list)
    for gt in gt_instances:
        gt_by_dataset[gt.dataset].append(gt)
    for pred in pred_instances:
        pred_by_dataset[pred.dataset].append(pred)

    per_dataset: dict[str, Any] = {}
    overall_ap_gt: list[GTInstance] = []
    overall_ap_pred: list[PredInstance] = []
    for dataset in sorted(set(gt_by_dataset) | set(pred_by_dataset) | set(datasets)):
        if dataset in PCKH_DATASETS:
            per_dataset[dataset] = compute_mpii_pckh(gt_by_dataset.get(dataset, []), pred_by_dataset.get(dataset, []))
        elif dataset in AP_DATASETS:
            gt_ds = gt_by_dataset.get(dataset, [])
            pred_ds = pred_by_dataset.get(dataset, [])
            metrics = compute_oks_ap(gt_ds, pred_ds)
            if dataset == "crowdpose":
                for difficulty in ("easy", "medium", "hard"):
                    image_ids = {gt.image_id for gt in gt_ds if gt.difficulty == difficulty}
                    gt_part = [gt for gt in gt_ds if gt.difficulty == difficulty]
                    pred_part = [pred for pred in pred_ds if pred.image_id in image_ids]
                    diff_metrics = compute_oks_ap(gt_part, pred_part)
                    metrics[f"AP_{difficulty}"] = diff_metrics.get("AP", 0.0)
                    metrics[f"AP50_{difficulty}"] = diff_metrics.get("AP50", 0.0)
                    metrics[f"num_gt_{difficulty}"] = diff_metrics.get("num_gt", 0.0)
            per_dataset[dataset] = metrics
            overall_ap_gt.extend(gt_ds)
            overall_ap_pred.extend(pred_ds)
    overall_ap = compute_oks_ap(overall_ap_gt, overall_ap_pred) if overall_ap_gt else {}
    return {
        "overall_ap": overall_ap,
        "per_dataset": per_dataset,
        "num_gt": len(gt_instances),
        "num_predictions": len(pred_instances),
        "method": {
            "ap": "COCO-style OKS AP with thresholds 0.50:0.95 and schema-specific keypoint sigmas.",
            "pckh": "MPII PCKh@0.5 using head_top-to-upper_neck/head-neck distance when available.",
            "crowdpose": "CrowdPose easy/medium/hard split by image crowdIndex: <0.1, 0.1-0.8, >=0.8.",
        },
    }


def compute_pose_metrics_from_targets(
    prediction_rows: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    *,
    dataset_root: Path | None = None,
    split: str = "val",
) -> dict[str, Any]:
    crowd_index = load_crowdpose_crowd_index(dataset_root, split) if dataset_root is not None else {}
    gt_instances = targets_to_gt_instances(targets, crowd_index_by_image=crowd_index)
    return compute_pose_metrics(prediction_rows, gt_instances=gt_instances, dataset_root=dataset_root, split=split)
