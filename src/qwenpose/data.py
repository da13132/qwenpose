from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Thread
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch.utils.data import ConcatDataset, Dataset

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional for minimal envs.
    tqdm = None

from .schemas import (
    SCHEMA_TO_ID,
    UNION_KEYPOINTS,
    UNION_TO_ID,
    coco_to_union,
    crowdpose_to_union,
    get_schema,
    mpii_to_union,
)


TASK_TO_ID = {"ALL_POSE": 0, "REF_POSE": 1}
RECORD_CACHE_VERSION = 11

ALL_POSE_PROMPT = "Locate all the instances that match the following description: person."

DATASET_BOX_CONTEXT_SCALE = {
    "coco": 1.15,
    "mpii": 1.00,
    "crowdpose": 1.10,
    "aic": 1.15,
    "refhuman": 1.15,
}

DATASET_BOX_JITTER = {
    "coco": (0.05, 0.03),
    "mpii": (0.10, 0.00),
    "crowdpose": (0.05, 0.02),
    "aic": (0.05, 0.03),
    "refhuman": (0.05, 0.03),
}


@dataclass(frozen=True)
class PoseAugmentConfig:
    enabled: bool = False
    horizontal_flip_prob: float = 0.5
    affine_prob: float = 0.8
    rotate_degrees: float = 15.0
    scale_min: float = 0.85
    scale_max: float = 1.15
    translate_fraction: float = 0.08
    color_prob: float = 0.8
    brightness: float = 0.20
    contrast: float = 0.20
    saturation: float = 0.20
    hue: float = 0.05
    grayscale_prob: float = 0.05
    blur_prob: float = 0.10
    blur_sigma_min: float = 0.10
    blur_sigma_max: float = 1.50
    erase_prob: float = 0.15
    erase_area_min: float = 0.02
    erase_area_max: float = 0.10


@dataclass(frozen=True)
class PoseAugmentSpec:
    matrix: torch.Tensor
    horizontal_flip: bool
    brightness_factor: float = 1.0
    contrast_factor: float = 1.0
    saturation_factor: float = 1.0
    hue_shift: float = 0.0
    grayscale: bool = False
    blur_sigma: float = 0.0
    erase_rects: tuple[tuple[int, int, int, int], ...] = ()


_LEFT_RIGHT_PAIRS = (
    ("left_eye", "right_eye"),
    ("left_ear", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_elbow", "right_elbow"),
    ("left_wrist", "right_wrist"),
    ("left_hip", "right_hip"),
    ("left_knee", "right_knee"),
    ("left_ankle", "right_ankle"),
)
UNION_FLIP_PERMUTATION = list(range(len(UNION_KEYPOINTS)))
for _left_name, _right_name in _LEFT_RIGHT_PAIRS:
    _left_idx = UNION_TO_ID[_left_name]
    _right_idx = UNION_TO_ID[_right_name]
    UNION_FLIP_PERMUTATION[_left_idx] = _right_idx
    UNION_FLIP_PERMUTATION[_right_idx] = _left_idx
UNION_FLIP_PERMUTATION = torch.tensor(UNION_FLIP_PERMUTATION, dtype=torch.long)


def _distributed_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _is_data_log_process() -> bool:
    return _distributed_rank() == 0


def _data_log(message: str) -> None:
    if _is_data_log_process():
        print(message, flush=True)


def _format_duration(seconds: float) -> str:
    total_seconds = max(int(round(float(seconds))), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _cache_lock_path(cache_path: Path) -> Path:
    return cache_path.with_name(f"{cache_path.name}.lock")


def _touch_path(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)


def _start_cache_build_heartbeat(lock_path: Path, interval_seconds: float = 5.0) -> tuple[Event, Thread]:
    stop_event = Event()

    def _heartbeat() -> None:
        _touch_path(lock_path)
        while not stop_event.wait(interval_seconds):
            try:
                _touch_path(lock_path)
            except Exception:
                # Best effort only; readers will still fall back to timeout logic.
                pass

    thread = Thread(target=_heartbeat, name=f"cache-heartbeat-{lock_path.name}", daemon=True)
    thread.start()
    return stop_event, thread


@dataclass
class PoseRecord:
    image_path: Path
    width: int
    height: int
    # Normalized boxes used to condition the PoseHead.
    boxes_xyxy: torch.Tensor
    # Normalized reference boxes used only for coordinate-loss scaling.
    loss_boxes_xyxy: torch.Tensor
    # Reference areas normalized by full-image area, used by OKS.
    loss_areas: torch.Tensor
    keypoints: torch.Tensor
    keypoint_valid: torch.Tensor
    visibility_valid: torch.Tensor
    box_context_scale: torch.Tensor
    box_jitter_scale: torch.Tensor
    box_jitter_shift: torch.Tensor
    schema: str
    task: str
    prompt: str
    ref_text: str = ""
    ref_target: int = -1
    dataset_name: str = ""
    image_id: str = ""


def xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, w, h = [float(v) for v in box]
    return [x, y, x + max(w, 0.0), y + max(h, 0.0)]


def clamp_box_xyxy(box: list[float], width: float, height: float) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = min(max(x1, 0.0), width)
    y1 = min(max(y1, 0.0), height)
    x2 = min(max(x2, 0.0), width)
    y2 = min(max(y2, 0.0), height)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def normalize_boxes(
    boxes: list[list[float]],
    width: float,
    height: float,
    *,
    clamp: bool = True,
) -> torch.Tensor:
    if not boxes:
        return torch.zeros(0, 4, dtype=torch.float32)
    out = torch.tensor(boxes, dtype=torch.float32)
    out[:, [0, 2]] /= max(float(width), 1.0)
    out[:, [1, 3]] /= max(float(height), 1.0)
    return out.clamp_(0.0, 1.0) if clamp else out


def normalize_areas(areas: list[float], width: float, height: float) -> torch.Tensor:
    if not areas:
        return torch.zeros(0, dtype=torch.float32)
    image_area = max(float(width) * float(height), 1.0)
    return torch.tensor(areas, dtype=torch.float32).div_(image_area).clamp_(min=1e-8)


def box_area_abs(box: list[float]) -> float:
    x1, y1, x2, y2 = [float(v) for v in box]
    return max(x2 - x1, 0.0) * max(y2 - y1, 0.0)


def box_from_keypoints(keypoints: torch.Tensor, valid: torch.Tensor) -> list[float] | None:
    if valid.sum().item() == 0:
        return None
    pts = keypoints[valid, :2]
    x1, y1 = pts.min(dim=0).values.tolist()
    x2, y2 = pts.max(dim=0).values.tolist()
    pad_x = max((x2 - x1) * 0.15, 0.03)
    pad_y = max((y2 - y1) * 0.15, 0.03)
    return [max(x1 - pad_x, 0.0), max(y1 - pad_y, 0.0), min(x2 + pad_x, 1.0), min(y2 + pad_y, 1.0)]


def _translation_matrix(tx: float, ty: float) -> torch.Tensor:
    return torch.tensor(
        [[1.0, 0.0, float(tx)], [0.0, 1.0, float(ty)], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )


def sample_pose_augment_spec(
    width: int,
    height: int,
    config: PoseAugmentConfig,
    rng: random.Random | None = None,
) -> PoseAugmentSpec:
    rng = rng or random
    width = max(int(width), 1)
    height = max(int(height), 1)
    matrix = torch.eye(3, dtype=torch.float64)
    do_affine = config.enabled and rng.random() < float(config.affine_prob)
    if do_affine:
        angle = math.radians(rng.uniform(-float(config.rotate_degrees), float(config.rotate_degrees)))
        scale = rng.uniform(float(config.scale_min), float(config.scale_max))
        tx = rng.uniform(-float(config.translate_fraction), float(config.translate_fraction)) * width
        ty = rng.uniform(-float(config.translate_fraction), float(config.translate_fraction)) * height
        cos_a = math.cos(angle) * scale
        sin_a = math.sin(angle) * scale
        rotate_scale = torch.tensor(
            [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
        )
        cx = width * 0.5
        cy = height * 0.5
        matrix = (
            _translation_matrix(tx, ty)
            @ _translation_matrix(cx, cy)
            @ rotate_scale
            @ _translation_matrix(-cx, -cy)
        )

    horizontal_flip = bool(config.enabled and rng.random() < float(config.horizontal_flip_prob))
    if horizontal_flip:
        flip = torch.tensor(
            [[-1.0, 0.0, float(width)], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
        )
        matrix = flip @ matrix

    brightness_factor = contrast_factor = saturation_factor = 1.0
    hue_shift = 0.0
    if config.enabled and rng.random() < float(config.color_prob):
        brightness_factor = rng.uniform(1.0 - float(config.brightness), 1.0 + float(config.brightness))
        contrast_factor = rng.uniform(1.0 - float(config.contrast), 1.0 + float(config.contrast))
        saturation_factor = rng.uniform(1.0 - float(config.saturation), 1.0 + float(config.saturation))
        hue_shift = rng.uniform(-float(config.hue), float(config.hue))

    grayscale = bool(config.enabled and rng.random() < float(config.grayscale_prob))
    blur_sigma = 0.0
    if config.enabled and rng.random() < float(config.blur_prob):
        blur_sigma = rng.uniform(float(config.blur_sigma_min), float(config.blur_sigma_max))

    erase_rects: list[tuple[int, int, int, int]] = []
    if config.enabled and rng.random() < float(config.erase_prob):
        count = 1 if rng.random() < 0.75 else 2
        image_area = float(width * height)
        for _ in range(count):
            area = image_area * rng.uniform(float(config.erase_area_min), float(config.erase_area_max))
            aspect = math.exp(rng.uniform(math.log(0.5), math.log(2.0)))
            erase_w = min(max(int(round(math.sqrt(area * aspect))), 1), width)
            erase_h = min(max(int(round(math.sqrt(area / aspect))), 1), height)
            x1 = rng.randint(0, max(width - erase_w, 0))
            y1 = rng.randint(0, max(height - erase_h, 0))
            erase_rects.append((x1, y1, x1 + erase_w, y1 + erase_h))

    return PoseAugmentSpec(
        matrix=matrix,
        horizontal_flip=horizontal_flip,
        brightness_factor=brightness_factor,
        contrast_factor=contrast_factor,
        saturation_factor=saturation_factor,
        hue_shift=hue_shift,
        grayscale=grayscale,
        blur_sigma=blur_sigma,
        erase_rects=tuple(erase_rects),
    )


def _transform_xy(points: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    if points.numel() == 0:
        return points.clone()
    dtype = points.dtype
    ones = torch.ones((*points.shape[:-1], 1), dtype=torch.float64, device=points.device)
    homogeneous = torch.cat([points.to(torch.float64), ones], dim=-1)
    transformed = homogeneous @ matrix.to(device=points.device, dtype=torch.float64).T
    return transformed[..., :2].to(dtype=dtype)


def transform_pose_boxes(
    boxes: torch.Tensor,
    matrix: torch.Tensor,
    width: int,
    height: int,
    *,
    clamp: bool,
) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.clone()
    scale = boxes.new_tensor([width, height, width, height])
    pixel = boxes * scale
    x1, y1, x2, y2 = pixel.unbind(dim=-1)
    corners = torch.stack(
        [
            torch.stack([x1, y1], dim=-1),
            torch.stack([x2, y1], dim=-1),
            torch.stack([x2, y2], dim=-1),
            torch.stack([x1, y2], dim=-1),
        ],
        dim=-2,
    )
    transformed = _transform_xy(corners, matrix)
    mins = transformed.min(dim=-2).values
    maxs = transformed.max(dim=-2).values
    out = torch.cat([mins, maxs], dim=-1) / scale
    return out.clamp_(0.0, 1.0) if clamp else out


def transform_pose_keypoints(
    keypoints: torch.Tensor,
    valid: torch.Tensor,
    visibility_valid: torch.Tensor,
    matrix: torch.Tensor,
    width: int,
    height: int,
    *,
    horizontal_flip: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if keypoints.numel() == 0:
        return keypoints.clone(), valid.clone(), visibility_valid.clone()
    out = keypoints.clone()
    pixel = out[..., :2] * out.new_tensor([width, height])
    out[..., :2] = _transform_xy(pixel, matrix) / out.new_tensor([width, height])
    if horizontal_flip:
        permutation = UNION_FLIP_PERMUTATION.to(device=out.device)
        out = out.index_select(1, permutation)
        valid = valid.index_select(1, permutation)
        visibility_valid = visibility_valid.index_select(1, permutation)
    in_bounds = (
        (out[..., 0] >= 0.0)
        & (out[..., 0] <= 1.0)
        & (out[..., 1] >= 0.0)
        & (out[..., 1] <= 1.0)
    )
    valid = valid & in_bounds
    visibility_valid = visibility_valid & in_bounds
    out[~valid] = 0.0
    return out, valid, visibility_valid


def apply_pose_image_augmentation(image: Image.Image, spec: PoseAugmentSpec) -> Image.Image:
    identity = torch.eye(3, dtype=spec.matrix.dtype, device=spec.matrix.device)
    if not torch.allclose(spec.matrix, identity):
        inverse = torch.linalg.inv(spec.matrix).cpu().numpy()
        coefficients = tuple(float(v) for v in inverse[:2].reshape(-1))
        transform_mode = getattr(Image, "Transform", Image).AFFINE
        image = image.transform(
            image.size,
            transform_mode,
            coefficients,
            resample=Image.Resampling.BILINEAR,
            fillcolor=(127, 127, 127),
        )
    if spec.brightness_factor != 1.0:
        image = ImageEnhance.Brightness(image).enhance(spec.brightness_factor)
    if spec.contrast_factor != 1.0:
        image = ImageEnhance.Contrast(image).enhance(spec.contrast_factor)
    if spec.saturation_factor != 1.0:
        image = ImageEnhance.Color(image).enhance(spec.saturation_factor)
    if spec.hue_shift != 0.0:
        hsv = np.asarray(image.convert("HSV"), dtype=np.uint8).copy()
        shift = int(round(spec.hue_shift * 255.0))
        hsv[..., 0] = (hsv[..., 0].astype(np.int16) + shift).astype(np.uint8)
        image = Image.fromarray(hsv, mode="HSV").convert("RGB")
    if spec.grayscale:
        image = ImageOps.grayscale(image).convert("RGB")
    if spec.blur_sigma > 0.0:
        image = image.filter(ImageFilter.GaussianBlur(radius=float(spec.blur_sigma)))
    if spec.erase_rects:
        array = np.asarray(image, dtype=np.uint8).copy()
        for x1, y1, x2, y2 in spec.erase_rects:
            array[y1:y2, x1:x2] = 127
        image = Image.fromarray(array, mode="RGB")
    return image


def pil_to_uint8_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def pil_to_local_rgb_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    resized = image.resize((int(image_size), int(image_size)), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy()).permute(2, 0, 1).contiguous()


def mpii_boxes_from_center_scale(
    center: list[float] | tuple[float, float],
    scale: float,
    width: float,
    height: float,
    *,
    padding: float = 1.25,
) -> tuple[list[float], list[float], float] | None:
    """Build MMPose-compatible MPII condition/loss boxes.

    MPII stores a MATLAB 1-based center and a scalar scale normalized by 200
    pixels. MMPose shifts the center downward by ``15 * scale`` pixels, converts
    the center to 0-based coordinates, uses ``scale * 200`` as the reference
    square, and applies 1.25 padding in ``GetBBoxCenterScale`` before affine
    cropping. We reproduce that geometry without applying an affine image warp.

    Returns:
        condition_box: 1.25-padded square used by the PoseHead.
        loss_box: unpadded ``scale * 200`` square used for coordinate scaling.
        loss_area: MMPose MPII area, ``base_side**2 * 0.53`` in pixel units.
    """
    if center is None or len(center) < 2:
        return None
    try:
        scale_value = float(scale)
        cx = float(center[0]) - 1.0
        cy = float(center[1]) - 1.0 + 15.0 * scale_value
    except Exception:
        return None
    if (
        not math.isfinite(cx)
        or not math.isfinite(cy)
        or not math.isfinite(scale_value)
        or scale_value <= 0.0
    ):
        return None

    base_side = scale_value * 200.0
    if base_side <= 1.0:
        return None
    condition_side = base_side * max(float(padding), 1e-4)

    def square(side: float) -> list[float]:
        half = side * 0.5
        return [cx - half, cy - half, cx + half, cy + half]

    condition_box = clamp_box_xyxy(square(condition_side), width, height)
    # Keep the full unpadded reference square even when it extends beyond the
    # image. MMPose's affine crop pads outside-image regions instead of shrinking
    # the person scale, and this box is used only as a loss normalization scale.
    loss_box = square(base_side)
    loss_area = max(base_side * base_side * 0.53, 1.0)
    return condition_box, loss_box, loss_area


def read_image_tensor(path: Path, image_size: int) -> tuple[torch.Tensor, int, int]:
    """Read one RGB image and create the fixed-size local visual tensor."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        width, height = img.size
        tensor = pil_to_local_rgb_tensor(img, image_size)
    return tensor, width, height


class PoseRecordDataset(Dataset):
    def __init__(
        self,
        records: list[PoseRecord],
        max_instances: int = 80,
        image_size: int = 640,
        load_image_tensors: bool = True,
        load_vision_images: bool = False,
        augment_config: PoseAugmentConfig | None = None,
        use_prompts: bool = True,
    ) -> None:
        self.records = records
        self.max_instances = max_instances
        self.image_size = image_size
        self.load_image_tensors = load_image_tensors
        self.load_vision_images = load_vision_images
        self.augment_config = augment_config or PoseAugmentConfig(enabled=False)
        self.use_prompts = bool(use_prompts)

    def __len__(self) -> int:
        return len(self.records)

    def _record_for_index(self, index: int) -> PoseRecord:
        return self.records[index]

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self._record_for_index(index)
        n = min(record.boxes_xyxy.shape[0], self.max_instances)
        boxes = record.boxes_xyxy[:n].clone()
        loss_boxes = record.loss_boxes_xyxy[:n].clone()
        loss_areas = record.loss_areas[:n].clone()
        keypoints = record.keypoints[:n].clone()
        valid = record.keypoint_valid[:n].clone()
        visibility_valid = record.visibility_valid[:n].clone()
        box_context_scale = record.box_context_scale[:n].clone()
        box_jitter_scale = record.box_jitter_scale[:n].clone()
        box_jitter_shift = record.box_jitter_shift[:n].clone()
        ref_target = record.ref_target if record.ref_target < n else -1

        needs_image = self.load_image_tensors or self.load_vision_images
        if needs_image:
            with Image.open(record.image_path) as handle:
                source_image = handle.convert("RGB").copy()
            width, height = source_image.size
            augment_rng = random.Random(
                random.getrandbits(64) ^ (_distributed_rank() << 32)
            )
            spec = sample_pose_augment_spec(
                width,
                height,
                self.augment_config,
                rng=augment_rng,
            )
            if self.augment_config.enabled:
                transformed_boxes = transform_pose_boxes(
                    boxes, spec.matrix, width, height, clamp=True
                )
                transformed_loss_boxes = transform_pose_boxes(
                    loss_boxes, spec.matrix, width, height, clamp=False
                )
                transformed_keypoints, transformed_valid, transformed_visibility_valid = (
                    transform_pose_keypoints(
                        keypoints,
                        valid,
                        visibility_valid,
                        spec.matrix,
                        width,
                        height,
                        horizontal_flip=spec.horizontal_flip,
                    )
                )
                determinant = abs(float(torch.det(spec.matrix[:2, :2])))
                transformed_loss_areas = loss_areas * max(determinant, 1e-8)
                keep = (
                    (transformed_boxes[:, 2] - transformed_boxes[:, 0] > 1e-4)
                    & (transformed_boxes[:, 3] - transformed_boxes[:, 1] > 1e-4)
                )
                if n > 0 and bool(keep.any()):
                    kept_indices = torch.nonzero(keep, as_tuple=False).flatten()
                    boxes = transformed_boxes[keep]
                    loss_boxes = transformed_loss_boxes[keep]
                    loss_areas = transformed_loss_areas[keep]
                    keypoints = transformed_keypoints[keep]
                    valid = transformed_valid[keep]
                    visibility_valid = transformed_visibility_valid[keep]
                    box_context_scale = box_context_scale[keep]
                    box_jitter_scale = box_jitter_scale[keep]
                    box_jitter_shift = box_jitter_shift[keep]
                    if ref_target >= 0:
                        matches = torch.nonzero(kept_indices == ref_target, as_tuple=False).flatten()
                        ref_target = int(matches[0]) if matches.numel() > 0 else -1
                elif n > 0:
                    # Do not produce an empty-supervision sample because of a rare
                    # aggressive random affine. Keep photometric augmentation only.
                    spec = PoseAugmentSpec(
                        matrix=torch.eye(3, dtype=torch.float64),
                        horizontal_flip=False,
                        brightness_factor=spec.brightness_factor,
                        contrast_factor=spec.contrast_factor,
                        saturation_factor=spec.saturation_factor,
                        hue_shift=spec.hue_shift,
                        grayscale=spec.grayscale,
                        blur_sigma=spec.blur_sigma,
                        erase_rects=spec.erase_rects,
                    )
            augmented_image = apply_pose_image_augmentation(source_image, spec)
            image = (
                pil_to_local_rgb_tensor(augmented_image, self.image_size)
                if self.load_image_tensors
                else torch.zeros(3, 1, 1, dtype=torch.float32)
            )
            vision_image = (
                pil_to_uint8_tensor(augmented_image)
                if self.load_vision_images
                else None
            )
            augmentation_matrix = spec.matrix.to(dtype=torch.float32)
            augmented = bool(self.augment_config.enabled)
        else:
            image = torch.zeros(3, 1, 1, dtype=torch.float32)
            vision_image = None
            augmentation_matrix = torch.eye(3, dtype=torch.float32)
            augmented = False

        return {
            "image": image,
            "vision_image": vision_image,
            "image_path": str(record.image_path),
            "schema_id": torch.tensor(SCHEMA_TO_ID[record.schema], dtype=torch.long),
            "task_id": torch.tensor(TASK_TO_ID[record.task], dtype=torch.long),
            "target": {
                "boxes": boxes,
                "loss_boxes": loss_boxes,
                "loss_areas": loss_areas,
                "keypoints": keypoints,
                "keypoint_valid": valid,
                "visibility_valid": visibility_valid,
                "box_context_scale": box_context_scale,
                "box_jitter_scale": box_jitter_scale,
                "box_jitter_shift": box_jitter_shift,
                "ref_target": torch.tensor(ref_target, dtype=torch.long),
                "dataset": record.dataset_name,
                "image_id": record.image_id,
                "schema": record.schema,
                "width": record.width,
                "height": record.height,
                "augmentation_matrix": augmentation_matrix,
                "augmented": augmented,
            },
            "prompt": record.prompt if self.use_prompts else "",
            "ref_text": record.ref_text if self.use_prompts else "",
        }


class EpochRandomRefHumanDataset(PoseRecordDataset):
    """Expose a fixed number of captions per person with epoch-wise rotation.

    Each person instance contributes at most ``captions_per_instance`` samples
    to one epoch. Caption order is shuffled reproducibly once per instance and
    rotated by epoch, so the default value of one never expands an instance
    into every expression but also does not freeze one expression forever.
    """

    def __init__(
        self,
        records: list[PoseRecord],
        *,
        captions_per_instance: int = 1,
        seed: int = 42,
        max_samples: int | None = None,
        **kwargs: Any,
    ) -> None:
        limit = max(int(captions_per_instance), 1)
        grouped_records: dict[
            tuple[str, tuple[float, float, float, float]], list[PoseRecord]
        ] = defaultdict(list)
        for record in records:
            grouped_records[_refhuman_instance_key(record)].append(record)

        representatives: list[PoseRecord] = []
        caption_pools: list[list[PoseRecord]] = []
        caption_slots: list[int] = []
        caption_orders: list[list[int]] = []
        sample_pool_indices: list[int] = []
        for pool_index, pool in enumerate(grouped_records.values()):
            order = list(range(len(pool)))
            random.Random(int(seed) + pool_index * 9973).shuffle(order)
            caption_pools.append(pool)
            caption_orders.append(order)
            for slot in range(min(limit, len(pool))):
                representatives.append(pool[order[slot]])
                caption_slots.append(slot)
                sample_pool_indices.append(pool_index)

        if max_samples is not None:
            keep = max(int(max_samples), 0)
            representatives = representatives[:keep]
            caption_slots = caption_slots[:keep]
            sample_pool_indices = sample_pool_indices[:keep]

        super().__init__(representatives, **kwargs)
        self.caption_pools = caption_pools
        self.caption_orders = caption_orders
        self.caption_slots = caption_slots
        self.sample_pool_indices = sample_pool_indices
        # DataLoader persistent workers retain their dataset copies. A shared
        # scalar lets the main process change captions without recreating them.
        self._shared_epoch = torch.zeros((), dtype=torch.long).share_memory_()

    def set_epoch(self, epoch: int) -> None:
        self._shared_epoch.fill_(max(int(epoch), 0))

    def _record_for_index(self, index: int) -> PoseRecord:
        pool_index = self.sample_pool_indices[index]
        pool = self.caption_pools[pool_index]
        order = self.caption_orders[pool_index]
        slot = self.caption_slots[index]
        epoch = int(self._shared_epoch.item())
        return pool[order[(epoch + slot) % len(order)]]


class InterleavedPoseDataset(Dataset):
    """Dataset-level weighted fair round-robin sampler.

    Manual weights are dataset traversal multipliers, not normalized
    probabilities: ``coco:3,refhuman:0.5`` contributes three complete COCO
    traversals and half of RefHuman to one epoch. Fractional traversals advance
    their start offset every epoch, so 0.5 visits the first half and then the
    second half instead of repeatedly sampling the same subset.
    """

    def __init__(
        self,
        named_datasets: list[tuple[str, Dataset]],
        weights: dict[str, float] | None,
        seed: int = 42,
        epoch_size: int | None = None,
    ) -> None:
        if not named_datasets:
            raise ValueError("InterleavedPoseDataset requires at least one dataset.")
        self.named_datasets = [(name.lower(), dataset) for name, dataset in named_datasets]
        self.names = [name for name, _ in self.named_datasets]
        self.datasets = [dataset for _, dataset in self.named_datasets]
        self.auto_weights = weights is None
        self.multipliers = (
            [1.0 for _ in self.datasets]
            if weights is None
            else [float(weights.get(name, 1.0)) for name in self.names]
        )
        basis = [self.sample_count_for_epoch(idx, 0) for idx in range(len(self.datasets))]
        self.schedule = self._build_weighted_schedule(basis)
        self.weights = [self.schedule.count(idx) for idx in range(len(self.datasets))]
        self.slot_offsets: list[int] = []
        self.schedule_slots_by_dataset: list[list[int]] = [[] for _ in self.datasets]
        running_counts = [0 for _ in self.datasets]
        for schedule_idx, dataset_idx in enumerate(self.schedule):
            self.slot_offsets.append(running_counts[dataset_idx])
            self.schedule_slots_by_dataset[dataset_idx].append(schedule_idx)
            running_counts[dataset_idx] += 1
        if not self.schedule:
            raise ValueError("Interleave schedule is empty.")
        self.epoch_size = int(epoch_size or sum(basis))
        self.offsets: list[int] = []
        self.strides: list[int] = []
        for dataset_idx, dataset in enumerate(self.datasets):
            n = len(dataset)
            rng = random.Random(seed + dataset_idx * 9973)
            self.offsets.append(rng.randrange(n) if n > 1 else 0)
            self.strides.append(self._coprime_stride(n, rng))

    def sample_count_for_epoch(self, dataset_idx: int, epoch: int) -> int:
        """Return this source's fixed per-epoch sample budget.

        ``ceil`` avoids silently dropping a tiny positive multiplier. The
        rotating start below guarantees that every source record is eventually
        reached; padding at the global-batch boundary is handled separately by
        the batch sampler.
        """
        del epoch  # Reserved for a future exact variable-size remainder mode.
        n = len(self.datasets[int(dataset_idx)])
        multiplier = self.multipliers[int(dataset_idx)]
        if n <= 0 or multiplier <= 0.0:
            return 0
        return max(int(math.ceil(n * multiplier - 1e-12)), 1)

    def sample_start_for_epoch(self, dataset_idx: int, epoch: int) -> int:
        """Return the continuation offset for fractional dataset traversals."""
        n = len(self.datasets[int(dataset_idx)])
        if n <= 0:
            return 0
        multiplier = self.multipliers[int(dataset_idx)]
        return int(math.floor(max(int(epoch), 0) * n * multiplier + 1e-12)) % n

    def sample_linear_indices_for_epoch(self, dataset_idx: int, epoch: int) -> list[int]:
        """Build a fixed-size budget from one non-overlapping traversal slice.

        For odd-sized datasets, a 0.5 slice alternates between floor/ceil unique
        records. The smaller half is padded only from itself, so it never borrows
        a record from the following epoch's half while DataLoader length remains
        stable for checkpoint resume and scheduler calculations.
        """
        dataset_idx = int(dataset_idx)
        epoch = max(int(epoch), 0)
        n = len(self.datasets[dataset_idx])
        multiplier = self.multipliers[dataset_idx]
        budget = self.sample_count_for_epoch(dataset_idx, epoch)
        if n <= 0 or multiplier <= 0.0 or budget <= 0:
            return []
        start = int(math.floor(epoch * n * multiplier + 1e-12))
        end = int(math.floor((epoch + 1) * n * multiplier + 1e-12))
        values = list(range(start, end))
        if not values:
            values = [start]
        if len(values) < budget:
            source = list(values)
            values.extend(source[idx % len(source)] for idx in range(budget - len(values)))
        return values[:budget]

    @staticmethod
    def _build_weighted_schedule(counts: list[int]) -> list[int]:
        total = sum(max(int(count), 0) for count in counts)
        if total <= 0:
            return []
        used = [0 for _ in counts]
        schedule: list[int] = []
        for step in range(total):
            candidates = [idx for idx, count in enumerate(counts) if used[idx] < count]
            best = max(
                candidates,
                key=lambda idx: (
                    ((step + 1) * counts[idx] / total) - used[idx],
                    counts[idx],
                    -idx,
                ),
            )
            schedule.append(best)
            used[best] += 1
        return schedule

    @staticmethod
    def _coprime_stride(n: int, rng: random.Random) -> int:
        if n <= 1:
            return 1
        stride = rng.randrange(1, n)
        while math.gcd(stride, n) != 1:
            stride += 1
            if stride >= n:
                stride = 1
        return stride

    def __len__(self) -> int:
        return self.epoch_size

    def set_epoch(self, epoch: int) -> None:
        for dataset in self.datasets:
            setter = getattr(dataset, "set_epoch", None)
            if callable(setter):
                setter(int(epoch))

    def dataset_index_at(self, index: int) -> int:
        schedule_idx = index % len(self.schedule)
        return self.schedule[schedule_idx]

    def global_index_for_dataset_linear(self, dataset_idx: int, local_linear: int) -> int:
        """Return a global index that maps to a dataset-local linear slot.

        DataLoader batch samplers can use this to assemble homogeneous batches
        while still reusing this dataset's deterministic offset/stride mapping.
        """
        dataset_idx = int(dataset_idx)
        local_linear = int(local_linear)
        if dataset_idx < 0 or dataset_idx >= len(self.datasets):
            raise IndexError(f"dataset_idx out of range: {dataset_idx}")
        slots = self.schedule_slots_by_dataset[dataset_idx]
        if not slots:
            raise ValueError(f"No schedule slots for dataset index {dataset_idx}")
        weight = max(int(self.weights[dataset_idx]), 1)
        slot = slots[local_linear % weight]
        round_idx = local_linear // weight
        return round_idx * len(self.schedule) + slot

    def __getitem__(self, index: int) -> dict[str, Any]:
        schedule_idx = index % len(self.schedule)
        round_idx = index // len(self.schedule)
        dataset_idx = self.schedule[schedule_idx]
        local_linear = round_idx * self.weights[dataset_idx] + self.slot_offsets[schedule_idx]
        dataset = self.datasets[dataset_idx]
        local_index = (self.offsets[dataset_idx] + local_linear * self.strides[dataset_idx]) % len(dataset)
        item = dataset[local_index]
        item["source_dataset"] = self.names[dataset_idx]
        return item

    def schedule_description(self, max_items: int = 32) -> str:
        shown = ",".join(self.names[idx] for idx in self.schedule[:max_items])
        if len(self.schedule) > max_items:
            shown += f",... ({len(self.schedule)} slots)"
        mode = "auto_one_traversal" if self.auto_weights else "manual_traversal_multiplier"
        total = max(sum(self.weights), 1)
        counts = ",".join(
            f"{name}:{multiplier:g}x={count}({count / total:.1%})"
            for name, multiplier, count in zip(self.names, self.multipliers, self.weights)
        )
        return f"{mode} counts=[{counts}] first=[{shown}]"


def set_pose_dataset_epoch(dataset: Dataset, epoch: int) -> None:
    """Propagate an epoch cursor through mixed/concatenated datasets."""
    setter = getattr(dataset, "set_epoch", None)
    if callable(setter):
        setter(int(epoch))
        return
    for child in getattr(dataset, "datasets", []):
        set_pose_dataset_epoch(child, int(epoch))


def pose_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    vision_images = [item["vision_image"] for item in batch]
    if all(image is None for image in vision_images):
        vision_images = None
    elif any(image is None for image in vision_images):
        raise ValueError("A batch cannot mix materialized and path-backed vision images.")
    return {
        "images": torch.stack([item["image"] for item in batch], dim=0),
        "vision_images": vision_images,
        "schema_ids": torch.stack([item["schema_id"] for item in batch], dim=0),
        "task_ids": torch.stack([item["task_id"] for item in batch], dim=0),
        "targets": [item["target"] for item in batch],
        "prompts": [item["prompt"] for item in batch],
        "ref_texts": [item.get("ref_text", "") for item in batch],
        "image_paths": [item["image_path"] for item in batch],
        "source_datasets": [item.get("source_dataset", item["target"].get("dataset", "")) for item in batch],
    }


def _record_if_valid(
    records: list[PoseRecord],
    image_path: Path,
    width: int,
    height: int,
    boxes: list[list[float]],
    keypoints: list[torch.Tensor],
    valid: list[torch.Tensor],
    schema: str,
    task: str,
    prompt: str,
    dataset_name: str,
    image_id: str,
    ref_text: str = "",
    ref_target: int = -1,
    *,
    loss_boxes: list[list[float]] | None = None,
    loss_areas: list[float] | None = None,
    visibility_valid: list[torch.Tensor] | None = None,
    box_context_scales: list[float] | None = None,
    box_jitter_scales: list[float] | None = None,
    box_jitter_shifts: list[float] | None = None,
) -> None:
    if not image_path.exists() or not boxes:
        return
    n = len(boxes)
    if not (len(keypoints) == len(valid) == n):
        raise ValueError("PoseRecord fields must have the same instance count.")
    loss_boxes = boxes if loss_boxes is None else loss_boxes
    loss_areas = [box_area_abs(box) for box in loss_boxes] if loss_areas is None else loss_areas
    visibility_valid = valid if visibility_valid is None else visibility_valid
    context_default = DATASET_BOX_CONTEXT_SCALE.get(dataset_name, 1.0)
    jitter_scale_default, jitter_shift_default = DATASET_BOX_JITTER.get(dataset_name, (0.0, 0.0))
    box_context_scales = [context_default] * n if box_context_scales is None else box_context_scales
    box_jitter_scales = [jitter_scale_default] * n if box_jitter_scales is None else box_jitter_scales
    box_jitter_shifts = [jitter_shift_default] * n if box_jitter_shifts is None else box_jitter_shifts
    if not all(
        len(items) == n
        for items in (
            loss_boxes,
            loss_areas,
            visibility_valid,
            box_context_scales,
            box_jitter_scales,
            box_jitter_shifts,
        )
    ):
        raise ValueError("PoseRecord auxiliary fields must match the box count.")

    records.append(
        PoseRecord(
            image_path=image_path,
            width=width,
            height=height,
            boxes_xyxy=normalize_boxes(boxes, width, height),
            loss_boxes_xyxy=normalize_boxes(loss_boxes, width, height, clamp=False),
            loss_areas=normalize_areas(loss_areas, width, height),
            keypoints=torch.stack(keypoints, dim=0),
            keypoint_valid=torch.stack(valid, dim=0),
            visibility_valid=torch.stack(visibility_valid, dim=0),
            box_context_scale=torch.tensor(box_context_scales, dtype=torch.float32),
            box_jitter_scale=torch.tensor(box_jitter_scales, dtype=torch.float32),
            box_jitter_shift=torch.tensor(box_jitter_shifts, dtype=torch.float32),
            schema=schema,
            task=task,
            prompt=prompt,
            ref_text=ref_text,
            ref_target=ref_target,
            dataset_name=dataset_name,
            image_id=image_id,
        )
    )


def aic_to_union(
    flat_keypoints: list[float] | list[int],
    image_width: float,
    image_height: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Treat every labeled AIC joint as a positive visibility target.

    In the local AIC annotations, both visible (`v == 1`) and occluded
    (`v == 2`) joints supervise coordinates and visibility. Missing/outside
    joints (`v >= 3`) remain excluded.
    """
    if isinstance(flat_keypoints, dict):
        flat_keypoints = list(flat_keypoints.values())
    spec = get_schema("AIC14")
    keypoints = torch.zeros(len(UNION_KEYPOINTS), 3, dtype=torch.float32)
    valid = torch.zeros(len(UNION_KEYPOINTS), dtype=torch.bool)
    visibility_valid = torch.zeros(len(UNION_KEYPOINTS), dtype=torch.bool)
    if len(flat_keypoints) != len(spec.keypoints) * 3:
        raise ValueError(f"AIC14 expects {len(spec.keypoints) * 3} values, got {len(flat_keypoints)}")
    width = max(float(image_width), 1.0)
    height = max(float(image_height), 1.0)
    for local_idx, union_idx in enumerate(spec.indices.tolist()):
        x, y, v = flat_keypoints[local_idx * 3 : local_idx * 3 + 3]
        visibility = float(v)
        if visibility <= 0 or visibility >= 3:
            continue
        keypoints[union_idx, 0] = float(x) / width
        keypoints[union_idx, 1] = float(y) / height
        keypoints[union_idx, 2] = 1.0
        valid[union_idx] = True
        visibility_valid[union_idx] = True
    return keypoints.clamp_(0.0, 1.0), valid, visibility_valid


def _source_signature(paths: list[Path]) -> list[dict[str, Any]]:
    signature = []
    for path in paths:
        stat = path.stat() if path.exists() else None
        signature.append(
            {
                "path": str(path.resolve()),
                "size": None if stat is None else stat.st_size,
                "mtime_ns": None if stat is None else stat.st_mtime_ns,
            }
        )
    return signature


def _record_cache_path(
    cache_dir: Path,
    dataset_name: str,
    root: Path,
    split: str,
    max_samples: int | None,
    source_paths: list[Path],
    extra_cache_key: dict[str, Any] | None = None,
) -> Path:
    payload = {
        "version": RECORD_CACHE_VERSION,
        "dataset": dataset_name.lower(),
        "root": str(root.resolve()),
        "split": split,
        "max_samples": max_samples,
        "sources": _source_signature(source_paths),
    }
    if extra_cache_key is not None:
        payload["extra"] = extra_cache_key
    digest = hashlib.blake2b(json.dumps(payload, sort_keys=True).encode("utf-8"), digest_size=16).hexdigest()
    return cache_dir / f"{dataset_name.lower()}_{split}_{digest}.pt"


def _load_records_cached(
    *,
    dataset_name: str,
    root: Path,
    split: str,
    max_samples: int | None,
    source_paths: list[Path],
    cache_dir: Path | None,
    disable_cache: bool,
    builder,
    extra_cache_key: dict[str, Any] | None = None,
) -> list[PoseRecord]:
    if cache_dir is None or disable_cache:
        _data_log(f"Building records without cache for {dataset_name} split={split}.")
        return builder()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _record_cache_path(
        cache_dir,
        dataset_name,
        root,
        split,
        max_samples,
        source_paths,
        extra_cache_key=extra_cache_key,
    )
    lock_path = _cache_lock_path(cache_path)
    if cache_path.exists():
        cache_size_mb = cache_path.stat().st_size / (1024 * 1024)
        _data_log(
            f"Loading record cache for {dataset_name} split={split}: "
            f"{cache_path} ({cache_size_mb:.1f} MB)"
        )
        started = time.perf_counter()
        records = torch.load(cache_path, map_location="cpu", weights_only=False)
        _data_log(
            f"Loaded record cache for {dataset_name} split={split}: "
            f"{cache_path} ({time.perf_counter() - started:.2f}s)"
        )
        return records
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    stale_lock_seconds = float(os.environ.get("QWENPOSE_RECORD_CACHE_STALE_LOCK_SECONDS", "7200"))
    wait_timeout = float(os.environ.get("QWENPOSE_RECORD_CACHE_WAIT_SECONDS", "1800"))
    wait_status_seconds = float(os.environ.get("QWENPOSE_RECORD_CACHE_WAIT_STATUS_SECONDS", "60"))
    if world_size > 1 and rank != 0:
        wait_started = time.perf_counter()
        last_status = wait_started
        while True:
            if cache_path.exists():
                records = torch.load(cache_path, map_location="cpu", weights_only=False)
                _data_log(
                    f"Loaded record cache for {dataset_name} split={split}: "
                    f"{cache_path} after waiting {time.perf_counter() - wait_started:.2f}s"
                )
                return records
            if lock_path.exists():
                try:
                    lock_age = time.time() - lock_path.stat().st_mtime
                except FileNotFoundError:
                    lock_age = 0.0
                now = time.perf_counter()
                if now - last_status >= wait_status_seconds:
                    _data_log(
                        f"Waiting for record cache build on rank 0: {dataset_name} split={split} "
                        f"(waited={_format_duration(now - wait_started)}, lock_age={_format_duration(lock_age)})"
                    )
                    last_status = now
                if lock_age <= stale_lock_seconds:
                    time.sleep(1.0)
                    continue
                _data_log(
                    f"Cache lock became stale for {dataset_name} split={split} "
                    f"(lock_age={_format_duration(lock_age)}); rebuilding on rank {rank}."
                )
                break
            if time.perf_counter() - wait_started >= wait_timeout:
                _data_log(f"Timed out waiting for record cache on rank {rank}; rebuilding {dataset_name} split={split}.")
                break
            time.sleep(1.0)
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(lock_fd)
    except FileExistsError:
        if cache_path.exists():
            records = torch.load(cache_path, map_location="cpu", weights_only=False)
            _data_log(f"Loaded record cache for {dataset_name} split={split}: {cache_path}")
            return records
        if rank == 0:
            _data_log(
                f"Another rank is already building record cache for {dataset_name} split={split}; "
                f"waiting on {lock_path}."
            )
        wait_started = time.perf_counter()
        while True:
            if cache_path.exists():
                records = torch.load(cache_path, map_location="cpu", weights_only=False)
                _data_log(
                    f"Loaded record cache for {dataset_name} split={split}: "
                    f"{cache_path} after waiting {time.perf_counter() - wait_started:.2f}s"
                )
                return records
            try:
                lock_age = time.time() - lock_path.stat().st_mtime
            except FileNotFoundError:
                lock_age = stale_lock_seconds + 1.0
            if lock_age > stale_lock_seconds:
                _data_log(
                    f"Cache lock became stale for {dataset_name} split={split} "
                    f"while waiting on {lock_path}; retrying cache build acquisition."
                )
                lock_path.unlink(missing_ok=True)
                return _load_records_cached(
                    dataset_name=dataset_name,
                    root=root,
                    split=split,
                    max_samples=max_samples,
                    source_paths=source_paths,
                    cache_dir=cache_dir,
                    disable_cache=disable_cache,
                    builder=builder,
                    extra_cache_key=extra_cache_key,
                )
            time.sleep(1.0)

    started = time.perf_counter()
    stop_event, heartbeat_thread = _start_cache_build_heartbeat(lock_path)
    try:
        _data_log(f"Building record cache for {dataset_name} split={split}: {cache_path}")
        records = builder()
        tmp_path = cache_path.with_suffix(".tmp")
        torch.save(records, tmp_path)
        tmp_path.replace(cache_path)
        _data_log(
            f"Saved record cache for {dataset_name} split={split}: "
            f"{cache_path} ({time.perf_counter() - started:.2f}s)"
        )
        return records
    finally:
        stop_event.set()
        heartbeat_thread.join(timeout=1.0)
        lock_path.unlink(missing_ok=True)


def _load_aic_image_sizes(
    *,
    root: Path,
    split: str,
    annotations: list[dict[str, Any]],
    image_root: Path,
    max_images: int | None,
    cache_dir: Path | None,
    disable_cache: bool,
    show_progress: bool,
) -> dict[str, tuple[int, int]]:
    ann_path = resolve_aic_annotation_path(root, split)
    source_paths = [ann_path, image_root]
    target_annotations = annotations if max_images is None else annotations[: max(int(max_images), 0)]
    cache_path = None
    if max_images is None and cache_dir is not None:
        cache_path = _record_cache_path(
            cache_dir,
            "aic_image_sizes",
            root,
            split,
            None,
            source_paths,
        )
    if cache_path is not None and cache_path.exists() and not disable_cache:
        started = time.perf_counter()
        size_map = torch.load(cache_path, map_location="cpu", weights_only=False)
        _data_log(
            f"Loaded AIC image-size cache for split={split}: {cache_path} "
            f"({time.perf_counter() - started:.2f}s)"
        )
        return size_map

    iterator = target_annotations
    progress_bar = None
    if show_progress and tqdm is not None and _is_data_log_process():
        progress_bar = tqdm(
            target_annotations,
            total=len(target_annotations),
            desc=f"aic sizes {split}",
            unit="image",
            dynamic_ncols=True,
            mininterval=0.5,
        )
        iterator = progress_bar

    started = time.perf_counter()
    size_map: dict[str, tuple[int, int]] = {}
    for index, item in enumerate(iterator, start=1):
        image_id = str(item["image_id"])
        image_path = image_root / f"{image_id}.jpg"
        if not image_path.exists():
            continue
        with Image.open(image_path) as img:
            size_map[image_id] = tuple(int(v) for v in img.size)
        if progress_bar is not None and index % 2000 == 0:
            elapsed = time.perf_counter() - started
            avg = elapsed / max(index, 1)
            eta = avg * max(len(target_annotations) - index, 0)
            progress_bar.set_postfix(
                cached=len(size_map),
                elapsed=_format_duration(elapsed),
                eta=_format_duration(eta),
                refresh=False,
            )
    if progress_bar is not None:
        progress_bar.close()

    if cache_path is not None and not disable_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(".tmp")
        torch.save(size_map, tmp_path)
        tmp_path.replace(cache_path)
        _data_log(
            f"Saved AIC image-size cache for split={split}: {cache_path} "
            f"({time.perf_counter() - started:.2f}s)"
        )
    return size_map


def _normalize_aic_split(split: str) -> str:
    split_name = str(split).lower()
    if split_name in {"val", "valid", "validation"}:
        return "val"
    # The local tree has the official train and validation files separately;
    # keep trainval as the historical train-only behavior unless a caller
    # explicitly asks for val.
    return "train"


def resolve_aic_annotation_path(root: Path, split: str = "train") -> Path:
    normalized_split = _normalize_aic_split(split)
    if normalized_split == "val":
        candidates = [
            root / "ai_challenger_keypoint_validation_20170911" / "keypoint_validation_annotations_20170911.json",
        ]
    else:
        candidates = [
            root / "ai_challenger_keypoint_train_annotations_20170909" / "keypoint_train_annotations_20170909.json",
            root / "ai_challenger_keypoint_train_20170902" / "keypoint_train_annotations_20170902.json",
        ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find the AIC {normalized_split} annotation JSON. Tried: "
        f"{', '.join(str(path) for path in candidates)}"
    )


def resolve_aic_image_root(root: Path, split: str = "train") -> Path:
    normalized_split = _normalize_aic_split(split)
    if normalized_split == "val":
        candidates = [
            root / "ai_challenger_keypoint_validation_20170911" / "keypoint_validation_images_20170911",
        ]
    else:
        candidates = [
            root / "ai_challenger_keypoint_train_20170902" / "keypoint_train_images_20170902",
        ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not find the AIC {normalized_split} image directory. Tried: "
        f"{', '.join(str(path) for path in candidates)}"
    )


def load_coco_records(
    root: Path,
    split: str = "train2017",
    max_samples: int | None = None,
) -> list[PoseRecord]:
    ann_path = root / "annotations" / f"person_keypoints_{split}.json"
    image_root = root / split
    with ann_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    images = {img["id"]: img for img in data["images"]}
    anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        anns_by_image[ann["image_id"]].append(ann)
    records: list[PoseRecord] = []
    for image_id, anns in anns_by_image.items():
        img = images[image_id]
        boxes, kpts, masks, visibility_masks = [], [], [], []
        for ann in anns:
            kp, valid, visibility_valid = coco_to_union(
                ann["keypoints"], img["width"], img["height"]
            )
            boxes.append(clamp_box_xyxy(xywh_to_xyxy(ann["bbox"]), img["width"], img["height"]))
            kpts.append(kp)
            masks.append(valid)
            visibility_masks.append(visibility_valid)
        _record_if_valid(
            records,
            image_root / img["file_name"],
            img["width"],
            img["height"],
            boxes,
            kpts,
            masks,
            "COCO17",
            "ALL_POSE",
            ALL_POSE_PROMPT,
            "coco",
            str(image_id),
            visibility_valid=visibility_masks,
        )
        if max_samples and len(records) >= max_samples:
            break
    return records


def load_crowdpose_records(root: Path, split: str = "train", max_samples: int | None = None) -> list[PoseRecord]:
    ann_path = root / "annotations" / f"mmpose_crowdpose_{split}.json"
    image_root = root / "annotations" / "images"
    data = json.load(open(ann_path, encoding="utf-8"))
    images = {img["id"]: img for img in data["images"]}
    anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        anns_by_image[ann["image_id"]].append(ann)
    records: list[PoseRecord] = []
    for image_id, anns in anns_by_image.items():
        img = images[image_id]
        boxes, kpts, masks, visibility_masks = [], [], [], []
        for ann in anns:
            kp, valid, visibility_valid = crowdpose_to_union(
                ann["keypoints"], img["width"], img["height"]
            )
            boxes.append(clamp_box_xyxy(xywh_to_xyxy(ann["bbox"]), img["width"], img["height"]))
            kpts.append(kp)
            masks.append(valid)
            visibility_masks.append(visibility_valid)
        _record_if_valid(
            records,
            image_root / img["file_name"],
            img["width"],
            img["height"],
            boxes,
            kpts,
            masks,
            "CrowdPose14",
            "ALL_POSE",
            ALL_POSE_PROMPT,
            "crowdpose",
            str(image_id),
            visibility_valid=visibility_masks,
        )
        if max_samples and len(records) >= max_samples:
            break
    return records


def load_aic_records(
    root: Path,
    split: str = "train",
    max_samples: int | None = None,
    cache_dir: Path | None = None,
    disable_cache: bool = False,
    show_progress: bool = True,
) -> list[PoseRecord]:
    ann_path = resolve_aic_annotation_path(root, split)
    image_root = resolve_aic_image_root(root, split)
    with ann_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    image_size_map = _load_aic_image_sizes(
        root=root,
        split=split,
        annotations=data,
        image_root=image_root,
        max_images=max_samples,
        cache_dir=cache_dir,
        disable_cache=disable_cache,
        show_progress=show_progress,
    )
    records: list[PoseRecord] = []
    iterator = data
    progress_bar = None
    if show_progress and tqdm is not None and _is_data_log_process():
        progress_bar = tqdm(
            data,
            total=len(data),
            desc=f"aic records {split}",
            unit="image",
            dynamic_ncols=True,
            mininterval=0.5,
        )
        iterator = progress_bar
    started = time.perf_counter()
    for index, item in enumerate(iterator, start=1):
        image_path = image_root / f"{item['image_id']}.jpg"
        if not image_path.exists():
            continue
        size = image_size_map.get(str(item["image_id"]))
        if size is None:
            with Image.open(image_path) as img:
                size = tuple(int(v) for v in img.size)
            image_size_map[str(item["image_id"])] = size
        width, height = size
        boxes, kpts, masks, visibility_masks = [], [], [], []
        for human_id, box in item.get("human_annotations", {}).items():
            annotation = item.get("keypoint_annotations", {}).get(human_id)
            if annotation is None:
                kp = torch.zeros(len(UNION_KEYPOINTS), 3, dtype=torch.float32)
                valid = torch.zeros(len(UNION_KEYPOINTS), dtype=torch.bool)
                visibility_valid = torch.zeros(len(UNION_KEYPOINTS), dtype=torch.bool)
            else:
                kp, valid, visibility_valid = aic_to_union(annotation, width, height)
            boxes.append(clamp_box_xyxy(box, width, height))
            kpts.append(kp)
            masks.append(valid)
            visibility_masks.append(visibility_valid)
        _record_if_valid(
            records,
            image_path,
            width,
            height,
            boxes,
            kpts,
            masks,
            "AIC14",
            "ALL_POSE",
            ALL_POSE_PROMPT,
            "aic",
            str(item["image_id"]),
            visibility_valid=visibility_masks,
        )
        if progress_bar is not None and index % 2000 == 0:
            elapsed = time.perf_counter() - started
            avg = elapsed / max(index, 1)
            eta = avg * max(len(data) - index, 0)
            progress_bar.set_postfix(
                records=len(records),
                elapsed=_format_duration(elapsed),
                eta=_format_duration(eta),
                refresh=False,
            )
        if max_samples and len(records) >= max_samples:
            break
    if progress_bar is not None:
        progress_bar.close()
    return records


def load_mpii_records(root: Path, split: str = "train", max_samples: int | None = None) -> list[PoseRecord]:
    ann_path = root / "annotations" / f"mpii_{split}.json"
    image_root = root / "images"
    with ann_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    anns_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ann in data:
        anns_by_image[ann["image"]].append(ann)
    records: list[PoseRecord] = []
    for image_name, anns in anns_by_image.items():
        image_path = image_root / image_name
        if not image_path.exists():
            continue
        with Image.open(image_path) as img:
            width, height = img.size
        boxes, loss_boxes, loss_areas = [], [], []
        kpts, masks, visibility_masks = [], [], []
        for ann in anns:
            kp, valid, visibility_valid = mpii_to_union(
                ann["joints"], ann["joints_vis"], width, height
            )
            geometry = mpii_boxes_from_center_scale(
                ann.get("center"), ann.get("scale", 0.0), width, height
            )
            if geometry is None:
                continue
            condition_box, loss_box, loss_area = geometry
            boxes.append(condition_box)
            loss_boxes.append(loss_box)
            loss_areas.append(loss_area)
            kpts.append(kp)
            masks.append(valid)
            visibility_masks.append(visibility_valid)
        _record_if_valid(
            records,
            image_path,
            width,
            height,
            boxes,
            kpts,
            masks,
            "MPII16",
            "ALL_POSE",
            ALL_POSE_PROMPT,
            "mpii",
            image_name,
            loss_boxes=loss_boxes,
            loss_areas=loss_areas,
            visibility_valid=visibility_masks,
        )
        if max_samples and len(records) >= max_samples:
            break
    return records


def load_refhuman_records(
    root: Path,
    split: str = "train",
    max_samples: int | None = None,
) -> list[PoseRecord]:
    ann_path = root / f"RefHuman_{split}.json"
    image_root = root / "images"
    with ann_path.open(encoding="utf-8") as handle:
        data = json.load(handle)

    grouped_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_candidate_keys: dict[str, set[tuple[Any, ...]]] = defaultdict(set)
    seen_caption_texts: dict[tuple[str, tuple[Any, ...]], set[str]] = defaultdict(set)
    expression_rows: list[tuple[dict[str, Any], dict[str, Any], tuple[Any, ...], str]] = []

    for img, ann in zip(data["images"], data["annotations"]):
        group_key = f"{img['file_name']}::{img.get('original_id', img.get('origin_id', img['file_name']))}"
        candidate_key = (
            ann.get("original_id", ann.get("origin_id", ann.get("id"))),
            tuple(round(float(v), 2) for v in ann["bbox"]),
        )
        if candidate_key not in seen_candidate_keys[group_key]:
            grouped_candidates[group_key].append({"image": img, "ann": ann, "candidate_key": candidate_key})
            seen_candidate_keys[group_key].add(candidate_key)
        caption = img.get("caption", "").strip()
        target_slot = (group_key, candidate_key)
        if caption in seen_caption_texts[target_slot]:
            continue
        seen_caption_texts[target_slot].add(caption)
        expression_rows.append((img, ann, candidate_key, caption))

    records: list[PoseRecord] = []
    for img, ann, target_key, caption in expression_rows:
        group_key = f"{img['file_name']}::{img.get('original_id', img.get('origin_id', img['file_name']))}"
        candidates = grouped_candidates[group_key]
        boxes, kpts, masks, visibility_masks = [], [], [], []
        ref_target = -1
        for idx, cand in enumerate(candidates):
            cand_ann = cand["ann"]
            kp, valid, visibility_valid = coco_to_union(
                cand_ann["keypoints"], img["width"], img["height"]
            )
            if cand["candidate_key"] == target_key:
                ref_target = len(boxes)
            boxes.append(clamp_box_xyxy(xywh_to_xyxy(cand_ann["bbox"]), img["width"], img["height"]))
            kpts.append(kp)
            masks.append(valid)
            visibility_masks.append(visibility_valid)
        if ref_target < 0:
            continue
        prompt = f'Locate a single person that matches the following description: "{caption}".'
        _record_if_valid(
            records,
            image_root / img["file_name"],
            img["width"],
            img["height"],
            boxes,
            kpts,
            masks,
            "COCO17",
            "REF_POSE",
            prompt,
            "refhuman",
            f"{img.get('id')}",
            ref_text=caption,
            ref_target=ref_target,
            visibility_valid=visibility_masks,
        )
        if max_samples and len(records) >= max_samples:
            break
    return records


def _refhuman_instance_key(record: PoseRecord) -> tuple[str, tuple[float, float, float, float]]:
    if record.ref_target < 0 or record.ref_target >= int(record.boxes_xyxy.shape[0]):
        return (str(record.image_path), (0.0, 0.0, 0.0, 0.0))
    target_box = record.boxes_xyxy[int(record.ref_target)]
    return (
        str(record.image_path),
        tuple(round(float(value), 6) for value in target_box.tolist()),
    )


def build_datasets(
    dataset_root: Path,
    names: list[str],
    max_instances: int,
    image_size: int = 640,
    load_image_tensors: bool = True,
    load_vision_images: bool = False,
    augment_config: PoseAugmentConfig | None = None,
    use_prompts: bool = True,
    split: str = "train",
    max_samples_per_dataset: int | None = None,
    refhuman_max_captions_per_instance: int = 1,
    mixing_strategy: str = "interleave",
    dataset_mix_weights: str | dict[str, float] | None = None,
    seed: int = 42,
    record_cache_dir: Path | None = Path(".cache/qwenpose_records"),
    disable_record_cache: bool = False,
    show_progress: bool = True,
) -> Dataset:
    named_datasets: list[tuple[str, Dataset]] = []
    weights = parse_dataset_mix_weights(dataset_mix_weights)
    overall_started = time.perf_counter()
    progress_bar = None
    total_datasets = len(names)
    if show_progress and tqdm is not None and _is_data_log_process():
        progress_bar = tqdm(
            total=total_datasets,
            desc=f"load data {split}",
            unit="dataset",
            dynamic_ncols=True,
            mininterval=0.5,
        )
    for dataset_index, name in enumerate(names, start=1):
        name = name.lower()
        dataset_started = time.perf_counter()
        if progress_bar is not None:
            progress_bar.set_postfix_str(f"current={name}")
        if name == "coco":
            coco_split = "val2017" if split == "val" else "train2017"
            root = dataset_root / "coco"
            source_paths = [root / "annotations" / f"person_keypoints_{coco_split}.json"]
            records = _load_records_cached(
                dataset_name=name,
                root=root,
                split=coco_split,
                max_samples=max_samples_per_dataset,
                source_paths=source_paths,
                cache_dir=record_cache_dir,
                disable_cache=disable_record_cache,
                builder=lambda root=root, coco_split=coco_split: load_coco_records(
                    root,
                    split=coco_split,
                    max_samples=max_samples_per_dataset,
                ),
            )
        elif name == "aic":
            root = dataset_root / "aic"
            source_paths = [resolve_aic_annotation_path(root, split), resolve_aic_image_root(root, split)]
            records = _load_records_cached(
                dataset_name=name,
                root=root,
                split=split,
                max_samples=max_samples_per_dataset,
                source_paths=source_paths,
                cache_dir=record_cache_dir,
                disable_cache=disable_record_cache,
                builder=lambda root=root: load_aic_records(
                    root,
                    split=split,
                    max_samples=max_samples_per_dataset,
                    cache_dir=record_cache_dir,
                    disable_cache=disable_record_cache,
                    show_progress=show_progress,
                ),
            )
        elif name == "mpii":
            root = dataset_root / "mpii"
            source_paths = [root / "annotations" / f"mpii_{split}.json"]
            records = _load_records_cached(
                dataset_name=name,
                root=root,
                split=split,
                max_samples=max_samples_per_dataset,
                source_paths=source_paths,
                cache_dir=record_cache_dir,
                disable_cache=disable_record_cache,
                builder=lambda root=root: load_mpii_records(root, split=split, max_samples=max_samples_per_dataset),
            )
        elif name == "crowdpose":
            root = dataset_root / "crowdpose"
            source_paths = [root / "annotations" / f"mmpose_crowdpose_{split}.json"]
            records = _load_records_cached(
                dataset_name=name,
                root=root,
                split=split,
                max_samples=max_samples_per_dataset,
                source_paths=source_paths,
                cache_dir=record_cache_dir,
                disable_cache=disable_record_cache,
                builder=lambda root=root: load_crowdpose_records(
                    root,
                    split=split,
                    max_samples=max_samples_per_dataset,
                ),
            )
        elif name == "refhuman":
            root = dataset_root / "refhuman"
            source_paths = [root / f"RefHuman_{split}.json"]
            records = _load_records_cached(
                dataset_name=name,
                root=root,
                split=split,
                max_samples=None,
                source_paths=source_paths,
                cache_dir=record_cache_dir,
                disable_cache=disable_record_cache,
                # This cache already contains every unique caption. Keep the
                # legacy key value so existing 674 MB caches remain reusable;
                # epoch-wise selection now happens in the dataset wrapper.
                extra_cache_key={
                    "refhuman_caption_pool_mode": "all_unique_captions_then_sample",
                },
                builder=lambda root=root: load_refhuman_records(
                    root,
                    split=split,
                    max_samples=None,
                ),
            )
            if refhuman_max_captions_per_instance == 0 and max_samples_per_dataset is not None:
                records = records[: max(int(max_samples_per_dataset), 0)]
        else:
            raise ValueError(f"Unknown dataset {name!r}")
        if not records:
            raise RuntimeError(f"Dataset {name!r} produced no records.")
        dataset_kwargs = {
            "max_instances": max_instances,
            "image_size": image_size,
            "load_image_tensors": load_image_tensors,
            "load_vision_images": load_vision_images,
            "augment_config": augment_config,
            "use_prompts": use_prompts,
        }
        if name == "refhuman" and refhuman_max_captions_per_instance > 0:
            pose_dataset: Dataset = EpochRandomRefHumanDataset(
                records,
                captions_per_instance=refhuman_max_captions_per_instance,
                seed=seed,
                max_samples=max_samples_per_dataset,
                **dataset_kwargs,
            )
        else:
            pose_dataset = PoseRecordDataset(records, **dataset_kwargs)
        named_datasets.append((name, pose_dataset))
        effective_record_count = len(pose_dataset)
        dataset_elapsed = time.perf_counter() - dataset_started
        total_elapsed = time.perf_counter() - overall_started
        remaining_count = max(total_datasets - dataset_index, 0)
        avg_per_dataset = total_elapsed / max(dataset_index, 1)
        eta_seconds = avg_per_dataset * remaining_count
        _data_log(
            f"Loaded {effective_record_count:6d} records from {name} split={split} "
            f"(dataset_time={dataset_elapsed:.2f}s, elapsed={_format_duration(total_elapsed)}, "
            f"eta={_format_duration(eta_seconds)})"
        )
        if progress_bar is not None:
            progress_bar.update(1)
            progress_bar.set_postfix(
                dataset=name,
                records=effective_record_count,
                last_s=f"{dataset_elapsed:.1f}",
                eta=_format_duration(eta_seconds),
                refresh=False,
            )
    if progress_bar is not None:
        progress_bar.close()
    if len(named_datasets) == 1:
        return named_datasets[0][1]
    if mixing_strategy == "concat_shuffle":
        return ConcatDataset([dataset for _, dataset in named_datasets])
    if mixing_strategy != "interleave":
        raise ValueError(f"Unknown mixing_strategy {mixing_strategy!r}")
    mixed = InterleavedPoseDataset(named_datasets, weights=weights, seed=seed)
    _data_log(f"Interleave schedule: {mixed.schedule_description()} (epoch_size={len(mixed)})")
    return mixed


def parse_dataset_mix_weights(spec: str | dict[str, float] | None) -> dict[str, float] | None:
    if spec is None or spec == "" or str(spec).lower() == "auto":
        return None
    raw_items = spec.items() if isinstance(spec, dict) else None
    if raw_items is None:
        parsed_items: list[tuple[str, str]] = []
        for chunk in str(spec).split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if ":" not in chunk:
                raise ValueError(f"Bad dataset weight item {chunk!r}; expected name:weight.")
            name, value = chunk.split(":", 1)
            parsed_items.append((name, value))
        raw_items = parsed_items
    weights: dict[str, float] = {}
    for raw_name, raw_value in raw_items:
        name = str(raw_name).strip().lower()
        if not name:
            raise ValueError("Dataset weight name cannot be empty.")
        value = float(raw_value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"Dataset weight for {name!r} must be finite and non-negative, got {raw_value!r}.")
        weights[name] = value
    if not weights:
        raise ValueError("dataset_mix_weights must contain at least one name:weight item.")
    return weights
