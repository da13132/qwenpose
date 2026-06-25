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
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional for minimal envs.
    tqdm = None

from .schemas import SCHEMA_TO_ID, UNION_KEYPOINTS, get_schema, mpii_to_union, schema_to_union


TASK_TO_ID = {"ALL_POSE": 0, "REF_POSE": 1}
RECORD_CACHE_VERSION = 7


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
    boxes_xyxy: torch.Tensor
    keypoints: torch.Tensor
    keypoint_valid: torch.Tensor
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


def normalize_boxes(boxes: list[list[float]], width: float, height: float) -> torch.Tensor:
    if not boxes:
        return torch.zeros(0, 4, dtype=torch.float32)
    out = torch.tensor(boxes, dtype=torch.float32)
    out[:, [0, 2]] /= max(float(width), 1.0)
    out[:, [1, 3]] /= max(float(height), 1.0)
    return out.clamp_(0.0, 1.0)


def box_from_keypoints(keypoints: torch.Tensor, valid: torch.Tensor) -> list[float] | None:
    if valid.sum().item() == 0:
        return None
    pts = keypoints[valid, :2]
    x1, y1 = pts.min(dim=0).values.tolist()
    x2, y2 = pts.max(dim=0).values.tolist()
    pad_x = max((x2 - x1) * 0.15, 0.03)
    pad_y = max((y2 - y1) * 0.15, 0.03)
    return [max(x1 - pad_x, 0.0), max(y1 - pad_y, 0.0), min(x2 + pad_x, 1.0), min(y2 + pad_y, 1.0)]


def read_image_tensor(path: Path, image_size: int) -> tuple[torch.Tensor, int, int]:
    """Read an RGB image for the lightweight pose visual branch.

    Qwen still reads the original image path separately; this tensor is a
    fixed-size, normalized RGB view used only by QwenPoseModel's local visual
    branch.
    """
    with Image.open(path) as img:
        img = img.convert("RGB")
        width, height = img.size
        img = img.resize((image_size, image_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return tensor, width, height


class PoseRecordDataset(Dataset):
    def __init__(
        self,
        records: list[PoseRecord],
        max_instances: int = 80,
        image_size: int = 256,
        load_image_tensors: bool = True,
    ) -> None:
        self.records = records
        self.max_instances = max_instances
        self.image_size = image_size
        self.load_image_tensors = load_image_tensors

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        if self.load_image_tensors:
            image, _, _ = read_image_tensor(record.image_path, self.image_size)
        else:
            # Keep a tiny placeholder so the model can fall back to the pure-Qwen
            # path without forcing a second PIL read in the DataLoader workers.
            image = torch.zeros(3, 1, 1, dtype=torch.float32)
        n = min(record.boxes_xyxy.shape[0], self.max_instances)
        boxes = record.boxes_xyxy[:n].clone()
        keypoints = record.keypoints[:n].clone()
        valid = record.keypoint_valid[:n].clone()
        ref_target = record.ref_target
        if ref_target >= n:
            ref_target = -1
        text = record.prompt
        return {
            "image": image,
            "image_path": str(record.image_path),
            "schema_id": torch.tensor(SCHEMA_TO_ID[record.schema], dtype=torch.long),
            "task_id": torch.tensor(TASK_TO_ID[record.task], dtype=torch.long),
            "target": {
                "boxes": boxes,
                "keypoints": keypoints,
                "keypoint_valid": valid,
                "ref_target": torch.tensor(ref_target, dtype=torch.long),
                "dataset": record.dataset_name,
                "image_id": record.image_id,
                "schema": record.schema,
                "width": record.width,
                "height": record.height,
            },
            "prompt": text,
        }


class InterleavedPoseDataset(Dataset):
    """Dataset-level weighted fair round-robin sampler.

    When weights are None, the schedule is proportional to the actual number
    of records loaded from each dataset, so one epoch keeps the natural dataset
    size ratio while still interleaving sources instead of block-concatenating.
    """

    def __init__(
        self,
        named_datasets: list[tuple[str, Dataset]],
        weights: dict[str, int] | None,
        seed: int = 42,
        epoch_size: int | None = None,
    ) -> None:
        if not named_datasets:
            raise ValueError("InterleavedPoseDataset requires at least one dataset.")
        self.named_datasets = [(name.lower(), dataset) for name, dataset in named_datasets]
        self.names = [name for name, _ in self.named_datasets]
        self.datasets = [dataset for _, dataset in self.named_datasets]
        self.auto_weights = weights is None
        basis = (
            [len(dataset) for dataset in self.datasets]
            if weights is None
            else [max(int(weights.get(name, 1)), 1) for name in self.names]
        )
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
        self.epoch_size = int(epoch_size or sum(len(dataset) for dataset in self.datasets))
        self.offsets: list[int] = []
        self.strides: list[int] = []
        for dataset_idx, dataset in enumerate(self.datasets):
            n = len(dataset)
            rng = random.Random(seed + dataset_idx * 9973)
            self.offsets.append(rng.randrange(n) if n > 1 else 0)
            self.strides.append(self._coprime_stride(n, rng))

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
        mode = "auto_size_proportional" if self.auto_weights else "manual_weighted"
        total = max(sum(self.weights), 1)
        counts = ",".join(
            f"{name}:{count}({count / total:.1%})"
            for name, count in zip(self.names, self.weights)
        )
        return f"{mode} counts=[{counts}] first=[{shown}]"


def pose_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": torch.stack([item["image"] for item in batch], dim=0),
        "schema_ids": torch.stack([item["schema_id"] for item in batch], dim=0),
        "task_ids": torch.stack([item["task_id"] for item in batch], dim=0),
        "targets": [item["target"] for item in batch],
        "prompts": [item["prompt"] for item in batch],
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
) -> None:
    if not image_path.exists() or not boxes:
        return
    records.append(
        PoseRecord(
            image_path=image_path,
            width=width,
            height=height,
            boxes_xyxy=normalize_boxes(boxes, width, height),
            keypoints=torch.stack(keypoints, dim=0),
            keypoint_valid=torch.stack(valid, dim=0),
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert AIC14 keypoints while keeping visible vs occluded states.

    In the local AIC annotations:
    - `v == 1` means labeled and visible.
    - `v == 2` means labeled but occluded / not visible.
    - `v >= 3` behaves like missing / outside image and is often paired with (0, 0).

    We keep `v in {1, 2}` as valid supervision for coordinates, but only set
    the visibility target channel to 1.0 for `v == 1`.
    """
    if isinstance(flat_keypoints, dict):
        flat_keypoints = list(flat_keypoints.values())
    spec = get_schema("AIC14")
    keypoints = torch.zeros(len(UNION_KEYPOINTS), 3, dtype=torch.float32)
    valid = torch.zeros(len(UNION_KEYPOINTS), dtype=torch.bool)
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
        keypoints[union_idx, 2] = 1.0 if int(round(visibility)) == 1 else 0.0
        valid[union_idx] = True
    return keypoints.clamp_(0.0, 1.0), valid


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
    ann_path = resolve_aic_annotation_path(root)
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


def resolve_aic_annotation_path(root: Path) -> Path:
    candidates = [
        root / "ai_challenger_keypoint_train_annotations_20170909" / "keypoint_train_annotations_20170909.json",
        root / "ai_challenger_keypoint_train_20170902" / "keypoint_train_annotations_20170902.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find the AIC annotation JSON. Supported layouts are: "
        "ai_challenger_keypoint_train_annotations_20170909/keypoint_train_annotations_20170909.json "
        "or ai_challenger_keypoint_train_20170902/keypoint_train_annotations_20170902.json. "
        f"Tried: {', '.join(str(path) for path in candidates)}"
    )


def resolve_aic_image_root(root: Path) -> Path:
    candidates = [
        root / "ai_challenger_keypoint_train_20170902" / "keypoint_train_images_20170902",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find the AIC image directory. Expected: "
        "ai_challenger_keypoint_train_20170902/keypoint_train_images_20170902. "
        f"Tried: {', '.join(str(path) for path in candidates)}"
    )


def load_coco_records(
    root: Path,
    split: str = "train2017",
    max_samples: int | None = None,
) -> list[PoseRecord]:
    ann_path = root / "annotations" / f"person_keypoints_{split}.json"
    image_root = root / split
    data = json.load(open(ann_path, encoding="utf-8"))
    images = {img["id"]: img for img in data["images"]}
    anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0) or ann.get("num_keypoints", 0) <= 0:
            continue
        anns_by_image[ann["image_id"]].append(ann)
    records: list[PoseRecord] = []
    for image_id, anns in anns_by_image.items():
        img = images[image_id]
        boxes, kpts, masks = [], [], []
        for ann in anns:
            kp, valid = schema_to_union(ann["keypoints"], "COCO17", img["width"], img["height"])
            if valid.sum().item() == 0:
                continue
            boxes.append(clamp_box_xyxy(xywh_to_xyxy(ann["bbox"]), img["width"], img["height"]))
            kpts.append(kp)
            masks.append(valid)
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
            "Locate all the instances that match the following description: person. "
            "Estimate the human pose for each located person using the COCO17 keypoint schema.",
            "coco",
            str(image_id),
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
        if ann.get("iscrowd", 0) or ann.get("num_keypoints", 0) <= 0:
            continue
        anns_by_image[ann["image_id"]].append(ann)
    records: list[PoseRecord] = []
    for image_id, anns in anns_by_image.items():
        img = images[image_id]
        boxes, kpts, masks = [], [], []
        for ann in anns:
            kp, valid = schema_to_union(ann["keypoints"], "CrowdPose14", img["width"], img["height"])
            if valid.sum().item() == 0:
                continue
            boxes.append(clamp_box_xyxy(xywh_to_xyxy(ann["bbox"]), img["width"], img["height"]))
            kpts.append(kp)
            masks.append(valid)
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
            "Locate all the instances that match the following description: person. "
            "Estimate the human pose for each located person. "
            "Use the available keypoint schema.",
            "crowdpose",
            str(image_id),
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
    if split not in ("train", "trainval"):
        _data_log(f"AIC split {split!r} is not available in this local tree; falling back to train.")
    ann_path = resolve_aic_annotation_path(root)
    image_root = resolve_aic_image_root(root)
    data = json.load(open(ann_path, encoding="utf-8"))
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
        boxes, kpts, masks = [], [], []
        for human_id, box in item.get("human_annotations", {}).items():
            if human_id not in item.get("keypoint_annotations", {}):
                continue
            kp, valid = aic_to_union(item["keypoint_annotations"][human_id], width, height)
            if valid.sum().item() == 0:
                continue
            boxes.append(clamp_box_xyxy(box, width, height))
            kpts.append(kp)
            masks.append(valid)
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
            "Locate all the instances that match the following description: person. "
            "Estimate the human pose for each located person using the AIC14 keypoint schema.",
            "aic",
            str(item["image_id"]),
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
    data = json.load(open(ann_path, encoding="utf-8"))
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
        boxes, kpts, masks = [], [], []
        for ann in anns:
            kp, valid = mpii_to_union(ann["joints"], ann["joints_vis"], width, height)
            if valid.sum().item() == 0:
                continue
            box_norm = box_from_keypoints(kp, valid)
            if box_norm is None:
                continue
            boxes.append(
                [
                    box_norm[0] * width,
                    box_norm[1] * height,
                    box_norm[2] * width,
                    box_norm[3] * height,
                ]
            )
            kpts.append(kp)
            masks.append(valid)
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
            "Locate all the instances that match the following description: person. "
            "Estimate the human pose for each located person using the MPII16 keypoint schema.",
            "mpii",
            image_name,
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
    data = json.load(open(ann_path, encoding="utf-8"))

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
        boxes, kpts, masks = [], [], []
        ref_target = -1
        for idx, cand in enumerate(candidates):
            cand_ann = cand["ann"]
            kp, valid = schema_to_union(cand_ann["keypoints"], "COCO17", img["width"], img["height"])
            if valid.sum().item() == 0:
                continue
            if cand["candidate_key"] == target_key:
                ref_target = len(boxes)
            boxes.append(clamp_box_xyxy(xywh_to_xyxy(cand_ann["bbox"]), img["width"], img["height"]))
            kpts.append(kp)
            masks.append(valid)
        if ref_target < 0:
            continue
        prompt = (
            "Locate a single person that matches the following description: "
            f"\"{caption}\". "
            "Estimate the human pose for the located person using the COCO17 keypoint schema."
        )
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


def sample_refhuman_records(
    records: list[PoseRecord],
    max_captions_per_instance: int,
    seed: int,
) -> list[PoseRecord]:
    limit = max(int(max_captions_per_instance), 0)
    if limit == 0:
        return records

    grouped_records: dict[tuple[str, tuple[float, float, float, float]], list[PoseRecord]] = defaultdict(list)
    for record in records:
        grouped_records[_refhuman_instance_key(record)].append(record)

    # 启动时按 seed 对每个人体实例的 caption 做一次随机抽样，训练过程中保持固定。
    rng = random.Random(int(seed))
    selected_ids: set[int] = set()
    for group_records in grouped_records.values():
        if len(group_records) <= limit:
            selected_ids.update(id(record) for record in group_records)
            continue
        sampled_records = rng.sample(group_records, k=limit)
        selected_ids.update(id(record) for record in sampled_records)
    return [record for record in records if id(record) in selected_ids]


def build_datasets(
    dataset_root: Path,
    names: list[str],
    max_instances: int,
    image_size: int = 256,
    load_image_tensors: bool = True,
    split: str = "train",
    max_samples_per_dataset: int | None = None,
    refhuman_max_captions_per_instance: int = 2,
    mixing_strategy: str = "interleave",
    dataset_mix_weights: str | dict[str, int] | None = None,
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
            source_paths = [resolve_aic_annotation_path(root)]
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
                extra_cache_key={
                    "refhuman_caption_pool_mode": "all_unique_captions_then_sample",
                },
                builder=lambda root=root: load_refhuman_records(
                    root,
                    split=split,
                    max_samples=None,
                ),
            )
            records = sample_refhuman_records(
                records,
                max_captions_per_instance=refhuman_max_captions_per_instance,
                seed=seed,
            )
            if max_samples_per_dataset is not None:
                records = records[: max(int(max_samples_per_dataset), 0)]
        else:
            raise ValueError(f"Unknown dataset {name!r}")
        if not records:
            raise RuntimeError(f"Dataset {name!r} produced no records.")
        named_datasets.append(
            (
                name,
                PoseRecordDataset(
                    records,
                    max_instances=max_instances,
                    image_size=image_size,
                    load_image_tensors=load_image_tensors,
                ),
            )
        )
        dataset_elapsed = time.perf_counter() - dataset_started
        total_elapsed = time.perf_counter() - overall_started
        remaining_count = max(total_datasets - dataset_index, 0)
        avg_per_dataset = total_elapsed / max(dataset_index, 1)
        eta_seconds = avg_per_dataset * remaining_count
        _data_log(
            f"Loaded {len(records):6d} records from {name} split={split} "
            f"(dataset_time={dataset_elapsed:.2f}s, elapsed={_format_duration(total_elapsed)}, "
            f"eta={_format_duration(eta_seconds)})"
        )
        if progress_bar is not None:
            progress_bar.update(1)
            progress_bar.set_postfix(
                dataset=name,
                records=len(records),
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


def parse_dataset_mix_weights(spec: str | dict[str, int] | None) -> dict[str, int] | None:
    if spec is None or spec == "" or str(spec).lower() == "auto":
        return None
    if isinstance(spec, dict):
        return {str(k).lower(): max(int(v), 1) for k, v in spec.items()}
    weights: dict[str, int] = {}
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"Bad dataset weight item {chunk!r}; expected name:weight.")
        name, value = chunk.split(":", 1)
        weights[name.strip().lower()] = max(int(value), 1)
    return weights
