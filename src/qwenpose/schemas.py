from __future__ import annotations

from dataclasses import dataclass

import torch


UNION_KEYPOINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "neck",
    "head_top",
    "pelvis",
    "thorax",
    "upper_neck",
]

SCHEMA_KEYPOINTS = {
    "COCO17": [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ],
    "AIC14": [
        "right_shoulder",
        "right_elbow",
        "right_wrist",
        "left_shoulder",
        "left_elbow",
        "left_wrist",
        "right_hip",
        "right_knee",
        "right_ankle",
        "left_hip",
        "left_knee",
        "left_ankle",
        "head_top",
        "neck",
    ],
    "MPII16": [
        "right_ankle",
        "right_knee",
        "right_hip",
        "left_hip",
        "left_knee",
        "left_ankle",
        "pelvis",
        "thorax",
        "upper_neck",
        "head_top",
        "right_wrist",
        "right_elbow",
        "right_shoulder",
        "left_shoulder",
        "left_elbow",
        "left_wrist",
    ],
    "CrowdPose14": [
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "head_top",
        "neck",
    ],
}

SCHEMA_NAMES = tuple(SCHEMA_KEYPOINTS.keys())
SCHEMA_TO_ID = {name: idx for idx, name in enumerate(SCHEMA_NAMES)}
ID_TO_SCHEMA = {idx: name for name, idx in SCHEMA_TO_ID.items()}
UNION_TO_ID = {name: idx for idx, name in enumerate(UNION_KEYPOINTS)}
SCHEMA_INDICES = {
    name: torch.tensor([UNION_TO_ID[joint] for joint in joints], dtype=torch.long)
    for name, joints in SCHEMA_KEYPOINTS.items()
}


def _default_sigmas() -> dict[str, float]:
    # COCO keypoint sigmas after the standard /10 scaling. Extra joints use
    # conservative torso/head-scale defaults and can be tuned on validation sets.
    coco = {
        "nose": 0.026,
        "left_eye": 0.025,
        "right_eye": 0.025,
        "left_ear": 0.035,
        "right_ear": 0.035,
        "left_shoulder": 0.079,
        "right_shoulder": 0.079,
        "left_elbow": 0.072,
        "right_elbow": 0.072,
        "left_wrist": 0.062,
        "right_wrist": 0.062,
        "left_hip": 0.107,
        "right_hip": 0.107,
        "left_knee": 0.087,
        "right_knee": 0.087,
        "left_ankle": 0.089,
        "right_ankle": 0.089,
    }
    extra = {
        "neck": 0.079,
        "head_top": 0.035,
        "pelvis": 0.107,
        "thorax": 0.079,
        "upper_neck": 0.079,
    }
    return {**coco, **extra}


UNION_SIGMAS = torch.tensor(
    [_default_sigmas()[name] for name in UNION_KEYPOINTS], dtype=torch.float32
)


@dataclass(frozen=True)
class SchemaSpec:
    name: str
    schema_id: int
    keypoints: list[str]
    indices: torch.Tensor


def get_schema(name: str) -> SchemaSpec:
    if name not in SCHEMA_KEYPOINTS:
        raise KeyError(f"Unknown pose schema {name!r}. Available: {sorted(SCHEMA_KEYPOINTS)}")
    return SchemaSpec(
        name=name,
        schema_id=SCHEMA_TO_ID[name],
        keypoints=SCHEMA_KEYPOINTS[name],
        indices=SCHEMA_INDICES[name],
    )


def _schema_to_union_impl(
    flat_keypoints: list[float] | list[int],
    schema_name: str,
    image_width: float,
    image_height: float,
    *,
    visibility_target: str = "valid",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert schema-specific keypoints to fixed union tensors.

    Args:
        visibility_target: How to fill the third channel used by the visibility
            BCE target. ``valid`` keeps the previous behavior: every supervised
            coordinate has target 1.0. ``coco`` treats ``v > 0`` as coordinate
            supervision, but only ``v > 1`` as visible.

    Returns:
        keypoints: [U, 3] normalized x/y and confidence target.
        valid: [U] bool mask. Only true keypoints contribute to coordinate losses.
    """
    spec = get_schema(schema_name)
    keypoints = torch.zeros(len(UNION_KEYPOINTS), 3, dtype=torch.float32)
    valid = torch.zeros(len(UNION_KEYPOINTS), dtype=torch.bool)
    if len(flat_keypoints) != len(spec.keypoints) * 3:
        raise ValueError(
            f"{schema_name} expects {len(spec.keypoints) * 3} values, got {len(flat_keypoints)}"
        )
    width = max(float(image_width), 1.0)
    height = max(float(image_height), 1.0)
    for local_idx, union_idx in enumerate(spec.indices.tolist()):
        x, y, v = flat_keypoints[local_idx * 3 : local_idx * 3 + 3]
        visibility = float(v)
        if visibility <= 0:
            continue
        keypoints[union_idx, 0] = float(x) / width
        keypoints[union_idx, 1] = float(y) / height
        if visibility_target == "coco":
            keypoints[union_idx, 2] = 1.0 if visibility > 1.0 else 0.0
        elif visibility_target == "valid":
            keypoints[union_idx, 2] = 1.0
        else:
            raise ValueError(f"Unknown visibility target mode: {visibility_target!r}")
        valid[union_idx] = True
    return keypoints.clamp_(0.0, 1.0), valid


def schema_to_union(
    flat_keypoints: list[float] | list[int],
    schema_name: str,
    image_width: float,
    image_height: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _schema_to_union_impl(flat_keypoints, schema_name, image_width, image_height)


def coco_to_union(
    flat_keypoints: list[float] | list[int],
    image_width: float,
    image_height: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert COCO-style keypoints and preserve visible-vs-occluded labels.

    COCO-style annotations use ``v == 0`` for missing, ``v == 1`` for labeled
    but not visible, and ``v == 2`` for labeled and visible. Coordinates with
    ``v in {1, 2}`` remain valid supervision, while the visibility target is 1
    only for visible joints.
    """
    return _schema_to_union_impl(
        flat_keypoints,
        "COCO17",
        image_width,
        image_height,
        visibility_target="coco",
    )


def crowdpose_to_union(
    flat_keypoints: list[float] | list[int],
    image_width: float,
    image_height: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert CrowdPose keypoints and preserve visible-vs-occluded labels.

    The local MMPose CrowdPose annotation names the 13th joint ``head``. To keep
    the existing union output dimension and checkpoint format stable, this
    schema maps that joint to the union ``head_top`` slot consistently in both
    training and evaluation.
    """
    return _schema_to_union_impl(
        flat_keypoints,
        "CrowdPose14",
        image_width,
        image_height,
        visibility_target="coco",
    )


def mpii_to_union(
    joints: list[list[float]],
    joints_vis: list[int] | list[float],
    image_width: float,
    image_height: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat: list[float] = []
    for joint, vis in zip(joints, joints_vis):
        flat.extend([float(joint[0]), float(joint[1]), float(vis)])
    return schema_to_union(flat, "MPII16", image_width, image_height)

