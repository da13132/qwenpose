from __future__ import annotations

from dataclasses import dataclass
import math
import json
from pathlib import Path
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torchvision.ops import roi_align as torchvision_roi_align
except Exception:  # pragma: no cover - torchvision may be unavailable in minimal envs.
    torchvision_roi_align = None

from .schemas import SCHEMA_INDICES, SCHEMA_NAMES, UNION_KEYPOINTS, UNION_TO_ID
from .spatial_features import MultiScaleSpatialFeatureBatch, SpatialFeatureBatch


def _box_iou_diagonal_xyxy(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
) -> torch.Tensor:
    """Return pairwise-aligned IoU for equally shaped ``[..., 4]`` boxes."""
    top_left = torch.maximum(boxes1[..., :2], boxes2[..., :2])
    bottom_right = torch.minimum(boxes1[..., 2:], boxes2[..., 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(dim=-1)
    area1 = (boxes1[..., 2:] - boxes1[..., :2]).clamp(min=0).prod(dim=-1)
    area2 = (boxes2[..., 2:] - boxes2[..., :2]).clamp(min=0).prod(dim=-1)
    union = (area1 + area2 - intersection).clamp(min=1e-8)
    return intersection / union


def apply_refhuman_box_refinement_safety(
    refined_boxes: torch.Tensor,
    input_boxes: torch.Tensor,
    box_mask: torch.Tensor,
    task_ids: torch.Tensor,
    *,
    minimum_iou: float = 0.30,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep inference-time RefHuman refinement local to Locate's grounded box.

    The regression head remains fully supervised during training. At inference,
    only RefHuman boxes whose refinement has drifted far from their Locate input
    are restored to that input box; ordinary all-person pose queries are unchanged.
    """
    overlap = _box_iou_diagonal_xyxy(refined_boxes, input_boxes)
    fallback_mask = (
        task_ids.to(device=refined_boxes.device).eq(1)[:, None]
        & box_mask.to(device=refined_boxes.device).bool()
        & overlap.lt(float(minimum_iou))
    )
    safe_boxes = torch.where(fallback_mask[..., None], input_boxes, refined_boxes)
    return safe_boxes, fallback_mask


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, depth: int = 3) -> None:
        super().__init__()
        layers = []
        dim = in_dim
        for _ in range(depth - 1):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.GELU())
            dim = hidden_dim
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _group_count(hidden_dim: int, max_groups: int = 32) -> int:
    for groups in range(min(max_groups, hidden_dim), 0, -1):
        if hidden_dim % groups == 0:
            return groups
    return 1


def _padded_grid_from_normalized_points(
    points: torch.Tensor,
    spatial_shapes: torch.Tensor,
    padded_height: int,
    padded_width: int,
) -> torch.Tensor:
    """Map per-image [0,1] coordinates into a top-left padded feature tensor."""
    scales = torch.stack(
        [
            spatial_shapes[:, 1].to(dtype=points.dtype) / max(float(padded_width), 1.0),
            spatial_shapes[:, 0].to(dtype=points.dtype) / max(float(padded_height), 1.0),
        ],
        dim=-1,
    ).to(device=points.device)
    view_shape = [int(points.shape[0])] + [1] * (points.ndim - 2) + [2]
    return points * scales.view(*view_shape) * 2.0 - 1.0


def _masked_spatial_mean(
    feature_map: torch.Tensor,
    spatial_shapes: torch.Tensor,
) -> torch.Tensor:
    height, width = feature_map.shape[-2:]
    rows = torch.arange(height, device=feature_map.device)[None, :, None]
    cols = torch.arange(width, device=feature_map.device)[None, None, :]
    mask = (rows < spatial_shapes[:, 0, None, None]) & (
        cols < spatial_shapes[:, 1, None, None]
    )
    weights = mask[:, None].to(dtype=feature_map.dtype)
    return (feature_map * weights).sum(dim=(-2, -1)) / weights.sum(
        dim=(-2, -1)
    ).clamp(min=1.0)


def _repeat_feature_levels(
    feature_maps: list[torch.Tensor],
    spatial_shapes: list[torch.Tensor] | torch.Tensor,
    num_scales: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    maps = list(feature_maps[:num_scales])
    if isinstance(spatial_shapes, torch.Tensor):
        shapes = [spatial_shapes for _ in maps]
    else:
        shapes = list(spatial_shapes[:num_scales])
    while len(maps) < num_scales:
        maps.append(maps[-1])
        shapes.append(shapes[-1])
    if len(shapes) != len(maps):
        raise ValueError("Every deformable feature level requires its own spatial shape tensor.")
    return maps, shapes


def _radial_offset_bias(num_scales: int, num_points: int) -> torch.Tensor:
    """Non-collapsed Deformable-DETR offset initialization around each reference."""
    if num_points <= 0:
        return torch.zeros(num_scales, 0, 2)
    biases = []
    for level_idx in range(num_scales):
        phase = 0.0 if level_idx % 2 == 0 else math.pi / max(float(num_points), 1.0)
        angles = torch.arange(num_points, dtype=torch.float32) * (
            2.0 * math.pi / float(num_points)
        ) + phase
        radius = 0.5
        biases.append(torch.stack([angles.cos(), angles.sin()], dim=-1) * radius)
    pattern = torch.stack(biases, dim=0).clamp(-0.95, 0.95)
    return torch.atanh(pattern).reshape(-1)


def _apply_two_level_scale_prior(
    logits: torch.Tensor,
    boxes_wh: torch.Tensor,
    p3_shapes: torch.Tensor,
    *,
    num_points: int,
    strength: float,
    center_cells: float,
    temperature: float,
) -> torch.Tensor:
    """Bias tiny people toward P2 while preserving fully learned soft routing."""
    if logits.shape[-2] != 2 or strength <= 0:
        return logits
    p3_wh = torch.stack([p3_shapes[:, 1], p3_shapes[:, 0]], dim=-1).to(
        device=boxes_wh.device, dtype=boxes_wh.dtype
    )
    size_cells = (boxes_wh * p3_wh[:, None]).amin(dim=-1)
    p2_gate = torch.sigmoid(
        (float(center_cells) - size_cells) / max(float(temperature), 1e-4)
    ).clamp(1e-4, 1.0 - 1e-4)
    prior = torch.stack([p2_gate.log(), (1.0 - p2_gate).log()], dim=-1)
    while prior.ndim < logits.ndim - 1:
        prior = prior.unsqueeze(-2)
    prior = prior.unsqueeze(-1).expand(*logits.shape[:-1], num_points)
    return logits + float(strength) * prior


class JointDeformableKeypointAttention(nn.Module):
    """Joint-centric sparse sampling over the Locate spatial feature."""

    def __init__(
        self,
        hidden_dim: int,
        feature_dim: int,
        num_scales: int = 3,
        num_points: int = 4,
        offset_scale: float = 0.35,
        min_radius_cells: float = 2.0,
        scale_prior_strength: float = 0.5,
        scale_prior_center_cells: float = 6.0,
        scale_prior_temperature: float = 1.5,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.feature_dim = int(feature_dim)
        self.num_scales = max(int(num_scales), 1)
        self.num_points = max(int(num_points), 1)
        self.offset_scale = float(offset_scale)
        self.min_radius_cells = max(float(min_radius_cells), 0.0)
        self.scale_prior_strength = max(float(scale_prior_strength), 0.0)
        self.scale_prior_center_cells = float(scale_prior_center_cells)
        self.scale_prior_temperature = max(float(scale_prior_temperature), 1e-4)
        sample_count = self.num_scales * self.num_points
        self.offset_head = nn.Linear(hidden_dim, sample_count * 2)
        self.weight_head = nn.Linear(hidden_dim, sample_count)
        self.level_projections = nn.ModuleList(
            [nn.Linear(self.feature_dim, hidden_dim) for _ in range(self.num_scales)]
        )
        self.context_proj = MLP(hidden_dim * 3, hidden_dim, hidden_dim, depth=2)
        self.scale = nn.Parameter(torch.tensor(-1.0))
        nn.init.zeros_(self.offset_head.weight)
        with torch.no_grad():
            self.offset_head.bias.copy_(
                _radial_offset_bias(self.num_scales, self.num_points).to(
                    dtype=self.offset_head.bias.dtype
                )
            )
        nn.init.zeros_(self.weight_head.weight)
        nn.init.zeros_(self.weight_head.bias)
        self._zero_init_last_linear(self.context_proj)

    def forward(
        self,
        tokens: torch.Tensor,
        reference_xy: torch.Tensor,
        box_wh: torch.Tensor,
        feature_maps: list[torch.Tensor],
        spatial_shapes: list[torch.Tensor] | torch.Tensor,
    ) -> torch.Tensor:
        if tokens.numel() == 0 or not feature_maps:
            return tokens
        # Coordinate heads are supervised at their own stage.  Treat their
        # output as a fixed spatial reference while sampling the next stage,
        # matching iterative Deformable-DETR/DINO refinement and avoiding the
        # very large derivative of Fourier/grid-sampling paths in bf16.
        reference_xy = reference_xy.detach()
        box_wh = box_wh.detach()
        maps, level_shapes = _repeat_feature_levels(
            feature_maps, spatial_shapes, self.num_scales
        )
        b, q, k, c = tokens.shape
        sample_count = self.num_scales * self.num_points
        token_input = tokens.to(dtype=self.offset_head.weight.dtype)
        offsets = torch.tanh(self.offset_head(token_input).float()).to(dtype=tokens.dtype)
        offsets = offsets.view(b, q, k, self.num_scales, self.num_points, 2)
        weight_logits = self.weight_head(token_input).float().view(
            b, q, k, self.num_scales, self.num_points
        )
        if self.num_scales == 2:
            weight_logits = _apply_two_level_scale_prior(
                weight_logits,
                box_wh.float(),
                level_shapes[1],
                num_points=self.num_points,
                strength=self.scale_prior_strength,
                center_cells=self.scale_prior_center_cells,
                temperature=self.scale_prior_temperature,
            )
        weights = weight_logits.flatten(-2).softmax(dim=-1).view_as(weight_logits)
        weights = weights.to(dtype=tokens.dtype)

        sampled_scales = []
        for scale_idx, feature_map in enumerate(maps):
            shape = level_shapes[scale_idx]
            minimum = torch.stack(
                [
                    self.min_radius_cells / shape[:, 1].clamp(min=1).to(dtype=box_wh.dtype),
                    self.min_radius_cells / shape[:, 0].clamp(min=1).to(dtype=box_wh.dtype),
                ],
                dim=-1,
            ).to(device=box_wh.device)[:, None, :]
            radius = torch.maximum(
                box_wh.to(dtype=tokens.dtype) * self.offset_scale,
                minimum.to(dtype=tokens.dtype),
            ).view(b, q, 1, 1, 2)
            points = (
                reference_xy.to(dtype=tokens.dtype).unsqueeze(3)
                + offsets[:, :, :, scale_idx] * radius
            ).clamp(0.0, 1.0)
            sampled = self._sample_points(feature_map, points, shape)
            projection = self.level_projections[scale_idx]
            sampled = projection(sampled.to(dtype=projection.weight.dtype)).to(dtype=tokens.dtype)
            sampled_scales.append(sampled)
        sampled_all = torch.stack(sampled_scales, dim=3).reshape(b, q, k, sample_count, c)
        sampled = (sampled_all * weights.reshape(b, q, k, sample_count, 1)).sum(dim=3)
        point_pe = point_fourier_pe(reference_xy.to(dtype=tokens.dtype), c)
        update = self.context_proj(torch.cat([tokens, sampled, point_pe], dim=-1))
        return tokens + self.scale.sigmoid().to(dtype=update.dtype) * update

    @staticmethod
    def _sample_points(
        feature_map: torch.Tensor,
        points: torch.Tensor,
        spatial_shapes: torch.Tensor,
    ) -> torch.Tensor:
        b, channels, feature_h, feature_w = feature_map.shape
        q, k, p = points.shape[1], points.shape[2], points.shape[3]
        grid = _padded_grid_from_normalized_points(
            points.to(device=feature_map.device, dtype=feature_map.dtype),
            spatial_shapes,
            feature_h,
            feature_w,
        )
        grid = grid.view(b, q * k * p, 1, 2)
        sampled = F.grid_sample(feature_map, grid, align_corners=False)
        sampled = sampled.squeeze(-1).transpose(1, 2).view(b, q, k, p, channels)
        return sampled

    @staticmethod
    def _zero_init_last_linear(module: nn.Module) -> None:
        for child in reversed(list(module.modules())):
            if isinstance(child, nn.Linear):
                nn.init.zeros_(child.weight)
                if child.bias is not None:
                    nn.init.zeros_(child.bias)
                return


class MultiScaleDeformableEncoderLayer(nn.Module):
    """Learn cross-scale context on native P2/P3/P4 grids.

    Each level keeps its own native grid. Tokens at every valid grid location
    use deformable sampling over all feature levels, followed by a local FFN.
    This avoids fixed-size resizing or top-down FPN fusion while making the
    multi-scale memory itself trainable before the pose decoder consumes it.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_levels: int,
        num_points: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        c = int(hidden_dim)
        self.attention = JointDeformableKeypointAttention(
            c,
            feature_dim=c,
            num_scales=int(num_levels),
            num_points=int(num_points),
            offset_scale=0.20,
            min_radius_cells=1.0,
            scale_prior_strength=0.0,
        )
        # Pose decoder residuals start at zero for migration stability, but the
        # newly introduced feature encoder must learn cross-scale routing from
        # optimizer step one. Give its final context projection a small nonzero
        # initialization so offsets, weights and P4 all receive immediate grads.
        for child in reversed(list(self.attention.context_proj.modules())):
            if isinstance(child, nn.Linear):
                nn.init.xavier_uniform_(child.weight, gain=0.1)
                if child.bias is not None:
                    nn.init.zeros_(child.bias)
                break
        self.norm1 = nn.LayerNorm(c)
        self.norm2 = nn.LayerNorm(c)
        self.ffn = nn.Sequential(
            nn.Linear(c, c * 4),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(c * 4, c),
        )
        self.dropout = nn.Dropout(float(dropout))

    @staticmethod
    def _reference_grid(
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        y, x = torch.meshgrid(
            (torch.arange(height, device=device, dtype=torch.float32) + 0.5)
            / max(float(height), 1.0),
            (torch.arange(width, device=device, dtype=torch.float32) + 0.5)
            / max(float(width), 1.0),
            indexing="ij",
        )
        return torch.stack([x, y], dim=-1).reshape(1, height * width, 1, 2).expand(
            batch_size, -1, -1, -1
        ).to(dtype=dtype)

    def forward(
        self,
        feature_maps: list[torch.Tensor],
        spatial_shapes: list[torch.Tensor],
        valid_masks: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        for level_idx, feature in enumerate(feature_maps):
            b, c, h, w = feature.shape
            tokens = feature.flatten(2).transpose(1, 2)
            normalized = self.norm1(tokens)
            reference = self._reference_grid(b, h, w, feature.device, feature.dtype)
            level_shape = spatial_shapes[level_idx]
            person_scale = torch.stack(
                [
                    2.0 / level_shape[:, 1].clamp(min=1).to(dtype=feature.dtype),
                    2.0 / level_shape[:, 0].clamp(min=1).to(dtype=feature.dtype),
                ],
                dim=-1,
            )[:, None].expand(-1, h * w, -1)
            updated = self.attention(
                normalized[:, :, None, :],
                reference,
                person_scale,
                feature_maps,
                spatial_shapes,
            )[:, :, 0]
            tokens = tokens + self.dropout(updated - normalized)
            tokens = tokens + self.dropout(self.ffn(self.norm2(tokens)))
            valid = valid_masks[level_idx].flatten(1)[..., None].to(dtype=tokens.dtype)
            outputs.append(
                (tokens * valid).transpose(1, 2).reshape(b, c, h, w)
            )
        return outputs


class GroupedPoseDecoderLayer(nn.Module):
    """GroupPose person/joint interaction without ROI memory.

    Tokens are grouped as one explicit instance token followed by active joint
    tokens. The layer performs within-person interaction and same-role
    cross-person interaction. Whole-image P2/P3/P4 deformable attention is
    applied immediately after this layer by the pose decoder.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        c = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.within_norm = nn.LayerNorm(c)
        self.within_attention = nn.MultiheadAttention(
            c, self.num_heads, dropout=float(dropout), batch_first=True
        )
        self.same_role_norm = nn.LayerNorm(c)
        self.same_role_attention = nn.MultiheadAttention(
            c, self.num_heads, dropout=float(dropout), batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(c)
        self.ffn = nn.Sequential(
            nn.Linear(c, c * 4),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(c * 4, c),
        )
        self.residual_dropout = nn.Dropout(float(dropout))

    @staticmethod
    def _safe_padding_mask(valid: torch.Tensor) -> torch.Tensor:
        padding = ~valid.bool()
        return padding.masked_fill(padding.all(dim=1, keepdim=True), False)

    def forward(
        self,
        tokens: torch.Tensor,
        role_position: torch.Tensor,
        token_valid: torch.Tensor,
        same_role_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, num_people, num_roles, c = tokens.shape
        if num_people == 0:
            return tokens

        valid_f = token_valid[..., None].to(dtype=tokens.dtype)

        normalized = self.within_norm(tokens)
        within_query = (normalized + role_position).reshape(
            b * num_people, num_roles, c
        )
        within_value = normalized.reshape(b * num_people, num_roles, c)
        within_valid = token_valid.reshape(b * num_people, num_roles)
        within_update = self.within_attention(
            within_query,
            within_query,
            within_value,
            key_padding_mask=self._safe_padding_mask(within_valid),
            need_weights=False,
        )[0].view(b, num_people, num_roles, c)
        tokens = tokens + self.residual_dropout(within_update) * valid_f

        normalized = self.same_role_norm(tokens)
        same_query = (normalized + role_position).permute(0, 2, 1, 3).reshape(
            b * num_roles, num_people, c
        )
        same_value = normalized.permute(0, 2, 1, 3).reshape(
            b * num_roles, num_people, c
        )
        same_valid = token_valid.permute(0, 2, 1).reshape(
            b * num_roles, num_people
        )
        same_update = self.same_role_attention(
            same_query,
            same_query,
            same_value,
            attn_mask=same_role_attention_mask,
            key_padding_mask=self._safe_padding_mask(same_valid),
            need_weights=False,
        )[0]
        same_update = same_update.view(b, num_roles, num_people, c).permute(
            0, 2, 1, 3
        )
        tokens = tokens + self.residual_dropout(same_update) * valid_f

        ffn_update = self.ffn(self.ffn_norm(tokens))
        return (tokens + self.residual_dropout(ffn_update) * valid_f) * valid_f


def canonical_joint_priors() -> torch.Tensor:
    """Fallback joint locations inside a normalized person box.

    ``left`` and ``right`` follow anatomical person coordinates. In the common
    front-facing image convention, the person's left side lies on the image's
    right, so left-joint x priors are greater than their right-joint partners.
    Dataset-specific priors loaded by :func:`build_schema_joint_priors` override
    these values when available.
    """
    priors = {
        "nose": (0.50, 0.16),
        "left_eye": (0.56, 0.13),
        "right_eye": (0.44, 0.13),
        "left_ear": (0.62, 0.16),
        "right_ear": (0.38, 0.16),
        "left_shoulder": (0.64, 0.32),
        "right_shoulder": (0.36, 0.32),
        "left_elbow": (0.70, 0.48),
        "right_elbow": (0.30, 0.48),
        "left_wrist": (0.74, 0.64),
        "right_wrist": (0.26, 0.64),
        "left_hip": (0.58, 0.58),
        "right_hip": (0.42, 0.58),
        "left_knee": (0.60, 0.76),
        "right_knee": (0.40, 0.76),
        "left_ankle": (0.61, 0.92),
        "right_ankle": (0.39, 0.92),
        "neck": (0.50, 0.28),
        "head_top": (0.50, 0.06),
        "pelvis": (0.50, 0.60),
        "thorax": (0.50, 0.38),
        "upper_neck": (0.50, 0.24),
        "crowdpose_head": (0.50, 0.12),
    }
    return torch.tensor([priors[name] for name in UNION_KEYPOINTS], dtype=torch.float32)


def build_schema_joint_priors(path: str | None) -> torch.Tensor:
    """Build ``[schema, union_joint, xy]`` priors from an optional JSON file."""
    fallback = canonical_joint_priors()
    priors = fallback.unsqueeze(0).repeat(len(SCHEMA_NAMES), 1, 1)
    if not path:
        return priors

    prior_path = Path(path)
    if not prior_path.is_file():
        warnings.warn(
            f"Schema joint prior file does not exist: {prior_path}; using fallback priors. "
            "A checkpoint load may overwrite this persistent buffer.",
            stacklevel=2,
        )
        return priors
    with prior_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Schema joint prior file must contain a JSON object.")

    for schema_idx, schema_name in enumerate(SCHEMA_NAMES):
        schema_payload = payload.get(schema_name, {})
        if not isinstance(schema_payload, dict):
            raise ValueError(f"Prior entry for {schema_name} must be an object.")
        for joint_name, xy in schema_payload.items():
            if joint_name not in UNION_TO_ID:
                raise KeyError(f"Unknown joint {joint_name!r} in prior file for {schema_name}.")
            if not isinstance(xy, (list, tuple)) or len(xy) != 2:
                raise ValueError(f"Prior for {schema_name}/{joint_name} must be [x, y].")
            priors[schema_idx, UNION_TO_ID[joint_name]] = torch.tensor(
                [float(xy[0]), float(xy[1])], dtype=torch.float32
            ).clamp(0.02, 0.98)
    return priors


def nonsemantic_joint_reference_points(
    count: int,
    *,
    low: float = 0.2,
    high: float = 0.8,
) -> torch.Tensor:
    """Return deterministic, spatially dispersed points without a pose prior.

    Halton coordinates cover the person box without assigning anatomical
    meaning to any location.  They break the all-joints-at-center symmetry for
    deformable sampling, while remaining learnable after initialization.
    """

    def radical_inverse(index: int, base: int) -> float:
        value = 0.0
        factor = 1.0 / float(base)
        while index > 0:
            value += factor * float(index % base)
            index //= base
            factor /= float(base)
        return value

    if count <= 0:
        return torch.zeros(0, 2, dtype=torch.float32)
    if not 0.0 < low < high < 1.0:
        raise ValueError("Joint reference bounds must satisfy 0 < low < high < 1.")
    unit = torch.tensor(
        [
            (radical_inverse(index, 2), radical_inverse(index, 3))
            for index in range(1, int(count) + 1)
        ],
        dtype=torch.float32,
    )
    return low + (high - low) * unit


def box_fourier_pe(boxes: torch.Tensor, hidden_dim: int) -> torch.Tensor:
    """Fourier PE for normalized xyxy boxes."""
    wh = (boxes[..., 2:] - boxes[..., :2]).clamp(min=1e-4)
    log_wh = 0.2 * torch.log(wh) + 0.5
    geom = torch.cat([boxes[..., :2], log_wh], dim=-1)
    pe = _bounded_fourier_features(geom, hidden_dim)
    if pe.shape[-1] < hidden_dim:
        pe = F.pad(pe, (0, hidden_dim - pe.shape[-1]))
    return pe[..., :hidden_dim].to(dtype=boxes.dtype)


def _bounded_fourier_features(values: torch.Tensor, hidden_dim: int, max_log2_freq: float = 8.0) -> torch.Tensor:
    """Stable Fourier features for low-dimensional normalized geometry.

    The old 2**arange schedule reaches extreme frequencies when hidden_dim is
    large. In fp16/bf16 that can overflow or make sin/cos return NaNs, which then
    poisons Hungarian matching. A bounded log-spaced schedule keeps the same
    periodic inductive bias without entering the numerically hostile range.
    """
    num_coords = int(values.shape[-1])
    num_freq = max(hidden_dim // max(2 * num_coords, 1), 1)
    values_f = values.float()
    freq = torch.linspace(
        0.0,
        max_log2_freq,
        num_freq,
        device=values.device,
        dtype=torch.float32,
    )
    freq = (2.0 ** freq) * torch.pi
    feat = values_f[..., None] * freq
    pe = torch.cat([torch.sin(feat), torch.cos(feat)], dim=-1).flatten(-2)
    if pe.shape[-1] < hidden_dim:
        pe = F.pad(pe, (0, hidden_dim - pe.shape[-1]))
    return pe[..., :hidden_dim]


def point_fourier_pe(points: torch.Tensor, hidden_dim: int) -> torch.Tensor:
    """Fourier PE for normalized xy points."""
    pe = _bounded_fourier_features(points, hidden_dim)
    if pe.shape[-1] < hidden_dim:
        pe = F.pad(pe, (0, hidden_dim - pe.shape[-1]))
    return pe[..., :hidden_dim].to(dtype=points.dtype)


def boxes_from_cxcywh(raw: torch.Tensor) -> torch.Tensor:
    cxcy = raw[..., :2].sigmoid()
    wh = raw[..., 2:].sigmoid() * 0.9
    xy1 = cxcy - wh * 0.5
    xy2 = cxcy + wh * 0.5
    return torch.cat([xy1, xy2], dim=-1).clamp(0.0, 1.0)


def expand_boxes_xyxy(boxes: torch.Tensor, scale: float) -> torch.Tensor:
    scale = max(float(scale), 1e-4)
    center = (boxes[..., :2] + boxes[..., 2:]) * 0.5
    wh = (boxes[..., 2:] - boxes[..., :2]).clamp(min=1e-4) * scale
    xy1 = center - wh * 0.5
    xy2 = center + wh * 0.5
    return torch.cat([xy1, xy2], dim=-1).clamp(0.0, 1.0)


def refine_boxes_xyxy(boxes: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """DAB/DINO-style inverse-sigmoid box refinement in normalized cxcywh space."""
    center = (boxes[..., :2] + boxes[..., 2:]) * 0.5
    wh = (boxes[..., 2:] - boxes[..., :2]).clamp(min=1e-4)
    cxcywh = torch.cat([center, wh], dim=-1).clamp(1e-4, 1.0 - 1e-4)
    refined = torch.sigmoid(torch.logit(cxcywh) + deltas.float()).to(dtype=boxes.dtype)
    center = refined[..., :2]
    wh = refined[..., 2:].clamp(min=1e-4)
    xy1 = center - wh * 0.5
    xy2 = center + wh * 0.5
    return torch.cat([xy1, xy2], dim=-1).clamp(0.0, 1.0)


def refine_external_boxes_xyxy(
    boxes: torch.Tensor,
    deltas: torch.Tensor,
    *,
    center_fraction: float = 0.15,
    minimum_scale: float = 0.75,
    maximum_scale: float = 4.0 / 3.0,
) -> torch.Tensor:
    """Apply a bounded local correction without allowing identity-changing jumps."""
    center = (boxes[..., :2] + boxes[..., 2:]) * 0.5
    wh = (boxes[..., 2:] - boxes[..., :2]).clamp(min=1e-4)
    delta_center = (
        torch.tanh(deltas[..., :2].float())
        * float(center_fraction)
        * wh.float()
    )
    log_min = math.log(max(float(minimum_scale), 1e-4))
    log_max = math.log(max(float(maximum_scale), float(minimum_scale) + 1e-4))
    size_control = torch.tanh(deltas[..., 2:].float())
    log_scale = torch.where(
        size_control >= 0,
        size_control * log_max,
        -size_control * log_min,
    )
    refined_center = center.float() + delta_center
    refined_wh = wh.float() * log_scale.exp()
    xy1 = refined_center - refined_wh * 0.5
    xy2 = refined_center + refined_wh * 0.5
    return torch.cat([xy1, xy2], dim=-1).to(dtype=boxes.dtype).clamp(0.0, 1.0)


def box_soft_gate(
    boxes: torch.Tensor,
    box_mask: torch.Tensor,
    height: int,
    width: int,
    expand_ratio: float = 0.15,
    alpha: float = 20.0,
) -> torch.Tensor:
    """Build a differentiable per-image spatial gate from normalized boxes."""
    b, n, _ = boxes.shape
    y, x = torch.meshgrid(
        torch.linspace(0, 1, height, device=boxes.device, dtype=boxes.dtype),
        torch.linspace(0, 1, width, device=boxes.device, dtype=boxes.dtype),
        indexing="ij",
    )
    xy1 = boxes[..., :2]
    xy2 = boxes[..., 2:]
    wh = (xy2 - xy1).clamp(min=1e-4)
    xy1 = (xy1 - wh * (expand_ratio * 0.5)).clamp(0.0, 1.0)
    xy2 = (xy2 + wh * (expand_ratio * 0.5)).clamp(0.0, 1.0)
    x = x.view(1, 1, height, width)
    y = y.view(1, 1, height, width)
    x1 = xy1[..., 0].view(b, n, 1, 1)
    y1 = xy1[..., 1].view(b, n, 1, 1)
    x2 = xy2[..., 0].view(b, n, 1, 1)
    y2 = xy2[..., 1].view(b, n, 1, 1)
    gate = (
        torch.sigmoid(alpha * (x - x1))
        * torch.sigmoid(alpha * (x2 - x))
        * torch.sigmoid(alpha * (y - y1))
        * torch.sigmoid(alpha * (y2 - y))
    )
    gate = gate.masked_fill(~box_mask.view(b, n, 1, 1), 0.0)
    return gate.max(dim=1, keepdim=True).values


@dataclass
class QwenPoseConfig:
    hidden_dim: int = 448
    external_dim: int = 2560
    high_res_external_dim: int = 0
    pose_decoder_layers: int = 3
    refinement_steps: int = 1
    decoder_heads: int = 8
    dropout: float = 0.0
    # Tight refined boxes initialize joints; this scale only enlarges the
    # deformable-attention context used to gather surrounding limb evidence.
    box_condition_scale: float = 1.15
    use_refinement: bool = True
    pose_feature_channels: int = 256
    use_native_spatial_features: bool = True
    deformable_points: int = 4
    deformable_min_radius_cells: float = 2.0
    deformable_scale_prior_strength: float = 0.5
    deformable_scale_prior_center_cells: float = 6.0
    deformable_scale_prior_temperature: float = 1.5
    enable_keypoint_denoising: bool = True
    ref_text_scale: float = 0.2
    legacy_checkpoint_compat: bool = False
    enable_person_confidence_head: bool = True
    person_confidence_rescue: bool = False
    # In the unified detector/grounder path, learned person queries replace
    # externally generated coordinate strings.  One forward pass predicts all
    # people; RefHuman then selects one of those people with ``ref_match_head``.
    use_global_person_queries: bool = True
    num_person_queries: int = 60
    num_ref_queries: int = 4
    multiscale_encoder_layers: int = 2
    multiscale_encoder_points: int = 4
    use_detrpose_architecture: bool = True
    # The default starts from a schema-aligned anatomical reference and predicts
    # an instance-conditioned residual before iterative decoder refinement.
    pose_coordinate_init: str = "anatomical_dynamic"
    schema_joint_priors_path: str | None = "configs/schema_joint_priors.json"
    dynamic_reference_offset_scale: float = 1.5


class QwenPoseModel(nn.Module):
    def __init__(self, config: QwenPoseConfig) -> None:
        super().__init__()
        self.config = config
        if not bool(config.use_detrpose_architecture):
            raise ValueError("Only the active DETRPose/GroupPose architecture is supported.")
        c = int(config.hidden_dim)
        pose_feature_dim = int(config.pose_feature_channels)
        coordinate_init = str(config.pose_coordinate_init).strip().lower()
        coordinate_modes = {
            "anatomical_dynamic",
            "learned_spread",
            "box_center",
            "schema_prior",
        }
        if coordinate_init not in coordinate_modes:
            raise ValueError(
                "pose_coordinate_init must be anatomical_dynamic, learned_spread, "
                f"box_center, or schema_prior, got {config.pose_coordinate_init!r}."
            )
        self.config.pose_coordinate_init = coordinate_init
        if not config.use_native_spatial_features:
            raise ValueError("Only native-grid Locate spatial features are supported.")
        if config.legacy_checkpoint_compat:
            raise ValueError(
                "legacy_checkpoint_compat is not supported by the Locate-only pose feature. "
                "Train a new Stage1 checkpoint for this architecture."
            )
        self.base_feature_levels = 2 if int(config.high_res_external_dim) > 0 else 1
        self.num_feature_levels = (
            self.base_feature_levels + 1
            if bool(config.use_detrpose_architecture)
            else self.base_feature_levels
        )
        self.high_res_feature_proj = (
            nn.Conv2d(int(config.high_res_external_dim), pose_feature_dim, 1)
            if self.base_feature_levels == 2
            else None
        )
        self.external_feature_proj = nn.Conv2d(config.external_dim, pose_feature_dim, 1)
        self.p4_feature_proj = (
            nn.Sequential(
                nn.Conv2d(pose_feature_dim, pose_feature_dim, 3, stride=2, padding=1),
                nn.GroupNorm(_group_count(pose_feature_dim), pose_feature_dim),
                nn.GELU(),
            )
            if bool(config.use_detrpose_architecture)
            else None
        )
        self.feature_level_embeddings = nn.Parameter(
            torch.zeros(self.num_feature_levels, pose_feature_dim)
        )
        nn.init.normal_(self.feature_level_embeddings, mean=0.0, std=0.02)
        self.external_text_proj = nn.Sequential(
            nn.LayerNorm(config.external_dim),
            nn.Linear(config.external_dim, c),
            nn.GELU(),
        )
        self.ref_candidate_proj = nn.Sequential(
            nn.LayerNorm(c),
            nn.Linear(c, c),
            nn.GELU(),
        )
        self.ref_text_match_proj = nn.Sequential(
            nn.LayerNorm(c),
            nn.Linear(c, c),
            nn.GELU(),
        )
        self.ref_match_head = MLP(c * 3, c, 1, depth=3)
        ref_output = self.ref_match_head.net[-1]
        if not isinstance(ref_output, nn.Linear):
            raise TypeError("RefHuman match head must end with a Linear layer.")
        nn.init.normal_(ref_output.weight, mean=0.0, std=1e-3)
        if ref_output.bias is not None:
            nn.init.zeros_(ref_output.bias)
        self.human_context_proj = nn.Sequential(
            nn.LayerNorm(pose_feature_dim),
            nn.Linear(pose_feature_dim, c),
            nn.GELU(),
            nn.Linear(c, c),
        )
        self.multiscale_encoder = nn.ModuleList(
            [
                MultiScaleDeformableEncoderLayer(
                    pose_feature_dim,
                    num_levels=self.num_feature_levels,
                    num_points=config.multiscale_encoder_points,
                    dropout=float(config.dropout),
                )
                for _ in range(max(int(config.multiscale_encoder_layers), 1))
            ]
        )
        self.proposal_objectness_heads = nn.ModuleList(
            [nn.Conv2d(pose_feature_dim, 1, 1) for _ in range(self.num_feature_levels)]
        )
        self.proposal_offset_heads = nn.ModuleList(
            [nn.Conv2d(pose_feature_dim, 2, 1) for _ in range(self.num_feature_levels)]
        )
        self.proposal_scale_heads = nn.ModuleList(
            [nn.Conv2d(pose_feature_dim, 2, 1) for _ in range(self.num_feature_levels)]
        )
        self.proposal_token_proj = nn.Sequential(
            nn.LayerNorm(pose_feature_dim),
            nn.Linear(pose_feature_dim, c),
            nn.GELU(),
            nn.Linear(c, c),
        )
        self.proposal_text_proj = nn.Sequential(
            nn.LayerNorm(c),
            nn.Linear(c, pose_feature_dim),
        )
        self.proposal_source_embed = nn.Embedding(3, c)
        nn.init.normal_(self.proposal_source_embed.weight, mean=0.0, std=0.02)
        self.external_box_token_proj = nn.Sequential(
            nn.LayerNorm(pose_feature_dim * self.num_feature_levels),
            nn.Linear(pose_feature_dim * self.num_feature_levels, c),
            nn.GELU(),
            nn.Linear(c, c),
        )
        # The pre-pose head locally corrects an external GT/Locate proposal
        # before it initializes joint references. The existing post-pose head
        # remains the canonical final bbox output head.
        self.pre_pose_box_norm = nn.LayerNorm(c)
        self.pre_pose_box_refine_head = MLP(c, c, 4, depth=3)
        self.external_box_refine_head = MLP(c, c, 4, depth=3)
        self._zero_init_last_linear(self.pre_pose_box_refine_head)
        self._zero_init_last_linear(self.external_box_refine_head)
        for objectness_head in self.proposal_objectness_heads:
            nn.init.zeros_(objectness_head.weight)
            nn.init.zeros_(objectness_head.bias)
        for offset_head in self.proposal_offset_heads:
            nn.init.zeros_(offset_head.weight)
            nn.init.zeros_(offset_head.bias)
        for scale_head in self.proposal_scale_heads:
            nn.init.zeros_(scale_head.weight)
            nn.init.constant_(scale_head.bias, -3.0)
        self.person_query_embed: nn.Embedding | None = None
        if config.use_global_person_queries:
            num_queries = max(int(config.num_person_queries), 1)
            self.config.num_person_queries = num_queries
            self.person_query_embed = nn.Embedding(num_queries, c)
            nn.init.normal_(self.person_query_embed.weight, mean=0.0, std=0.02)

        # Schema identity controls only the active joint set. Joint semantics
        # come from the shared union embedding, never a dataset-specific route.
        # Keeping a learnable schema embedding here would let shared joints route
        # into separate dataset-specific predictors, weakening cross-dataset transfer.
        self.schema_embed = nn.Embedding(len(SCHEMA_NAMES), c)
        nn.init.zeros_(self.schema_embed.weight)
        self.schema_embed.weight.requires_grad_(False)
        self.task_embed = nn.Embedding(2, c)
        self.joint_embed = nn.Embedding(len(UNION_KEYPOINTS), c)
        self.pose_token_type_embed = nn.Embedding(2, c)
        nn.init.normal_(self.pose_token_type_embed.weight, mean=0.0, std=0.02)
        self.joint_reference_logits = (
            nn.Parameter(
                torch.logit(
                    nonsemantic_joint_reference_points(len(UNION_KEYPOINTS))
                )
            )
            if coordinate_init == "learned_spread"
            else None
        )
        self.reference_offset_head = (
            MLP(c * 2, c, 2, depth=3)
            if coordinate_init == "anatomical_dynamic"
            else None
        )
        if self.reference_offset_head is not None:
            self._zero_init_last_linear(self.reference_offset_head)
        # Training-only keypoint DN branches share the complete pose decoder.
        # This tiny type embedding tells positive reconstruction queries apart
        # from contrastive negative queries; it is never used at inference.
        self.keypoint_dn_type_embed = (
            nn.Embedding(2, c) if config.enable_keypoint_denoising else None
        )
        if self.keypoint_dn_type_embed is not None:
            nn.init.normal_(self.keypoint_dn_type_embed.weight, mean=0.0, std=0.02)
        # Dynamic anatomical references and legacy fixed-prior evaluation both
        # need a checkpoint-contained schema prior tensor. Ablation modes keep
        # the old state-dict behavior and do not read the prior file.
        self.register_buffer(
            "schema_joint_priors",
            build_schema_joint_priors(config.schema_joint_priors_path)
            if coordinate_init in {"anatomical_dynamic", "schema_prior"}
            else None,
            persistent=True,
        )
        self.box_query_proj = nn.Sequential(
            nn.LayerNorm(c),
            MLP(c, c, c, depth=2),
            nn.LayerNorm(c),
        )
        self.instance_query_norm = nn.LayerNorm(c)
        self.pose_instance_token_norm = nn.LayerNorm(c)
        self.pose_query_norm = nn.LayerNorm(c)
        max_schema_keypoints = max(int(indices.numel()) for indices in SCHEMA_INDICES.values())
        schema_joint_indices = torch.zeros(len(SCHEMA_NAMES), max_schema_keypoints, dtype=torch.long)
        schema_joint_valid = torch.zeros(len(SCHEMA_NAMES), max_schema_keypoints, dtype=torch.bool)
        for schema_id, schema_name in enumerate(SCHEMA_NAMES):
            indices = SCHEMA_INDICES[schema_name].long()
            schema_joint_indices[schema_id, : indices.numel()] = indices
            schema_joint_valid[schema_id, : indices.numel()] = True
        self.register_buffer("schema_joint_indices", schema_joint_indices, persistent=False)
        self.register_buffer("schema_joint_valid", schema_joint_valid, persistent=False)
        pose_decoder_layers = max(int(config.pose_decoder_layers), 1)
        self.pose_group_decoder_layers = nn.ModuleList(
            [
                GroupedPoseDecoderLayer(
                    c,
                    num_heads=config.decoder_heads,
                    dropout=float(config.dropout),
                )
                for _ in range(pose_decoder_layers)
            ]
        )
        decoder_offset_scales = (0.25, 0.15, 0.08)
        decoder_min_radius_cells = (2.0, 1.0, 0.5)
        self.pose_decoder_deformable_attention = nn.ModuleList(
            [
                JointDeformableKeypointAttention(
                    c,
                    feature_dim=pose_feature_dim,
                    num_scales=self.num_feature_levels,
                    num_points=config.deformable_points,
                    offset_scale=decoder_offset_scales[
                        min(layer_idx, len(decoder_offset_scales) - 1)
                    ],
                    min_radius_cells=decoder_min_radius_cells[
                        min(layer_idx, len(decoder_min_radius_cells) - 1)
                    ],
                    scale_prior_strength=config.deformable_scale_prior_strength,
                    scale_prior_center_cells=config.deformable_scale_prior_center_cells,
                    scale_prior_temperature=config.deformable_scale_prior_temperature,
                )
                for layer_idx in range(pose_decoder_layers)
            ]
        )
        self.pose_decoder_coordinate_heads = nn.ModuleList(
            [MLP(c, c, 2, depth=3) for _ in range(pose_decoder_layers)]
        )
        for coordinate_head in self.pose_decoder_coordinate_heads:
            self._zero_init_last_linear(coordinate_head)
        # Keep joint presence/visibility separate from the instance score used
        # by COCO keypoint AP.  The third keypoint channel is useful for drawing
        # and downstream consumers, but it must never gate the pose AP score.
        self.keypoint_visibility_head = MLP(c, c, 1, depth=3)

        # The two public AP logits are direct, quality-aware person scores.  In
        # particular, neither is multiplied by proposal objectness at inference.
        # Pose-LQE adds local evidence sampled at the final joint coordinates to
        # an instance-token base score; the box head similarly reads a small ROI
        # from the independently regressed box.
        self.pose_score_base_head = MLP(c, c, 1, depth=3)
        self.pose_lqe_feature_proj = nn.Sequential(
            nn.LayerNorm(pose_feature_dim * self.num_feature_levels),
            nn.Linear(pose_feature_dim * self.num_feature_levels, c),
            nn.GELU(),
        )
        self.pose_lqe_joint_head = MLP(c * 2, c, 1, depth=2)
        self.pose_lqe_residual_head = MLP(c + 3, c, 1, depth=3)
        self.box_lqe_feature_proj = nn.Sequential(
            nn.LayerNorm(pose_feature_dim),
            nn.Linear(pose_feature_dim, c),
            nn.GELU(),
        )
        self.box_score_head = MLP(c * 3, c, 1, depth=3)
        if config.use_refinement:
            self.local_proj = nn.Conv2d(pose_feature_dim, c, 1)
            joint_context_layer = nn.TransformerEncoderLayer(
                d_model=c,
                nhead=config.decoder_heads,
                dim_feedforward=c * 4,
                dropout=float(config.dropout),
                activation="gelu",
                batch_first=True,
            )
            self.joint_context = nn.TransformerEncoder(joint_context_layer, num_layers=1)
            refinement_steps = max(int(config.refinement_steps), 1)
            self.refine_heads = nn.ModuleList([MLP(c * 3, c, 2, depth=3) for _ in range(refinement_steps)])
            self.refine_token_fusers = nn.ModuleList([MLP(c * 3, c, c, depth=2) for _ in range(refinement_steps)])
            self.refine_patch_weight_heads = nn.ModuleList(
                [nn.Linear(c, 9) for _ in range(refinement_steps)]
            )
            initial_scales = torch.tensor(
                [0.75, 0.50, 0.35] + [0.35] * max(refinement_steps - 3, 0),
                dtype=torch.float32,
            )[:refinement_steps]
            self.refine_step_scales = nn.Parameter(torch.logit(initial_scales))
            for head in self.refine_heads:
                self._zero_init_last_linear(head)
            for fuser in self.refine_token_fusers:
                self._zero_init_last_linear(fuser)
            for weight_head in self.refine_patch_weight_heads:
                nn.init.zeros_(weight_head.weight)
                nn.init.zeros_(weight_head.bias)
        else:
            self.local_proj = None
            self.joint_context = None
            self.refine_heads = None
            self.refine_token_fusers = None
            self.refine_patch_weight_heads = None
            self.refine_step_scales = None
        self._zero_init_last_linear(self.keypoint_visibility_head)
        self._zero_init_last_linear(self.pose_score_base_head)
        self._zero_init_last_linear(self.pose_lqe_residual_head)
        self._zero_init_last_linear(self.box_score_head)
        foreground_prior_bias = -math.log((1.0 - 0.01) / 0.01)
        for score_head in (self.pose_score_base_head, self.box_score_head):
            final_layer = score_head.net[-1]
            if not isinstance(final_layer, nn.Linear):
                raise TypeError("Direct AP score heads must end with Linear layers.")
            if final_layer.bias is not None:
                nn.init.constant_(final_layer.bias, foreground_prior_bias)


    def build_locate_pose_features(
        self,
        external_feature_map: torch.Tensor | SpatialFeatureBatch | MultiScaleSpatialFeatureBatch,
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        int,
        int,
    ]:
        """Build trainable native-grid P2/P3/P4 memory without resizing or ROI."""

        def as_spatial_batch(value: torch.Tensor | SpatialFeatureBatch) -> SpatialFeatureBatch:
            if isinstance(value, SpatialFeatureBatch):
                return value
            batch_size, _, height, width = value.shape
            shapes = torch.tensor(
                [[height, width]] * batch_size,
                device=value.device,
                dtype=torch.long,
            )
            return SpatialFeatureBatch(value, shapes)

        if isinstance(external_feature_map, MultiScaleSpatialFeatureBatch):
            if self.base_feature_levels != 2 or self.high_res_feature_proj is None:
                raise ValueError(
                    "Received Locate P2/P3 features but PoseHead was not configured "
                    "with high_res_external_dim."
                )
            if len(external_feature_map.levels) != 2:
                raise ValueError("Locate multi-scale input must contain exactly P2 and P3.")
            raw_levels = list(external_feature_map.levels)
            projections = [self.high_res_feature_proj, self.external_feature_proj]
        else:
            if self.base_feature_levels != 1:
                raise ValueError(
                    "PoseHead expects true Locate P2/P3 features but received one feature level."
                )
            raw_levels = [as_spatial_batch(external_feature_map)]
            projections = [self.external_feature_proj]

        projected_levels: list[torch.Tensor] = []
        spatial_shapes: list[torch.Tensor] = []
        valid_masks: list[torch.Tensor] = []
        for level_idx, (raw_level, projection) in enumerate(zip(raw_levels, projections)):
            projection_dtype = projection.weight.dtype
            projected_maps = []
            level_embed = self.feature_level_embeddings[level_idx].to(
                device=raw_level.device,
                dtype=projection_dtype,
            ).view(-1, 1, 1)
            for batch_idx, (height, width) in enumerate(
                raw_level.spatial_shapes.detach().cpu().tolist()
            ):
                native_map = raw_level.tensor[
                    batch_idx : batch_idx + 1, :, : int(height), : int(width)
                ]
                projected = projection(native_map.to(dtype=projection_dtype)).squeeze(0)
                projected_maps.append(projected + level_embed)
            projected_batch = SpatialFeatureBatch.from_maps(projected_maps)
            projected_levels.append(projected_batch.tensor)
            spatial_shapes.append(projected_batch.spatial_shapes)
            valid_masks.append(projected_batch.valid_mask())
        if self.p4_feature_proj is not None:
            p3_batch = SpatialFeatureBatch(projected_levels[-1], spatial_shapes[-1])
            p4_maps: list[torch.Tensor] = []
            for batch_idx, (height, width) in enumerate(
                p3_batch.spatial_shapes.detach().cpu().tolist()
            ):
                native = p3_batch.tensor[
                    batch_idx : batch_idx + 1, :, : int(height), : int(width)
                ]
                p4 = self.p4_feature_proj(native).squeeze(0)
                p4 = p4 + self.feature_level_embeddings[-1].to(
                    device=p4.device, dtype=p4.dtype
                ).view(-1, 1, 1)
                p4_maps.append(p4)
            p4_batch = SpatialFeatureBatch.from_maps(p4_maps)
            projected_levels.append(p4_batch.tensor)
            spatial_shapes.append(p4_batch.spatial_shapes)
            valid_masks.append(p4_batch.valid_mask())

        if len(projected_levels) != self.num_feature_levels:
            raise RuntimeError(
                f"Expected {self.num_feature_levels} pose feature levels, got {len(projected_levels)}."
            )
        for encoder_layer in self.multiscale_encoder:
            projected_levels = encoder_layer(projected_levels, spatial_shapes, valid_masks)
        return projected_levels, spatial_shapes, valid_masks, min(1, len(projected_levels) - 1), 0

    @staticmethod
    def _sample_feature_at_points(
        feature_map: torch.Tensor,
        points: torch.Tensor,
        spatial_shapes: torch.Tensor,
    ) -> torch.Tensor:
        b, channels, height, width = feature_map.shape
        grid = _padded_grid_from_normalized_points(
            points.to(device=feature_map.device, dtype=feature_map.dtype),
            spatial_shapes,
            height,
            width,
        ).view(b, -1, 1, 2)
        sampled = F.grid_sample(feature_map, grid, align_corners=False)
        return sampled.squeeze(-1).transpose(1, 2)

    def _build_internal_pose_proposals(
        self,
        feature_maps: list[torch.Tensor],
        spatial_shapes: list[torch.Tensor],
        valid_masks: list[torch.Tensor],
        text_embed: torch.Tensor,
        task_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select fixed-count person groups from dense P2/P3/P4 predictions."""
        candidate_tokens: list[torch.Tensor] = []
        candidate_scores: list[torch.Tensor] = []
        candidate_text_scores: list[torch.Tensor] = []
        candidate_centers: list[torch.Tensor] = []
        candidate_scales: list[torch.Tensor] = []
        text_feature = F.normalize(
            self.proposal_text_proj(text_embed).float(), dim=-1
        )
        for level_idx, feature in enumerate(feature_maps):
            b, channels, height, width = feature.shape
            valid = valid_masks[level_idx]
            objectness = self.proposal_objectness_heads[level_idx](feature).flatten(1)
            offsets = torch.tanh(
                self.proposal_offset_heads[level_idx](feature).float()
            ).permute(0, 2, 3, 1).reshape(b, height * width, 2)
            scales = self.proposal_scale_heads[level_idx](feature).float().permute(
                0, 2, 3, 1
            ).reshape(b, height * width, 2)
            y, x = torch.meshgrid(
                torch.arange(height, device=feature.device, dtype=torch.float32),
                torch.arange(width, device=feature.device, dtype=torch.float32),
                indexing="ij",
            )
            base = torch.stack([x + 0.5, y + 0.5], dim=-1).reshape(1, -1, 2)
            native_wh = torch.stack(
                [spatial_shapes[level_idx][:, 1], spatial_shapes[level_idx][:, 0]],
                dim=-1,
            ).to(device=feature.device, dtype=torch.float32)[:, None]
            centers = (base + 0.5 * offsets) / native_wh.clamp(min=1.0)
            minimum_scale = (2.0 / native_wh.clamp(min=1.0)).clamp(0.02, 0.5)
            # A hard maximum would completely block scale-head gradients on
            # coarse levels. Interpolate continuously from the native-grid
            # minimum to 0.95 so P2/P3/P4 all learn person extent immediately.
            person_scale = minimum_scale + (0.95 - minimum_scale) * scales.sigmoid()
            flat_tokens = feature.flatten(2).transpose(1, 2)
            normalized_tokens = F.normalize(flat_tokens.float(), dim=-1)
            text_scores = torch.einsum(
                "bnc,bc->bn", normalized_tokens, text_feature
            )
            # Conv heads start at zero. Break score ties by native normalized
            # coordinates rather than padded flat indices so batching cannot
            # change proposal order for an otherwise identical image.
            tie_break = -1e-6 * (
                centers[..., 1] * 2.0
                + centers[..., 0]
                + float(level_idx) * 4.0
            )
            objectness = objectness + tie_break.to(dtype=objectness.dtype)
            text_scores = text_scores + tie_break
            valid_flat = valid.flatten(1)
            invalid_value = torch.finfo(objectness.dtype).min
            candidate_scores.append(objectness.masked_fill(~valid_flat, invalid_value))
            candidate_text_scores.append(
                text_scores.to(dtype=objectness.dtype).masked_fill(~valid_flat, invalid_value)
            )
            candidate_tokens.append(flat_tokens)
            candidate_centers.append(centers.to(dtype=feature.dtype))
            candidate_scales.append(person_scale.to(dtype=feature.dtype))

        all_tokens = torch.cat(candidate_tokens, dim=1)
        all_scores = torch.cat(candidate_scores, dim=1)
        all_text_scores = torch.cat(candidate_text_scores, dim=1)
        all_centers = torch.cat(candidate_centers, dim=1)
        all_scales = torch.cat(candidate_scales, dim=1)
        b = int(all_tokens.shape[0])
        query_count = max(int(self.config.num_person_queries), 1)
        ref_count = min(max(int(self.config.num_ref_queries), 0), query_count)
        level_lengths = [int(tokens.shape[1]) for tokens in candidate_tokens]
        level_offsets: list[int] = []
        running_offset = 0
        for level_length in level_lengths:
            level_offsets.append(running_offset)
            running_offset += level_length

        def allocate_level_counts(total: int) -> list[int]:
            """Reserve candidates across P2/P3/P4 instead of letting P2 dominate."""
            total = max(int(total), 0)
            level_count = len(level_lengths)
            if level_count == 1:
                return [total]
            if level_count == 2:
                weights = [0.6, 0.4]
            else:
                weights = [0.5, 0.3, 0.2] + [0.0] * (level_count - 3)
            weight_sum = max(sum(weights), 1e-8)
            raw = [total * weight / weight_sum for weight in weights]
            counts = [int(value) for value in raw]
            remainder = total - sum(counts)
            order = sorted(
                range(level_count),
                key=lambda idx: (raw[idx] - counts[idx], -idx),
                reverse=True,
            )
            for idx in order[:remainder]:
                counts[idx] += 1
            if total >= level_count:
                for idx in range(level_count):
                    if counts[idx] > 0:
                        continue
                    donor = max(range(level_count), key=lambda item: counts[item])
                    if counts[donor] > 1:
                        counts[donor] -= 1
                        counts[idx] += 1
            return counts

        def padded_topk(values: torch.Tensor, count: int) -> torch.Tensor:
            count = max(int(count), 0)
            if count == 0:
                return torch.zeros(0, device=values.device, dtype=torch.long)
            invalid_floor = torch.finfo(values.dtype).min * 0.5
            valid_indices = torch.nonzero(
                torch.isfinite(values) & values.gt(invalid_floor), as_tuple=False
            ).flatten()
            available = int(valid_indices.numel())
            if available <= 0:
                return torch.zeros(count, device=values.device, dtype=torch.long)
            local = torch.topk(
                values[valid_indices], k=min(count, available), dim=0
            ).indices
            selected = valid_indices[local]
            if int(selected.numel()) < count:
                repeat = selected[
                    torch.arange(
                        count - int(selected.numel()), device=values.device
                    ).remainder(max(int(selected.numel()), 1))
                ]
                selected = torch.cat([selected, repeat], dim=0)
            return selected

        def levelwise_topk(values: torch.Tensor, count: int) -> torch.Tensor:
            selected: list[torch.Tensor] = []
            for level_idx, level_count in enumerate(allocate_level_counts(count)):
                if level_count <= 0:
                    continue
                start = level_offsets[level_idx]
                end = start + level_lengths[level_idx]
                selected.append(padded_topk(values[start:end], level_count) + start)
            if not selected:
                return torch.zeros(0, device=values.device, dtype=torch.long)
            return torch.cat(selected, dim=0)

        selected_indices: list[torch.Tensor] = []
        selected_ref_masks: list[torch.Tensor] = []
        for batch_idx in range(b):
            is_ref = int(task_ids[batch_idx].detach().item()) == 1 and ref_count > 0
            if is_ref:
                text_idx = levelwise_topk(all_text_scores[batch_idx], ref_count)
                generic_count = query_count - ref_count
                generic_idx = levelwise_topk(all_scores[batch_idx], generic_count)
                selected_indices.append(torch.cat([text_idx, generic_idx], dim=0))
                selected_ref_masks.append(
                    torch.cat(
                        [
                            torch.ones(ref_count, device=feature_maps[0].device, dtype=torch.bool),
                            torch.zeros(generic_count, device=feature_maps[0].device, dtype=torch.bool),
                        ],
                        dim=0,
                    )
                )
            else:
                selected_indices.append(
                    levelwise_topk(all_scores[batch_idx], query_count)
                )
                selected_ref_masks.append(
                    torch.zeros(query_count, device=feature_maps[0].device, dtype=torch.bool)
                )
        indices = torch.stack(selected_indices, dim=0)
        ref_query_mask = torch.stack(selected_ref_masks, dim=0)
        gather_c = indices[..., None].expand(-1, -1, all_tokens.shape[-1])
        gather_xy = indices[..., None].expand(-1, -1, 2)
        tokens = torch.gather(all_tokens, 1, gather_c)
        centers = torch.gather(all_centers, 1, gather_xy).clamp(0.0, 1.0)
        scales = torch.gather(all_scales, 1, gather_xy).clamp(0.02, 0.95)
        scores = torch.gather(all_scores, 1, indices)
        selected_text_scores = torch.gather(all_text_scores, 1, indices)
        boxes = torch.cat([centers - 0.5 * scales, centers + 0.5 * scales], dim=-1).clamp(
            0.0, 1.0
        )
        tokens = self.proposal_token_proj(tokens.to(dtype=next(self.proposal_token_proj.parameters()).dtype))
        tokens = tokens + self.proposal_source_embed.weight[0].to(
            device=tokens.device, dtype=tokens.dtype
        )
        return boxes, tokens, scores, ref_query_mask, selected_text_scores

    def _merge_external_box_proposals(
        self,
        internal_boxes: torch.Tensor,
        internal_tokens: torch.Tensor,
        internal_scores: torch.Tensor,
        internal_ref_mask: torch.Tensor,
        internal_text_scores: torch.Tensor,
        feature_maps: list[torch.Tensor],
        spatial_shapes: list[torch.Tensor],
        task_ids: torch.Tensor,
        external_boxes: torch.Tensor | None,
        external_mask: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Use Locate boxes as optional query priors, never as visual boundaries."""
        b, query_count = internal_boxes.shape[:2]
        device = internal_boxes.device
        proposal_mask = torch.ones(b, query_count, device=device, dtype=torch.bool)
        source_ids = torch.zeros(b, query_count, device=device, dtype=torch.long)
        if external_boxes is None or external_mask is None:
            return (
                internal_boxes,
                internal_tokens,
                internal_scores,
                internal_ref_mask,
                internal_text_scores,
                source_ids,
            )
        external_boxes = external_boxes.to(device=device, dtype=internal_boxes.dtype).clamp(0.0, 1.0)
        external_mask = external_mask.to(device=device).bool()
        boxes = internal_boxes.clone()
        tokens = internal_tokens.clone()
        scores = internal_scores.clone()
        ref_mask = internal_ref_mask.clone()
        text_scores = internal_text_scores.clone()
        # A single P3 center sample is too fragile for truncated, occluded, or
        # crowded people. Pool a compact 3x3 grid from every active P2/P3/P4
        # level and fuse the scale descriptors into the external query token.
        context_boxes = expand_boxes_xyxy(
            external_boxes, float(self.config.box_condition_scale)
        )
        pooled_levels: list[torch.Tensor] = []
        for level_idx, feature_map in enumerate(feature_maps):
            pooled = self._sample_box_feature_maps(
                feature_map,
                context_boxes,
                3,
                spatial_shapes[level_idx],
            ).mean(dim=(-2, -1))
            pooled_levels.append(pooled)
        pooled_multiscale = torch.cat(pooled_levels, dim=-1)
        external_tokens = self.external_box_token_proj(
            pooled_multiscale.to(
                dtype=next(self.external_box_token_proj.parameters()).dtype
            )
        ) + self.proposal_source_embed.weight[1].to(
            device=device, dtype=internal_tokens.dtype
        )
        for batch_idx in range(b):
            valid = torch.nonzero(external_mask[batch_idx], as_tuple=False).flatten()
            max_external = 1 if int(task_ids[batch_idx].detach().item()) == 1 else query_count
            valid = valid[:max_external]
            count = min(int(valid.numel()), query_count)
            if count <= 0:
                continue
            boxes[batch_idx, :count] = external_boxes[batch_idx, valid[:count]]
            tokens[batch_idx, :count] = external_tokens[batch_idx, valid[:count]]
            scores[batch_idx, :count] = 5.0
            source_ids[batch_idx, :count] = 1
            text_scores[batch_idx, :count] = 0.0
            if int(task_ids[batch_idx].detach().item()) == 1:
                ref_mask[batch_idx, : max(count, min(int(self.config.num_ref_queries), query_count))] = True
        return boxes, tokens, scores, ref_mask, text_scores, source_ids

    def _run_detrpose_pose_branch(
        self,
        *,
        feature_maps: list[torch.Tensor],
        spatial_shapes: list[torch.Tensor],
        local_level_idx: int,
        text_embed: torch.Tensor,
        schema_ids: torch.Tensor,
        task_ids: torch.Tensor,
        proposal_boxes: torch.Tensor,
        proposal_tokens: torch.Tensor,
        proposal_mask: torch.Tensor,
        ref_query_mask: torch.Tensor,
        initial_keypoints: torch.Tensor | None = None,
        initial_keypoint_valid: torch.Tensor | None = None,
        initial_query_mask: torch.Tensor | None = None,
        dn_labels: torch.Tensor | None = None,
        dn_group_ids: torch.Tensor | None = None,
    ) -> dict[str, object]:
        """Decode pose groups directly against whole-image P2/P3/P4 memory."""
        feature_map = feature_maps[min(1, len(feature_maps) - 1)]
        b, num_people = int(proposal_boxes.shape[0]), int(proposal_boxes.shape[1])
        c = int(self.config.hidden_dim)
        proposal_mask = proposal_mask.to(device=feature_map.device).bool()
        # The person box is a spatial prior, not the coordinate system of the
        # final keypoints.  Stop pose/ref gradients at this boundary so slow box
        # regression cannot drag the direct image-coordinate pose head.
        boxes = proposal_boxes.detach().to(
            device=feature_map.device, dtype=feature_map.dtype
        ).clamp(0.0, 1.0)
        box_wh = (boxes[..., 2:] - boxes[..., :2]).clamp(min=0.02)
        box_center = ((boxes[..., :2] + boxes[..., 2:]) * 0.5).clamp(0.0, 1.0)
        box_embed = self.box_query_proj(box_fourier_pe(boxes, c))
        global_visual = _masked_spatial_mean(
            feature_map, spatial_shapes[min(1, len(spatial_shapes) - 1)]
        )
        image_embed = self.human_context_proj(
            global_visual.to(dtype=next(self.human_context_proj.parameters()).dtype)
        )
        text_condition = (
            float(self.config.ref_text_scale)
            * ref_query_mask[..., None].to(dtype=text_embed.dtype)
            * text_embed[:, None, :]
        )
        instance = self.instance_query_norm(
            proposal_tokens.to(dtype=box_embed.dtype)
            + box_embed
            + image_embed[:, None, :]
            + text_condition
        )

        schema_joint_indices = self.schema_joint_indices[schema_ids]
        schema_joint_valid = self.schema_joint_valid[schema_ids]
        active_k = max(int(schema_joint_valid.sum(dim=1).max().item()), 1)
        schema_joint_indices = schema_joint_indices[:, :active_k]
        schema_joint_valid = schema_joint_valid[:, :active_k]
        schema_scatter_map = self._build_schema_scatter_map(
            schema_joint_indices,
            schema_joint_valid,
            union_dim=len(UNION_KEYPOINTS),
            dtype=feature_map.dtype,
        )
        joint_base = self.joint_embed(schema_joint_indices).view(b, 1, active_k, c)
        task = self.task_embed(task_ids).view(b, 1, 1, c)
        joint_type = self.pose_token_type_embed.weight[1].view(1, 1, 1, c)
        joint_text = (
            float(self.config.ref_text_scale)
            * ref_query_mask[:, :, None, None].to(dtype=text_embed.dtype)
            * text_embed[:, None, None, :]
        )
        pose_tokens = self.pose_query_norm(
            instance[:, :, None, :]
            + joint_base
            + task
            + box_embed[:, :, None, :]
            + joint_text
            + joint_type
        )
        instance_type = self.pose_token_type_embed.weight[0].view(1, 1, c)
        instance_token = self.pose_instance_token_norm(instance + instance_type)

        schema_prior_active: torch.Tensor | None = None
        if self.schema_joint_priors is not None:
            schema_prior_all = self.schema_joint_priors.to(
                device=feature_map.device, dtype=feature_map.dtype
            )[schema_ids]
            schema_prior_active = torch.gather(
                schema_prior_all,
                dim=1,
                index=schema_joint_indices[..., None].expand(-1, -1, 2),
            )
        if self.config.pose_coordinate_init == "anatomical_dynamic":
            if schema_prior_active is None or self.reference_offset_head is None:
                raise RuntimeError("Dynamic anatomical references require priors and offset head.")
            expanded_joint = joint_base.expand(-1, num_people, -1, -1)
            expanded_instance = instance[:, :, None, :].expand(-1, -1, active_k, -1)
            offset = self.reference_offset_head(
                torch.cat([expanded_instance, expanded_joint], dim=-1).to(
                    dtype=next(self.reference_offset_head.parameters()).dtype
                )
            ).float()
            relative = torch.sigmoid(
                torch.logit(schema_prior_active.float().clamp(1e-4, 1.0 - 1e-4))[:, None]
                + float(self.config.dynamic_reference_offset_scale) * torch.tanh(offset)
            ).to(dtype=feature_map.dtype)
        elif self.config.pose_coordinate_init == "schema_prior":
            if schema_prior_active is None:
                raise RuntimeError("Schema-prior mode requires priors.")
            relative = schema_prior_active[:, None].expand(-1, num_people, -1, -1)
        elif self.config.pose_coordinate_init == "learned_spread":
            if self.joint_reference_logits is None:
                raise RuntimeError("Learned-spread mode requires joint reference logits.")
            union_reference = self.joint_reference_logits.sigmoid().to(
                device=feature_map.device, dtype=feature_map.dtype
            )
            relative_active = torch.gather(
                union_reference[None].expand(b, -1, -1),
                dim=1,
                index=schema_joint_indices[..., None].expand(-1, -1, 2),
            )
            relative = relative_active[:, None].expand(-1, num_people, -1, -1)
        else:
            relative = boxes.new_full((b, num_people, active_k, 2), 0.5)
        current_reference_xy = (
            boxes[..., None, :2] + relative * box_wh[..., None, :]
        ).clamp(1e-4, 1.0 - 1e-4)

        active_initial_query_mask: torch.Tensor | None = None
        if initial_keypoints is not None:
            active_initial_query_mask = (
                proposal_mask
                if initial_query_mask is None
                else initial_query_mask.to(device=feature_map.device).bool() & proposal_mask
            )
            gather_index = schema_joint_indices[:, None, :, None].expand(
                b, num_people, active_k, 2
            )
            initial_active = torch.gather(
                initial_keypoints.to(device=feature_map.device, dtype=feature_map.dtype),
                dim=2,
                index=gather_index,
            ).clamp(1e-4, 1.0 - 1e-4)
            if initial_keypoint_valid is not None:
                valid_active = torch.gather(
                    initial_keypoint_valid.to(device=feature_map.device).bool(),
                    dim=2,
                    index=schema_joint_indices[:, None, :].expand(
                        b, num_people, active_k
                    ),
                )
                initial_active = torch.where(
                    valid_active[..., None], initial_active, current_reference_xy
                )
            current_reference_xy = torch.where(
                active_initial_query_mask[..., None, None],
                initial_active,
                current_reference_xy,
            )
            if dn_labels is not None and self.keypoint_dn_type_embed is not None:
                dn_type = self.keypoint_dn_type_embed(
                    dn_labels.to(device=feature_map.device, dtype=torch.long).clamp(0, 1)
                )[:, :, None, :]
                pose_tokens = pose_tokens + dn_type * active_initial_query_mask[
                    ..., None, None
                ].to(dtype=dn_type.dtype)

        pose_valid = (
            schema_joint_valid[:, None, :].expand(b, num_people, active_k)
            & proposal_mask[:, :, None]
        )
        group_tokens = torch.cat([instance_token[:, :, None, :], pose_tokens], dim=2)
        group_valid = torch.cat([proposal_mask[:, :, None], pose_valid], dim=2)
        same_role_attention_mask = self._build_group_pose_attention_mask(
            box_mask=proposal_mask,
            dn_query_mask=active_initial_query_mask,
            dn_group_ids=dn_group_ids,
            num_roles=active_k + 1,
        )

        decoder_reference_xy_steps: list[torch.Tensor] = []
        person_center = box_center
        person_scale = (
            box_wh * float(self.config.box_condition_scale)
        ).clamp(0.02, 0.95)
        for decoder_idx, (
            decoder_layer,
            decoder_deformable,
            decoder_coordinate_head,
        ) in enumerate(
            zip(
                self.pose_group_decoder_layers,
                self.pose_decoder_deformable_attention,
                self.pose_decoder_coordinate_heads,
            )
        ):
            role_reference = torch.cat(
                [person_center[:, :, None, :], current_reference_xy.detach()], dim=2
            )
            role_position = point_fourier_pe(role_reference, c)
            group_tokens = decoder_layer(
                group_tokens,
                role_position,
                group_valid,
                same_role_attention_mask,
            )
            pose_tokens = decoder_deformable(
                group_tokens[:, :, 1:],
                current_reference_xy,
                person_scale,
                feature_maps,
                spatial_shapes,
            )
            reference_delta = decoder_coordinate_head(pose_tokens).float()
            current_reference_xy = torch.sigmoid(
                torch.logit(current_reference_xy.float().clamp(1e-4, 1.0 - 1e-4))
                + reference_delta
            ).to(dtype=feature_map.dtype)
            decoder_reference_xy_steps.append(current_reference_xy)
            valid_xy = pose_valid[..., None]
            safe_min = torch.where(
                valid_xy, current_reference_xy, torch.ones_like(current_reference_xy)
            ).amin(dim=2)
            safe_max = torch.where(
                valid_xy, current_reference_xy, torch.zeros_like(current_reference_xy)
            ).amax(dim=2)
            estimated_center = (safe_min + safe_max) * 0.5
            estimated_scale = ((safe_max - safe_min) * 1.20).clamp(0.02, 0.95)
            has_pose = pose_valid.any(dim=2)
            person_center = torch.where(has_pose[..., None], estimated_center, person_center).detach()
            person_scale = torch.where(has_pose[..., None], estimated_scale, person_scale).detach()
            group_tokens = torch.cat([group_tokens[:, :, :1], pose_tokens], dim=2)

        coarse_keypoint_xy = current_reference_xy
        keypoint_xy = current_reference_xy
        refine_keypoint_xy_steps: list[torch.Tensor] = []
        if (
            self.refine_heads is not None
            and self.refine_token_fusers is not None
            and self.refine_patch_weight_heads is not None
            and self.joint_context is not None
            and self.local_proj is not None
        ):
            local_feature_map = self.local_proj(feature_maps[local_level_idx])
            local_shapes = spatial_shapes[local_level_idx]
            context_mask = ~pose_valid.reshape(b * num_people, active_k)
            context_mask = context_mask.masked_fill(
                context_mask.all(dim=1, keepdim=True), False
            )
            # The DETRPose path intentionally keeps a single final P2 enhancement.
            refine_head = self.refine_heads[0]
            token_fuser = self.refine_token_fusers[0]
            patch_weight_head = self.refine_patch_weight_heads[0]
            pose_tokens = self.joint_context(
                pose_tokens.reshape(b * num_people, active_k, c),
                src_key_padding_mask=context_mask,
            ).reshape(b, num_people, active_k, c)
            patch_logits = patch_weight_head(
                pose_tokens.to(dtype=patch_weight_head.weight.dtype)
            )
            local = self._sample_local_patch_features(
                local_feature_map,
                keypoint_xy.detach(),
                person_scale.detach(),
                patch_logits,
                local_shapes,
                patch_size=3,
                radius_scale=0.03,
                min_radius_cells=1.0,
            )
            point_pe = point_fourier_pe(keypoint_xy.detach(), c)
            refine_input = torch.cat([pose_tokens, local, point_pe], dim=-1)
            delta = torch.tanh(refine_head(refine_input)).float()
            scale = self.refine_step_scales[0].sigmoid().float()
            keypoint_xy = torch.sigmoid(
                torch.logit(keypoint_xy.float().clamp(1e-4, 1.0 - 1e-4))
                + delta * scale
            ).to(dtype=feature_map.dtype)
            refine_keypoint_xy_steps.append(keypoint_xy)
            pose_tokens = pose_tokens + token_fuser(refine_input)

        schema_visibility_logits = self.keypoint_visibility_head(pose_tokens)
        keypoint_visibility = schema_visibility_logits.sigmoid()
        valid_f = pose_valid.to(dtype=pose_tokens.dtype).unsqueeze(-1)
        pooled_pose = (pose_tokens * valid_f).sum(dim=2) / valid_f.sum(dim=2).clamp(min=1.0)
        final_instance = self.pose_instance_token_norm(
            group_tokens[:, :, 0] + pooled_pose
        )
        pred_pose_logits, pose_lqe_joint_logits = self._predict_pose_score_logits(
            pose_tokens=pose_tokens,
            instance_tokens=final_instance,
            keypoint_xy=keypoint_xy,
            pose_valid=pose_valid,
            feature_maps=feature_maps,
            spatial_shapes=spatial_shapes,
        )

        valid_xy = pose_valid[..., None]
        pose_min = torch.where(valid_xy, keypoint_xy, torch.ones_like(keypoint_xy)).amin(dim=2)
        pose_max = torch.where(valid_xy, keypoint_xy, torch.zeros_like(keypoint_xy)).amax(dim=2)
        final_boxes = torch.cat(
            [pose_min - 0.05 * (pose_max - pose_min), pose_max + 0.05 * (pose_max - pose_min)],
            dim=-1,
        ).clamp(0.0, 1.0)
        final_boxes = torch.where(
            pose_valid.any(dim=2)[..., None], final_boxes, boxes
        )

        aux_visibility = keypoint_visibility.detach()
        keypoints = self._scatter_schema_keypoints(
            torch.cat([keypoint_xy, keypoint_visibility], dim=-1), schema_scatter_map
        )
        coarse_keypoints = self._scatter_schema_keypoints(
            torch.cat([coarse_keypoint_xy, aux_visibility], dim=-1), schema_scatter_map
        )
        decoder_keypoints = [
            self._scatter_schema_keypoints(
                torch.cat([xy, aux_visibility], dim=-1), schema_scatter_map
            )
            for xy in decoder_reference_xy_steps
        ]
        refine_keypoints = [
            self._scatter_schema_keypoints(
                torch.cat([xy, aux_visibility], dim=-1), schema_scatter_map
            )
            for xy in refine_keypoint_xy_steps
        ]
        return {
            "pose_boxes": final_boxes,
            "instance_emb": final_instance,
            "pred_pose_logits": pred_pose_logits,
            "pose_lqe_joint_logits": self._scatter_schema_keypoints(
                pose_lqe_joint_logits.unsqueeze(-1), schema_scatter_map
            ).squeeze(-1),
            "keypoints": keypoints,
            "keypoint_valid_mask": schema_scatter_map.bool().any(dim=1),
            "pred_keypoint_visibility_logits": self._scatter_schema_keypoints(
                schema_visibility_logits, schema_scatter_map
            ).squeeze(-1),
            "decoder_keypoints": decoder_keypoints,
            "coarse_keypoints": coarse_keypoints,
            "deform_keypoints": coarse_keypoints,
            "refine_keypoints": refine_keypoints,
            "schema_joint_indices": schema_joint_indices,
            "schema_joint_valid": schema_joint_valid,
        }

    def _forward_detrpose(
        self,
        *,
        schema_ids: torch.Tensor,
        task_ids: torch.Tensor,
        feature_maps: list[torch.Tensor],
        spatial_shapes: list[torch.Tensor],
        valid_masks: list[torch.Tensor],
        local_level_idx: int,
        text_embed: torch.Tensor,
        external_boxes: torch.Tensor | None,
        external_box_mask: torch.Tensor | None,
        keypoint_dn_noisy_keypoints: torch.Tensor | None,
        keypoint_dn_mask: torch.Tensor | None,
        keypoint_dn_labels: torch.Tensor | None,
        keypoint_dn_target_keypoints: torch.Tensor | None,
        keypoint_dn_target_valid: torch.Tensor | None,
        keypoint_dn_target_boxes: torch.Tensor | None,
        keypoint_dn_target_areas: torch.Tensor | None,
        keypoint_dn_source_indices: torch.Tensor | None,
        keypoint_dn_group_ids: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        internal = self._build_internal_pose_proposals(
            feature_maps, spatial_shapes, valid_masks, text_embed, task_ids
        )
        (
            proposal_boxes,
            proposal_tokens,
            proposal_scores,
            ref_query_mask,
            proposal_text_scores,
            source_ids,
        ) = self._merge_external_box_proposals(
            internal[0],
            internal[1],
            internal[2],
            internal[3],
            internal[4],
            feature_maps,
            spatial_shapes,
            task_ids,
            external_boxes,
            external_box_mask,
        )
        b, main_count = proposal_boxes.shape[:2]
        has_external_proposals = external_boxes is not None and external_box_mask is not None
        proposal_mask = (
            source_ids.eq(1)
            if has_external_proposals
            else torch.ones(
                b, main_count, device=proposal_boxes.device, dtype=torch.bool
            )
        )
        if self.person_query_embed is not None:
            proposal_tokens = proposal_tokens + self.person_query_embed.weight[
                :main_count
            ].to(device=proposal_tokens.device, dtype=proposal_tokens.dtype)[None]

        # Correct only injected GT/Locate proposals before they initialize the
        # pose decoder. Internal dense proposals keep their original references.
        c = int(self.config.hidden_dim)
        box_embed = self.box_query_proj(box_fourier_pe(proposal_boxes, c))
        context_level_idx = min(1, len(feature_maps) - 1)
        global_visual = _masked_spatial_mean(
            feature_maps[context_level_idx], spatial_shapes[context_level_idx]
        )
        image_embed = self.human_context_proj(
            global_visual.to(dtype=next(self.human_context_proj.parameters()).dtype)
        )
        pre_pose_instance = self.pre_pose_box_norm(
            proposal_tokens.to(dtype=box_embed.dtype)
            + box_embed
            + image_embed[:, None, :]
        )
        pre_pose_deltas = self.pre_pose_box_refine_head(pre_pose_instance).to(
            dtype=proposal_boxes.dtype
        )
        pre_pose_refined_boxes = refine_external_boxes_xyxy(
            proposal_boxes, pre_pose_deltas
        )
        external_source_mask = source_ids.eq(1)
        pose_proposal_boxes = torch.where(
            external_source_mask[..., None],
            pre_pose_refined_boxes,
            proposal_boxes,
        )
        pre_pose_refinement_fallback_mask = torch.zeros_like(external_source_mask)
        if not self.training:
            safe_pose_boxes, pre_fallback = apply_refhuman_box_refinement_safety(
                pose_proposal_boxes,
                proposal_boxes,
                external_source_mask,
                task_ids,
            )
            pose_proposal_boxes = torch.where(
                external_source_mask[..., None], safe_pose_boxes, pose_proposal_boxes
            )
            pre_pose_refinement_fallback_mask = pre_fallback

        # Keep the box-to-pose boundary explicit: box regression learns from
        # box supervision, while pose losses cannot drag a trusted external box
        # away from its person. Internal proposals retain their historical path.
        pose_condition_boxes = torch.where(
            external_source_mask[..., None],
            pose_proposal_boxes.detach(),
            proposal_boxes,
        )

        pose_dn_count = 0
        combined_boxes = pose_condition_boxes
        combined_tokens = proposal_tokens
        combined_mask = proposal_mask
        combined_ref_mask = ref_query_mask
        combined_initial_keypoints: torch.Tensor | None = None
        combined_initial_valid: torch.Tensor | None = None
        combined_initial_query_mask: torch.Tensor | None = None
        combined_labels: torch.Tensor | None = None
        combined_groups: torch.Tensor | None = None
        pose_dn_mask: torch.Tensor | None = None
        if (
            self.training
            and self.config.enable_keypoint_denoising
            and keypoint_dn_noisy_keypoints is not None
            and keypoint_dn_mask is not None
            and keypoint_dn_labels is not None
            and keypoint_dn_target_valid is not None
            and int(keypoint_dn_noisy_keypoints.shape[1]) > 0
        ):
            pose_dn_count = int(keypoint_dn_noisy_keypoints.shape[1])
            pose_dn_mask = keypoint_dn_mask.to(device=proposal_boxes.device).bool()
            if keypoint_dn_target_boxes is not None:
                dn_boxes = keypoint_dn_target_boxes.to(
                    device=proposal_boxes.device, dtype=proposal_boxes.dtype
                ).clamp(0.0, 1.0)
            else:
                noisy_xy = keypoint_dn_noisy_keypoints[..., :2].to(
                    device=proposal_boxes.device, dtype=proposal_boxes.dtype
                )
                dn_min = noisy_xy.amin(dim=2)
                dn_max = noisy_xy.amax(dim=2)
                dn_boxes = torch.cat([dn_min, dn_max], dim=-1).clamp(0.0, 1.0)
            dn_context_boxes = expand_boxes_xyxy(
                dn_boxes, float(self.config.box_condition_scale)
            )
            dn_pooled_levels: list[torch.Tensor] = []
            for level_idx, feature_map in enumerate(feature_maps):
                dn_pooled = self._sample_box_feature_maps(
                    feature_map,
                    dn_context_boxes,
                    3,
                    spatial_shapes[level_idx],
                ).mean(dim=(-2, -1))
                dn_pooled_levels.append(dn_pooled)
            dn_multiscale = torch.cat(dn_pooled_levels, dim=-1)
            dn_tokens = self.external_box_token_proj(
                dn_multiscale.to(
                    dtype=next(self.external_box_token_proj.parameters()).dtype
                )
            ) + self.proposal_source_embed.weight[2].to(
                device=proposal_boxes.device, dtype=proposal_tokens.dtype
            )
            combined_boxes = torch.cat([pose_condition_boxes, dn_boxes], dim=1)
            combined_tokens = torch.cat([proposal_tokens, dn_tokens], dim=1)
            combined_mask = torch.cat([proposal_mask, pose_dn_mask], dim=1)
            combined_ref_mask = torch.cat(
                [ref_query_mask, torch.zeros_like(pose_dn_mask)], dim=1
            )
            union_count = int(keypoint_dn_noisy_keypoints.shape[2])
            combined_initial_keypoints = torch.cat(
                [
                    keypoint_dn_noisy_keypoints.new_zeros(
                        b, main_count, union_count, 2
                    ),
                    keypoint_dn_noisy_keypoints,
                ],
                dim=1,
            )
            combined_initial_valid = torch.cat(
                [
                    torch.zeros(
                        b, main_count, union_count,
                        device=proposal_boxes.device, dtype=torch.bool
                    ),
                    keypoint_dn_target_valid.to(device=proposal_boxes.device).bool(),
                ],
                dim=1,
            )
            combined_initial_query_mask = torch.cat(
                [torch.zeros_like(proposal_mask), pose_dn_mask], dim=1
            )
            combined_labels = torch.cat(
                [
                    torch.zeros(b, main_count, device=proposal_boxes.device),
                    keypoint_dn_labels.to(device=proposal_boxes.device),
                ],
                dim=1,
            )
            dn_groups = (
                keypoint_dn_group_ids.to(device=proposal_boxes.device, dtype=torch.long)
                if keypoint_dn_group_ids is not None
                else torch.arange(pose_dn_count, device=proposal_boxes.device)[None].expand(b, -1)
            )
            combined_groups = torch.cat(
                [
                    torch.full(
                        (b, main_count), -1,
                        device=proposal_boxes.device, dtype=torch.long
                    ),
                    dn_groups,
                ],
                dim=1,
            )

        decoded = self._run_detrpose_pose_branch(
            feature_maps=feature_maps,
            spatial_shapes=spatial_shapes,
            local_level_idx=local_level_idx,
            text_embed=text_embed,
            schema_ids=schema_ids,
            task_ids=task_ids,
            proposal_boxes=combined_boxes,
            proposal_tokens=combined_tokens,
            proposal_mask=combined_mask,
            ref_query_mask=combined_ref_mask,
            initial_keypoints=combined_initial_keypoints,
            initial_keypoint_valid=combined_initial_valid,
            initial_query_mask=combined_initial_query_mask,
            dn_labels=combined_labels,
            dn_group_ids=combined_groups,
        )

        def main_slice(value: torch.Tensor) -> torch.Tensor:
            return value[:, :main_count]

        pred_pose_logits = main_slice(decoded["pred_pose_logits"])
        proposal_objectness_logits = proposal_scores
        instance = main_slice(decoded["instance_emb"])
        external_box_deltas = self.external_box_refine_head(instance).to(
            dtype=proposal_boxes.dtype
        )
        externally_refined_boxes = refine_external_boxes_xyxy(
            pose_proposal_boxes.detach(), external_box_deltas
        )
        pred_boxes = torch.where(
            external_source_mask[..., None],
            externally_refined_boxes,
            proposal_boxes,
        )
        refinement_fallback_mask = torch.zeros_like(external_source_mask)
        if not self.training:
            safe_boxes, ref_fallback = apply_refhuman_box_refinement_safety(
                pred_boxes,
                proposal_boxes,
                external_source_mask,
                task_ids,
            )
            pred_boxes = torch.where(
                external_source_mask[..., None], safe_boxes, pred_boxes
            )
            refinement_fallback_mask = ref_fallback
        pred_box_logits = self._predict_box_score_logits(
            proposal_tokens=proposal_tokens,
            instance_tokens=instance,
            pred_boxes=pred_boxes,
            feature_maps=feature_maps,
            spatial_shapes=spatial_shapes,
            local_level_idx=local_level_idx,
        )
        ref_candidate = self.ref_candidate_proj(instance)
        ref_text = self.ref_text_match_proj(text_embed).unsqueeze(1).expand_as(ref_candidate)
        ref_logits = self.ref_match_head(
            torch.cat([ref_candidate, ref_text, ref_candidate * ref_text], dim=-1)
        ).squeeze(-1)
        # Keep text-conditioned proposal selection differentiable. The selected
        # dense visual/text similarity contributes only to the reserved RefHuman
        # candidates; Locate-box candidates use the decoded instance score alone.
        ref_logits = ref_logits + proposal_text_scores.to(dtype=ref_logits.dtype) * (
            ref_query_mask & source_ids.eq(0)
        ).to(dtype=ref_logits.dtype)
        ref_logits = torch.where(
            task_ids.eq(1).to(device=ref_logits.device)[:, None],
            ref_logits,
            torch.full_like(ref_logits, -10.0),
        )
        outputs: dict[str, torch.Tensor] = {
            "pose_set_prediction": torch.tensor(True, device=proposal_boxes.device),
            "pred_logits": pred_box_logits.unsqueeze(-1),
            # Canonical box output is always the independently regressed human
            # proposal.  It is never reconstructed from the final keypoints.
            "pred_boxes": pred_boxes,
            "pred_keypoints": main_slice(decoded["keypoints"]),
            "pred_box_logits": pred_box_logits,
            "pred_pose_logits": pred_pose_logits,
            "pred_keypoint_visibility_logits": main_slice(
                decoded["pred_keypoint_visibility_logits"]
            ),
            # Proposal objectness is an auxiliary matching/top-Q signal only.
            "proposal_objectness_logits_aux": proposal_objectness_logits,
            "box_objectness_logits": proposal_objectness_logits,
            # Compatibility aliases retain tensor availability but now carry
            # the direct AP logits rather than factors that consumers multiply.
            "person_class_logits": pred_box_logits,
            "person_logits": pred_box_logits,
            "box_quality_logits": pred_box_logits,
            "pose_quality_logits": pred_pose_logits,
            "decoded_pose_quality_logits": pred_pose_logits,
            "aux_box_outputs": [],
            "pose_score_head_available": True,
            "person_confidence_head_available": True,
            "person_confidence_rescue": True,
            "input_boxes": proposal_boxes,
            "pre_pose_boxes": pose_proposal_boxes,
            "boxes": pred_boxes,
            # Compatibility alias only.  Consumers must use ``pred_boxes`` for
            # detection and this envelope only for diagnostics.
            "pose_boxes": main_slice(decoded["pose_boxes"]),
            "debug_keypoint_envelope": main_slice(decoded["pose_boxes"]),
            "box_mask": proposal_mask,
            "proposal_source_ids": source_ids,
            "ref_query_mask": ref_query_mask,
            "pre_pose_box_refinement_fallback_mask": pre_pose_refinement_fallback_mask,
            "ref_box_refinement_fallback_mask": refinement_fallback_mask,
            "pre_pose_box_refinement_deltas": pre_pose_deltas,
            "external_box_refinement_deltas": external_box_deltas,
            "keypoints": main_slice(decoded["keypoints"]),
            "keypoint_valid_mask": decoded["keypoint_valid_mask"],
            "keypoint_confidence_logits": main_slice(
                decoded["pred_keypoint_visibility_logits"]
            ),
            "pose_lqe_joint_logits": main_slice(decoded["pose_lqe_joint_logits"]),
            "decoder_keypoints": [main_slice(value) for value in decoded["decoder_keypoints"]],
            "coarse_keypoints": main_slice(decoded["coarse_keypoints"]),
            "deform_keypoints": main_slice(decoded["deform_keypoints"]),
            "refine_keypoints": [main_slice(value) for value in decoded["refine_keypoints"]],
            "ref_logits": ref_logits,
            "ref_candidate_embed": ref_candidate,
            "ref_text_embed": ref_text[:, 0],
            "instance_emb": instance,
            "schema_joint_indices": decoded["schema_joint_indices"],
            "schema_joint_valid": decoded["schema_joint_valid"],
        }
        if self.training and self.keypoint_dn_type_embed is not None:
            outputs["keypoint_dn_graph_anchor"] = self.keypoint_dn_type_embed.weight.sum() * 0.0
        if pose_dn_count > 0 and pose_dn_mask is not None:
            start, end = main_count, main_count + pose_dn_count
            outputs.update(
                {
                    "keypoint_dn_keypoints": decoded["keypoints"][:, start:end],
                    "keypoint_dn_decoder_keypoints": [
                        value[:, start:end] for value in decoded["decoder_keypoints"]
                    ],
                    "keypoint_dn_coarse_keypoints": decoded["coarse_keypoints"][:, start:end],
                    "keypoint_dn_deform_keypoints": decoded["deform_keypoints"][:, start:end],
                    "keypoint_dn_refine_keypoints": [
                        value[:, start:end] for value in decoded["refine_keypoints"]
                    ],
                    "keypoint_dn_confidence_logits": decoded[
                        "pred_keypoint_visibility_logits"
                    ][:, start:end],
                    "keypoint_dn_pose_quality_logits": decoded["pred_pose_logits"][:, start:end],
                    "keypoint_dn_mask": pose_dn_mask,
                    "keypoint_dn_labels": keypoint_dn_labels,
                    "keypoint_dn_target_keypoints": keypoint_dn_target_keypoints,
                    "keypoint_dn_target_valid": keypoint_dn_target_valid,
                    "keypoint_dn_target_boxes": keypoint_dn_target_boxes,
                    "keypoint_dn_target_areas": keypoint_dn_target_areas,
                    "keypoint_dn_source_indices": keypoint_dn_source_indices,
                    "keypoint_dn_group_ids": keypoint_dn_group_ids,
                }
            )
        return outputs

    def forward(
        self,
        schema_ids: torch.Tensor,
        task_ids: torch.Tensor,
        images: torch.Tensor | None = None,
        external_feature_map: torch.Tensor | SpatialFeatureBatch | MultiScaleSpatialFeatureBatch | None = None,
        external_text_embed: torch.Tensor | None = None,
        cached_text_embed: torch.Tensor | None = None,
        cached_text_mask: torch.Tensor | None = None,
        target_boxes: torch.Tensor | None = None,
        target_box_mask: torch.Tensor | None = None,
        dn_boxes: torch.Tensor | None = None,
        dn_box_mask: torch.Tensor | None = None,
        dn_labels: torch.Tensor | None = None,
        dn_target_boxes: torch.Tensor | None = None,
        dn_group_ids: torch.Tensor | None = None,
        dn_source_indices: torch.Tensor | None = None,
        keypoint_dn_noisy_keypoints: torch.Tensor | None = None,
        keypoint_dn_mask: torch.Tensor | None = None,
        keypoint_dn_labels: torch.Tensor | None = None,
        keypoint_dn_target_keypoints: torch.Tensor | None = None,
        keypoint_dn_target_valid: torch.Tensor | None = None,
        keypoint_dn_target_boxes: torch.Tensor | None = None,
        keypoint_dn_target_areas: torch.Tensor | None = None,
        keypoint_dn_source_indices: torch.Tensor | None = None,
        keypoint_dn_group_ids: torch.Tensor | None = None,
        keypoint_dn_box_query_indices: torch.Tensor | None = None,
        pose_condition_box_mode: str = "refined_detached",
    ) -> dict[str, torch.Tensor]:
        if external_feature_map is None:
            raise ValueError("QwenPoseModel requires backbone external_feature_map.")
        batch_size = int(external_feature_map.shape[0])
        feature_device = external_feature_map.device
        if external_text_embed is None:
            external_text_embed = torch.zeros(
                batch_size,
                int(self.config.external_dim),
                device=feature_device,
                dtype=external_feature_map.dtype,
            )
        if cached_text_embed is not None:
            cached = cached_text_embed.to(
                device=feature_device, dtype=external_text_embed.dtype
            )
            if cached.shape != external_text_embed.shape:
                raise ValueError(
                    "cached text embedding shape must match backbone text embedding: "
                    f"cached={tuple(cached.shape)} backbone={tuple(external_text_embed.shape)}."
                )
            mask = (
                torch.ones(batch_size, device=feature_device, dtype=torch.bool)
                if cached_text_mask is None
                else cached_text_mask.to(device=feature_device).bool().view(batch_size)
            )
            external_text_embed = torch.where(
                mask[:, None], cached, external_text_embed
            )
        if pose_condition_box_mode not in {"refined_detached", "input"}:
            raise ValueError(
                "pose_condition_box_mode must be 'refined_detached' or 'input', "
                f"got {pose_condition_box_mode!r}."
            )
        if bool(self.config.use_detrpose_architecture):
            (
                feature_maps,
                spatial_shapes,
                spatial_valid_masks,
                _,
                local_level_idx,
            ) = self.build_locate_pose_features(external_feature_map)
            text_dtype = next(self.external_text_proj.parameters()).dtype
            text_embed = self.external_text_proj(
                external_text_embed.to(dtype=text_dtype)
            )
            return self._forward_detrpose(
                schema_ids=schema_ids,
                task_ids=task_ids,
                feature_maps=feature_maps,
                spatial_shapes=spatial_shapes,
                valid_masks=spatial_valid_masks,
                local_level_idx=local_level_idx,
                text_embed=text_embed,
                external_boxes=target_boxes,
                external_box_mask=target_box_mask,
                keypoint_dn_noisy_keypoints=keypoint_dn_noisy_keypoints,
                keypoint_dn_mask=keypoint_dn_mask,
                keypoint_dn_labels=keypoint_dn_labels,
                keypoint_dn_target_keypoints=keypoint_dn_target_keypoints,
                keypoint_dn_target_valid=keypoint_dn_target_valid,
                keypoint_dn_target_boxes=keypoint_dn_target_boxes,
                keypoint_dn_target_areas=keypoint_dn_target_areas,
                keypoint_dn_source_indices=keypoint_dn_source_indices,
                keypoint_dn_group_ids=keypoint_dn_group_ids,
            )

    def initialize_person_confidence_from_visibility(self) -> None:
        raise RuntimeError(
            "Legacy visibility-head rescue is incompatible with the Locate-only architecture. "
            "Train a new Stage1 checkpoint."
        )

    def _build_group_pose_attention_mask(
        self,
        *,
        box_mask: torch.Tensor,
        dn_query_mask: torch.Tensor | None,
        dn_group_ids: torch.Tensor | None,
        num_roles: int,
    ) -> torch.Tensor | None:
        """Build the asymmetric main/DN mask for same-role attention.

        Main queries cannot read DN queries. DN queries may read main queries,
        while DN queries from different groups remain mutually isolated.
        """
        b, num_people = box_mask.shape
        if dn_query_mask is None and dn_group_ids is None:
            return None
        if dn_query_mask is None:
            assert dn_group_ids is not None
            dn_query_mask = dn_group_ids.to(device=box_mask.device).ge(0)
        dn_queries = dn_query_mask.to(device=box_mask.device).bool() & box_mask
        if dn_queries.shape != (b, num_people):
            raise ValueError(
                "keypoint DN query mask must have shape "
                f"{(b, num_people)}, got {tuple(dn_queries.shape)}."
            )
        if dn_group_ids is None:
            group_ids = torch.full(
                (b, num_people),
                -1,
                device=box_mask.device,
                dtype=torch.long,
            )
        else:
            group_ids = dn_group_ids.to(device=box_mask.device, dtype=torch.long)
            if group_ids.shape != (b, num_people):
                raise ValueError(
                    "keypoint DN group IDs must have shape "
                    f"{(b, num_people)}, got {tuple(group_ids.shape)}."
                )

        main_queries = box_mask & ~dn_queries
        main_to_dn = main_queries[:, :, None] & dn_queries[:, None, :]
        valid_dn_groups = dn_queries & group_ids.ge(0)
        cross_group = (
            valid_dn_groups[:, :, None]
            & valid_dn_groups[:, None, :]
            & group_ids[:, :, None].ne(group_ids[:, None, :])
        )
        per_sample = main_to_dn | cross_group
        diagonal = torch.eye(
            num_people, device=box_mask.device, dtype=torch.bool
        )[None]
        per_sample = per_sample & ~diagonal
        per_role = per_sample[:, None].expand(
            b, int(num_roles), num_people, num_people
        ).reshape(b * int(num_roles), num_people, num_people)
        return per_role.repeat_interleave(int(self.config.decoder_heads), dim=0)

    @staticmethod
    def _build_schema_scatter_map(
        schema_joint_indices: torch.Tensor,
        schema_joint_valid: torch.Tensor,
        union_dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        scatter_map = F.one_hot(schema_joint_indices, num_classes=union_dim).to(dtype=dtype)
        return scatter_map * schema_joint_valid[..., None].to(dtype=dtype)

    @staticmethod
    def _scatter_schema_keypoints(
        values: torch.Tensor,
        schema_scatter_map: torch.Tensor,
    ) -> torch.Tensor:
        return torch.einsum("bqkd,bku->bqud", values, schema_scatter_map)

    @staticmethod
    def _sample_local_features(
        feature_map: torch.Tensor,
        keypoint_xy: torch.Tensor,
        spatial_shapes: torch.Tensor,
    ) -> torch.Tensor:
        b, c, feature_h, feature_w = feature_map.shape
        q, u = keypoint_xy.shape[1], keypoint_xy.shape[2]
        grid = _padded_grid_from_normalized_points(
            keypoint_xy.to(device=feature_map.device, dtype=feature_map.dtype),
            spatial_shapes,
            feature_h,
            feature_w,
        )
        grid = grid.view(b, q * u, 1, 2)
        sampled = F.grid_sample(feature_map, grid, align_corners=False)
        sampled = sampled.squeeze(-1).transpose(1, 2).view(b, q, u, c)
        return sampled

    @staticmethod
    def _sample_local_patch_features(
        feature_map: torch.Tensor,
        keypoint_xy: torch.Tensor,
        box_wh: torch.Tensor,
        patch_logits: torch.Tensor | None,
        spatial_shapes: torch.Tensor,
        *,
        patch_size: int = 3,
        radius_scale: float = 0.08,
        min_radius_cells: float = 2.0,
    ) -> torch.Tensor:
        b, c, _, _ = feature_map.shape
        q, u = keypoint_xy.shape[1], keypoint_xy.shape[2]
        patch_size = max(int(patch_size), 1)
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, patch_size, device=feature_map.device, dtype=feature_map.dtype),
            torch.linspace(-1.0, 1.0, patch_size, device=feature_map.device, dtype=feature_map.dtype),
            indexing="ij",
        )
        offsets = torch.stack([x, y], dim=-1).view(1, 1, 1, patch_size * patch_size, 2)
        radius = box_wh.to(device=feature_map.device, dtype=feature_map.dtype) * float(radius_scale)
        feature_h, feature_w = feature_map.shape[-2:]
        minimum = torch.stack(
            [
                float(min_radius_cells) / spatial_shapes[:, 1].clamp(min=1).to(dtype=radius.dtype),
                float(min_radius_cells) / spatial_shapes[:, 0].clamp(min=1).to(dtype=radius.dtype),
            ],
            dim=-1,
        ).to(device=radius.device)[:, None, :]
        radius = torch.maximum(radius, minimum).view(b, q, 1, 1, 2)
        points = keypoint_xy.to(device=feature_map.device, dtype=feature_map.dtype).unsqueeze(3) + offsets * radius
        points = points.clamp(0.0, 1.0)
        grid = _padded_grid_from_normalized_points(
            points,
            spatial_shapes,
            feature_h,
            feature_w,
        ).view(b, q * u * patch_size * patch_size, 1, 2)
        sampled = F.grid_sample(feature_map, grid, align_corners=False)
        sampled = sampled.squeeze(-1).transpose(1, 2).view(b, q, u, patch_size * patch_size, c)
        if patch_logits is None or patch_logits.shape[-1] != patch_size * patch_size:
            weights = sampled.new_full((b, q, u, patch_size * patch_size), 1.0 / float(patch_size * patch_size))
        else:
            weights = patch_logits.to(device=feature_map.device).float().softmax(dim=-1).to(dtype=sampled.dtype)
        return (sampled * weights.unsqueeze(-1)).sum(dim=3)

    @staticmethod
    def _sample_box_feature_maps(
        feature_map: torch.Tensor,
        boxes: torch.Tensor,
        roi_size: int,
        spatial_shapes: torch.Tensor,
    ) -> torch.Tensor:
        b, c, _, _ = feature_map.shape
        num_boxes = boxes.shape[1]
        if num_boxes == 0:
            return feature_map.new_zeros(b, 0, c, roi_size, roi_size)
        if torchvision_roi_align is not None:
            # Refined boxes carry gradients. Clone before numerical safety edits
            # so ROIAlign preparation never mutates an autograd view in-place.
            flat_boxes = boxes.to(dtype=feature_map.dtype).reshape(b * num_boxes, 4).clone()
            per_sample_scales = torch.stack(
                [
                    spatial_shapes[:, 1],
                    spatial_shapes[:, 0],
                    spatial_shapes[:, 1],
                    spatial_shapes[:, 0],
                ],
                dim=-1,
            ).to(device=feature_map.device, dtype=flat_boxes.dtype)
            flat_boxes = flat_boxes * per_sample_scales[:, None, :].expand(
                -1, num_boxes, -1
            ).reshape(b * num_boxes, 4)
            # Keep zero-padded/degenerate boxes numerically safe without any
            # in-place slice writes, because refined boxes carry gradients.
            xy1 = flat_boxes[:, :2]
            xy2 = torch.maximum(flat_boxes[:, 2:], xy1 + 1e-4)
            flat_boxes = torch.cat([xy1, xy2], dim=-1)
            batch_indices = (
                torch.arange(b, device=feature_map.device)
                .repeat_interleave(num_boxes)
                .to(dtype=feature_map.dtype)
                .unsqueeze(1)
            )
            rois = torch.cat([batch_indices, flat_boxes], dim=1)
            # Some torchvision builds do not provide a CUDA roi_align kernel for
            # bfloat16. Run roi_align in float32 and cast back so the rest of the
            # pose head can stay in the original mixed-precision dtype.
            roi_feature_map = feature_map
            roi_rois = rois
            if feature_map.dtype == torch.bfloat16:
                roi_feature_map = feature_map.float()
                roi_rois = rois.float()
            pooled = torchvision_roi_align(
                roi_feature_map,
                roi_rois,
                output_size=(roi_size, roi_size),
                spatial_scale=1.0,
                sampling_ratio=-1,
                aligned=True,
            )
            return pooled.to(dtype=feature_map.dtype).view(b, num_boxes, c, roi_size, roi_size)
        y, x = torch.meshgrid(
            torch.linspace(0.0, 1.0, roi_size, device=feature_map.device, dtype=feature_map.dtype),
            torch.linspace(0.0, 1.0, roi_size, device=feature_map.device, dtype=feature_map.dtype),
            indexing="ij",
        )
        base = torch.stack([x, y], dim=-1).view(1, 1, roi_size, roi_size, 2)
        sample_boxes = boxes.to(dtype=feature_map.dtype)
        xy1 = sample_boxes[..., :2].unsqueeze(2).unsqueeze(2)
        wh = (sample_boxes[..., 2:] - sample_boxes[..., :2]).clamp(min=1e-4).unsqueeze(2).unsqueeze(2)
        normalized_points = (xy1 + base * wh).clamp(0.0, 1.0)
        grid = _padded_grid_from_normalized_points(
            normalized_points,
            spatial_shapes,
            int(feature_map.shape[-2]),
            int(feature_map.shape[-1]),
        )
        flat_grid = grid.view(b * num_boxes, roi_size, roi_size, 2)
        batch_indices = torch.arange(b, device=feature_map.device, dtype=torch.long).repeat_interleave(num_boxes)
        flat_features = feature_map.index_select(0, batch_indices)
        sampled = F.grid_sample(flat_features, flat_grid, align_corners=False)
        return sampled.view(b, num_boxes, c, roi_size, roi_size)

    def _predict_pose_score_logits(
        self,
        *,
        pose_tokens: torch.Tensor,
        instance_tokens: torch.Tensor,
        keypoint_xy: torch.Tensor,
        pose_valid: torch.Tensor,
        feature_maps: list[torch.Tensor],
        spatial_shapes: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict a direct OKS-aware pose logit with local quality evidence."""
        b, q, k, _ = pose_tokens.shape
        sample_points = keypoint_xy.detach().reshape(b, q * k, 2)
        sampled_levels = []
        for feature_map, shapes in zip(feature_maps, spatial_shapes):
            sampled = self._sample_feature_at_points(
                feature_map, sample_points, shapes
            ).reshape(b, q, k, -1)
            sampled_levels.append(sampled)
        local_features = torch.cat(sampled_levels, dim=-1)
        local_dtype = next(self.pose_lqe_feature_proj.parameters()).dtype
        local_tokens = self.pose_lqe_feature_proj(
            local_features.to(dtype=local_dtype)
        ).to(dtype=pose_tokens.dtype)
        joint_dtype = next(self.pose_lqe_joint_head.parameters()).dtype
        joint_lqe_logits = self.pose_lqe_joint_head(
            torch.cat([pose_tokens, local_tokens], dim=-1).to(dtype=joint_dtype)
        ).squeeze(-1).to(dtype=pose_tokens.dtype)

        valid = pose_valid.to(device=joint_lqe_logits.device).bool()
        joint_scores = joint_lqe_logits.float().sigmoid()
        valid_f = valid.float()
        mean_score = (joint_scores * valid_f).sum(dim=-1) / valid_f.sum(
            dim=-1
        ).clamp(min=1.0)
        topk_count = max(k // 2, 1)
        top_values = torch.where(
            valid, joint_scores, torch.full_like(joint_scores, -1.0)
        ).topk(topk_count, dim=-1).values
        top_valid = top_values.ge(0.0)
        top_mean = torch.where(top_valid, top_values, torch.zeros_like(top_values)).sum(
            dim=-1
        ) / top_valid.float().sum(dim=-1).clamp(min=1.0)
        max_score = torch.where(
            valid, joint_scores, torch.zeros_like(joint_scores)
        ).amax(dim=-1)
        statistics = torch.stack([mean_score, top_mean, max_score], dim=-1).to(
            dtype=instance_tokens.dtype
        )

        score_dtype = next(self.pose_score_base_head.parameters()).dtype
        base_logits = self.pose_score_base_head(
            instance_tokens.to(dtype=score_dtype)
        ).squeeze(-1)
        residual_logits = self.pose_lqe_residual_head(
            torch.cat(
                [instance_tokens.to(dtype=score_dtype), statistics.to(dtype=score_dtype)],
                dim=-1,
            )
        ).squeeze(-1)
        return (
            (base_logits + residual_logits).to(dtype=pose_tokens.dtype),
            joint_lqe_logits,
        )

    def _predict_box_score_logits(
        self,
        *,
        proposal_tokens: torch.Tensor,
        instance_tokens: torch.Tensor,
        pred_boxes: torch.Tensor,
        feature_maps: list[torch.Tensor],
        spatial_shapes: list[torch.Tensor],
        local_level_idx: int,
    ) -> torch.Tensor:
        """Predict the direct person/bbox AP logit from token and ROI evidence."""
        local_roi = self._sample_box_feature_maps(
            feature_maps[local_level_idx],
            pred_boxes.detach(),
            3,
            spatial_shapes[local_level_idx],
        ).mean(dim=(-2, -1))
        local_dtype = next(self.box_lqe_feature_proj.parameters()).dtype
        local_token = self.box_lqe_feature_proj(
            local_roi.to(dtype=local_dtype)
        ).to(dtype=proposal_tokens.dtype)
        score_dtype = next(self.box_score_head.parameters()).dtype
        score_input = torch.cat(
            [proposal_tokens, instance_tokens, local_token], dim=-1
        ).to(dtype=score_dtype)
        return self.box_score_head(score_input).squeeze(-1).to(
            dtype=pred_boxes.dtype
        )

    @staticmethod
    def _zero_init_last_linear(module: nn.Module) -> None:
        for child in reversed(list(module.modules())):
            if isinstance(child, nn.Linear):
                nn.init.zeros_(child.weight)
                if child.bias is not None:
                    nn.init.zeros_(child.bias)
                return

    @staticmethod
    def _zero_init_last_conv(module: nn.Module) -> None:
        for child in reversed(list(module.modules())):
            if isinstance(child, nn.Conv2d):
                nn.init.zeros_(child.weight)
                if child.bias is not None:
                    nn.init.zeros_(child.bias)
                return


def apply_keypoint_decode_mode(
    outputs: dict[str, torch.Tensor],
    mode: str = "regression",
    fusion_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Return direct regression coordinates; distributional decoding was removed."""
    del fusion_weight
    mode = str(mode).strip().lower()
    if mode != "regression":
        raise ValueError(
            f"Unsupported keypoint decode mode: {mode!r}; only regression is available."
        )
    return outputs


def topk_keypoint_confidence(
    keypoints: torch.Tensor,
    keypoint_valid_mask: torch.Tensor,
    fraction: float = 0.5,
) -> torch.Tensor:
    """Score each pose by the mean confidence of its strongest half joints."""
    if keypoints.ndim != 4 or keypoints.shape[-1] < 3:
        raise ValueError("keypoints must have shape [B,Q,U,3+].")
    if keypoint_valid_mask.ndim == 1:
        keypoint_valid_mask = keypoint_valid_mask.view(1, -1).expand(keypoints.shape[0], -1)
    if keypoint_valid_mask.shape != (keypoints.shape[0], keypoints.shape[2]):
        raise ValueError(
            "keypoint_valid_mask must have shape [B,U] matching keypoints; "
            f"got {tuple(keypoint_valid_mask.shape)} for {tuple(keypoints.shape)}."
        )
    fraction = min(max(float(fraction), 1e-6), 1.0)
    scores = keypoints[..., 2].float().clamp(0.0, 1.0)
    output = scores.new_zeros(scores.shape[:2])
    for batch_idx in range(scores.shape[0]):
        valid = keypoint_valid_mask[batch_idx].to(device=scores.device).bool()
        count = int(valid.sum().item())
        if count <= 0:
            continue
        top_count = max(int((count * fraction) + 0.999999), 1)
        output[batch_idx] = scores[batch_idx, :, valid].topk(
            min(top_count, count),
            dim=-1,
        ).values.mean(dim=-1)
    return output


def count_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
