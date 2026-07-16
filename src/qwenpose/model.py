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


class SinePositionEncoding(nn.Module):
    """Standard DETR-style normalized 2D sine/cosine position encoding."""

    def __init__(self, hidden_dim: int, temperature: float = 10000.0) -> None:
        super().__init__()
        if hidden_dim % 4 != 0:
            raise ValueError("hidden_dim must be divisible by 4 for 2D sine PE.")
        self.hidden_dim = int(hidden_dim)
        self.temperature = float(temperature)

    def forward(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        if height <= 0 or width <= 0:
            raise ValueError(f"Position-encoding shape must be positive, got {(height, width)}.")
        y, x = torch.meshgrid(
            (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / float(height),
            (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / float(width),
            indexing="ij",
        )
        num_pos_feats = self.hidden_dim // 2
        dim_t = torch.arange(num_pos_feats, device=device, dtype=torch.float32)
        dim_t = self.temperature ** (
            2.0 * torch.div(dim_t, 2, rounding_mode="floor") / float(num_pos_feats)
        )
        scale = 2.0 * math.pi
        pos_x = x[..., None] * scale / dim_t
        pos_y = y[..., None] * scale / dim_t
        pos_x = torch.stack(
            [pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()], dim=-1
        ).flatten(-2)
        pos_y = torch.stack(
            [pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()], dim=-1
        ).flatten(-2)
        return torch.cat([pos_y, pos_x], dim=-1).view(height * width, self.hidden_dim)


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


class HumanBoxDeformableAttention(nn.Module):
    """Box-relative sparse cross-attention over the Locate spatial feature."""

    def __init__(
        self,
        hidden_dim: int,
        feature_dim: int,
        num_scales: int = 3,
        num_points: int = 4,
        offset_scale: float = 0.5,
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
        self.context_proj = MLP(hidden_dim * 2, hidden_dim, hidden_dim, depth=2)
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
        boxes: torch.Tensor,
        feature_maps: list[torch.Tensor],
        spatial_shapes: list[torch.Tensor] | torch.Tensor,
    ) -> torch.Tensor:
        if tokens.numel() == 0 or not feature_maps:
            return tokens
        maps, level_shapes = _repeat_feature_levels(
            feature_maps, spatial_shapes, self.num_scales
        )
        b, q, c = tokens.shape
        sample_count = self.num_scales * self.num_points
        token_input = tokens.to(dtype=self.offset_head.weight.dtype)
        offsets = torch.tanh(self.offset_head(token_input).float()).to(dtype=tokens.dtype)
        offsets = offsets.view(b, q, self.num_scales, self.num_points, 2)
        center = (boxes[..., :2] + boxes[..., 2:]) * 0.5
        box_wh = (boxes[..., 2:] - boxes[..., :2]).clamp(min=1e-4)
        weight_logits = self.weight_head(token_input).float().view(
            b, q, self.num_scales, self.num_points
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
            ).unsqueeze(2)
            points = (
                center.to(dtype=tokens.dtype).unsqueeze(2)
                + offsets[:, :, scale_idx] * radius
            ).clamp(0.0, 1.0)
            sampled = self._sample_points(feature_map, points, shape)
            projection = self.level_projections[scale_idx]
            sampled = projection(sampled.to(dtype=projection.weight.dtype)).to(dtype=tokens.dtype)
            sampled_scales.append(sampled)
        sampled_all = torch.stack(sampled_scales, dim=2).reshape(
            b, q, sample_count, c
        )
        sampled = (
            sampled_all * weights.reshape(b, q, sample_count, 1)
        ).sum(dim=2)
        update = self.context_proj(torch.cat([tokens, sampled], dim=-1))
        return tokens + self.scale.sigmoid().to(dtype=update.dtype) * update

    @staticmethod
    def _sample_points(
        feature_map: torch.Tensor,
        points: torch.Tensor,
        spatial_shapes: torch.Tensor,
    ) -> torch.Tensor:
        b, channels, feature_h, feature_w = feature_map.shape
        q, p = points.shape[1], points.shape[2]
        grid = _padded_grid_from_normalized_points(
            points.to(device=feature_map.device, dtype=feature_map.dtype),
            spatial_shapes,
            feature_h,
            feature_w,
        )
        grid = grid.view(b, q * p, 1, 2)
        sampled = F.grid_sample(feature_map, grid, align_corners=False)
        return sampled.squeeze(-1).transpose(1, 2).view(b, q, p, channels)

    @staticmethod
    def _zero_init_last_linear(module: nn.Module) -> None:
        for child in reversed(list(module.modules())):
            if isinstance(child, nn.Linear):
                nn.init.zeros_(child.weight)
                if child.bias is not None:
                    nn.init.zeros_(child.bias)
                return


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


class GroupedPoseDecoderLayer(nn.Module):
    """GroupPose-style person/joint decoder layer with ROI cross-attention.

    Tokens are grouped as one explicit instance token followed by active joint
    tokens. The layer alternates within-person interaction, same-role
    cross-person interaction, per-person ROI cross-attention and an FFN.
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
        self.cross_norm = nn.LayerNorm(c)
        self.roi_cross_attention = nn.MultiheadAttention(
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
        roi_memory: torch.Tensor,
        same_role_attention_mask: torch.Tensor | None,
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

        normalized = self.cross_norm(tokens)
        cross_query = (normalized + role_position).reshape(
            b * num_people, num_roles, c
        )
        cross_update = self.roi_cross_attention(
            cross_query,
            roi_memory,
            roi_memory,
            need_weights=False,
        )[0].view(b, num_people, num_roles, c)
        tokens = tokens + self.residual_dropout(cross_update) * valid_f

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
    refinement_steps: int = 3
    decoder_heads: int = 8
    dropout: float = 0.0
    box_condition_scale: float = 1.25
    pose_roi_size: int = 16
    use_refinement: bool = True
    pose_feature_channels: int = 256
    use_native_spatial_features: bool = True
    human_decoder_layers: int = 2
    deformable_points: int = 4
    deformable_min_radius_cells: float = 2.0
    deformable_scale_prior_strength: float = 0.5
    deformable_scale_prior_center_cells: float = 6.0
    deformable_scale_prior_temperature: float = 1.5
    enable_box_denoising: bool = True
    enable_keypoint_denoising: bool = True
    ref_text_scale: float = 0.2
    enable_ref_visual_modulation: bool = True
    legacy_checkpoint_compat: bool = False
    enable_person_confidence_head: bool = True
    person_confidence_rescue: bool = False
    # In the unified detector/grounder path, learned person queries replace
    # externally generated coordinate strings.  One forward pass predicts all
    # people; RefHuman then selects one of those people with ``ref_match_head``.
    use_global_person_queries: bool = False
    num_person_queries: int = 80
    # The default starts from a schema-aligned anatomical reference and predicts
    # an instance-conditioned residual before iterative decoder refinement.
    pose_coordinate_init: str = "anatomical_dynamic"
    schema_joint_priors_path: str | None = "configs/schema_joint_priors.json"
    dynamic_reference_offset_scale: float = 1.5


class QwenPoseModel(nn.Module):
    def __init__(self, config: QwenPoseConfig) -> None:
        super().__init__()
        self.config = config
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
        self.num_feature_levels = 2 if int(config.high_res_external_dim) > 0 else 1
        self.high_res_feature_proj = (
            nn.Conv2d(int(config.high_res_external_dim), pose_feature_dim, 1)
            if self.num_feature_levels == 2
            else None
        )
        self.external_feature_proj = nn.Conv2d(config.external_dim, pose_feature_dim, 1)
        self.feature_level_embeddings = nn.Parameter(
            torch.zeros(self.num_feature_levels, pose_feature_dim)
        )
        nn.init.normal_(self.feature_level_embeddings, mean=0.0, std=0.02)
        self.external_text_proj = nn.Sequential(
            nn.LayerNorm(config.external_dim),
            nn.Linear(config.external_dim, c),
            nn.GELU(),
        )
        self.ref_visual_modulators = nn.ModuleList(
            [nn.Linear(c, pose_feature_dim * 2) for _ in range(self.num_feature_levels)]
        )
        # A zero scalar gate keeps Stage3 initialization exactly identical to
        # Stage2, while the randomly initialized FiLM projection gives the gate
        # a non-zero first-step gradient. Once the gate opens, both learn jointly.
        self.ref_visual_gates = nn.Parameter(torch.zeros(self.num_feature_levels))
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
        self.human_query_norm = nn.LayerNorm(c)
        self.person_query_embed: nn.Embedding | None = None
        self.person_query_reference_logits: nn.Parameter | None = None
        if config.use_global_person_queries:
            num_queries = max(int(config.num_person_queries), 1)
            self.config.num_person_queries = num_queries
            self.person_query_embed = nn.Embedding(num_queries, c)
            nn.init.normal_(self.person_query_embed.weight, mean=0.0, std=0.02)

            # Spread the initial DAB references over the whole image.  They are
            # learnable, but start as upright person-shaped windows so every
            # region receives a useful detection gradient from the first step.
            columns = max(int(math.ceil(math.sqrt(num_queries * 1.25))), 1)
            rows = max(int(math.ceil(num_queries / columns)), 1)
            query_index = torch.arange(num_queries, dtype=torch.float32)
            centers_x = (query_index.remainder(columns) + 0.5) / float(columns)
            centers_y = (query_index.div(columns, rounding_mode="floor") + 0.5) / float(rows)
            widths = torch.full_like(centers_x, min(1.6 / columns, 0.8))
            heights = torch.full_like(centers_y, min(2.4 / rows, 0.8))
            initial_cxcywh = torch.stack(
                [centers_x, centers_y, widths / 0.9, heights / 0.9], dim=-1
            ).clamp(1e-4, 1.0 - 1e-4)
            self.person_query_reference_logits = nn.Parameter(torch.logit(initial_cxcywh))
        self.human_decoder_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=c,
                    nhead=config.decoder_heads,
                    dim_feedforward=c * 4,
                    dropout=float(config.dropout),
                    activation="gelu",
                    batch_first=True,
                )
                for _ in range(max(int(config.human_decoder_layers), 1))
            ]
        )
        self.human_deformable_attention = nn.ModuleList(
            [
                HumanBoxDeformableAttention(
                    c,
                    feature_dim=pose_feature_dim,
                    num_scales=self.num_feature_levels,
                    num_points=config.deformable_points,
                    offset_scale=0.5,
                    min_radius_cells=config.deformable_min_radius_cells,
                    scale_prior_strength=config.deformable_scale_prior_strength,
                    scale_prior_center_cells=config.deformable_scale_prior_center_cells,
                    scale_prior_temperature=config.deformable_scale_prior_temperature,
                )
                for _ in self.human_decoder_layers
            ]
        )
        self.human_box_heads = nn.ModuleList(
            [MLP(c, c, 4, depth=3) for _ in self.human_decoder_layers]
        )
        self.human_objectness_heads = nn.ModuleList(
            [MLP(c, c, 1, depth=3) for _ in self.human_decoder_layers]
        )
        for box_head in self.human_box_heads:
            self._zero_init_last_linear(box_head)
        for objectness_head in self.human_objectness_heads:
            self._zero_init_last_linear(objectness_head)
        self.roi_memory_proj = nn.Conv2d(pose_feature_dim, c, kernel_size=1)
        self.pos_encoding = SinePositionEncoding(c)

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
        self.roi_memory_norm = nn.LayerNorm(c)
        self.roi_pool_proj = nn.Sequential(
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
        self.deformable_joint_attention = JointDeformableKeypointAttention(
            c,
            feature_dim=pose_feature_dim,
            num_scales=self.num_feature_levels,
            num_points=config.deformable_points,
            offset_scale=0.08,
            min_radius_cells=0.5,
            scale_prior_strength=config.deformable_scale_prior_strength,
            scale_prior_center_cells=config.deformable_scale_prior_center_cells,
            scale_prior_temperature=config.deformable_scale_prior_temperature,
        )
        self.coarse_xy_head = MLP(c, c, 2, depth=3)
        self.pose_xy_head = MLP(c, c, 2, depth=3)
        self.pose_vis_head = None
        self.pose_confidence_head = MLP(c, c, 1, depth=3)
        self.person_confidence_head = (
            MLP(c, c, 1, depth=3)
            if (config.enable_person_confidence_head or config.person_confidence_rescue)
            else None
        )
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
        self._zero_init_last_linear(self.coarse_xy_head)
        self._zero_init_last_linear(self.pose_xy_head)
        # v now means localization confidence, not physical visibility. A zero
        # final layer gives an unbiased 0.5 probability before the new target is
        # learned, while allowing the migrated hidden layers to be reused.
        if self.pose_confidence_head is not None:
            self._zero_init_last_linear(self.pose_confidence_head)
        if self.person_confidence_head is not None:
            self._zero_init_last_linear(self.person_confidence_head)

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
        """Project true P2/P3 native grids independently before dynamic padding."""

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
            if self.num_feature_levels != 2 or self.high_res_feature_proj is None:
                raise ValueError(
                    "Received Locate P2/P3 features but PoseHead was not configured "
                    "with high_res_external_dim."
                )
            if len(external_feature_map.levels) != 2:
                raise ValueError("Locate multi-scale input must contain exactly P2 and P3.")
            raw_levels = list(external_feature_map.levels)
            projections = [self.high_res_feature_proj, self.external_feature_proj]
            roi_level_idx = 1
            local_level_idx = 0
        else:
            if self.num_feature_levels != 1:
                raise ValueError(
                    "PoseHead expects true Locate P2/P3 features but received one feature level."
                )
            raw_levels = [as_spatial_batch(external_feature_map)]
            projections = [self.external_feature_proj]
            roi_level_idx = 0
            local_level_idx = 0

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
        return (
            projected_levels,
            spatial_shapes,
            valid_masks,
            roi_level_idx,
            local_level_idx,
        )

    def forward(
        self,
        schema_ids: torch.Tensor,
        task_ids: torch.Tensor,
        images: torch.Tensor | None = None,
        external_feature_map: torch.Tensor | SpatialFeatureBatch | MultiScaleSpatialFeatureBatch | None = None,
        external_text_embed: torch.Tensor | None = None,
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
        if external_feature_map is None or external_text_embed is None:
            raise ValueError(
                "QwenPoseModel requires backbone external_feature_map and external_text_embed."
            )
        if pose_condition_box_mode not in {"refined_detached", "input"}:
            raise ValueError(
                "pose_condition_box_mode must be 'refined_detached' or 'input', "
                f"got {pose_condition_box_mode!r}."
            )
        use_person_queries = bool(self.config.use_global_person_queries)
        if not use_person_queries and (target_boxes is None or target_box_mask is None):
            raise ValueError("QwenPoseModel requires external box conditions when person queries are disabled.")
        feature_device = external_feature_map.device
        batch_size = int(external_feature_map.shape[0])
        if use_person_queries:
            if self.person_query_reference_logits is None or self.person_query_embed is None:
                raise RuntimeError("Global person-query mode was enabled without person-query parameters.")
            target_boxes = boxes_from_cxcywh(
                self.person_query_reference_logits.to(
                    device=feature_device, dtype=torch.float32
                )
            )[None].expand(batch_size, -1, -1)
            target_box_mask = torch.ones(
                batch_size,
                int(self.config.num_person_queries),
                device=feature_device,
                dtype=torch.bool,
            )
        else:
            assert target_boxes is not None and target_box_mask is not None
            target_boxes = target_boxes.to(device=feature_device, dtype=torch.float32).clamp(0.0, 1.0)
            target_box_mask = target_box_mask.to(device=feature_device).bool()
        (
            feature_maps,
            spatial_shapes,
            spatial_valid_masks,
            roi_level_idx,
            local_level_idx,
        ) = self.build_locate_pose_features(external_feature_map)
        text_dtype = next(self.external_text_proj.parameters()).dtype
        text_embed = self.external_text_proj(external_text_embed.to(dtype=text_dtype))
        b = int(target_boxes.shape[0])
        c = int(self.config.hidden_dim)
        ref_task_gate = task_ids.eq(1).to(device=text_embed.device, dtype=text_embed.dtype)
        # Person proposals and shared visual maps stay caption-independent in
        # the unified path. Language is consumed later by Ref Match and the
        # shared RefHuman pose-query conditioning path.
        if self.config.enable_ref_visual_modulation and not use_person_queries:
            conditioned_maps: list[torch.Tensor] = []
            task_gate = ref_task_gate.view(b, 1, 1, 1)
            for level_idx, feature in enumerate(feature_maps):
                modulator = self.ref_visual_modulators[level_idx]
                modulation = modulator(
                    text_embed.to(dtype=modulator.weight.dtype)
                ).to(dtype=feature.dtype)
                gamma, beta = modulation.chunk(2, dim=-1)
                gamma = torch.tanh(gamma).view(b, -1, 1, 1)
                beta = beta.view(b, -1, 1, 1)
                task_level_gate = task_gate.to(
                    device=feature.device,
                    dtype=feature.dtype,
                ) * torch.tanh(self.ref_visual_gates[level_idx]).to(
                    device=feature.device,
                    dtype=feature.dtype,
                )
                conditioned_maps.append(
                    (
                        feature * (1.0 + task_level_gate * gamma)
                        + task_level_gate * beta
                    )
                    * spatial_valid_masks[level_idx][:, None].to(dtype=feature.dtype)
                )
            feature_maps = conditioned_maps
        feature_map = feature_maps[roi_level_idx]
        main_count = int(target_boxes.shape[1])
        box_mask = target_box_mask

        all_boxes = target_boxes.to(dtype=feature_map.dtype)
        all_mask = box_mask
        dn_count = 0
        if (
            self.config.enable_box_denoising
            and dn_boxes is not None
            and dn_box_mask is not None
            and int(dn_boxes.shape[1]) > 0
        ):
            dn_boxes = dn_boxes.to(device=feature_map.device, dtype=feature_map.dtype).clamp(0.0, 1.0)
            dn_box_mask = dn_box_mask.to(device=feature_map.device).bool()
            dn_count = int(dn_boxes.shape[1])
            all_boxes = torch.cat([all_boxes, dn_boxes], dim=1)
            all_mask = torch.cat([all_mask, dn_box_mask], dim=1)

        total_count = int(all_boxes.shape[1])
        attention_mask = None
        if dn_count > 0:
            per_sample_mask = torch.zeros(
                b,
                total_count,
                total_count,
                device=feature_map.device,
                dtype=torch.bool,
            )
            per_sample_mask[:, :main_count, main_count:] = True
            per_sample_mask[:, main_count:, :main_count] = True
            if dn_group_ids is not None:
                group_ids = dn_group_ids.to(device=feature_map.device, dtype=torch.long)
                group_a = group_ids[:, :, None]
                group_b = group_ids[:, None, :]
                different_group = (group_a != group_b) & (group_a >= 0) & (group_b >= 0)
                per_sample_mask[:, main_count:, main_count:] = different_group
            # MultiheadAttention accepts one mask per batch/head. Keeping the
            # DN-group mask sample-specific avoids over-masking heterogeneous batches.
            attention_mask = per_sample_mask.repeat_interleave(
                int(self.config.decoder_heads), dim=0
            )

        def pooled_human_context(current_boxes: torch.Tensor) -> torch.Tensor:
            # Dense person identity/context remains anchored to P3; P2/P3 are
            # both available through the sparse deformable branch below.
            pooled = self._sample_box_feature_maps(
                feature_maps[roi_level_idx],
                current_boxes,
                3,
                spatial_shapes[roi_level_idx],
            ).mean(dim=(-2, -1))
            projection_dtype = next(self.human_context_proj.parameters()).dtype
            return self.human_context_proj(pooled.to(dtype=projection_dtype))

        current_boxes = all_boxes
        human_context = pooled_human_context(current_boxes)
        human_box_embed = self.box_query_proj(box_fourier_pe(current_boxes, c))
        query_text = current_boxes.new_zeros((b, 1, c))
        if not use_person_queries:
            query_text = (
                float(self.config.ref_text_scale)
                * ref_task_gate[:, None, None]
                * text_embed[:, None, :]
            )
        learned_queries = current_boxes.new_zeros((b, total_count, c))
        if use_person_queries:
            learned_queries[:, :main_count] = self.person_query_embed.weight.to(
                device=current_boxes.device, dtype=current_boxes.dtype
            )[None]
        human_tokens = self.human_query_norm(
            human_box_embed + human_context + query_text + learned_queries
        )
        padding_mask = ~all_mask
        safe_padding_mask = padding_mask.masked_fill(
            padding_mask.all(dim=1, keepdim=True),
            False,
        )
        aux_box_outputs: list[dict[str, torch.Tensor]] = []
        objectness_logits = current_boxes.new_zeros((b, total_count))
        for layer_idx, layer in enumerate(self.human_decoder_layers):
            human_tokens = layer(
                human_tokens,
                src_mask=attention_mask,
                src_key_padding_mask=safe_padding_mask,
            )
            human_tokens = self.human_deformable_attention[layer_idx](
                human_tokens,
                current_boxes,
                feature_maps,
                spatial_shapes,
            )
            deltas = self.human_box_heads[layer_idx](human_tokens)
            current_boxes = refine_boxes_xyxy(current_boxes, deltas)
            objectness_logits = self.human_objectness_heads[layer_idx](human_tokens).squeeze(-1)
            current_boxes = torch.where(all_mask[..., None], current_boxes, all_boxes)
            objectness_logits = torch.where(
                all_mask,
                objectness_logits,
                torch.full_like(objectness_logits, -10.0),
            )
            aux_box_outputs.append(
                {
                    "pred_boxes": current_boxes[:, :main_count],
                    "objectness_logits": objectness_logits[:, :main_count],
                }
            )
            if layer_idx + 1 < len(self.human_decoder_layers):
                # Each box-head layer receives a detached reference, matching
                # iterative DAB/DINO refinement. Earlier layers retain their own
                # auxiliary losses and do not absorb unstable later gradients.
                current_boxes = current_boxes.detach()

        refined_boxes = current_boxes[:, :main_count]
        refinement_fallback_mask = torch.zeros_like(box_mask, dtype=torch.bool)
        if not self.training and not use_person_queries:
            # LocateAnything owns RefHuman target identification; the box head
            # may refine that proposal locally but must not silently move it to
            # another person at inference time.
            refined_boxes, refinement_fallback_mask = apply_refhuman_box_refinement_safety(
                refined_boxes,
                all_boxes[:, :main_count],
                box_mask,
                task_ids,
            )
        box_objectness_logits = objectness_logits[:, :main_count]
        main_human_tokens = human_tokens[:, :main_count]
        ref_candidate = self.ref_candidate_proj(main_human_tokens)
        ref_text = self.ref_text_match_proj(text_embed).unsqueeze(1).expand_as(ref_candidate)
        ref_pair = torch.cat(
            [ref_candidate, ref_text, ref_candidate * ref_text],
            dim=-1,
        )
        ref_logits = self.ref_match_head(ref_pair).squeeze(-1)
        ref_active = task_ids.eq(1).to(device=ref_logits.device)[:, None] & box_mask
        ref_logits = torch.where(
            ref_active,
            ref_logits,
            torch.full_like(ref_logits, -10.0),
        )
        input_boxes = target_boxes.to(dtype=feature_map.dtype)

        # Build a hierarchical DN path: each noisy skeleton is conditioned on
        # the refined *positive box-DN query* from the same source person and
        # group. The box is detached at the box->pose boundary so pose-DN does
        # not distort LocateAnything grounding or the human-box decoder.
        pose_dn_count = 0
        pose_dn_mask: torch.Tensor | None = None
        paired_dn_boxes: torch.Tensor | None = None
        source_indices: torch.Tensor | None = None
        if (
            self.training
            and self.config.enable_keypoint_denoising
            and dn_count > 0
            and keypoint_dn_noisy_keypoints is not None
            and keypoint_dn_mask is not None
            and keypoint_dn_labels is not None
            and keypoint_dn_target_valid is not None
            and keypoint_dn_source_indices is not None
            and keypoint_dn_group_ids is not None
            and keypoint_dn_box_query_indices is not None
            and int(keypoint_dn_noisy_keypoints.shape[1]) > 0
        ):
            pose_dn_count = int(keypoint_dn_noisy_keypoints.shape[1])
            box_query_indices = keypoint_dn_box_query_indices.to(
                device=feature_map.device, dtype=torch.long
            )
            if box_query_indices.shape != (b, pose_dn_count):
                raise ValueError(
                    "keypoint-DN box query indices must have shape "
                    f"{(b, pose_dn_count)}, got {tuple(box_query_indices.shape)}."
                )
            source_indices = keypoint_dn_source_indices.to(
                device=feature_map.device, dtype=torch.long
            )
            source_valid = source_indices.ge(0)
            box_index_valid = box_query_indices.ge(0) & box_query_indices.lt(dn_count)
            safe_box_indices = box_query_indices.clamp(
                min=0, max=max(dn_count - 1, 0)
            )
            dn_refined_boxes = current_boxes[:, main_count:]
            paired_dn_boxes = torch.gather(
                dn_refined_boxes.detach(),
                dim=1,
                index=safe_box_indices[..., None].expand(-1, -1, 4),
            )
            paired_box_valid = torch.gather(
                all_mask[:, main_count:], dim=1, index=safe_box_indices
            )
            if dn_labels is not None:
                paired_box_valid = paired_box_valid & torch.gather(
                    dn_labels.to(device=feature_map.device).gt(0.5),
                    dim=1,
                    index=safe_box_indices,
                )
            pose_dn_mask = (
                keypoint_dn_mask.to(device=feature_map.device).bool()
                & source_valid
                & box_index_valid
                & paired_box_valid
            )
            paired_dn_boxes = torch.where(
                pose_dn_mask[..., None],
                paired_dn_boxes,
                torch.zeros_like(paired_dn_boxes),
            )

        # The normal path keeps the historical box->pose detach boundary.  Joint
        # soft-box training feeds differentiable Locate boxes into the same main
        # PoseHead pass so downstream pose/box error can update both PoseHead and Locate LoRA.
        combined_boxes = (
            input_boxes
            if pose_condition_box_mode == "input"
            else refined_boxes.detach()
        )
        combined_box_mask = box_mask
        combined_initial_keypoints: torch.Tensor | None = None
        combined_initial_valid: torch.Tensor | None = None
        combined_initial_query_mask: torch.Tensor | None = None
        combined_dn_labels: torch.Tensor | None = None
        combined_dn_groups: torch.Tensor | None = None
        if pose_dn_count > 0 and paired_dn_boxes is not None and pose_dn_mask is not None:
            union_count = int(keypoint_dn_noisy_keypoints.shape[2])
            main_pose_boxes = (
                input_boxes
                if pose_condition_box_mode == "input"
                else refined_boxes.detach()
            )
            combined_boxes = torch.cat([main_pose_boxes, paired_dn_boxes], dim=1)
            combined_box_mask = torch.cat([box_mask, pose_dn_mask], dim=1)
            main_initial = keypoint_dn_noisy_keypoints.new_zeros(
                (b, main_count, union_count, 2)
            )
            combined_initial_keypoints = torch.cat(
                [main_initial, keypoint_dn_noisy_keypoints], dim=1
            )
            main_valid = torch.zeros(
                b,
                main_count,
                union_count,
                device=feature_map.device,
                dtype=torch.bool,
            )
            combined_initial_valid = torch.cat(
                [main_valid, keypoint_dn_target_valid.to(device=feature_map.device).bool()],
                dim=1,
            )
            combined_initial_query_mask = torch.cat(
                [torch.zeros_like(box_mask), pose_dn_mask], dim=1
            )
            combined_dn_labels = torch.cat(
                [
                    torch.zeros(b, main_count, device=feature_map.device),
                    keypoint_dn_labels.to(device=feature_map.device),
                ],
                dim=1,
            )
            combined_dn_groups = torch.cat(
                [
                    torch.full(
                        (b, main_count),
                        -1,
                        device=feature_map.device,
                        dtype=torch.long,
                    ),
                    keypoint_dn_group_ids.to(
                        device=feature_map.device, dtype=torch.long
                    ),
                ],
                dim=1,
            )

        combined_pose = self._run_pose_branch(
            feature_maps=feature_maps,
            spatial_shapes=spatial_shapes,
            roi_level_idx=roi_level_idx,
            local_level_idx=local_level_idx,
            text_embed=text_embed,
            ref_task_gate=(torch.zeros_like(ref_task_gate) if use_person_queries else ref_task_gate),
            schema_ids=schema_ids,
            task_ids=task_ids,
            refined_boxes=combined_boxes,
            box_mask=combined_box_mask,
            initial_keypoints=combined_initial_keypoints,
            initial_keypoint_valid=combined_initial_valid,
            initial_query_mask=combined_initial_query_mask,
            dn_labels=combined_dn_labels,
            dn_group_ids=combined_dn_groups,
        )

        def query_slice(value: torch.Tensor, start: int, end: int) -> torch.Tensor:
            return value[:, start:end]

        main_pose = {
            "pose_boxes": query_slice(combined_pose["pose_boxes"], 0, main_count),
            "instance_emb": query_slice(combined_pose["instance_emb"], 0, main_count),
            "pose_quality_logits": query_slice(
                combined_pose["pose_quality_logits"], 0, main_count
            ),
            "keypoints": query_slice(combined_pose["keypoints"], 0, main_count),
            "keypoint_valid_mask": combined_pose["keypoint_valid_mask"],
            "keypoint_confidence_logits": query_slice(
                combined_pose["keypoint_confidence_logits"], 0, main_count
            ),
            "decoder_keypoints": [
                query_slice(value, 0, main_count)
                for value in combined_pose["decoder_keypoints"]
            ],
            "coarse_keypoints": query_slice(
                combined_pose["coarse_keypoints"], 0, main_count
            ),
            "deform_keypoints": query_slice(
                combined_pose["deform_keypoints"], 0, main_count
            ),
            "refine_keypoints": [
                query_slice(value, 0, main_count)
                for value in combined_pose["refine_keypoints"]
            ],
            "schema_joint_indices": combined_pose["schema_joint_indices"],
            "schema_joint_valid": combined_pose["schema_joint_valid"],
        }
        boxes = main_pose["pose_boxes"]
        instance = main_pose["instance_emb"]
        pose_quality_logits = main_pose["pose_quality_logits"]
        objectness_prob = box_objectness_logits.sigmoid()
        pose_quality_prob = pose_quality_logits.sigmoid()
        combined_prob = (objectness_prob * pose_quality_prob).clamp(1e-5, 1.0 - 1e-5)
        person_logits = torch.where(
            box_mask,
            torch.logit(combined_prob),
            torch.full_like(combined_prob, -10.0),
        )
        person_confidence_head_available = self.person_confidence_head is not None
        outputs = {
            # Canonical DETR/GroupPose-style names.
            "pred_logits": person_logits.unsqueeze(-1),
            "pred_boxes": refined_boxes,
            "pred_keypoints": main_pose["keypoints"],
            # Explicit detection/pose quality decomposition.
            "box_objectness_logits": box_objectness_logits,
            "pose_quality_logits": pose_quality_logits,
            "aux_box_outputs": aux_box_outputs[:-1],
            # Backwards-compatible project names.
            "person_logits": person_logits,
            "person_confidence_head_available": person_confidence_head_available,
            "person_confidence_rescue": person_confidence_head_available,
            "input_boxes": input_boxes,
            "boxes": refined_boxes,
            "pose_boxes": boxes,
            "box_mask": box_mask,
            "ref_box_refinement_fallback_mask": refinement_fallback_mask,
            "keypoints": main_pose["keypoints"],
            "keypoint_valid_mask": main_pose["keypoint_valid_mask"],
            "keypoint_confidence_logits": main_pose["keypoint_confidence_logits"],
            "decoder_keypoints": main_pose["decoder_keypoints"],
            "coarse_keypoints": main_pose["coarse_keypoints"],
            "deform_keypoints": main_pose["deform_keypoints"],
            "ref_logits": ref_logits,
            "instance_emb": instance,
            "schema_joint_indices": main_pose["schema_joint_indices"],
            "schema_joint_valid": main_pose["schema_joint_valid"],
        }
        if self.training and self.keypoint_dn_type_embed is not None:
            # Some images contain people but no annotated/valid keypoints, so
            # prepare_keypoint_denoising() can legitimately return no DN batch
            # on one distributed rank while another rank has DN queries.  This
            # embedding is the only parameter used exclusively by the DN pose
            # branch (2 * hidden_dim = 896 parameters in the default model).
            # Keep a zero-valued autograd edge on every rank so ZeRO-2 reduces
            # identical parameter buckets without changing the objective.
            outputs["keypoint_dn_graph_anchor"] = (
                self.keypoint_dn_type_embed.weight.sum() * 0.0
            )
        if dn_count > 0:
            outputs.update(
                {
                    "dn_pred_boxes": current_boxes[:, main_count:],
                    "dn_objectness_logits": objectness_logits[:, main_count:],
                    "dn_box_mask": all_mask[:, main_count:],
                    "dn_labels": dn_labels.to(device=feature_map.device).float()
                    if dn_labels is not None
                    else all_mask[:, main_count:].float(),
                    "dn_target_boxes": dn_target_boxes.to(
                        device=feature_map.device,
                        dtype=current_boxes.dtype,
                    )
                    if dn_target_boxes is not None
                    else current_boxes[:, main_count:].detach(),
                }
            )
        if main_pose["decoder_keypoints"]:
            outputs["decoder_keypoints"] = main_pose["decoder_keypoints"]
        if main_pose["refine_keypoints"]:
            outputs["refine_keypoints"] = main_pose["refine_keypoints"]

        if pose_dn_count > 0 and pose_dn_mask is not None and source_indices is not None:
            pose_start = main_count
            pose_end = main_count + pose_dn_count
            outputs.update(
                {
                    "keypoint_dn_keypoints": query_slice(
                        combined_pose["keypoints"], pose_start, pose_end
                    ),
                    "keypoint_dn_decoder_keypoints": [
                        query_slice(value, pose_start, pose_end)
                        for value in combined_pose["decoder_keypoints"]
                    ],
                    "keypoint_dn_coarse_keypoints": query_slice(
                        combined_pose["coarse_keypoints"], pose_start, pose_end
                    ),
                    "keypoint_dn_deform_keypoints": query_slice(
                        combined_pose["deform_keypoints"], pose_start, pose_end
                    ),
                    "keypoint_dn_refine_keypoints": [
                        query_slice(value, pose_start, pose_end)
                        for value in combined_pose["refine_keypoints"]
                    ],
                    "keypoint_dn_confidence_logits": query_slice(
                        combined_pose["keypoint_confidence_logits"],
                        pose_start,
                        pose_end,
                    ),
                    "keypoint_dn_pose_quality_logits": query_slice(
                        combined_pose["pose_quality_logits"], pose_start, pose_end
                    ),
                    "keypoint_dn_mask": pose_dn_mask,
                    "keypoint_dn_labels": keypoint_dn_labels.to(
                        device=feature_map.device, dtype=feature_map.dtype
                    ),
                    "keypoint_dn_target_keypoints": keypoint_dn_target_keypoints,
                    "keypoint_dn_target_valid": keypoint_dn_target_valid,
                    "keypoint_dn_target_boxes": keypoint_dn_target_boxes,
                    "keypoint_dn_target_areas": keypoint_dn_target_areas,
                    "keypoint_dn_source_indices": source_indices,
                    "keypoint_dn_group_ids": keypoint_dn_group_ids,
                    "keypoint_dn_box_query_indices": keypoint_dn_box_query_indices,
                }
            )
        return outputs

    def _run_pose_branch(
        self,
        *,
        feature_maps: list[torch.Tensor],
        spatial_shapes: list[torch.Tensor],
        roi_level_idx: int,
        local_level_idx: int,
        text_embed: torch.Tensor,
        ref_task_gate: torch.Tensor,
        schema_ids: torch.Tensor,
        task_ids: torch.Tensor,
        refined_boxes: torch.Tensor,
        box_mask: torch.Tensor,
        initial_keypoints: torch.Tensor | None = None,
        initial_keypoint_valid: torch.Tensor | None = None,
        initial_query_mask: torch.Tensor | None = None,
        dn_labels: torch.Tensor | None = None,
        dn_group_ids: torch.Tensor | None = None,
    ) -> dict[str, object]:
        """Run normal and training-only paired-DN persons in one pose stack.

        ``initial_query_mask`` marks the DN slots that receive noisy keypoint
        references. Main queries cannot read DN slots, DN queries may read main
        slots, and different DN groups remain isolated in same-role attention.
        """
        feature_map = feature_maps[roi_level_idx]
        b, num_boxes = int(refined_boxes.shape[0]), int(refined_boxes.shape[1])
        c = int(self.config.hidden_dim)
        boxes = expand_boxes_xyxy(
            refined_boxes.to(dtype=feature_map.dtype), self.config.box_condition_scale
        )
        box_mask = box_mask.to(device=feature_map.device).bool()
        box_embed = self.box_query_proj(box_fourier_pe(refined_boxes, c))
        roi_size = max(int(self.config.pose_roi_size), 2)
        # The dense ROI is intentionally P3-only: it anchors person identity and
        # whole-body structure without compressing P2 into another fixed memory.
        roi_source = self._sample_box_feature_maps(
            feature_maps[roi_level_idx],
            boxes,
            roi_size,
            spatial_shapes[roi_level_idx],
        )
        roi_source = roi_source * box_mask.view(b, num_boxes, 1, 1, 1).to(
            dtype=roi_source.dtype
        )
        flat_roi = roi_source.reshape(
            b * num_boxes, roi_source.shape[2], roi_size, roi_size
        )
        roi_features = self.roi_memory_proj(flat_roi).view(
            b, num_boxes, c, roi_size, roi_size
        )
        roi_embed = self.roi_pool_proj(roi_features.mean(dim=(-2, -1)))
        global_visual = _masked_spatial_mean(
            feature_maps[roi_level_idx], spatial_shapes[roi_level_idx]
        )
        image_embed = self.human_context_proj(
            global_visual.to(dtype=next(self.human_context_proj.parameters()).dtype)
        )
        instance_text = (
            float(self.config.ref_text_scale)
            * ref_task_gate[:, None, None]
            * text_embed[:, None, :]
        )
        instance = self.instance_query_norm(
            box_embed + roi_embed + image_embed[:, None, :] + instance_text
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
        box_pe = box_embed.view(b, num_boxes, 1, c)
        text_condition = (
            float(self.config.ref_text_scale)
            * ref_task_gate.view(b, 1, 1, 1)
            * text_embed.view(b, 1, 1, c)
        )
        joint_type = self.pose_token_type_embed.weight[1].view(1, 1, 1, c)
        pose_tokens = (
            instance[:, :, None, :]
            + joint_base
            + task
            + box_pe
            + text_condition
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
                raise RuntimeError(
                    "Dynamic anatomical mode requires schema priors and a reference offset head."
                )
            expanded_joint = joint_base.expand(-1, num_boxes, -1, -1)
            expanded_instance = instance[:, :, None, :].expand(-1, -1, active_k, -1)
            reference_input = torch.cat([expanded_instance, expanded_joint], dim=-1)
            reference_dtype = next(self.reference_offset_head.parameters()).dtype
            reference_offset = self.reference_offset_head(
                reference_input.to(dtype=reference_dtype)
            ).float()
            reference_offset = (
                float(self.config.dynamic_reference_offset_scale)
                * torch.tanh(reference_offset)
            )
            prior_logits = torch.logit(
                schema_prior_active.float().clamp(1e-4, 1.0 - 1e-4)
            )[:, None]
            default_reference_rel = torch.sigmoid(
                prior_logits + reference_offset
            ).to(dtype=feature_map.dtype)
        elif self.config.pose_coordinate_init == "schema_prior":
            if schema_prior_active is None:
                raise RuntimeError("Legacy schema-prior mode requires its prior buffer.")
            default_reference_rel = schema_prior_active[:, None].expand(
                -1, num_boxes, -1, -1
            )
        elif self.config.pose_coordinate_init == "learned_spread":
            if self.joint_reference_logits is None:
                raise RuntimeError("Learned-spread mode requires joint reference logits.")
            union_reference = self.joint_reference_logits.sigmoid().to(
                device=feature_map.device,
                dtype=feature_map.dtype,
            )
            active_reference = torch.gather(
                union_reference[None].expand(b, -1, -1),
                dim=1,
                index=schema_joint_indices[..., None].expand(-1, -1, 2),
            )
            default_reference_rel = active_reference[:, None].expand(
                -1, num_boxes, -1, -1
            )
        else:
            default_reference_rel = boxes.new_full(
                (b, num_boxes, active_k, 2), 0.5
            )

        initial_active: torch.Tensor | None = None
        active_initial_query_mask: torch.Tensor | None = None
        if initial_keypoints is not None:
            if initial_query_mask is None:
                active_initial_query_mask = box_mask
            else:
                active_initial_query_mask = initial_query_mask.to(
                    device=feature_map.device
                ).bool()
                if active_initial_query_mask.shape != (b, num_boxes):
                    raise ValueError(
                        "initial pose query mask must have shape "
                        f"{(b, num_boxes)}, got {tuple(active_initial_query_mask.shape)}."
                    )
                active_initial_query_mask = active_initial_query_mask & box_mask
            initial_keypoints = initial_keypoints.to(
                device=feature_map.device, dtype=feature_map.dtype
            )
            gather_index = schema_joint_indices[:, None, :, None].expand(
                b, num_boxes, active_k, 2
            )
            initial_active = torch.gather(initial_keypoints, dim=2, index=gather_index)
            wh_for_reference = (boxes[..., 2:] - boxes[..., :2]).clamp(min=1e-4)
            fallback_absolute = (
                boxes[..., None, :2]
                + default_reference_rel.detach() * wh_for_reference[..., None, :]
            )
            if initial_keypoint_valid is not None:
                initial_valid_active = torch.gather(
                    initial_keypoint_valid.to(device=feature_map.device).bool(),
                    dim=2,
                    index=schema_joint_indices[:, None, :].expand(
                        b, num_boxes, active_k
                    ),
                )
                initial_active = torch.where(
                    initial_valid_active[..., None], initial_active, fallback_absolute
                )
            initial_active = initial_active.clamp(0.0, 1.0)
            if dn_labels is not None and self.keypoint_dn_type_embed is not None:
                type_indices = dn_labels.to(
                    device=feature_map.device, dtype=torch.long
                ).clamp(0, 1)
                type_embed = self.keypoint_dn_type_embed(type_indices)[:, :, None, :]
                pose_tokens = pose_tokens + type_embed * active_initial_query_mask[
                    ..., None, None
                ].to(dtype=type_embed.dtype)

        pose_tokens = self.pose_query_norm(pose_tokens)
        pose_valid = (
            schema_joint_valid[:, None, :].expand(b, num_boxes, active_k)
            & box_mask[:, :, None]
        )
        wh = (boxes[..., 2:] - boxes[..., :2]).clamp(min=1e-4)
        current_reference_rel = default_reference_rel.clamp(1e-4, 1.0 - 1e-4)
        if initial_active is not None:
            initial_rel = (
                (initial_active - boxes[..., None, :2]) / wh[..., None, :]
            ).clamp(1e-4, 1.0 - 1e-4)
            if active_initial_query_mask is None:
                current_reference_rel = initial_rel
            else:
                current_reference_rel = torch.where(
                    active_initial_query_mask[..., None, None],
                    initial_rel,
                    current_reference_rel,
                )

        group_tokens = torch.cat([instance_token[:, :, None, :], pose_tokens], dim=2)
        group_valid = torch.cat([box_mask[:, :, None], pose_valid], dim=2)
        same_role_attention_mask = self._build_group_pose_attention_mask(
            box_mask=box_mask,
            dn_query_mask=active_initial_query_mask,
            dn_group_ids=dn_group_ids,
            num_roles=active_k + 1,
        )

        roi_memory = roi_features.flatten(3).permute(0, 1, 3, 2)
        roi_pe = self.pos_encoding(roi_size, roi_size, feature_map.device).to(
            dtype=roi_memory.dtype
        )
        roi_memory = self.roi_memory_norm(
            roi_memory + roi_pe.view(1, 1, roi_size * roi_size, c)
        ).reshape(b * num_boxes, roi_size * roi_size, c)

        decoder_reference_xy_steps: list[torch.Tensor] = []
        center_reference = current_reference_rel.new_full((b, num_boxes, 1, 2), 0.5)
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
                [center_reference, current_reference_rel.detach()], dim=2
            )
            role_position = point_fourier_pe(role_reference, c)
            group_tokens = decoder_layer(
                group_tokens,
                role_position,
                group_valid,
                roi_memory,
                same_role_attention_mask,
            )
            pose_tokens = group_tokens[:, :, 1:]
            current_reference_xy = (
                boxes[..., None, :2]
                + current_reference_rel.detach() * wh[..., None, :]
            ).clamp(0.0, 1.0)
            pose_tokens = decoder_deformable(
                pose_tokens,
                current_reference_xy,
                wh,
                feature_maps,
                spatial_shapes,
            )
            reference_base = (
                current_reference_rel
                if decoder_idx == 0
                else current_reference_rel.detach()
            )
            reference_delta = decoder_coordinate_head(pose_tokens).float()
            current_reference_rel = torch.sigmoid(
                torch.logit(reference_base.float().clamp(1e-4, 1.0 - 1e-4))
                + reference_delta
            ).to(dtype=feature_map.dtype)
            decoder_reference_xy_steps.append(
                (
                    boxes[..., None, :2]
                    + current_reference_rel * wh[..., None, :]
                ).clamp(0.0, 1.0)
            )
            group_tokens = torch.cat(
                [group_tokens[:, :, :1], pose_tokens], dim=2
            )

        coarse_tokens = pose_tokens
        coarse_rel_xy = torch.sigmoid(
            torch.logit(current_reference_rel.float().clamp(1e-4, 1.0 - 1e-4))
            + self.coarse_xy_head(coarse_tokens).float()
        ).to(dtype=feature_map.dtype)
        coarse_reference_xy = (
            boxes[..., None, :2] + coarse_rel_xy * wh[..., None, :]
        ).clamp(0.0, 1.0)
        pose_tokens = self.deformable_joint_attention(
            coarse_tokens, coarse_reference_xy, wh, feature_maps, spatial_shapes
        )

        coarse_rel_detached = (
            (coarse_reference_xy.detach() - boxes[..., None, :2])
            / wh[..., None, :]
        ).clamp(1e-4, 1.0 - 1e-4)
        # Every coordinate mode now uses the final grouped-decoder/coarse output
        # as the sole deformable-stage reference. This keeps the iterative path
        # identical for main and DN queries.
        deform_base_logits = torch.logit(coarse_rel_detached.float())
        rel_xy = torch.sigmoid(
            deform_base_logits + self.pose_xy_head(pose_tokens).float()
        ).to(dtype=feature_map.dtype)
        keypoint_xy = (
            boxes[..., None, :2] + rel_xy * wh[..., None, :]
        ).clamp(0.0, 1.0)
        deform_keypoint_xy = keypoint_xy

        refine_keypoint_xy_steps: list[torch.Tensor] = []
        if (
            self.refine_heads is not None
            and self.refine_token_fusers is not None
            and self.refine_patch_weight_heads is not None
            and self.joint_context is not None
            and self.local_proj is not None
        ):
            # Final coordinate refinement reads only the true high-resolution
            # P2 grid when present; samples remain free to move outside the ROI.
            local_feature_map = self.local_proj(feature_maps[local_level_idx])
            local_shapes = spatial_shapes[local_level_idx]
            local_rows = torch.arange(
                local_feature_map.shape[-2], device=local_feature_map.device
            )[None, :, None]
            local_cols = torch.arange(
                local_feature_map.shape[-1], device=local_feature_map.device
            )[None, None, :]
            local_valid = (
                (local_rows < local_shapes[:, 0, None, None])
                & (local_cols < local_shapes[:, 1, None, None])
            )
            local_feature_map = local_feature_map * local_valid[:, None].to(
                dtype=local_feature_map.dtype
            )
            context_mask = ~pose_valid.reshape(b * num_boxes, active_k)
            context_mask = context_mask.masked_fill(
                context_mask.all(dim=1, keepdim=True), False
            )
            refine_radius_scales = (0.06, 0.03, 0.015)
            refine_min_radius_cells = (2.0, 1.0, 0.5)
            for refine_idx, (refine_head, token_fuser, patch_weight_head) in enumerate(
                zip(
                    self.refine_heads,
                    self.refine_token_fusers,
                    self.refine_patch_weight_heads,
                )
            ):
                pose_tokens = self.joint_context(
                    pose_tokens.reshape(b * num_boxes, active_k, c),
                    src_key_padding_mask=context_mask,
                ).reshape(b, num_boxes, active_k, c)
                patch_logits = patch_weight_head(
                    pose_tokens.to(dtype=patch_weight_head.weight.dtype)
                )
                local = self._sample_local_patch_features(
                    local_feature_map,
                    keypoint_xy.detach(),
                    wh.detach(),
                    patch_logits,
                    local_shapes,
                    patch_size=3,
                    radius_scale=refine_radius_scales[
                        min(refine_idx, len(refine_radius_scales) - 1)
                    ],
                    min_radius_cells=refine_min_radius_cells[
                        min(refine_idx, len(refine_min_radius_cells) - 1)
                    ],
                )
                point_pe = point_fourier_pe(keypoint_xy.detach(), c)
                refine_input = torch.cat([pose_tokens, local, point_pe], dim=-1)
                delta = torch.tanh(refine_head(refine_input)).float()
                scale = self.refine_step_scales[refine_idx].sigmoid().float()
                current_rel = (
                    (keypoint_xy.detach() - boxes[..., None, :2])
                    / wh[..., None, :]
                ).float().clamp(1e-4, 1.0 - 1e-4)
                refined_rel = torch.sigmoid(
                    torch.logit(current_rel) + delta * scale
                ).to(dtype=feature_map.dtype)
                keypoint_xy = (
                    boxes[..., None, :2] + refined_rel * wh[..., None, :]
                ).clamp(0.0, 1.0)
                refine_keypoint_xy_steps.append(keypoint_xy)
                pose_tokens = pose_tokens + token_fuser(refine_input)

        assert self.pose_confidence_head is not None
        schema_confidence_logits = self.pose_confidence_head(pose_tokens)
        keypoint_confidence = schema_confidence_logits.sigmoid()
        if self.person_confidence_head is not None:
            valid_f = pose_valid.to(dtype=pose_tokens.dtype).unsqueeze(-1)
            pooled_pose_tokens = (pose_tokens * valid_f).sum(dim=2) / valid_f.sum(
                dim=2
            ).clamp(min=1.0)
            confidence_dtype = next(self.person_confidence_head.parameters()).dtype
            pose_quality_logits = self.person_confidence_head(
                pooled_pose_tokens.to(dtype=confidence_dtype)
            ).squeeze(-1).to(dtype=boxes.dtype)
        else:
            pose_quality_logits = boxes.new_zeros((b, num_boxes))

        aux_confidence = keypoint_confidence.detach()
        coarse_keypoints = self._scatter_schema_keypoints(
            torch.cat([coarse_reference_xy, aux_confidence], dim=-1),
            schema_scatter_map,
        )
        deform_keypoints = self._scatter_schema_keypoints(
            torch.cat([deform_keypoint_xy, aux_confidence], dim=-1),
            schema_scatter_map,
        )
        keypoints = self._scatter_schema_keypoints(
            torch.cat([keypoint_xy, keypoint_confidence], dim=-1),
            schema_scatter_map,
        )
        keypoint_confidence_logits = self._scatter_schema_keypoints(
            schema_confidence_logits, schema_scatter_map
        ).squeeze(-1)
        decoder_keypoints = [
            self._scatter_schema_keypoints(
                torch.cat([xy, aux_confidence], dim=-1), schema_scatter_map
            )
            for xy in decoder_reference_xy_steps
        ]
        refine_keypoints = [
            self._scatter_schema_keypoints(
                torch.cat([xy, aux_confidence], dim=-1), schema_scatter_map
            )
            for xy in refine_keypoint_xy_steps
        ]
        return {
            "pose_boxes": boxes,
            "instance_emb": instance,
            "pose_quality_logits": pose_quality_logits,
            "keypoints": keypoints,
            "keypoint_valid_mask": schema_scatter_map.bool().any(dim=1),
            "keypoint_confidence_logits": keypoint_confidence_logits,
            "decoder_keypoints": decoder_keypoints,
            "coarse_keypoints": coarse_keypoints,
            "deform_keypoints": deform_keypoints,
            "refine_keypoints": refine_keypoints,
            "schema_joint_indices": schema_joint_indices,
            "schema_joint_valid": schema_joint_valid,
        }

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
