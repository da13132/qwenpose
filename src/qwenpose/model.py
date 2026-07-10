from __future__ import annotations

from dataclasses import dataclass
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
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        if hidden_dim % 4 != 0:
            raise ValueError("hidden_dim must be divisible by 4 for 2D sine PE.")
        self.hidden_dim = hidden_dim

    def forward(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        y, x = torch.meshgrid(
            torch.linspace(0, 1, height, device=device),
            torch.linspace(0, 1, width, device=device),
            indexing="ij",
        )
        omega = torch.arange(self.hidden_dim // 4, device=device, dtype=torch.float32)
        omega = 1.0 / (10000 ** (omega / max(len(omega), 1)))
        pe = torch.cat(
            [
                torch.sin(x[..., None] * omega),
                torch.cos(x[..., None] * omega),
                torch.sin(y[..., None] * omega),
                torch.cos(y[..., None] * omega),
            ],
            dim=-1,
        )
        return pe.view(height * width, self.hidden_dim)


def _group_count(hidden_dim: int, max_groups: int = 32) -> int:
    for groups in range(min(max_groups, hidden_dim), 0, -1):
        if hidden_dim % groups == 0:
            return groups
    return 1


class SpatialFeatureInjector(nn.Module):
    """Near-identity spatial adapter, following Qwen3-VL-Seg's stable injection idea."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(_group_count(hidden_dim), hidden_dim)
        self.depthwise = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim)
        self.scale = nn.Parameter(torch.tensor(1e-3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale.to(dtype=x.dtype) * F.gelu(self.depthwise(self.norm(x)))


class JointDeformableKeypointAttention(nn.Module):
    """Joint-centric sparse feature sampling after the person-level ROI decoder."""

    def __init__(
        self,
        hidden_dim: int,
        num_scales: int = 2,
        num_points: int = 4,
        offset_scale: float = 0.35,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_scales = max(int(num_scales), 1)
        self.num_points = max(int(num_points), 1)
        self.offset_scale = float(offset_scale)
        sample_count = self.num_scales * self.num_points
        self.offset_head = nn.Linear(hidden_dim, sample_count * 2)
        self.weight_head = nn.Linear(hidden_dim, sample_count)
        self.context_proj = MLP(hidden_dim * 3, hidden_dim, hidden_dim, depth=2)
        self.scale = nn.Parameter(torch.tensor(-2.0))
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)
        nn.init.zeros_(self.weight_head.weight)
        nn.init.zeros_(self.weight_head.bias)
        self._zero_init_last_linear(self.context_proj)

    def forward(
        self,
        tokens: torch.Tensor,
        reference_xy: torch.Tensor,
        box_wh: torch.Tensor,
        feature_maps: list[torch.Tensor],
    ) -> torch.Tensor:
        if tokens.numel() == 0 or not feature_maps:
            return tokens
        maps = list(feature_maps[: self.num_scales])
        while len(maps) < self.num_scales:
            maps.append(maps[-1])
        b, q, k, c = tokens.shape
        sample_count = self.num_scales * self.num_points
        token_input = tokens.to(dtype=self.offset_head.weight.dtype)
        offsets = torch.tanh(self.offset_head(token_input).float()).to(dtype=tokens.dtype)
        offsets = offsets.view(b, q, k, self.num_scales, self.num_points, 2)
        weights = self.weight_head(token_input).float().softmax(dim=-1).to(dtype=tokens.dtype)
        weights = weights.view(b, q, k, self.num_scales, self.num_points)
        radius = box_wh.to(dtype=tokens.dtype).view(b, q, 1, 1, 1, 2) * self.offset_scale
        points = reference_xy.to(dtype=tokens.dtype).view(b, q, k, 1, 1, 2) + offsets * radius
        points = points.clamp(0.0, 1.0)

        sampled_scales = []
        for scale_idx, feature_map in enumerate(maps):
            sampled = self._sample_points(feature_map, points[:, :, :, scale_idx])
            sampled_scales.append(sampled)
        sampled_all = torch.stack(sampled_scales, dim=3).reshape(b, q, k, sample_count, c)
        sampled = (sampled_all * weights.reshape(b, q, k, sample_count, 1)).sum(dim=3)
        point_pe = point_fourier_pe(reference_xy.to(dtype=tokens.dtype), c)
        update = self.context_proj(torch.cat([tokens, sampled, point_pe], dim=-1))
        return tokens + self.scale.sigmoid().to(dtype=update.dtype) * update

    @staticmethod
    def _sample_points(feature_map: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        b, channels, _, _ = feature_map.shape
        q, k, p = points.shape[1], points.shape[2], points.shape[3]
        grid = points.to(device=feature_map.device, dtype=feature_map.dtype) * 2.0 - 1.0
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


class ConvNormAct(nn.Module):
    """Lightweight RGB stem block used by the trainable visual pose branch."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.norm = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


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
    pose_decoder_layers: int = 1
    refinement_steps: int = 3
    decoder_heads: int = 8
    box_condition_scale: float = 1.2
    pose_roi_size: int = 16
    use_refinement: bool = True
    simcc_bins: int = 256
    rgb_low_gate_expand_ratio: float = 0.10
    rgb_refine_gate_expand_ratio: float = 0.15
    rgb_gate_background: float = 0.05
    schema_joint_priors_path: str | None = "configs/schema_joint_priors.json"


class QwenPoseModel(nn.Module):
    def __init__(self, config: QwenPoseConfig) -> None:
        super().__init__()
        self.config = config
        c = config.hidden_dim
        self.external_feature_proj = nn.Conv2d(config.external_dim, c, 1)
        self.spatial_injector = SpatialFeatureInjector(c)
        self.external_text_proj = nn.Sequential(
            nn.LayerNorm(config.external_dim),
            nn.Linear(config.external_dim, c),
            nn.GELU(),
        )
        rgb_c1 = max(c // 8, 16)
        rgb_c2 = max(c // 4, 32)
        rgb_c3 = max(c // 2, 64)
        self.rgb_stem = nn.Sequential(
            ConvNormAct(3, rgb_c1, stride=2),
            ConvNormAct(rgb_c1, rgb_c2, stride=2),
            ConvNormAct(rgb_c2, rgb_c3, stride=2),
            ConvNormAct(rgb_c3, c, stride=1),
        )
        self.rgb_low_fuse = nn.Sequential(
            nn.Conv2d(c * 2, c, 1),
            nn.GELU(),
            nn.Conv2d(c, c, 3, padding=1),
        )
        self.rgb_refine_fuse = nn.Sequential(
            nn.Conv2d(c * 2, c, 1),
            nn.GELU(),
            nn.Conv2d(c, c, 3, padding=1),
        )
        # Start close to the pure-Qwen path, then let training learn how much RGB
        # local detail to inject.
        self.rgb_low_scale = nn.Parameter(torch.tensor(-3.0))
        self.rgb_refine_scale = nn.Parameter(torch.tensor(-3.0))
        self._zero_init_last_conv(self.rgb_low_fuse)
        self._zero_init_last_conv(self.rgb_refine_fuse)
        self.pos_encoding = SinePositionEncoding(c)

        # Schema identity controls only the active joint set and geometric prior.
        # Keeping a learnable schema embedding here would let shared joints route
        # into separate dataset-specific predictors, weakening cross-dataset transfer.
        self.schema_embed = nn.Embedding(len(SCHEMA_NAMES), c)
        nn.init.zeros_(self.schema_embed.weight)
        self.schema_embed.weight.requires_grad_(False)
        self.task_embed = nn.Embedding(2, c)
        self.joint_embed = nn.Embedding(len(UNION_KEYPOINTS), c)
        self.register_buffer(
            "schema_joint_priors",
            build_schema_joint_priors(config.schema_joint_priors_path),
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
        same_joint_layer = nn.TransformerEncoderLayer(
            d_model=c,
            nhead=config.decoder_heads,
            dim_feedforward=c * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.same_joint_context = nn.TransformerEncoder(same_joint_layer, num_layers=1)
        self.instance_query_norm = nn.LayerNorm(c)
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
        pose_decoder_layer = nn.TransformerDecoderLayer(
            d_model=c,
            nhead=config.decoder_heads,
            dim_feedforward=c * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.pose_decoder = nn.TransformerDecoder(pose_decoder_layer, num_layers=config.pose_decoder_layers)
        self.deformable_joint_attention = JointDeformableKeypointAttention(
            c,
            num_scales=2,
            num_points=4,
            offset_scale=0.35,
        )
        self.coarse_xy_head = MLP(c, c, 2, depth=3)
        self.pose_xy_head = MLP(c, c, 2, depth=3)
        self.pose_vis_head = MLP(c, c, 1, depth=3)
        self.simcc_bins = max(int(config.simcc_bins), 0)
        if self.simcc_bins > 1:
            self.simcc_x_head = nn.Linear(c, self.simcc_bins)
            self.simcc_y_head = nn.Linear(c, self.simcc_bins)
        else:
            self.simcc_x_head = None
            self.simcc_y_head = None
        if config.use_refinement:
            self.local_proj = nn.Conv2d(c, c, 1)
            joint_context_layer = nn.TransformerEncoderLayer(
                d_model=c,
                nhead=config.decoder_heads,
                dim_feedforward=c * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
            )
            self.joint_context = nn.TransformerEncoder(joint_context_layer, num_layers=1)
            refinement_steps = max(int(config.refinement_steps), 1)
            refine_patch_points = 9
            self.refine_heads = nn.ModuleList([MLP(c * 3, c, 2, depth=3) for _ in range(refinement_steps)])
            self.refine_token_fusers = nn.ModuleList([MLP(c * 3, c, c, depth=2) for _ in range(refinement_steps)])
            self.refine_patch_weight_heads = nn.ModuleList(
                [nn.Linear(c, refine_patch_points) for _ in range(refinement_steps)]
            )
            self.refine_step_scales = nn.Parameter(torch.full((refinement_steps,), -1.4))
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

    def forward(
        self,
        schema_ids: torch.Tensor,
        task_ids: torch.Tensor,
        images: torch.Tensor | None = None,
        external_feature_map: torch.Tensor | None = None,
        external_text_embed: torch.Tensor | None = None,
        target_boxes: torch.Tensor | None = None,
        target_box_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if external_feature_map is None or external_text_embed is None:
            raise ValueError("QwenPoseModel now requires Qwen3-VL external_feature_map and external_text_embed.")
        if target_boxes is None or target_box_mask is None:
            raise ValueError("QwenPoseModel requires box conditions from the LLM/teacher-forced targets.")
        target_boxes = target_boxes.to(device=external_feature_map.device).clamp(0.0, 1.0)
        target_box_mask = target_box_mask.to(device=external_feature_map.device).bool()
        pose_boxes = expand_boxes_xyxy(target_boxes, self.config.box_condition_scale)
        dtype = self.external_feature_proj.weight.dtype
        feature_map = self.external_feature_proj(external_feature_map.to(dtype=dtype))
        feature_map = self.spatial_injector(feature_map)
        refine_feature_map = feature_map
        if images is not None and images.shape[-2] > 1 and images.shape[-1] > 1:
            rgb_feature = self.rgb_stem(images.to(device=feature_map.device, dtype=feature_map.dtype))
            rgb_low = F.interpolate(
                rgb_feature,
                size=feature_map.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            low_gate = self._rgb_box_gate(
                pose_boxes.to(dtype=feature_map.dtype),
                target_box_mask,
                feature_map.shape[-2],
                feature_map.shape[-1],
                expand_ratio=self.config.rgb_low_gate_expand_ratio,
            )
            rgb_low = rgb_low * low_gate
            low_delta = self.rgb_low_fuse(torch.cat([feature_map, rgb_low], dim=1))
            feature_map = feature_map + self.rgb_low_scale.sigmoid().to(dtype=feature_map.dtype) * low_delta

            qwen_high = F.interpolate(
                feature_map,
                size=rgb_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            high_gate = self._rgb_box_gate(
                pose_boxes.to(dtype=rgb_feature.dtype),
                target_box_mask,
                rgb_feature.shape[-2],
                rgb_feature.shape[-1],
                expand_ratio=self.config.rgb_refine_gate_expand_ratio,
            )
            rgb_refine = rgb_feature * high_gate
            refine_delta = self.rgb_refine_fuse(torch.cat([qwen_high, rgb_refine], dim=1))
            refine_feature_map = qwen_high + self.rgb_refine_scale.sigmoid().to(dtype=qwen_high.dtype) * refine_delta
        image_embed = feature_map.mean(dim=(2, 3))
        text_dtype = next(self.external_text_proj.parameters()).dtype
        text_embed = self.external_text_proj(external_text_embed.to(dtype=text_dtype))
        b, c, h, w = feature_map.shape
        ref_task_gate = task_ids.eq(1).to(device=text_embed.device, dtype=text_embed.dtype)

        input_boxes = target_boxes.to(dtype=feature_map.dtype)
        boxes = pose_boxes.to(dtype=feature_map.dtype)
        box_mask = target_box_mask
        num_boxes = boxes.shape[1]
        box_embed = self.box_query_proj(box_fourier_pe(boxes, c))
        roi_size = max(int(self.config.pose_roi_size), 2)
        roi_features = self._sample_box_feature_maps(feature_map, boxes, roi_size)
        roi_features = roi_features * box_mask.view(b, num_boxes, 1, 1, 1).to(dtype=roi_features.dtype)
        roi_pooled = roi_features.mean(dim=(-2, -1))
        roi_embed = self.roi_pool_proj(roi_pooled)
        instance_text = 0.2 * ref_task_gate[:, None, None] * text_embed[:, None, :]
        instance = self.instance_query_norm(
            box_embed + roi_embed + image_embed[:, None, :] + instance_text
        )
        person_logits = torch.where(
            box_mask,
            torch.full_like(boxes[..., 0], 10.0),
            torch.full_like(boxes[..., 0], -10.0),
        )
        ref_logits = person_logits

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
        schema_prior_all = self.schema_joint_priors.to(
            device=feature_map.device, dtype=feature_map.dtype
        )[schema_ids]
        joint_prior = torch.gather(
            schema_prior_all,
            dim=1,
            index=schema_joint_indices[..., None].expand(-1, -1, 2),
        )
        joint_prior_pe = point_fourier_pe(joint_prior, c).view(b, 1, active_k, c)
        task = self.task_embed(task_ids).view(b, 1, 1, c)
        box_pe = box_embed.view(b, num_boxes, 1, c)
        text_condition = (
            0.2
            * ref_task_gate.view(b, 1, 1, 1)
            * text_embed.view(b, 1, 1, c)
        )
        pose_tokens = self.pose_query_norm(
            instance[:, :, None, :]
            + joint_base
            + joint_prior_pe
            + task
            + box_pe
            + text_condition
        )
        pose_valid = schema_joint_valid[:, None, :].expand(b, num_boxes, active_k) & box_mask[:, :, None]
        same_joint_tokens = pose_tokens.permute(0, 2, 1, 3).reshape(b * active_k, num_boxes, c)
        same_joint_valid = (
            schema_joint_valid[:, :, None].expand(b, active_k, num_boxes) & box_mask[:, None, :]
        ).reshape(b * active_k, num_boxes)
        same_joint_padding_mask = ~same_joint_valid
        same_joint_padding_mask = same_joint_padding_mask.masked_fill(
            same_joint_padding_mask.all(dim=1, keepdim=True),
            False,
        )
        same_joint_tokens = self.same_joint_context(
            same_joint_tokens,
            src_key_padding_mask=same_joint_padding_mask,
        )
        pose_tokens = same_joint_tokens.view(b, active_k, num_boxes, c).permute(0, 2, 1, 3)

        roi_memory = roi_features.flatten(3).permute(0, 1, 3, 2)
        roi_pe = self.pos_encoding(roi_size, roi_size, feature_map.device).to(dtype=roi_memory.dtype)
        roi_memory = self.roi_memory_norm(roi_memory + roi_pe.view(1, 1, roi_size * roi_size, c))
        roi_memory = roi_memory.reshape(b * num_boxes, roi_size * roi_size, c)
        pose_tokens = pose_tokens.reshape(b * num_boxes, active_k, c)
        pose_padding_mask = ~pose_valid.reshape(b * num_boxes, active_k)
        pose_padding_mask = pose_padding_mask.masked_fill(pose_padding_mask.all(dim=1, keepdim=True), False)
        pose_tokens = self.pose_decoder(
            pose_tokens,
            roi_memory,
            tgt_key_padding_mask=pose_padding_mask,
        )
        pose_tokens = pose_tokens.view(b, num_boxes, active_k, c)

        wh = (boxes[..., 2:] - boxes[..., :2]).clamp(min=1e-4)
        prior_logits = torch.logit(joint_prior.clamp(1e-4, 1.0 - 1e-4)).view(b, 1, active_k, 2)
        coarse_tokens = pose_tokens
        coarse_rel_xy = torch.sigmoid(prior_logits + self.coarse_xy_head(coarse_tokens))
        coarse_reference_xy = boxes[..., None, :2] + coarse_rel_xy * wh[..., None, :]
        coarse_reference_xy = coarse_reference_xy.clamp(0.0, 1.0)
        simcc_coarse_x, simcc_coarse_y = self._simcc_logits(coarse_tokens)

        pose_tokens = self.deformable_joint_attention(
            coarse_tokens,
            coarse_reference_xy,
            wh,
            [feature_map, refine_feature_map],
        )

        deform_tokens = pose_tokens
        rel_xy = torch.sigmoid(prior_logits + self.pose_xy_head(pose_tokens))
        keypoint_xy = boxes[..., None, :2] + rel_xy * wh[..., None, :]
        keypoint_xy = keypoint_xy.clamp(0.0, 1.0)
        deform_keypoint_xy = keypoint_xy
        simcc_deform_x, simcc_deform_y = self._simcc_logits(deform_tokens)

        refine_keypoint_xy_steps: list[torch.Tensor] = []
        refine_simcc_x_steps: list[torch.Tensor] = []
        refine_simcc_y_steps: list[torch.Tensor] = []
        if (
            self.refine_heads is not None
            and self.refine_token_fusers is not None
            and self.refine_patch_weight_heads is not None
            and self.joint_context is not None
            and self.local_proj is not None
        ):
            local_feature_map = self.local_proj(refine_feature_map)
            context_mask = ~pose_valid.reshape(b * num_boxes, active_k)
            context_mask = context_mask.masked_fill(context_mask.all(dim=1, keepdim=True), False)
            for refine_idx, (refine_head, token_fuser, patch_weight_head) in enumerate(
                zip(self.refine_heads, self.refine_token_fusers, self.refine_patch_weight_heads)
            ):
                pose_tokens = self.joint_context(
                    pose_tokens.reshape(b * num_boxes, active_k, c),
                    src_key_padding_mask=context_mask,
                ).reshape(b, num_boxes, active_k, c)
                patch_logits = patch_weight_head(pose_tokens.to(dtype=patch_weight_head.weight.dtype))
                local = self._sample_local_patch_features(
                    local_feature_map,
                    keypoint_xy,
                    wh,
                    patch_logits,
                    patch_size=3,
                    radius_scale=0.12,
                )
                point_pe = point_fourier_pe(keypoint_xy, c)
                refine_input = torch.cat([pose_tokens, local, point_pe], dim=-1)
                delta = torch.tanh(refine_head(refine_input))
                scale = self.refine_step_scales[refine_idx].sigmoid().to(dtype=delta.dtype) * 0.35
                keypoint_xy = (keypoint_xy + delta * wh[..., None, :] * scale).clamp(0.0, 1.0)
                refine_keypoint_xy_steps.append(keypoint_xy)
                pose_tokens = pose_tokens + token_fuser(refine_input)
                simcc_refine_x, simcc_refine_y = self._simcc_logits(pose_tokens)
                if simcc_refine_x is not None and simcc_refine_y is not None:
                    refine_simcc_x_steps.append(simcc_refine_x)
                    refine_simcc_y_steps.append(simcc_refine_y)

        keypoint_vis = self.pose_vis_head(pose_tokens).sigmoid()
        aux_vis = keypoint_vis.detach()
        coarse_keypoints = self._scatter_schema_keypoints(
            torch.cat([coarse_reference_xy, aux_vis], dim=-1),
            schema_scatter_map,
        )
        deform_keypoints = self._scatter_schema_keypoints(
            torch.cat([deform_keypoint_xy, aux_vis], dim=-1),
            schema_scatter_map,
        )
        schema_keypoints = torch.cat([keypoint_xy, keypoint_vis], dim=-1)
        keypoints = self._scatter_schema_keypoints(schema_keypoints, schema_scatter_map)
        keypoint_valid_mask = schema_scatter_map.bool().any(dim=1)
        outputs = {
            "person_logits": person_logits,
            "boxes": input_boxes,
            "pose_boxes": boxes,
            "box_mask": box_mask,
            "keypoints": keypoints,
            "keypoint_valid_mask": keypoint_valid_mask,
            "coarse_keypoints": coarse_keypoints,
            "deform_keypoints": deform_keypoints,
            "ref_logits": ref_logits,
            "instance_emb": instance,
            "schema_joint_indices": schema_joint_indices,
            "schema_joint_valid": schema_joint_valid,
        }
        if simcc_coarse_x is not None and simcc_coarse_y is not None:
            outputs["simcc_coarse_x"] = simcc_coarse_x
            outputs["simcc_coarse_y"] = simcc_coarse_y
        if simcc_deform_x is not None and simcc_deform_y is not None:
            outputs["simcc_deform_x"] = simcc_deform_x
            outputs["simcc_deform_y"] = simcc_deform_y
        if refine_keypoint_xy_steps:
            outputs["refine_keypoints"] = [
                self._scatter_schema_keypoints(torch.cat([xy, aux_vis], dim=-1), schema_scatter_map)
                for xy in refine_keypoint_xy_steps
            ]
        if refine_simcc_x_steps and refine_simcc_y_steps:
            outputs["simcc_refine_x"] = refine_simcc_x_steps
            outputs["simcc_refine_y"] = refine_simcc_y_steps
        return outputs

    def _rgb_box_gate(
        self,
        boxes: torch.Tensor,
        box_mask: torch.Tensor,
        height: int,
        width: int,
        expand_ratio: float,
    ) -> torch.Tensor:
        gate = box_soft_gate(boxes, box_mask, height, width, expand_ratio=expand_ratio)
        background = min(max(float(self.config.rgb_gate_background), 0.0), 1.0)
        if background > 0.0:
            gate = background + (1.0 - background) * gate
        return gate

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

    def _simcc_logits(self, tokens: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.simcc_x_head is None or self.simcc_y_head is None:
            return None, None
        token_input = tokens.to(dtype=self.simcc_x_head.weight.dtype)
        return self.simcc_x_head(token_input), self.simcc_y_head(token_input)

    @staticmethod
    def _sample_local_features(feature_map: torch.Tensor, keypoint_xy: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = feature_map.shape
        q, u = keypoint_xy.shape[1], keypoint_xy.shape[2]
        grid = keypoint_xy * 2.0 - 1.0
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
        *,
        patch_size: int = 3,
        radius_scale: float = 0.08,
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
        radius = box_wh.to(device=feature_map.device, dtype=feature_map.dtype).view(b, q, 1, 1, 2)
        radius = radius * float(radius_scale)
        points = keypoint_xy.to(device=feature_map.device, dtype=feature_map.dtype).unsqueeze(3) + offsets * radius
        points = points.clamp(0.0, 1.0)
        grid = points.view(b, q * u * patch_size * patch_size, 1, 2) * 2.0 - 1.0
        sampled = F.grid_sample(feature_map, grid, align_corners=False)
        sampled = sampled.squeeze(-1).transpose(1, 2).view(b, q, u, patch_size * patch_size, c)
        if patch_logits is None or patch_logits.shape[-1] != patch_size * patch_size:
            weights = sampled.new_full((b, q, u, patch_size * patch_size), 1.0 / float(patch_size * patch_size))
        else:
            weights = patch_logits.to(device=feature_map.device).float().softmax(dim=-1).to(dtype=sampled.dtype)
        return (sampled * weights.unsqueeze(-1)).sum(dim=3)

    @staticmethod
    def _sample_box_feature_maps(feature_map: torch.Tensor, boxes: torch.Tensor, roi_size: int) -> torch.Tensor:
        b, c, _, _ = feature_map.shape
        num_boxes = boxes.shape[1]
        if num_boxes == 0:
            return feature_map.new_zeros(b, 0, c, roi_size, roi_size)
        if torchvision_roi_align is not None:
            feature_h, feature_w = feature_map.shape[-2:]
            flat_boxes = boxes.to(dtype=feature_map.dtype).reshape(b * num_boxes, 4)
            scales = flat_boxes.new_tensor([feature_w, feature_h, feature_w, feature_h])
            flat_boxes = flat_boxes * scales
            # Keep zero-padded/degenerate boxes numerically safe; the caller will
            # mask them out afterwards via box_mask.
            flat_boxes[:, 2] = torch.maximum(flat_boxes[:, 2], flat_boxes[:, 0] + 1e-4)
            flat_boxes[:, 3] = torch.maximum(flat_boxes[:, 3], flat_boxes[:, 1] + 1e-4)
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
        grid = (xy1 + base * wh).clamp(0.0, 1.0) * 2.0 - 1.0
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


def count_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
