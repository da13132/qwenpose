from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .schemas import UNION_KEYPOINTS, UNION_SIGMAS

UNION_TO_ID = {name: idx for idx, name in enumerate(UNION_KEYPOINTS)}
BONE_EDGES = tuple(
    (UNION_TO_ID[a], UNION_TO_ID[b])
    for a, b in (
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
        ("left_shoulder", "right_shoulder"),
        ("left_hip", "right_hip"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
    )
)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    wh = (boxes[..., 2:] - boxes[..., :2]).clamp(min=0)
    return wh[..., 0] * wh[..., 1]


def oks_loss(
    pred_keypoints: torch.Tensor,
    gt_keypoints: torch.Tensor,
    valid: torch.Tensor,
    gt_boxes: torch.Tensor,
    sigmas: torch.Tensor,
) -> torch.Tensor:
    if pred_keypoints.numel() == 0:
        return pred_keypoints.sum() * 0.0
    d2 = ((pred_keypoints[..., :2] - gt_keypoints[..., :2]) ** 2).sum(dim=-1)
    areas = box_area(gt_boxes).clamp(min=1e-6)[:, None]
    sigma2 = (sigmas.to(pred_keypoints.device)[None, :] ** 2).clamp(min=1e-6)
    oks = torch.exp(-d2 / (2.0 * areas * sigma2))
    valid_f = valid.float()
    oks_mean = (oks * valid_f).sum(dim=-1) / valid_f.sum(dim=-1).clamp(min=1.0)
    per_instance = -torch.log(oks_mean.clamp(min=1e-4)).clamp(max=10.0)
    return per_instance.sum()


def normalized_coord_loss(
    pred_keypoints: torch.Tensor,
    gt_keypoints: torch.Tensor,
    valid: torch.Tensor,
    gt_boxes: torch.Tensor,
) -> torch.Tensor:
    if pred_keypoints.numel() == 0:
        return pred_keypoints.sum() * 0.0
    scale = box_area(gt_boxes).sqrt().clamp(min=1e-3)[:, None, None]
    err = (pred_keypoints[..., :2] - gt_keypoints[..., :2]) / scale
    loss = F.smooth_l1_loss(err, torch.zeros_like(err), reduction="none").sum(dim=-1)
    valid_f = valid.float()
    per_instance = (loss * valid_f).sum(dim=-1) / valid_f.sum(dim=-1).clamp(min=1.0)
    return per_instance.sum()


def uncertainty_loss(
    pred_keypoints: torch.Tensor,
    pred_sigma: torch.Tensor | None,
    gt_keypoints: torch.Tensor,
    valid: torch.Tensor,
    gt_boxes: torch.Tensor,
) -> torch.Tensor:
    """Conservative box-normalized Laplace NLL for RLE-style uncertainty.

    This keeps the existing sigma head, but makes residuals comparable across
    human scales and avoids clamping the NLL to zero, so sigma can learn a real
    uncertainty trade-off instead of a clipped auxiliary score.
    """
    if pred_sigma is None or pred_keypoints.numel() == 0:
        return pred_keypoints.sum() * 0.0
    wh = (gt_boxes[:, 2:] - gt_boxes[:, :2]).clamp(min=1e-3)[:, None, :]
    residual = (pred_keypoints[..., :2] - gt_keypoints[..., :2]).abs() / wh
    residual = residual.clamp(max=10.0)
    sigma = F.softplus(pred_sigma).clamp(min=1e-4, max=10.0)
    loss = residual / sigma + torch.log(sigma)
    loss = loss.sum(dim=-1)
    valid_f = valid.float()
    per_instance = (loss * valid_f).sum(dim=-1) / valid_f.sum(dim=-1).clamp(min=1.0)
    return per_instance.sum()


def center_aux_loss(center_logits: torch.Tensor | None, targets: list[dict[str, torch.Tensor]]) -> torch.Tensor:
    if center_logits is None:
        return torch.tensor(0.0)
    b, _, h, w = center_logits.shape
    gt = torch.zeros_like(center_logits)
    yy, xx = torch.meshgrid(
        torch.arange(h, device=center_logits.device, dtype=center_logits.dtype),
        torch.arange(w, device=center_logits.device, dtype=center_logits.dtype),
        indexing="ij",
    )
    for batch_idx, target in enumerate(targets):
        boxes = target["boxes"].to(center_logits.device)
        if boxes.numel() == 0:
            continue
        centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5
        wh = (boxes[:, 2:] - boxes[:, :2]).clamp(min=1e-4)
        cx = centers[:, 0] * (w - 1)
        cy = centers[:, 1] * (h - 1)
        sigma = (wh.min(dim=-1).values * min(h, w) * 0.12).clamp(min=1.5, max=6.0)
        for i in range(int(boxes.shape[0])):
            dist2 = (xx - cx[i]) ** 2 + (yy - cy[i]) ** 2
            gt[batch_idx, 0] = torch.maximum(gt[batch_idx, 0], torch.exp(-dist2 / (2.0 * sigma[i] ** 2)))
    prob = center_logits.sigmoid()
    pos_loss = -torch.log(prob.clamp(min=1e-6)) * ((1.0 - prob) ** 2) * gt
    neg_loss = -torch.log((1.0 - prob).clamp(min=1e-6)) * (prob**2) * ((1.0 - gt) ** 4)
    return (pos_loss + neg_loss).mean()


@dataclass
class LossWeights:
    oks: float = 0.5
    coord: float = 3.0
    vis: float = 0.05
    uncertainty: float = 0.02
    aux_center: float = 0.05
    lm: float = 0.1
    hard_joint: float = 0.15
    hard_joint_fraction: float = 0.3
    simcc: float = 0.2
    deep_supervision: float = 0.3
    bone: float = 0.05


def compute_pose_losses(
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    task_ids: torch.Tensor,
    weights: LossWeights,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    clean_outputs = {}
    for key, value in outputs.items():
        if isinstance(value, list):
            clean_outputs[key] = [
                torch.nan_to_num(item.float(), nan=0.0, posinf=1.0, neginf=0.0)
                if torch.is_tensor(item) and torch.is_floating_point(item)
                else item
                for item in value
            ]
        elif torch.is_tensor(value) and torch.is_floating_point(value):
            clean_outputs[key] = torch.nan_to_num(value.float(), nan=0.0, posinf=1.0, neginf=0.0)
        else:
            clean_outputs[key] = value
    outputs = clean_outputs
    graph_terms = []
    for value in outputs.values():
        if torch.is_tensor(value) and torch.is_floating_point(value):
            graph_terms.append(value.sum() * 0.0)
        elif isinstance(value, list):
            graph_terms.extend(
                item.sum() * 0.0
                for item in value
                if torch.is_tensor(item) and torch.is_floating_point(item)
            )
    graph_anchor = sum(graph_terms) if graph_terms else outputs["keypoints"].sum() * 0.0
    if "box_mask" not in outputs:
        raise ValueError("QwenPose losses now require box-conditioned outputs with a box_mask.")
    return compute_box_conditioned_pose_losses(outputs, targets, weights, graph_anchor)


def compute_box_conditioned_pose_losses(
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    weights: LossWeights,
    graph_anchor: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    device = outputs["keypoints"].device
    sigmas = UNION_SIGMAS.to(device)
    box_mask = outputs["box_mask"].to(device).bool()
    total_oks = torch.tensor(0.0, device=device)
    total_coord = torch.tensor(0.0, device=device)
    total_hard_joint = torch.tensor(0.0, device=device)
    total_vis = torch.tensor(0.0, device=device)
    total_unc = torch.tensor(0.0, device=device)
    total_simcc = torch.tensor(0.0, device=device)
    total_deep = torch.tensor(0.0, device=device)
    total_bone = torch.tensor(0.0, device=device)
    total_joint_loss = torch.zeros(len(UNION_KEYPOINTS), device=device)
    total_joint_count = torch.zeros(len(UNION_KEYPOINTS), device=device)
    num_pos = 0
    num_visible_joints = 0
    num_schema_joints = 0
    num_hard_groups = 0
    num_deep_terms = 0

    refine_keypoints = outputs.get("refine_keypoints") or []
    simcc_x = outputs.get("simcc_x_logits")
    simcc_y = outputs.get("simcc_y_logits")
    schema_joint_indices_all = outputs.get("schema_joint_indices")
    schema_joint_valid_all = outputs.get("schema_joint_valid")
    pose_boxes_all = outputs.get("pose_boxes", outputs.get("boxes"))

    for b, target in enumerate(targets):
        valid_queries = torch.nonzero(box_mask[b], as_tuple=False).flatten()
        target_count = int(target["boxes"].shape[0])
        n = min(int(valid_queries.numel()), target_count)
        if n == 0:
            continue
        q = valid_queries[:n]
        gt_boxes = target["boxes"].to(device)[:n]
        gt_keypoints = target["keypoints"].to(device)[:n]
        gt_valid = target["keypoint_valid"].to(device)[:n]
        pred_keypoints = outputs["keypoints"][b, q]
        pred_sigma = outputs.get("keypoint_sigma")
        pred_sigma_b = pred_sigma[b, q] if pred_sigma is not None else None
        schema_valid = outputs["keypoint_valid_mask"][b].to(device).bool().view(1, -1).expand(n, -1)

        coord_joint = per_joint_normalized_coord_loss(pred_keypoints, gt_keypoints, gt_valid, gt_boxes)
        oks_joint = per_joint_oks_loss(pred_keypoints, gt_keypoints, gt_valid, gt_boxes, sigmas)
        visible_f = gt_valid.float()
        total_coord = total_coord + (coord_joint * visible_f).sum()
        total_oks = total_oks + (oks_joint * visible_f).sum()
        hard_source = coord_joint + oks_joint
        visible_hard = hard_source[gt_valid]
        if visible_hard.numel() > 0:
            hard_count = max(1, int(round(float(visible_hard.numel()) * weights.hard_joint_fraction)))
            hard_count = min(hard_count, int(visible_hard.numel()))
            total_hard_joint = total_hard_joint + torch.topk(visible_hard, hard_count).values.mean()
            num_hard_groups += 1
        joint_count = visible_f.sum(dim=0)
        total_joint_loss = total_joint_loss + (coord_joint * visible_f).sum(dim=0)
        total_joint_count = total_joint_count + joint_count
        total_unc = total_unc + uncertainty_loss(pred_keypoints, pred_sigma_b, gt_keypoints, gt_valid, gt_boxes)
        total_bone = total_bone + bone_consistency_loss(pred_keypoints, gt_keypoints, gt_valid, gt_boxes)
        for step_idx, refine_pred in enumerate(refine_keypoints):
            step_weight = min(1.0, 0.3 + 0.2 * float(step_idx))
            refine_pred_b = refine_pred[b, q]
            refine_coord = per_joint_normalized_coord_loss(refine_pred_b, gt_keypoints, gt_valid, gt_boxes)
            refine_oks = per_joint_oks_loss(refine_pred_b, gt_keypoints, gt_valid, gt_boxes, sigmas)
            total_deep = total_deep + step_weight * ((refine_coord + refine_oks) * visible_f).sum() / visible_f.sum().clamp(min=1.0)
            num_deep_terms += 1
        if (
            simcc_x is not None
            and simcc_y is not None
            and schema_joint_indices_all is not None
            and schema_joint_valid_all is not None
            and pose_boxes_all is not None
        ):
            total_simcc = total_simcc + simcc_loss(
                simcc_x[b, q],
                simcc_y[b, q],
                schema_joint_indices_all[b].to(device).long(),
                schema_joint_valid_all[b].to(device).bool(),
                pose_boxes_all[b, q].to(device),
                gt_keypoints,
                gt_valid,
            )
        vis_loss = F.binary_cross_entropy(
            pred_keypoints[..., 2].clamp(1e-6, 1.0 - 1e-6),
            gt_keypoints[..., 2],
            reduction="none",
        )
        schema_valid_f = schema_valid.float()
        total_vis = total_vis + (vis_loss * schema_valid_f).sum()
        num_pos += n
        num_visible_joints += int(gt_valid.sum().item())
        num_schema_joints += int(schema_valid.sum().item())

    denom = max(num_pos, 1)
    aux = center_aux_loss(outputs.get("aux_center_logits"), targets).to(device)
    loss_parts = {
        "loss_oks": total_oks / denom,
        "loss_coord": total_coord / denom,
        "loss_hard_joint": total_hard_joint / max(num_hard_groups, 1),
        "loss_vis": total_vis / denom,
        "loss_uncertainty": total_unc / denom,
        "loss_aux_center": aux,
        "loss_simcc": total_simcc / denom,
        "loss_deep_supervision": total_deep / max(num_deep_terms, 1),
        "loss_bone": total_bone / denom,
    }
    joint_means = total_joint_loss / total_joint_count.clamp(min=1.0)
    observed = total_joint_count > 0
    if observed.any():
        loss_parts["loss_coord_joint_mean"] = joint_means[observed].mean()
        loss_parts["loss_coord_joint_max"] = joint_means[observed].max()
    else:
        loss_parts["loss_coord_joint_mean"] = graph_anchor
        loss_parts["loss_coord_joint_max"] = graph_anchor
    total = (
        weights.oks * loss_parts["loss_oks"]
        + weights.coord * loss_parts["loss_coord"]
        + weights.hard_joint * loss_parts["loss_hard_joint"]
        + weights.vis * loss_parts["loss_vis"]
        + weights.uncertainty * loss_parts["loss_uncertainty"]
        + weights.aux_center * loss_parts["loss_aux_center"]
        + weights.simcc * loss_parts["loss_simcc"]
        + weights.deep_supervision * loss_parts["loss_deep_supervision"]
        + weights.bone * loss_parts["loss_bone"]
        + graph_anchor
    )
    loss_parts["loss_total"] = total
    return total, loss_parts


def per_joint_normalized_coord_loss(
    pred_keypoints: torch.Tensor,
    gt_keypoints: torch.Tensor,
    valid: torch.Tensor,
    gt_boxes: torch.Tensor,
) -> torch.Tensor:
    if pred_keypoints.numel() == 0:
        return pred_keypoints.new_zeros(pred_keypoints.shape[:-1])
    wh = (gt_boxes[:, 2:] - gt_boxes[:, :2]).clamp(min=1e-3)[:, None, :]
    err = (pred_keypoints[..., :2] - gt_keypoints[..., :2]) / wh
    err = err.clamp(-5.0, 5.0)
    loss = F.smooth_l1_loss(err, torch.zeros_like(err), reduction="none").sum(dim=-1)
    return loss.masked_fill(~valid, 0.0)


def per_joint_oks_loss(
    pred_keypoints: torch.Tensor,
    gt_keypoints: torch.Tensor,
    valid: torch.Tensor,
    gt_boxes: torch.Tensor,
    sigmas: torch.Tensor,
) -> torch.Tensor:
    if pred_keypoints.numel() == 0:
        return pred_keypoints.new_zeros(pred_keypoints.shape[:-1])
    d2 = ((pred_keypoints[..., :2] - gt_keypoints[..., :2]) ** 2).sum(dim=-1)
    d2 = d2.clamp(max=1.0)
    areas = box_area(gt_boxes).clamp(min=1e-6)[:, None]
    sigma2 = (sigmas.to(pred_keypoints.device)[None, :] ** 2).clamp(min=1e-6)
    oks = torch.exp(-d2 / (2.0 * areas * sigma2))
    return (1.0 - oks).masked_fill(~valid, 0.0)


def gaussian_soft_labels(position: torch.Tensor, bins: int, sigma: float = 2.0) -> torch.Tensor:
    centers = torch.arange(bins, device=position.device, dtype=position.dtype)
    dist2 = (centers.view(*([1] * position.ndim), bins) - position.unsqueeze(-1)) ** 2
    target = torch.exp(-dist2 / (2.0 * float(sigma) ** 2))
    return target / target.sum(dim=-1, keepdim=True).clamp(min=1e-6)


def simcc_loss(
    logits_x: torch.Tensor,
    logits_y: torch.Tensor,
    schema_joint_indices: torch.Tensor,
    schema_joint_valid: torch.Tensor,
    pose_boxes: torch.Tensor,
    gt_keypoints: torch.Tensor,
    gt_valid: torch.Tensor,
) -> torch.Tensor:
    if logits_x.numel() == 0 or logits_y.numel() == 0:
        return logits_x.sum() * 0.0
    bins = int(logits_x.shape[-1])
    active_k = int(logits_x.shape[-2])
    schema_joint_indices = schema_joint_indices[:active_k]
    schema_joint_valid = schema_joint_valid[:active_k]
    gt_active = gt_keypoints.index_select(1, schema_joint_indices)
    valid_active = gt_valid.index_select(1, schema_joint_indices) & schema_joint_valid.view(1, -1)
    wh = (pose_boxes[:, 2:] - pose_boxes[:, :2]).clamp(min=1e-4)[:, None, :]
    rel = ((gt_active[..., :2] - pose_boxes[:, None, :2]) / wh).clamp(0.0, 1.0)
    pos_x = rel[..., 0] * float(bins - 1)
    pos_y = rel[..., 1] * float(bins - 1)
    target_x = gaussian_soft_labels(pos_x, bins=bins)
    target_y = gaussian_soft_labels(pos_y, bins=bins)
    loss_x = -(target_x * F.log_softmax(logits_x.float(), dim=-1)).sum(dim=-1)
    loss_y = -(target_y * F.log_softmax(logits_y.float(), dim=-1)).sum(dim=-1)
    valid_f = valid_active.float()
    per_instance = ((loss_x + loss_y) * valid_f).sum(dim=-1) / valid_f.sum(dim=-1).clamp(min=1.0)
    return per_instance.sum()


def bone_consistency_loss(
    pred_keypoints: torch.Tensor,
    gt_keypoints: torch.Tensor,
    valid: torch.Tensor,
    gt_boxes: torch.Tensor,
) -> torch.Tensor:
    if pred_keypoints.numel() == 0 or not BONE_EDGES:
        return pred_keypoints.sum() * 0.0
    wh = (gt_boxes[:, 2:] - gt_boxes[:, :2]).clamp(min=1e-3)[:, None, :]
    pred_xy = pred_keypoints[..., :2] / wh
    gt_xy = gt_keypoints[..., :2] / wh
    losses = []
    for a, b in BONE_EDGES:
        edge_valid = valid[:, a] & valid[:, b]
        if not edge_valid.any():
            continue
        pred_vec = pred_xy[:, b] - pred_xy[:, a]
        gt_vec = gt_xy[:, b] - gt_xy[:, a]
        pred_len = pred_vec.norm(dim=-1).clamp(min=1e-6)
        gt_len = gt_vec.norm(dim=-1).clamp(min=1e-6)
        dir_loss = 1.0 - F.cosine_similarity(pred_vec, gt_vec, dim=-1).clamp(-1.0, 1.0)
        len_loss = F.smooth_l1_loss(
            torch.log(pred_len / gt_len).clamp(-5.0, 5.0),
            torch.zeros_like(pred_len),
            reduction="none",
        )
        edge_loss = (dir_loss + 0.2 * len_loss).masked_fill(~edge_valid, 0.0)
        losses.append(edge_loss.sum() / edge_valid.float().sum().clamp(min=1.0))
    if not losses:
        return pred_keypoints.sum() * 0.0
    return torch.stack(losses).mean()
