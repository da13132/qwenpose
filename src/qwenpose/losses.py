from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .schemas import UNION_KEYPOINTS, UNION_SIGMAS


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


@dataclass
class LossWeights:
    # Main pose supervision.
    oks: float = 0.2
    coord: float = 5.0
    vis: float = 0.05
    # Stage2 LM bbox supervision. Stage1 should keep this at 0.
    lm: float = 0.05
    # Experimental hard-joint mining. Kept for later ablation, default off.
    hard_joint: float = 0.0
    hard_joint_fraction: float = 0.2


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
        raise ValueError("QwenPose losses require box-conditioned outputs with a box_mask.")
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
    total_joint_loss = torch.zeros(len(UNION_KEYPOINTS), device=device)
    total_joint_count = torch.zeros(len(UNION_KEYPOINTS), device=device)
    num_pos = 0
    num_hard_groups = 0

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
        schema_valid = outputs["keypoint_valid_mask"][b].to(device).bool().view(1, -1).expand(n, -1)

        coord_joint = per_joint_normalized_coord_loss(pred_keypoints, gt_keypoints, gt_valid, gt_boxes)
        oks_joint = per_joint_oks_loss(pred_keypoints, gt_keypoints, gt_valid, gt_boxes, sigmas)
        visible_f = gt_valid.float()
        total_coord = total_coord + (coord_joint * visible_f).sum()
        total_oks = total_oks + (oks_joint * visible_f).sum()

        if weights.hard_joint > 0.0:
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

        vis_loss = F.binary_cross_entropy(
            pred_keypoints[..., 2].clamp(1e-6, 1.0 - 1e-6),
            gt_keypoints[..., 2],
            reduction="none",
        )
        schema_valid_f = schema_valid.float()
        total_vis = total_vis + (vis_loss * schema_valid_f).sum()
        num_pos += n

    denom = max(num_pos, 1)
    loss_parts = {
        "loss_oks": total_oks / denom,
        "loss_coord": total_coord / denom,
        "loss_hard_joint": total_hard_joint / max(num_hard_groups, 1),
        "loss_vis": total_vis / denom,
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
