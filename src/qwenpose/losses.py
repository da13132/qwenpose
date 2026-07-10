from __future__ import annotations

from dataclasses import dataclass
import math

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
    """Legacy aggregate OKS loss retained for external callers."""
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
    """Legacy aggregate coordinate loss retained for external callers."""
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
    oks: float = 0.5
    coord: float = 3.0
    image_coord: float = 5.0
    vis: float = 0.05
    # Stage2 LM bbox supervision. Stage1 should keep this at 0.
    lm: float = 0.05
    # Experimental hard-joint mining. Kept for later ablation, default off.
    hard_joint: float = 0.0
    hard_joint_fraction: float = 0.2
    # Auxiliary supervision defaults stay off for backwards-compatible direct
    # LossWeights construction; train_pose enables the full recipe explicitly.
    coarse_coord: float = 0.0
    deform_coord: float = 0.0
    refine_coords: tuple[float, ...] = ()
    simcc_coarse: float = 0.0
    simcc_deform: float = 0.0
    simcc_refine: tuple[float, ...] = ()
    simcc_sigma: float = 2.0


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


def _mean_valid_joints(per_joint: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    valid_f = valid.float()
    return (per_joint * valid_f).sum(dim=-1) / valid_f.sum(dim=-1).clamp(min=1.0)


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
    total_image_coord = torch.tensor(0.0, device=device)
    total_hard_joint = torch.tensor(0.0, device=device)
    total_vis = torch.tensor(0.0, device=device)
    total_coarse_coord = torch.tensor(0.0, device=device)
    total_deform_coord = torch.tensor(0.0, device=device)
    total_simcc_coarse = torch.tensor(0.0, device=device)
    total_simcc_deform = torch.tensor(0.0, device=device)
    total_joint_loss = torch.zeros(len(UNION_KEYPOINTS), device=device)
    total_joint_count = torch.zeros(len(UNION_KEYPOINTS), device=device)

    refine_keypoints = outputs.get("refine_keypoints", [])
    if not isinstance(refine_keypoints, list):
        refine_keypoints = []
    total_refine_coord = [torch.tensor(0.0, device=device) for _ in refine_keypoints]

    simcc_refine_x = outputs.get("simcc_refine_x", [])
    simcc_refine_y = outputs.get("simcc_refine_y", [])
    if not isinstance(simcc_refine_x, list) or not isinstance(simcc_refine_y, list):
        simcc_refine_x, simcc_refine_y = [], []
    simcc_refine_count = min(len(simcc_refine_x), len(simcc_refine_y))
    total_simcc_refine = [torch.tensor(0.0, device=device) for _ in range(simcc_refine_count)]

    num_pos = 0
    num_vis_instances = 0
    num_hard_groups = 0

    for b, target in enumerate(targets):
        valid_queries = torch.nonzero(box_mask[b], as_tuple=False).flatten()
        target_count = int(target["boxes"].shape[0])
        n = min(int(valid_queries.numel()), target_count)
        if n == 0:
            continue

        q = valid_queries[:n]
        gt_boxes = target["boxes"].to(device)[:n]
        loss_boxes = target.get("loss_boxes", target["boxes"]).to(device)[:n]
        default_areas = box_area(loss_boxes)
        loss_areas = target.get("loss_areas", default_areas).to(device)[:n].clamp(min=1e-8)
        gt_keypoints = target["keypoints"].to(device)[:n]
        gt_valid = target["keypoint_valid"].to(device)[:n].bool()
        gt_visibility_valid = target.get("visibility_valid", target["keypoint_valid"]).to(device)[:n].bool()
        pred_keypoints = outputs["keypoints"][b, q]
        schema_valid = outputs["keypoint_valid_mask"][b].to(device).bool().view(1, -1).expand(n, -1)

        coord_joint = per_joint_normalized_coord_loss(
            pred_keypoints, gt_keypoints, gt_valid, loss_boxes
        )
        image_coord_joint = per_joint_image_coord_loss(pred_keypoints, gt_keypoints, gt_valid)
        oks_joint = per_joint_oks_loss(
            pred_keypoints, gt_keypoints, gt_valid, loss_areas, sigmas
        )
        total_coord = total_coord + _mean_valid_joints(coord_joint, gt_valid).sum()
        total_image_coord = total_image_coord + _mean_valid_joints(image_coord_joint, gt_valid).sum()
        total_oks = total_oks + _mean_valid_joints(oks_joint, gt_valid).sum()

        if "coarse_keypoints" in outputs:
            coarse_joint = per_joint_normalized_coord_loss(
                outputs["coarse_keypoints"][b, q], gt_keypoints, gt_valid, loss_boxes
            )
            total_coarse_coord = total_coarse_coord + _mean_valid_joints(coarse_joint, gt_valid).sum()
        if "deform_keypoints" in outputs:
            deform_joint = per_joint_normalized_coord_loss(
                outputs["deform_keypoints"][b, q], gt_keypoints, gt_valid, loss_boxes
            )
            total_deform_coord = total_deform_coord + _mean_valid_joints(deform_joint, gt_valid).sum()
        for refine_idx, refine_pred in enumerate(refine_keypoints):
            refine_joint = per_joint_normalized_coord_loss(
                refine_pred[b, q], gt_keypoints, gt_valid, loss_boxes
            )
            total_refine_coord[refine_idx] = (
                total_refine_coord[refine_idx]
                + _mean_valid_joints(refine_joint, gt_valid).sum()
            )

        if "simcc_coarse_x" in outputs and "simcc_coarse_y" in outputs:
            total_simcc_coarse = total_simcc_coarse + simcc_box_loss(
                outputs["simcc_coarse_x"][b, q],
                outputs["simcc_coarse_y"][b, q],
                gt_keypoints,
                gt_valid,
                outputs["pose_boxes"][b, q],
                outputs["schema_joint_indices"][b],
                outputs["schema_joint_valid"][b],
                sigma=weights.simcc_sigma,
            )
        if "simcc_deform_x" in outputs and "simcc_deform_y" in outputs:
            total_simcc_deform = total_simcc_deform + simcc_box_loss(
                outputs["simcc_deform_x"][b, q],
                outputs["simcc_deform_y"][b, q],
                gt_keypoints,
                gt_valid,
                outputs["pose_boxes"][b, q],
                outputs["schema_joint_indices"][b],
                outputs["schema_joint_valid"][b],
                sigma=weights.simcc_sigma,
            )
        for refine_idx in range(simcc_refine_count):
            total_simcc_refine[refine_idx] = total_simcc_refine[refine_idx] + simcc_box_loss(
                simcc_refine_x[refine_idx][b, q],
                simcc_refine_y[refine_idx][b, q],
                gt_keypoints,
                gt_valid,
                outputs["pose_boxes"][b, q],
                outputs["schema_joint_indices"][b],
                outputs["schema_joint_valid"][b],
                sigma=weights.simcc_sigma,
            )

        if weights.hard_joint > 0.0:
            hard_source = coord_joint + oks_joint
            visible_hard = hard_source[gt_valid]
            if visible_hard.numel() > 0:
                hard_count = max(1, int(round(float(visible_hard.numel()) * weights.hard_joint_fraction)))
                hard_count = min(hard_count, int(visible_hard.numel()))
                total_hard_joint = total_hard_joint + torch.topk(visible_hard, hard_count).values.mean()
                num_hard_groups += 1

        valid_f = gt_valid.float()
        total_joint_loss = total_joint_loss + (coord_joint * valid_f).sum(dim=0)
        total_joint_count = total_joint_count + valid_f.sum(dim=0)

        vis_loss = F.binary_cross_entropy(
            pred_keypoints[..., 2].clamp(1e-6, 1.0 - 1e-6),
            gt_keypoints[..., 2],
            reduction="none",
        )
        visibility_mask = schema_valid & gt_visibility_valid
        has_visibility = visibility_mask.any(dim=-1)
        if has_visibility.any():
            vis_per_instance = _mean_valid_joints(vis_loss, visibility_mask)
            total_vis = total_vis + vis_per_instance[has_visibility].sum()
            num_vis_instances += int(has_visibility.sum().item())
        num_pos += n

    denom = max(num_pos, 1)
    loss_parts = {
        "loss_oks": total_oks / denom,
        "loss_coord": total_coord / denom,
        "loss_image_coord": total_image_coord / denom,
        "loss_hard_joint": total_hard_joint / max(num_hard_groups, 1),
        "loss_vis": total_vis / max(num_vis_instances, 1),
    }
    if "coarse_keypoints" in outputs:
        loss_parts["loss_coord_coarse"] = total_coarse_coord / denom
    if "deform_keypoints" in outputs:
        loss_parts["loss_coord_deform"] = total_deform_coord / denom
    for refine_idx, total_refine in enumerate(total_refine_coord, start=1):
        loss_parts[f"loss_coord_refine_{refine_idx}"] = total_refine / denom
    if "simcc_coarse_x" in outputs and "simcc_coarse_y" in outputs:
        loss_parts["loss_simcc_coarse"] = total_simcc_coarse / denom
    if "simcc_deform_x" in outputs and "simcc_deform_y" in outputs:
        loss_parts["loss_simcc_deform"] = total_simcc_deform / denom
    for refine_idx, total_refine in enumerate(total_simcc_refine, start=1):
        loss_parts[f"loss_simcc_refine_{refine_idx}"] = total_refine / denom

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
        + weights.image_coord * loss_parts["loss_image_coord"]
        + weights.hard_joint * loss_parts["loss_hard_joint"]
        + weights.vis * loss_parts["loss_vis"]
        + graph_anchor
    )
    total = total + weights.coarse_coord * loss_parts.get("loss_coord_coarse", graph_anchor)
    total = total + weights.deform_coord * loss_parts.get("loss_coord_deform", graph_anchor)
    for refine_idx, refine_weight in enumerate(_weight_sequence(weights.refine_coords), start=1):
        total = total + refine_weight * loss_parts.get(f"loss_coord_refine_{refine_idx}", graph_anchor)
    total = total + weights.simcc_coarse * loss_parts.get("loss_simcc_coarse", graph_anchor)
    total = total + weights.simcc_deform * loss_parts.get("loss_simcc_deform", graph_anchor)
    for refine_idx, refine_weight in enumerate(_weight_sequence(weights.simcc_refine), start=1):
        total = total + refine_weight * loss_parts.get(f"loss_simcc_refine_{refine_idx}", graph_anchor)
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


def per_joint_image_coord_loss(
    pred_keypoints: torch.Tensor,
    gt_keypoints: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    if pred_keypoints.numel() == 0:
        return pred_keypoints.new_zeros(pred_keypoints.shape[:-1])
    loss = F.smooth_l1_loss(
        pred_keypoints[..., :2],
        gt_keypoints[..., :2],
        reduction="none",
        beta=0.01,
    ).sum(dim=-1)
    return loss.masked_fill(~valid, 0.0)


def per_joint_oks_loss(
    pred_keypoints: torch.Tensor,
    gt_keypoints: torch.Tensor,
    valid: torch.Tensor,
    gt_areas: torch.Tensor,
    sigmas: torch.Tensor,
) -> torch.Tensor:
    if pred_keypoints.numel() == 0:
        return pred_keypoints.new_zeros(pred_keypoints.shape[:-1])
    d2 = ((pred_keypoints[..., :2] - gt_keypoints[..., :2]) ** 2).sum(dim=-1)
    d2 = d2.clamp(max=1.0)
    areas = gt_areas.to(pred_keypoints.device).clamp(min=1e-8)[:, None]
    sigma2 = (sigmas.to(pred_keypoints.device)[None, :] ** 2).clamp(min=1e-6)
    oks = torch.exp(-d2 / (2.0 * areas * sigma2))
    return (1.0 - oks).masked_fill(~valid, 0.0)


def simcc_box_loss(
    logits_x: torch.Tensor,
    logits_y: torch.Tensor,
    gt_keypoints: torch.Tensor,
    gt_valid: torch.Tensor,
    pose_boxes: torch.Tensor,
    schema_joint_indices: torch.Tensor,
    schema_joint_valid: torch.Tensor,
    sigma: float = 2.0,
) -> torch.Tensor:
    if logits_x.numel() == 0 or logits_y.numel() == 0:
        return logits_x.sum() * 0.0 + logits_y.sum() * 0.0
    bins = int(logits_x.shape[-1])
    if bins <= 1:
        return logits_x.sum() * 0.0 + logits_y.sum() * 0.0
    device = logits_x.device
    joint_indices = schema_joint_indices.to(device=device, dtype=torch.long)
    schema_valid = schema_joint_valid.to(device=device).bool()
    gt_schema_keypoints = gt_keypoints.to(device).index_select(1, joint_indices)
    gt_schema_valid = gt_valid.to(device).index_select(1, joint_indices) & schema_valid.view(1, -1)
    if not gt_schema_valid.any():
        return logits_x.sum() * 0.0 + logits_y.sum() * 0.0
    pose_boxes = pose_boxes.to(device=device, dtype=gt_schema_keypoints.dtype)
    box_wh = (pose_boxes[:, 2:] - pose_boxes[:, :2]).clamp(min=1e-4)
    target_xy = ((gt_schema_keypoints[..., :2] - pose_boxes[:, None, :2]) / box_wh[:, None, :]).clamp(0.0, 1.0)
    center = target_xy.float() * float(bins - 1)
    grid = torch.arange(bins, device=device, dtype=torch.float32).view(1, 1, bins)
    sigma = max(float(sigma), 1e-3)
    target_x = torch.exp(-0.5 * ((grid - center[..., 0:1]) / sigma) ** 2)
    target_y = torch.exp(-0.5 * ((grid - center[..., 1:2]) / sigma) ** 2)
    target_x = target_x / target_x.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    target_y = target_y / target_y.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    loss_x = -(target_x * F.log_softmax(logits_x.float(), dim=-1)).sum(dim=-1)
    loss_y = -(target_y * F.log_softmax(logits_y.float(), dim=-1)).sum(dim=-1)
    # Cross entropy grows with the number of SimCC bins: uniform predictions
    # produce CE=log(bins). Normalize by that baseline so SimCC remains an
    # auxiliary term whose scale is stable when SIMCC_BINS changes.
    log_bins = max(math.log(float(bins)), 1.0)
    per_joint = 0.5 * (loss_x + loss_y) / log_bins
    valid_f = gt_schema_valid.float()
    per_instance = (per_joint * valid_f).sum(dim=-1) / valid_f.sum(dim=-1).clamp(min=1.0)
    return per_instance.sum()


def _weight_sequence(value: tuple[float, ...] | list[float] | float | None) -> tuple[float, ...]:
    if value is None:
        return ()
    if isinstance(value, (int, float)):
        return (float(value),)
    return tuple(float(item) for item in value)
