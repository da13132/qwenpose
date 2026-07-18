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
    """Legacy aggregate OKS loss retained for external callers."""
    if pred_keypoints.numel() == 0:
        return pred_keypoints.sum() * 0.0
    d2 = ((pred_keypoints[..., :2] - gt_keypoints[..., :2]) ** 2).sum(dim=-1)
    areas = box_area(gt_boxes).clamp(min=1e-6)[:, None]
    variances = (
        2.0 * sigmas.to(device=pred_keypoints.device, dtype=pred_keypoints.dtype)
    ).square()[None, :].clamp(min=1e-8)
    oks = torch.exp(-d2 / (2.0 * areas * variances))
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
    # Deprecated compatibility field. Final pose no longer uses a second,
    # box-normalized coordinate objective; intermediate stages have dedicated
    # coarse/deform/refine weights below.
    coord: float = 0.0
    image_coord: float = 5.0
    # Legacy flag name; this now weights per-joint presence/visibility BCE.
    keypoint_confidence: float = 0.1
    # Direct per-joint localization quality trained against detached OKS.
    keypoint_quality: float = 0.1
    # Direct pose AP score trained with evaluator-aligned detached OKS targets.
    person_confidence: float = 0.0
    # RefHuman expression-to-person classification over the generated candidates.
    ref_match: float = 1.0
    # Legacy construction alias. When provided, it overrides the new field.
    vis: float | None = None
    # Stage2 LM bbox supervision. Stage1 should keep this at 0.
    lm: float = 0.05
    # Experimental hard-joint mining. Kept for later ablation, default off.
    hard_joint: float = 0.0
    hard_joint_fraction: float = 0.2
    # Human proposal refinement and denoising supervision.
    box_objectness: float = 1.0
    # Direct person/bbox AP score trained against detached matched IoU.
    box_quality: float = 1.0
    box_l1: float = 5.0
    box_giou: float = 2.0
    box_relative: float = 1.0
    box_dn: float = 1.0
    # Training-only box-conditioned OKS keypoint denoising.
    keypoint_dn: float = 1.0
    # Pose coordinate deep supervision.
    decoder_coords: tuple[float, ...] = ()
    coarse_coord: float = 0.0
    deform_coord: float = 0.0
    refine_coords: tuple[float, ...] = ()


def compute_pose_losses(
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    task_ids: torch.Tensor,
    weights: LossWeights,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # Keep loss arithmetic in float32, but never hide NaN/Inf model outputs.
    # The training loop performs a synchronized finite-value check before loss
    # construction so one bad rank cannot silently poison every optimizer shard.
    clean_outputs = {}
    for key, value in outputs.items():
        if isinstance(value, list):
            clean_outputs[key] = [
                item.float()
                if torch.is_tensor(item) and torch.is_floating_point(item)
                else item
                for item in value
            ]
        elif torch.is_tensor(value) and torch.is_floating_point(value):
            clean_outputs[key] = value.float()
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
    total, loss_parts = compute_box_conditioned_pose_losses(
        outputs,
        targets,
        weights,
        graph_anchor,
    )
    pose_set_prediction = bool(
        torch.is_tensor(outputs.get("pose_set_prediction"))
        and bool(outputs["pose_set_prediction"].detach().item())
    )
    if pose_set_prediction:
        box_total, box_parts = compute_pose_set_proposal_box_losses(
            outputs,
            targets,
            weights,
            graph_anchor,
        )
    else:
        box_total, box_parts = compute_box_refinement_losses(outputs, targets, weights)
    total = total + box_total
    loss_parts.update(box_parts)
    loss_parts["loss_total"] = total
    if weights.box_quality > 0.0:
        box_quality_loss, box_quality_parts = compute_box_quality_loss(
            outputs,
            targets,
            graph_anchor,
        )
        total = total + float(weights.box_quality) * box_quality_loss
        loss_parts.update(box_quality_parts)
        loss_parts["loss_total"] = total
    keypoint_dn_loss, keypoint_dn_parts = compute_keypoint_denoising_loss(
        outputs, weights, graph_anchor
    )
    total = total + float(weights.keypoint_dn) * keypoint_dn_loss
    loss_parts.update(keypoint_dn_parts)
    loss_parts["loss_total"] = total
    if weights.ref_match > 0.0:
        ref_loss, ref_parts = compute_refhuman_match_loss(
            outputs,
            targets,
            task_ids,
        )
        total = total + float(weights.ref_match) * ref_loss
        loss_parts.update(ref_parts)
        loss_parts["loss_total"] = total
    if weights.person_confidence > 0.0:
        person_loss, person_parts = compute_person_confidence_quality_loss(
            outputs,
            targets,
        )
        total = total + float(weights.person_confidence) * person_loss
        loss_parts.update(
            {
                key: value
                for key, value in person_parts.items()
                if key != "loss_total"
            }
        )
        loss_parts["loss_total"] = total
    return total, loss_parts


def compute_keypoint_denoising_loss(
    outputs: dict[str, torch.Tensor],
    weights: LossWeights,
    graph_anchor: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Positive pose reconstruction plus contrastive pose-quality rejection.

    Negative pose queries never alter box objectness and receive no coordinate
    loss. This is the essential adaptation from DETRPose to LocatePose's
    grounding-box -> pose hierarchy.
    """
    pred = outputs.get("keypoint_dn_keypoints")
    slot_mask = outputs.get("keypoint_dn_mask")
    labels = outputs.get("keypoint_dn_labels")
    target = outputs.get("keypoint_dn_target_keypoints")
    target_valid = outputs.get("keypoint_dn_target_valid")
    target_boxes = outputs.get("keypoint_dn_target_boxes")
    target_areas = outputs.get("keypoint_dn_target_areas")
    required = (pred, slot_mask, labels, target, target_valid, target_boxes, target_areas)
    zero_parts = {
        "loss_keypoint_dn": graph_anchor,
        "loss_keypoint_dn_oks": graph_anchor,
        "loss_keypoint_dn_coord": graph_anchor,
        "loss_keypoint_dn_image_coord": graph_anchor,
        "loss_keypoint_dn_confidence": graph_anchor,
        "loss_keypoint_dn_pose_quality": graph_anchor,
        "keypoint_dn_positive_count": graph_anchor.detach(),
        "keypoint_dn_negative_count": graph_anchor.detach(),
    }
    if not all(torch.is_tensor(value) for value in required):
        return graph_anchor, zero_parts
    assert isinstance(pred, torch.Tensor)
    assert isinstance(slot_mask, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert isinstance(target, torch.Tensor)
    assert isinstance(target_valid, torch.Tensor)
    assert isinstance(target_boxes, torch.Tensor)
    assert isinstance(target_areas, torch.Tensor)

    slot_mask = slot_mask.bool()
    labels = labels.float()
    target_valid = target_valid.bool()
    positive = slot_mask & labels.gt(0.5) & target_valid.any(dim=-1)
    negative = slot_mask & labels.le(0.5) & target_valid.any(dim=-1)
    sigmas = UNION_SIGMAS.to(device=pred.device, dtype=pred.dtype)
    loss_oks = graph_anchor
    loss_coord = graph_anchor
    loss_image_coord = graph_anchor
    loss_coarse = graph_anchor
    loss_deform = graph_anchor
    decoder_predictions = outputs.get("keypoint_dn_decoder_keypoints", [])
    if not isinstance(decoder_predictions, list):
        decoder_predictions = []
    loss_decoder = [graph_anchor for _ in decoder_predictions]
    refine_predictions = outputs.get("keypoint_dn_refine_keypoints", [])
    if not isinstance(refine_predictions, list):
        refine_predictions = []
    # The final refinement is exactly ``pred`` and must not be supervised a
    # second time. Only genuine intermediate predictions receive auxiliaries.
    auxiliary_refine_predictions = refine_predictions[:-1]
    loss_refine = [graph_anchor for _ in auxiliary_refine_predictions]

    if positive.any():
        pred_pos = pred[positive]
        target_pos = target[positive]
        valid_pos = target_valid[positive]
        boxes_pos = target_boxes[positive]
        areas_pos = target_areas[positive]
        loss_coord = _mean_valid_joints(
            per_joint_normalized_coord_loss(pred_pos, target_pos, valid_pos, boxes_pos),
            valid_pos,
        ).mean()
        loss_image_coord = _mean_valid_joints(
            per_joint_image_coord_loss(pred_pos, target_pos, valid_pos), valid_pos
        ).mean()
        loss_oks = _mean_valid_joints(
            per_joint_oks_loss(pred_pos, target_pos, valid_pos, areas_pos, sigmas),
            valid_pos,
        ).mean()
        for decoder_idx, decoder_pred in enumerate(decoder_predictions):
            if torch.is_tensor(decoder_pred):
                loss_decoder[decoder_idx] = _mean_valid_joints(
                    per_joint_normalized_coord_loss(
                        decoder_pred[positive], target_pos, valid_pos, boxes_pos
                    ),
                    valid_pos,
                ).mean()
        coarse_pred = outputs.get("keypoint_dn_coarse_keypoints")
        if torch.is_tensor(coarse_pred):
            loss_coarse = _mean_valid_joints(
                per_joint_normalized_coord_loss(
                    coarse_pred[positive], target_pos, valid_pos, boxes_pos
                ),
                valid_pos,
            ).mean()
        deform_pred = outputs.get("keypoint_dn_deform_keypoints")
        if torch.is_tensor(deform_pred):
            loss_deform = _mean_valid_joints(
                per_joint_normalized_coord_loss(
                    deform_pred[positive], target_pos, valid_pos, boxes_pos
                ),
                valid_pos,
            ).mean()
        for refine_idx, refine_pred in enumerate(auxiliary_refine_predictions):
            if torch.is_tensor(refine_pred):
                loss_refine[refine_idx] = _mean_valid_joints(
                    per_joint_normalized_coord_loss(
                        refine_pred[positive], target_pos, valid_pos, boxes_pos
                    ),
                    valid_pos,
                ).mean()

    confidence_loss = graph_anchor
    confidence_logits = outputs.get("keypoint_dn_confidence_logits")
    quality_score = pred.new_zeros(pred.shape[:-1])
    supervised = slot_mask[..., None] & target_valid
    if positive.any():
        quality_score[positive] = per_joint_oks_score(
            pred[positive][..., :2].detach(),
            target[positive][..., :2],
            target_areas[positive],
            sigmas,
        )
    # Negative DN skeletons are contrastive low-quality *instances* even if the
    # decoder moves one joint close to GT. Their per-joint presence labels are
    # nevertheless unchanged and are kept separate below.
    quality_score[negative] = 0.0
    if torch.is_tensor(confidence_logits) and supervised.any():
        per_joint_confidence = F.binary_cross_entropy_with_logits(
            confidence_logits, target[..., 2].float(), reduction="none"
        )
        weighted = per_joint_confidence * supervised.float()
        per_slot = weighted.sum(dim=-1) / supervised.float().sum(dim=-1).clamp(min=1.0)
        confidence_loss = per_slot[slot_mask & supervised.any(dim=-1)].mean()

    pose_quality_loss = graph_anchor
    pose_quality_logits = outputs.get("keypoint_dn_pose_quality_logits")
    quality_head_available = bool(outputs.get("person_confidence_head_available", True))
    if (
        quality_head_available
        and torch.is_tensor(pose_quality_logits)
        and slot_mask.any()
    ):
        valid_f = target_valid.float()
        pose_quality_target = (quality_score.detach() * valid_f).sum(dim=-1) / valid_f.sum(
            dim=-1
        ).clamp(min=1.0)
        pose_quality_target = torch.where(
            labels.gt(0.5), pose_quality_target, torch.zeros_like(pose_quality_target)
        )
        pose_quality_loss = F.binary_cross_entropy_with_logits(
            pose_quality_logits[slot_mask], pose_quality_target[slot_mask]
        )

    total = (
        float(weights.oks) * loss_oks
        + float(weights.image_coord) * loss_image_coord
        + _confidence_weight(weights) * confidence_loss
        + float(weights.person_confidence) * pose_quality_loss
        + float(weights.coarse_coord) * loss_coarse
        + float(weights.deform_coord) * loss_deform
    )
    for decoder_idx, decoder_weight in enumerate(
        _weight_sequence(weights.decoder_coords)
    ):
        if decoder_idx < len(loss_decoder):
            total = total + float(decoder_weight) * loss_decoder[decoder_idx]
    for refine_idx, refine_weight in enumerate(_weight_sequence(weights.refine_coords)):
        if refine_idx < len(loss_refine):
            total = total + float(refine_weight) * loss_refine[refine_idx]
    parts = {
        "loss_keypoint_dn": total,
        "loss_keypoint_dn_oks": loss_oks,
        "loss_keypoint_dn_coord": loss_coord,
        "loss_keypoint_dn_image_coord": loss_image_coord,
        "loss_keypoint_dn_confidence": confidence_loss,
        "loss_keypoint_dn_pose_quality": pose_quality_loss,
        "keypoint_dn_positive_count": positive.sum().detach().float(),
        "keypoint_dn_negative_count": negative.sum().detach().float(),
    }
    for decoder_idx, decoder_loss in enumerate(loss_decoder, start=1):
        parts[f"loss_keypoint_dn_decoder_{decoder_idx}"] = decoder_loss
    return total, parts


def _sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    targets = targets.to(device=logits.device, dtype=logits.dtype)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probabilities = logits.sigmoid()
    p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
    loss = ce * (1.0 - p_t).pow(float(gamma))
    if alpha >= 0.0:
        alpha_t = float(alpha) * targets + (1.0 - float(alpha)) * (1.0 - targets)
        loss = alpha_t * loss
    return loss


def _quality_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Varifocal-style loss for a direct quality-aware foreground logit."""
    targets = targets.to(device=logits.device, dtype=logits.dtype).clamp(0.0, 1.0)
    probabilities = logits.sigmoid()
    negative_weight = float(alpha) * probabilities.pow(float(gamma))
    weights = torch.where(targets.gt(0.0), targets, negative_weight)
    return F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    ) * weights


def _box_giou_diagonal(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    lt = torch.maximum(boxes1[..., :2], boxes2[..., :2])
    rb = torch.minimum(boxes1[..., 2:], boxes2[..., 2:])
    inter_wh = (rb - lt).clamp(min=0.0)
    inter = inter_wh[..., 0] * inter_wh[..., 1]
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    union = (area1 + area2 - inter).clamp(min=1e-8)
    iou = inter / union
    enclosing_lt = torch.minimum(boxes1[..., :2], boxes2[..., :2])
    enclosing_rb = torch.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclosing_wh = (enclosing_rb - enclosing_lt).clamp(min=0.0)
    enclosing = (enclosing_wh[..., 0] * enclosing_wh[..., 1]).clamp(min=1e-8)
    return iou - (enclosing - union) / enclosing


def _box_iou_diagonal(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    lt = torch.maximum(boxes1[..., :2], boxes2[..., :2])
    rb = torch.minimum(boxes1[..., 2:], boxes2[..., 2:])
    inter_wh = (rb - lt).clamp(min=0.0)
    intersection = inter_wh[..., 0] * inter_wh[..., 1]
    union = (box_area(boxes1) + box_area(boxes2) - intersection).clamp(min=1e-8)
    return intersection / union


def _box_regression_losses(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if pred_boxes.numel() == 0:
        zero = pred_boxes.sum() * 0.0
        return zero, zero, zero
    l1 = F.l1_loss(pred_boxes, target_boxes, reduction="none").sum(dim=-1)
    giou = 1.0 - _box_giou_diagonal(pred_boxes, target_boxes)
    pred_center = (pred_boxes[..., :2] + pred_boxes[..., 2:]) * 0.5
    target_center = (target_boxes[..., :2] + target_boxes[..., 2:]) * 0.5
    pred_wh = (pred_boxes[..., 2:] - pred_boxes[..., :2]).clamp(min=1e-4)
    target_wh = (target_boxes[..., 2:] - target_boxes[..., :2]).clamp(min=1e-4)
    center_error = (pred_center - target_center) / target_wh
    size_error = torch.log(pred_wh / target_wh)
    relative = F.smooth_l1_loss(
        torch.cat([center_error, size_error], dim=-1),
        torch.zeros_like(torch.cat([center_error, size_error], dim=-1)),
        reduction="none",
        beta=0.1,
    ).sum(dim=-1)
    return l1, giou, relative


def compute_box_quality_loss(
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    graph_anchor: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Train the direct person/bbox AP logit against detached matched IoU."""
    logits = outputs.get("pred_box_logits", outputs.get("box_quality_logits"))
    pred_boxes = outputs.get("pred_boxes")
    box_mask = outputs.get("box_mask")
    if not all(torch.is_tensor(value) for value in (logits, pred_boxes, box_mask)):
        return graph_anchor, {
            "loss_box_quality": graph_anchor,
            "box_quality_target_mean": graph_anchor.detach(),
            "box_quality_prediction_mean": graph_anchor.detach(),
            "box_quality_instances": graph_anchor.detach(),
        }
    assert isinstance(logits, torch.Tensor)
    assert isinstance(pred_boxes, torch.Tensor)
    assert isinstance(box_mask, torch.Tensor)
    total = logits.sum() * 0.0
    target_sum = logits.new_zeros(())
    prediction_sum = logits.new_zeros(())
    count = 0
    positive_count = 0
    for batch_idx, target in enumerate(targets):
        valid_queries = torch.nonzero(
            box_mask[batch_idx].bool(), as_tuple=False
        ).flatten()
        if valid_queries.numel() == 0:
            continue
        matched_gt_indices = target.get("matched_gt_indices")
        if torch.is_tensor(matched_gt_indices):
            matched = matched_gt_indices.to(device=logits.device)[valid_queries].ge(0)
        else:
            # Box-conditioned legacy batches have one positive query per row.
            positive_limit = min(
                int(valid_queries.numel()), int(target["boxes"].shape[0])
            )
            matched = torch.arange(
                int(valid_queries.numel()), device=logits.device
            ).lt(positive_limit)
        quality_target = logits.new_zeros(valid_queries.shape, dtype=torch.float32)
        if matched.any():
            positive_count += int(matched.sum().item())
            positive_queries = valid_queries[matched]
            target_boxes = target["boxes"].to(
                device=pred_boxes.device, dtype=pred_boxes.dtype
            )
            # Query-aligned targets place each matched GT at its query index.
            # Legacy targets are row-aligned with the leading valid queries.
            if torch.is_tensor(matched_gt_indices):
                positive_targets = target_boxes[positive_queries]
            else:
                positive_targets = target_boxes[: int(positive_queries.numel())]
            quality_target[matched] = _box_iou_diagonal(
                pred_boxes[batch_idx, positive_queries], positive_targets
            ).detach().float().clamp(0.0, 1.0)
        sample_logits = logits[batch_idx, valid_queries].float()
        total = total + _quality_focal_loss(sample_logits, quality_target).sum()
        target_sum = target_sum + quality_target.sum()
        prediction_sum = prediction_sum + sample_logits.sigmoid().detach().sum()
        count += int(valid_queries.numel())
    denom = max(positive_count, 1)
    loss = total / denom
    return loss, {
        "loss_box_quality": loss,
        "box_quality_target_mean": (target_sum / max(count, 1)).detach(),
        "box_quality_prediction_mean": (prediction_sum / max(count, 1)).detach(),
        "box_quality_instances": logits.new_tensor(float(count)),
        "box_quality_positive_instances": logits.new_tensor(float(positive_count)),
    }


def compute_pose_set_proposal_box_losses(
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    weights: LossWeights,
    graph_anchor: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Supervise the canonical independently regressed person boxes.

    The same query owns ``pred_boxes`` and ``pred_keypoints``; pose envelopes
    never enter this objective. Locate/Stage1-proxy queries (source id 1) have
    a learned residual box head, so they receive L1/GIoU supervision while their
    fixed foreground class prior remains excluded from objectness supervision.
    """
    proposal_boxes = outputs.get("pred_boxes", outputs.get("input_boxes"))
    pre_pose_boxes = outputs.get("pre_pose_boxes")
    objectness_logits = outputs.get("box_objectness_logits")
    box_mask = outputs.get("box_mask")
    if not all(
        torch.is_tensor(value)
        for value in (proposal_boxes, objectness_logits, box_mask)
    ):
        return graph_anchor, {
            "loss_box_objectness": graph_anchor,
            "loss_box_l1": graph_anchor,
            "loss_box_giou": graph_anchor,
            "loss_box_relative": graph_anchor,
            "loss_box_dn": graph_anchor,
            "box_valid_queries": graph_anchor.detach(),
            "box_positive_queries": graph_anchor.detach(),
        }
    assert isinstance(proposal_boxes, torch.Tensor)
    assert isinstance(objectness_logits, torch.Tensor)
    assert isinstance(box_mask, torch.Tensor)

    source_ids = outputs.get("proposal_source_ids")
    if torch.is_tensor(source_ids):
        objectness_mask = box_mask.bool() & source_ids.eq(0)
        regression_mask = box_mask.bool() & (source_ids.eq(0) | source_ids.eq(1))
    else:
        objectness_mask = box_mask.bool()
        regression_mask = box_mask.bool()

    objectness_sum = proposal_boxes.sum() * 0.0
    l1_sum = proposal_boxes.sum() * 0.0
    giou_sum = proposal_boxes.sum() * 0.0
    relative_sum = proposal_boxes.sum() * 0.0
    pre_pose_l1_sum = proposal_boxes.sum() * 0.0
    pre_pose_giou_sum = proposal_boxes.sum() * 0.0
    valid_count = 0
    objectness_positive_count = 0
    positive_count = 0
    pre_pose_positive_count = 0
    for batch_idx, target in enumerate(targets):
        target_boxes = target.get("boxes")
        matched_gt_indices = target.get("matched_gt_indices")
        if not torch.is_tensor(target_boxes) or not torch.is_tensor(matched_gt_indices):
            continue
        regression_queries = torch.nonzero(
            regression_mask[batch_idx], as_tuple=False
        ).flatten()
        regression_queries = regression_queries[regression_queries < min(
            int(target_boxes.shape[0]),
            int(matched_gt_indices.shape[0]),
        )]
        objectness_queries = torch.nonzero(
            objectness_mask[batch_idx], as_tuple=False
        ).flatten()
        objectness_queries = objectness_queries[objectness_queries < min(
            int(target_boxes.shape[0]),
            int(matched_gt_indices.shape[0]),
        )]
        matched_all = matched_gt_indices.to(device=proposal_boxes.device)
        if objectness_queries.numel() > 0:
            objectness_positive = matched_all[objectness_queries].ge(0)
            labels = objectness_positive.to(dtype=objectness_logits.dtype)
            objectness_sum = objectness_sum + _sigmoid_focal_loss(
                objectness_logits[batch_idx, objectness_queries], labels
            ).sum()
            valid_count += int(objectness_queries.numel())
            objectness_positive_count += int(objectness_positive.sum().item())
        if regression_queries.numel() > 0:
            regression_positive = matched_all[regression_queries].ge(0)
        else:
            regression_positive = torch.zeros(
                0, device=proposal_boxes.device, dtype=torch.bool
            )
        if regression_positive.any():
            positive_queries = regression_queries[regression_positive]
            gt_boxes = target_boxes.to(
                device=proposal_boxes.device,
                dtype=proposal_boxes.dtype,
            )[positive_queries]
            l1, giou, relative = _box_regression_losses(
                proposal_boxes[batch_idx, positive_queries],
                gt_boxes,
            )
            l1_sum = l1_sum + l1.sum()
            giou_sum = giou_sum + giou.sum()
            relative_sum = relative_sum + relative.sum()
            positive_count += int(regression_positive.sum().item())
            if torch.is_tensor(pre_pose_boxes) and torch.is_tensor(source_ids):
                external_positive = source_ids[batch_idx, positive_queries].eq(1)
                if external_positive.any():
                    pre_queries = positive_queries[external_positive]
                    pre_targets = gt_boxes[external_positive]
                    pre_l1, pre_giou, _ = _box_regression_losses(
                        pre_pose_boxes[batch_idx, pre_queries],
                        pre_targets.to(dtype=pre_pose_boxes.dtype),
                    )
                    pre_pose_l1_sum = pre_pose_l1_sum + pre_l1.sum()
                    pre_pose_giou_sum = pre_pose_giou_sum + pre_giou.sum()
                    pre_pose_positive_count += int(external_positive.sum().item())

    # DETR-style positive normalization prevents the many background proposals
    # from shrinking this auxiliary objectness signal toward zero.
    objectness_loss = objectness_sum / max(objectness_positive_count, 1)
    l1_loss = l1_sum / max(positive_count, 1)
    giou_loss = giou_sum / max(positive_count, 1)
    relative_loss = relative_sum / max(positive_count, 1)
    pre_pose_l1_loss = pre_pose_l1_sum / max(pre_pose_positive_count, 1)
    pre_pose_giou_loss = pre_pose_giou_sum / max(pre_pose_positive_count, 1)
    zero = proposal_boxes.sum() * 0.0
    total = (
        float(weights.box_objectness) * objectness_loss
        + float(weights.box_l1) * l1_loss
        + float(weights.box_giou) * giou_loss
        + float(weights.box_relative) * relative_loss
        + 0.5 * float(weights.box_l1) * pre_pose_l1_loss
        + 0.5 * float(weights.box_giou) * pre_pose_giou_loss
    )
    return total, {
        "loss_box_objectness": objectness_loss,
        "loss_box_l1": l1_loss,
        "loss_box_giou": giou_loss,
        "loss_box_relative": relative_loss,
        "loss_pre_pose_box_l1": pre_pose_l1_loss,
        "loss_pre_pose_box_giou": pre_pose_giou_loss,
        # BoxDN was removed from the pose-set graph; keep the metric key stable.
        "loss_box_dn": zero,
        "box_valid_queries": torch.as_tensor(
            float(valid_count), device=proposal_boxes.device
        ),
        "box_positive_queries": torch.as_tensor(
            float(positive_count), device=proposal_boxes.device
        ),
    }


def compute_box_refinement_losses(
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    weights: LossWeights,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pred_boxes = outputs.get("pred_boxes")
    objectness_logits = outputs.get("box_objectness_logits")
    box_mask = outputs.get("box_mask")
    if not torch.is_tensor(pred_boxes) or not torch.is_tensor(objectness_logits) or not torch.is_tensor(box_mask):
        anchor = outputs["keypoints"].sum() * 0.0
        return anchor, {
            "loss_box_objectness": anchor,
            "loss_box_l1": anchor,
            "loss_box_giou": anchor,
            "loss_box_relative": anchor,
            "loss_box_dn": anchor,
        }

    device = pred_boxes.device
    objectness_sum = pred_boxes.sum() * 0.0
    l1_sum = pred_boxes.sum() * 0.0
    giou_sum = pred_boxes.sum() * 0.0
    relative_sum = pred_boxes.sum() * 0.0
    valid_count = 0
    positive_count = 0

    def accumulate_layer(
        layer_boxes: torch.Tensor,
        layer_logits: torch.Tensor,
        layer_weight: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        layer_objectness = layer_boxes.sum() * 0.0
        layer_l1 = layer_boxes.sum() * 0.0
        layer_giou = layer_boxes.sum() * 0.0
        layer_relative = layer_boxes.sum() * 0.0
        layer_valid = 0
        layer_positive = 0
        for batch_idx, target in enumerate(targets):
            queries = torch.nonzero(box_mask[batch_idx].bool(), as_tuple=False).flatten()
            n = min(int(queries.numel()), int(target["boxes"].shape[0]))
            if n <= 0:
                continue
            queries = queries[:n]
            matched_gt_indices = target.get("matched_gt_indices")
            if isinstance(matched_gt_indices, torch.Tensor):
                positive = matched_gt_indices.to(device=device)[:n].ge(0)
            else:
                positive = torch.ones(n, device=device, dtype=torch.bool)
            labels = positive.to(dtype=layer_logits.dtype)
            layer_objectness = layer_objectness + _sigmoid_focal_loss(
                layer_logits[batch_idx, queries], labels
            ).sum() * float(layer_weight)
            layer_valid += n
            if positive.any():
                selected_queries = queries[positive]
                gt_boxes = target["boxes"].to(device=device, dtype=layer_boxes.dtype)[:n][positive]
                l1, giou, relative = _box_regression_losses(
                    layer_boxes[batch_idx, selected_queries], gt_boxes
                )
                layer_l1 = layer_l1 + l1.sum() * float(layer_weight)
                layer_giou = layer_giou + giou.sum() * float(layer_weight)
                layer_relative = layer_relative + relative.sum() * float(layer_weight)
                layer_positive += int(positive.sum().item())
        return layer_objectness, layer_l1, layer_giou, layer_relative, layer_valid, layer_positive

    main_values = accumulate_layer(pred_boxes, objectness_logits, 1.0)
    objectness_sum = objectness_sum + main_values[0]
    l1_sum = l1_sum + main_values[1]
    giou_sum = giou_sum + main_values[2]
    relative_sum = relative_sum + main_values[3]
    valid_count += main_values[4]
    positive_count += main_values[5]

    aux_outputs = outputs.get("aux_box_outputs", [])
    if isinstance(aux_outputs, list):
        for aux in aux_outputs:
            if not isinstance(aux, dict):
                continue
            aux_boxes = aux.get("pred_boxes")
            aux_logits = aux.get("objectness_logits")
            if torch.is_tensor(aux_boxes) and torch.is_tensor(aux_logits):
                aux_values = accumulate_layer(aux_boxes, aux_logits, 0.5)
                objectness_sum = objectness_sum + aux_values[0]
                l1_sum = l1_sum + aux_values[1]
                giou_sum = giou_sum + aux_values[2]
                relative_sum = relative_sum + aux_values[3]
                # Auxiliary layers are weighted additions to the final-layer
                # objective, not extra samples in the denominator.

    objectness_loss = objectness_sum / max(valid_count, 1)
    l1_loss = l1_sum / max(positive_count, 1)
    giou_loss = giou_sum / max(positive_count, 1)
    relative_loss = relative_sum / max(positive_count, 1)

    dn_loss = pred_boxes.sum() * 0.0
    dn_boxes = outputs.get("dn_pred_boxes")
    dn_logits = outputs.get("dn_objectness_logits")
    dn_mask = outputs.get("dn_box_mask")
    dn_labels = outputs.get("dn_labels")
    dn_targets = outputs.get("dn_target_boxes")
    if all(torch.is_tensor(value) for value in (dn_boxes, dn_logits, dn_mask, dn_labels, dn_targets)):
        assert isinstance(dn_boxes, torch.Tensor)
        assert isinstance(dn_logits, torch.Tensor)
        assert isinstance(dn_mask, torch.Tensor)
        assert isinstance(dn_labels, torch.Tensor)
        assert isinstance(dn_targets, torch.Tensor)
        valid = dn_mask.bool()
        if valid.any():
            dn_objectness = _sigmoid_focal_loss(dn_logits[valid], dn_labels[valid]).mean()
            positive = valid & dn_labels.gt(0.5)
            if positive.any():
                dn_l1, dn_giou, dn_relative = _box_regression_losses(
                    dn_boxes[positive], dn_targets[positive]
                )
                dn_loss = (
                    float(weights.box_objectness) * dn_objectness
                    + float(weights.box_l1) * dn_l1.mean()
                    + float(weights.box_giou) * dn_giou.mean()
                    + float(weights.box_relative) * dn_relative.mean()
                )
            else:
                dn_loss = float(weights.box_objectness) * dn_objectness

    total = (
        float(weights.box_objectness) * objectness_loss
        + float(weights.box_l1) * l1_loss
        + float(weights.box_giou) * giou_loss
        + float(weights.box_relative) * relative_loss
        + float(weights.box_dn) * dn_loss
    )
    return total, {
        "loss_box_objectness": objectness_loss,
        "loss_box_l1": l1_loss,
        "loss_box_giou": giou_loss,
        "loss_box_relative": relative_loss,
        "loss_box_dn": dn_loss,
    }


def compute_refhuman_match_loss(
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    task_ids: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Classify the referred person independently from pose/objectness quality.

    The target is the query index produced by Hungarian pose/proposal matching.
    Samples whose referred GT person did not receive a valid query match are
    excluded; forcing an arbitrary class would teach the semantic head the wrong
    person rather than improve grounding.
    """
    ref_logits = outputs.get("ref_logits")
    box_mask = outputs.get("box_mask")
    if not torch.is_tensor(ref_logits) or not torch.is_tensor(box_mask):
        anchor = outputs["keypoints"].sum() * 0.0
        return anchor, {
            "loss_ref_match": anchor,
            "ref_match_accuracy": anchor.detach(),
            "ref_match_margin": anchor.detach(),
            "ref_match_instances": anchor.detach(),
        }

    total_ce = ref_logits.sum() * 0.0
    total_rank = ref_logits.sum() * 0.0
    total_contrast = ref_logits.sum() * 0.0
    candidate_embed = outputs.get("ref_candidate_embed")
    text_embed = outputs.get("ref_text_embed")
    correct = ref_logits.new_zeros(())
    margin_sum = ref_logits.new_zeros(())
    count = 0
    for batch_idx, target in enumerate(targets):
        if int(task_ids[batch_idx].detach().cpu().item()) != 1:
            continue
        ref_target = target.get("ref_target")
        if not torch.is_tensor(ref_target) or ref_target.numel() != 1:
            continue
        target_query = int(ref_target.detach().cpu().item())
        valid_queries = torch.nonzero(
            box_mask[batch_idx].bool(), as_tuple=False
        ).flatten()
        if target_query < 0 or not bool((valid_queries == target_query).any().item()):
            continue
        logits = ref_logits[batch_idx, valid_queries].float()
        target_position = int(
            torch.nonzero(valid_queries == target_query, as_tuple=False)[0, 0].item()
        )
        target_tensor = torch.tensor(
            [target_position], device=logits.device, dtype=torch.long
        )
        total_ce = total_ce + F.cross_entropy(logits.unsqueeze(0), target_tensor)
        predicted_position = int(logits.argmax().item())
        correct = correct + float(predicted_position == target_position)
        positive_logit = logits[target_position]
        if logits.numel() > 1:
            negative_mask = torch.ones_like(logits, dtype=torch.bool)
            negative_mask[target_position] = False
            hardest_negative = logits[negative_mask].max()
            margin = positive_logit - hardest_negative
            margin_sum = margin_sum + margin
            total_rank = total_rank + F.relu(logits.new_tensor(0.3) - margin)
        if torch.is_tensor(candidate_embed) and torch.is_tensor(text_embed):
            candidates = F.normalize(
                candidate_embed[batch_idx, valid_queries].float(), dim=-1
            )
            text = F.normalize(text_embed[batch_idx].float(), dim=-1)
            contrast_logits = torch.matmul(candidates, text) / 0.07
            total_contrast = total_contrast + F.cross_entropy(
                contrast_logits.unsqueeze(0), target_tensor
            )
        count += 1

    denom = max(count, 1)
    loss_ce = total_ce / denom
    loss_rank = total_rank / denom
    loss_contrast = total_contrast / denom
    loss = loss_ce + 0.5 * loss_rank + 0.2 * loss_contrast
    return loss, {
        "loss_ref_match": loss,
        "loss_ref_match_ce": loss_ce,
        "loss_ref_match_rank": loss_rank,
        "loss_ref_match_contrast": loss_contrast,
        "ref_match_accuracy": (correct / denom).detach(),
        "ref_match_margin": (margin_sum / denom).detach(),
        "ref_match_instances": ref_logits.new_tensor(float(count)),
    }


def compute_person_confidence_quality_loss(
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Train the direct pose AP logit against detached evaluator-aligned OKS."""
    pose_quality_logits = outputs.get(
        "pred_pose_logits", outputs.get("pose_quality_logits")
    )
    if not torch.is_tensor(pose_quality_logits):
        raise ValueError("Pose-quality supervision requires tensor pose_quality_logits.")
    box_mask = outputs.get("box_mask")
    if not torch.is_tensor(box_mask):
        raise ValueError("Confidence rescue requires box_mask.")
    keypoints = outputs.get("keypoints")
    if not torch.is_tensor(keypoints):
        raise ValueError("Confidence rescue requires keypoints.")

    device = pose_quality_logits.device
    sigmas = UNION_SIGMAS.to(device=device)
    total_loss = pose_quality_logits.sum() * 0.0
    target_sum = pose_quality_logits.new_zeros(())
    prediction_sum = pose_quality_logits.new_zeros(())
    target_sq_sum = pose_quality_logits.new_zeros(())
    prediction_sq_sum = pose_quality_logits.new_zeros(())
    count = 0
    positive_count = 0

    for batch_idx, target in enumerate(targets):
        valid_queries = torch.nonzero(box_mask[batch_idx].bool(), as_tuple=False).flatten()
        target_count = int(target["boxes"].shape[0])
        n = min(int(valid_queries.numel()), target_count)
        if n <= 0:
            continue
        queries = valid_queries[:n]
        gt_keypoints = target["keypoints"].to(device=device)[:n]
        gt_valid = target["keypoint_valid"].to(device=device)[:n].bool()
        supervised = gt_valid.any(dim=-1)
        default_areas = box_area(target["boxes"].to(device=device)[:n])
        areas = target.get("loss_areas", default_areas).to(device=device)[:n]
        areas = areas.clamp(min=1e-8)

        matched_gt_indices = target.get("matched_gt_indices")
        if matched_gt_indices is not None:
            matched = matched_gt_indices.to(device=device)[:n].ge(0)
            # Generated queries are all supervised: unmatched boxes are negatives.
            include = torch.ones_like(matched, dtype=torch.bool)
        else:
            matched = supervised
            include = supervised
        if not include.any():
            continue

        instance_quality = pose_quality_logits.new_zeros((n,), dtype=torch.float32)
        quality_mask = matched & supervised
        if quality_mask.any():
            positive_count += int(quality_mask.sum().item())
            pred_xy = keypoints[batch_idx, queries[quality_mask], :, :2].detach()
            joint_quality = per_joint_oks_score(
                pred_xy,
                gt_keypoints[quality_mask, ..., :2],
                areas[quality_mask],
                sigmas,
            )
            valid_f = gt_valid[quality_mask].to(dtype=joint_quality.dtype)
            instance_quality[quality_mask] = (
                (joint_quality * valid_f).sum(dim=-1)
                / valid_f.sum(dim=-1).clamp(min=1.0)
            ).detach().clamp(0.0, 1.0)

        instance_quality = instance_quality[include]
        logits = pose_quality_logits[batch_idx, queries[include]].float()
        per_instance = _quality_focal_loss(logits, instance_quality.float())
        total_loss = total_loss + per_instance.sum()
        probabilities = logits.sigmoid().detach()
        target_sum = target_sum + instance_quality.sum()
        prediction_sum = prediction_sum + probabilities.sum()
        target_sq_sum = target_sq_sum + instance_quality.square().sum()
        prediction_sq_sum = prediction_sq_sum + probabilities.square().sum()
        count += int(instance_quality.numel())

    denom = max(positive_count, 1)
    loss = total_loss / denom
    metric_denom = max(count, 1)
    mean_target = target_sum / metric_denom
    mean_prediction = prediction_sum / metric_denom
    target_std = (target_sq_sum / metric_denom - mean_target.square()).clamp(min=0.0).sqrt()
    prediction_std = (
        prediction_sq_sum / metric_denom - mean_prediction.square()
    ).clamp(min=0.0).sqrt()
    return loss, {
        "loss_person_confidence": loss,
        "loss_total": loss,
        "person_quality_target_mean": mean_target.detach(),
        "person_quality_target_std": target_std.detach(),
        "person_confidence_mean": mean_prediction.detach(),
        "person_confidence_std": prediction_std.detach(),
        "person_confidence_instances": pose_quality_logits.new_tensor(float(count)),
        "pose_score_positive_instances": pose_quality_logits.new_tensor(
            float(positive_count)
        ),
    }


def compute_person_confidence_rescue_loss(
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Backward-compatible alias for confidence-rescue scripts and checkpoints."""
    return compute_person_confidence_quality_loss(outputs, targets)


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
    total_confidence = torch.tensor(0.0, device=device)
    total_keypoint_quality = torch.tensor(0.0, device=device)
    total_coarse_coord = torch.tensor(0.0, device=device)
    total_deform_coord = torch.tensor(0.0, device=device)
    total_joint_loss = torch.zeros(len(UNION_KEYPOINTS), device=device)
    total_joint_count = torch.zeros(len(UNION_KEYPOINTS), device=device)

    decoder_keypoints = outputs.get("decoder_keypoints", [])
    if not isinstance(decoder_keypoints, list):
        decoder_keypoints = []
    total_decoder_coord = [
        torch.tensor(0.0, device=device) for _ in decoder_keypoints
    ]

    refine_keypoints = outputs.get("refine_keypoints", [])
    if not isinstance(refine_keypoints, list):
        refine_keypoints = []
    # ``refine_keypoints[-1]`` is identical to the final ``keypoints`` output.
    # Excluding it here prevents duplicate final-stage coordinate supervision,
    # including when an older config still supplies one extra refine weight.
    auxiliary_refine_keypoints = refine_keypoints[:-1]
    total_refine_coord = [
        torch.tensor(0.0, device=device) for _ in auxiliary_refine_keypoints
    ]

    num_pos = 0
    num_confidence_instances = 0
    num_keypoint_quality_instances = 0
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
        pred_keypoints_all = outputs["keypoints"][b, q]
        schema_valid_all = outputs["keypoint_valid_mask"][b].to(device).bool().view(1, -1).expand(n, -1)

        # Per-joint visibility/presence belongs only to matched people.  It is
        # not an OKS score, and unmatched proposals must not flood this head with
        # artificial all-zero skeletons.
        supervised_people = gt_valid.any(dim=-1)
        matched_gt_indices = target.get("matched_gt_indices")
        if matched_gt_indices is not None:
            matched_people = matched_gt_indices.to(device)[:n].ge(0)
        else:
            matched_people = supervised_people

        visibility_valid = target.get("visibility_valid", gt_valid).to(device)[:n].bool()
        # Box-only annotations do not tell us that every joint is invisible.
        # Use schema-wide negatives only for people with at least one pose
        # coordinate, which covers partial/truncated annotated people without
        # turning wholly unannotated boxes into false all-zero skeletons.
        visibility_people = (
            matched_people & supervised_people & visibility_valid.any(dim=-1)
        )
        visibility_logits_all = outputs.get(
            "pred_keypoint_visibility_logits",
            outputs.get("keypoint_confidence_logits"),
        )
        if torch.is_tensor(visibility_logits_all) and visibility_people.any():
            visibility_logits = visibility_logits_all[b, q][visibility_people]
            visibility_targets = gt_keypoints[visibility_people, :, 2].float()
            visibility_mask = (
                schema_valid_all[visibility_people]
                & visibility_valid[visibility_people]
            )
            visibility_joint_loss = F.binary_cross_entropy_with_logits(
                visibility_logits,
                visibility_targets,
                reduction="none",
            )
            visibility_per_instance = (
                (visibility_joint_loss * visibility_mask.float()).sum(dim=-1)
                / visibility_mask.float().sum(dim=-1).clamp(min=1.0)
            )
            valid_visibility_people = visibility_mask.any(dim=-1)
            total_confidence = total_confidence + visibility_per_instance[
                valid_visibility_people
            ].sum()
            num_confidence_instances += int(valid_visibility_people.sum().item())

        if not supervised_people.any():
            continue
        q = q[supervised_people]
        gt_boxes = gt_boxes[supervised_people]
        loss_boxes = loss_boxes[supervised_people]
        loss_areas = loss_areas[supervised_people]
        gt_keypoints = gt_keypoints[supervised_people]
        gt_valid = gt_valid[supervised_people]
        n = int(supervised_people.sum().item())

        pred_keypoints = pred_keypoints_all[supervised_people]

        keypoint_quality_logits_all = outputs.get("pose_lqe_joint_logits")
        if torch.is_tensor(keypoint_quality_logits_all):
            keypoint_quality_logits = keypoint_quality_logits_all[b, q]
            keypoint_quality_targets = per_joint_oks_score(
                pred_keypoints[..., :2].detach(),
                gt_keypoints[..., :2],
                loss_areas,
                sigmas,
            ).detach()
            keypoint_quality_joint_loss = _quality_focal_loss(
                keypoint_quality_logits,
                keypoint_quality_targets,
            )
            keypoint_quality_per_instance = _mean_valid_joints(
                keypoint_quality_joint_loss,
                gt_valid,
            )
            total_keypoint_quality = (
                total_keypoint_quality + keypoint_quality_per_instance.sum()
            )
            num_keypoint_quality_instances += n

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

        for decoder_idx, decoder_pred in enumerate(decoder_keypoints):
            decoder_joint = per_joint_normalized_coord_loss(
                decoder_pred[b, q], gt_keypoints, gt_valid, loss_boxes
            )
            total_decoder_coord[decoder_idx] = (
                total_decoder_coord[decoder_idx]
                + _mean_valid_joints(decoder_joint, gt_valid).sum()
            )
        if "coarse_keypoints" in outputs:
            coarse_pred = outputs["coarse_keypoints"][b, q]
            coarse_joint = per_joint_normalized_coord_loss(
                coarse_pred, gt_keypoints, gt_valid, loss_boxes
            )
            total_coarse_coord = total_coarse_coord + _mean_valid_joints(coarse_joint, gt_valid).sum()
        if "deform_keypoints" in outputs:
            deform_joint = per_joint_normalized_coord_loss(
                outputs["deform_keypoints"][b, q], gt_keypoints, gt_valid, loss_boxes
            )
            total_deform_coord = total_deform_coord + _mean_valid_joints(deform_joint, gt_valid).sum()
        for refine_idx, refine_pred in enumerate(auxiliary_refine_keypoints):
            refine_joint = per_joint_normalized_coord_loss(
                refine_pred[b, q], gt_keypoints, gt_valid, loss_boxes
            )
            total_refine_coord[refine_idx] = (
                total_refine_coord[refine_idx]
                + _mean_valid_joints(refine_joint, gt_valid).sum()
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

        num_pos += n

    denom = max(num_pos, 1)
    loss_parts = {
        "loss_oks": total_oks / denom,
        "loss_coord": total_coord / denom,
        "loss_image_coord": total_image_coord / denom,
        "loss_hard_joint": total_hard_joint / max(num_hard_groups, 1),
        "loss_keypoint_confidence": total_confidence / max(num_confidence_instances, 1),
        "loss_keypoint_quality": total_keypoint_quality
        / max(num_keypoint_quality_instances, 1),
    }
    loss_parts["loss_keypoint_visibility"] = loss_parts[
        "loss_keypoint_confidence"
    ]
    for decoder_idx, total_decoder in enumerate(total_decoder_coord, start=1):
        loss_parts[f"loss_coord_decoder_{decoder_idx}"] = total_decoder / denom
    if "coarse_keypoints" in outputs:
        loss_parts["loss_coord_coarse"] = total_coarse_coord / denom
    if "deform_keypoints" in outputs:
        loss_parts["loss_coord_deform"] = total_deform_coord / denom
    for refine_idx, total_refine in enumerate(total_refine_coord, start=1):
        loss_parts[f"loss_coord_refine_{refine_idx}"] = total_refine / denom
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
        + weights.image_coord * loss_parts["loss_image_coord"]
        + weights.hard_joint * loss_parts["loss_hard_joint"]
        + _confidence_weight(weights) * loss_parts["loss_keypoint_confidence"]
        + weights.keypoint_quality * loss_parts["loss_keypoint_quality"]
        + graph_anchor
    )
    for decoder_idx, decoder_weight in enumerate(
        _weight_sequence(weights.decoder_coords), start=1
    ):
        total = total + decoder_weight * loss_parts.get(
            f"loss_coord_decoder_{decoder_idx}", graph_anchor
        )
    if "coarse_keypoints" in outputs:
        total = total + weights.coarse_coord * loss_parts["loss_coord_coarse"]
    total = total + weights.deform_coord * loss_parts.get("loss_coord_deform", graph_anchor)
    for refine_idx, refine_weight in enumerate(_weight_sequence(weights.refine_coords), start=1):
        total = total + refine_weight * loss_parts.get(f"loss_coord_refine_{refine_idx}", graph_anchor)
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
    oks = per_joint_oks_score(
        pred_keypoints[..., :2],
        gt_keypoints[..., :2],
        gt_areas,
        sigmas,
    )
    return (1.0 - oks).masked_fill(~valid, 0.0)


def per_joint_oks_score(
    pred_xy: torch.Tensor,
    gt_xy: torch.Tensor,
    gt_areas: torch.Tensor,
    sigmas: torch.Tensor,
) -> torch.Tensor:
    """Evaluator-aligned OKS shared by pose loss and confidence quality.

    COCO evaluation uses ``variances=(2*sigma)^2`` and divides the normalized
    squared distance by another factor of two. Keeping this exact expression in
    one function prevents the training loss and quality target from drifting
    away from the AP metric.
    """
    if pred_xy.numel() == 0:
        return pred_xy.new_zeros(pred_xy.shape[:-1])
    d2 = ((pred_xy - gt_xy) ** 2).sum(dim=-1)
    areas = gt_areas.to(device=pred_xy.device, dtype=pred_xy.dtype).clamp(min=1e-8)[:, None]
    variances = (2.0 * sigmas.to(device=pred_xy.device, dtype=pred_xy.dtype)) ** 2
    return torch.exp(-d2 / (2.0 * areas * variances[None, :].clamp(min=1e-8)))


def _confidence_weight(weights: LossWeights) -> float:
    return float(weights.keypoint_confidence if weights.vis is None else weights.vis)


def _weight_sequence(value: tuple[float, ...] | list[float] | float | None) -> tuple[float, ...]:
    if value is None:
        return ()
    if isinstance(value, (int, float)):
        return (float(value),)
    return tuple(float(item) for item in value)
