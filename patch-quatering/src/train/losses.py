"""Hungarian matching loss for DETR-style box detection."""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops import generalized_box_iou, box_convert


def matching_loss(
    cls_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    tile_exhaustive: bool,
    lam_cls: float = 1.0,
    lam_l1: float = 5.0,
    lam_giou: float = 2.0,
) -> torch.Tensor:
    """
    cls_logits:      (Q,) raw scores
    pred_boxes:      (Q, 4) (cx, cy, w, h) in [0, 1]
    gt_boxes:        (M, 4) (cx, cy, w, h) in [0, 1]
    tile_exhaustive: whether unmatched queries should be pushed to "no tree"
    """
    Q = cls_logits.shape[0]
    M = gt_boxes.shape[0]
    cls_prob = cls_logits.sigmoid()

    if M == 0:
        if tile_exhaustive:
            cls_loss = F.binary_cross_entropy_with_logits(
                cls_logits, torch.zeros(Q, device=cls_logits.device)
            )
        else:
            cls_loss = torch.tensor(0.0, device=cls_logits.device)
        return lam_cls * cls_loss

    pred_xyxy = box_convert(pred_boxes, in_fmt="cxcywh", out_fmt="xyxy")
    gt_xyxy   = box_convert(gt_boxes,   in_fmt="cxcywh", out_fmt="xyxy")

    cls_cost  = -cls_prob.unsqueeze(1).expand(Q, M)
    l1_cost   = torch.cdist(pred_boxes, gt_boxes, p=1)
    giou_cost = -generalized_box_iou(pred_xyxy, gt_xyxy)
    cost = lam_cls * cls_cost + lam_l1 * l1_cost + lam_giou * giou_cost

    q_idx, gt_idx = linear_sum_assignment(cost.detach().cpu().numpy())

    target = torch.zeros(Q, device=cls_logits.device)
    target[q_idx] = 1.0

    if tile_exhaustive:
        cls_loss = F.binary_cross_entropy_with_logits(cls_logits, target)
    else:
        matched_mask = torch.zeros(Q, dtype=torch.bool, device=cls_logits.device)
        matched_mask[q_idx] = True
        cls_loss = F.binary_cross_entropy_with_logits(
            cls_logits[matched_mask], target[matched_mask]
        )

    matched_pred = pred_boxes[q_idx]
    matched_gt   = gt_boxes[gt_idx]
    l1_loss = F.l1_loss(matched_pred, matched_gt)

    matched_pred_xyxy = box_convert(matched_pred, "cxcywh", "xyxy")
    matched_gt_xyxy   = box_convert(matched_gt,   "cxcywh", "xyxy")
    giou_loss = (
        1.0 - generalized_box_iou(matched_pred_xyxy, matched_gt_xyxy).diag()
    ).mean()

    return lam_cls * cls_loss + lam_l1 * l1_loss + lam_giou * giou_loss


def batch_matching_loss(
    cls_logits_batch: torch.Tensor,
    pred_boxes_batch: torch.Tensor,
    gt_boxes_list: list,
    exhaustive_list: list,
    lam_cls: float = 1.0,
    lam_l1: float = 5.0,
    lam_giou: float = 2.0,
) -> torch.Tensor:
    """Compute matching loss over a batch, averaging across tiles."""
    total = torch.tensor(0.0, device=cls_logits_batch.device)
    B = cls_logits_batch.shape[0]
    for i in range(B):
        total = total + matching_loss(
            cls_logits_batch[i],
            pred_boxes_batch[i],
            gt_boxes_list[i].to(cls_logits_batch.device),
            exhaustive_list[i],
            lam_cls, lam_l1, lam_giou,
        )
    return total / B
