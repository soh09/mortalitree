"""Hungarian matching loss for DETR-style box detection, plus the two DQ-DETR
auxiliary losses: the CCM count-category cross-entropy and the two-stage
encoder-proposal matching loss."""
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


# ---------------------------------------------------------------------------
# DQ-DETR auxiliary losses
# ---------------------------------------------------------------------------

def ccm_targets_from_boxes(gt_boxes_list: list, ccm_params, device) -> torch.Tensor:
    """Map each tile's GT box count to a count-category index.

    Mirrors DQ-DETR's engine.py: with ``ccm_params = [100, 300]`` a tile gets
    category 0 (<100), 1 (100-299), or 2 (>=300). Returns a LongTensor (B,).
    """
    targets = []
    for boxes in gt_boxes_list:
        num = int(boxes.shape[0])
        cat = 0
        for j, thresh in enumerate(ccm_params):
            if num >= thresh:
                cat = j + 1
        targets.append(cat)
    return torch.tensor(targets, dtype=torch.long, device=device)


def ccm_loss(count_logits: torch.Tensor, gt_boxes_list: list, ccm_params) -> torch.Tensor:
    """Cross-entropy on the CCM count category (DQ-DETR's CCM_LOSS)."""
    targets = ccm_targets_from_boxes(gt_boxes_list, ccm_params, count_logits.device)
    return F.cross_entropy(count_logits, targets)


def _sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """Per-element sigmoid focal loss (DETR two-stage encoder classification)."""
    prob = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss


def encoder_proposal_loss(
    enc_cls_logits: torch.Tensor,
    enc_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    tile_exhaustive: bool,
    top_t: int = 1200,
    lam_cls: float = 1.0,
    lam_l1: float = 5.0,
    lam_giou: float = 2.0,
) -> torch.Tensor:
    """Two-stage proposal loss for one tile (the DQ-DETR 'interm' supervision).

    Top-k DQS selection is non-differentiable, so the proposal scorer only learns
    if supervised directly. We Hungarian-match GT against the top-T proposals (by
    objectness -- i.e. the candidates DQS actually draws from, which are also the
    hardest negatives) and apply focal classification + L1/GIoU box losses.

    enc_cls_logits: (HW,)
    enc_boxes:      (HW, 4) (cx,cy,w,h) in [0,1]
    gt_boxes:       (M, 4)
    """
    device = enc_cls_logits.device
    HW = enc_cls_logits.shape[0]
    T = min(top_t, HW)
    sel = enc_cls_logits.topk(T, dim=0).indices
    cls_logits = enc_cls_logits[sel]                  # (T,)
    pred_boxes = enc_boxes[sel]                        # (T, 4)
    M = gt_boxes.shape[0]

    if M == 0:
        if tile_exhaustive:
            target = torch.zeros(T, device=device)
            return lam_cls * _sigmoid_focal_loss(cls_logits, target).mean()
        return torch.tensor(0.0, device=device)

    cls_prob = cls_logits.sigmoid()
    pred_xyxy = box_convert(pred_boxes, in_fmt="cxcywh", out_fmt="xyxy")
    gt_xyxy = box_convert(gt_boxes, in_fmt="cxcywh", out_fmt="xyxy")

    cls_cost = -cls_prob.unsqueeze(1).expand(T, M)
    l1_cost = torch.cdist(pred_boxes, gt_boxes, p=1)
    giou_cost = -generalized_box_iou(pred_xyxy, gt_xyxy)
    cost = lam_cls * cls_cost + lam_l1 * l1_cost + lam_giou * giou_cost
    q_idx, gt_idx = linear_sum_assignment(cost.detach().cpu().numpy())
    q_idx_t = torch.as_tensor(q_idx, device=device, dtype=torch.long)

    target = torch.zeros(T, device=device)
    target[q_idx_t] = 1.0
    if tile_exhaustive:
        # Focal loss over all T (matched -> 1, the hard top-T negatives -> 0).
        cls_loss = _sigmoid_focal_loss(cls_logits, target).mean()
    else:
        # Sparse tile: only supervise the matched positives.
        cls_loss = _sigmoid_focal_loss(cls_logits[q_idx_t], target[q_idx_t]).mean()

    matched_pred = pred_boxes[q_idx_t]
    matched_gt = gt_boxes[gt_idx]
    l1_loss = F.l1_loss(matched_pred, matched_gt)
    giou_loss = (1.0 - generalized_box_iou(
        box_convert(matched_pred, "cxcywh", "xyxy"),
        box_convert(matched_gt, "cxcywh", "xyxy"),
    ).diag()).mean()

    return lam_cls * cls_loss + lam_l1 * l1_loss + lam_giou * giou_loss


def batch_encoder_proposal_loss(
    enc_cls_logits_batch: torch.Tensor,
    enc_boxes_batch: torch.Tensor,
    gt_boxes_list: list,
    exhaustive_list: list,
    top_t: int = 1200,
    lam_cls: float = 1.0,
    lam_l1: float = 5.0,
    lam_giou: float = 2.0,
) -> torch.Tensor:
    """Two-stage proposal loss averaged across a batch of tiles."""
    total = torch.tensor(0.0, device=enc_cls_logits_batch.device)
    B = enc_cls_logits_batch.shape[0]
    for i in range(B):
        total = total + encoder_proposal_loss(
            enc_cls_logits_batch[i],
            enc_boxes_batch[i],
            gt_boxes_list[i].to(enc_cls_logits_batch.device),
            exhaustive_list[i],
            top_t, lam_cls, lam_l1, lam_giou,
        )
    return total / B
