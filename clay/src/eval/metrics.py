"""Evaluation metrics: per-tile MAE/F1, per-hectare aggregation, mAP, PR curves."""
from typing import Optional
import numpy as np
import torch
from torchvision.ops import box_iou, box_convert


def boxes_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert (cx,cy,w,h) normalized boxes to (x1,y1,x2,y2)."""
    if boxes.numel() == 0:
        return boxes
    return box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")


def match_predictions_to_gt(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_thresh: float = 0.5,
) -> tuple[list[bool], list[float]]:
    """
    Greedy match predicted boxes to GT by descending score.
    Returns (tp_flags, score_list) for a single tile.
    """
    if gt_boxes.numel() == 0:
        return [False] * len(pred_scores), pred_scores.tolist()
    if pred_boxes.numel() == 0:
        return [], []

    pred_xyxy = boxes_to_xyxy(pred_boxes)
    gt_xyxy   = boxes_to_xyxy(gt_boxes)
    iou_mat   = box_iou(pred_xyxy, gt_xyxy)   # (Q, M)

    order = pred_scores.argsort(descending=True)
    gt_matched = torch.zeros(gt_boxes.shape[0], dtype=torch.bool)
    tp_flags = []
    scores_ordered = []
    for i in order:
        best_iou, best_j = iou_mat[i].max(dim=0)
        if best_iou >= iou_thresh and not gt_matched[best_j]:
            gt_matched[best_j] = True
            tp_flags.append(True)
        else:
            tp_flags.append(False)
        scores_ordered.append(pred_scores[i].item())
    return tp_flags, scores_ordered


def compute_f1_at_iou(
    pred_boxes_list: list[torch.Tensor],
    pred_scores_list: list[torch.Tensor],
    gt_boxes_list: list[torch.Tensor],
    iou_thresh: float = 0.5,
    conf_thresh: float = 0.5,
) -> float:
    """F1 at a fixed confidence and IoU threshold across a dataset."""
    tp = fp = fn = 0
    for pred_b, pred_s, gt_b in zip(pred_boxes_list, pred_scores_list, gt_boxes_list):
        keep = pred_s >= conf_thresh
        pb, ps = pred_b[keep], pred_s[keep]
        tp_flags, _ = match_predictions_to_gt(pb, ps, gt_b, iou_thresh)
        tp += sum(tp_flags)
        fp += sum(1 - t for t in tp_flags)
        fn += max(0, gt_b.shape[0] - sum(tp_flags))
    prec = tp / max(1, tp + fp)
    rec  = tp / max(1, tp + fn)
    return 2 * prec * rec / max(1e-9, prec + rec)


def compute_count_metrics(
    pred_boxes_list: list[torch.Tensor],
    pred_scores_list: list[torch.Tensor],
    gt_boxes_list: list[torch.Tensor],
    conf_thresh: float = 0.5,
) -> dict:
    """Per-tile count MAE, RMSE, R²."""
    pred_counts, gt_counts = [], []
    for pred_b, pred_s, gt_b in zip(pred_boxes_list, pred_scores_list, gt_boxes_list):
        n_pred = int((pred_s >= conf_thresh).sum().item())
        n_gt   = gt_b.shape[0]
        pred_counts.append(n_pred)
        gt_counts.append(n_gt)
    pred_arr = np.array(pred_counts, dtype=float)
    gt_arr   = np.array(gt_counts,   dtype=float)
    errors   = pred_arr - gt_arr
    mae  = float(np.abs(errors).mean())
    rmse = float(np.sqrt((errors ** 2).mean()))
    ss_res = float(((gt_arr - pred_arr) ** 2).sum())
    ss_tot = float(((gt_arr - gt_arr.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / max(1e-9, ss_tot)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def compute_precision_recall_curve(
    pred_boxes_list: list[torch.Tensor],
    pred_scores_list: list[torch.Tensor],
    gt_boxes_list: list[torch.Tensor],
    iou_thresh: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (precision, recall, thresholds) arrays."""
    all_tp, all_scores = [], []
    total_gt = 0
    for pred_b, pred_s, gt_b in zip(pred_boxes_list, pred_scores_list, gt_boxes_list):
        tp_flags, scores = match_predictions_to_gt(pred_b, pred_s, gt_b, iou_thresh)
        all_tp.extend(tp_flags)
        all_scores.extend(scores)
        total_gt += gt_b.shape[0]

    if len(all_scores) == 0:
        return np.array([1.0]), np.array([0.0]), np.array([])

    order = np.argsort(all_scores)[::-1]
    tp_arr = np.array(all_tp)[order]
    cum_tp = np.cumsum(tp_arr)
    cum_fp = np.cumsum(~tp_arr)
    prec = cum_tp / (cum_tp + cum_fp + 1e-9)
    rec  = cum_tp / max(1, total_gt)
    thresholds = np.array(all_scores)[order]
    return prec, rec, thresholds


def compute_ap(prec: np.ndarray, rec: np.ndarray) -> float:
    """Area under precision-recall curve via 101-point interpolation."""
    prec = np.concatenate([[0.0], prec, [0.0]])
    rec  = np.concatenate([[0.0], rec,  [1.0]])
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    recall_thresholds = np.linspace(0, 1, 101)
    ap = 0.0
    for t in recall_thresholds:
        p = prec[rec >= t]
        ap += p.max() if len(p) else 0.0
    return ap / 101.0


def compute_map(
    pred_boxes_list: list[torch.Tensor],
    pred_scores_list: list[torch.Tensor],
    gt_boxes_list: list[torch.Tensor],
    iou_thresholds: Optional[list[float]] = None,
) -> dict:
    """mAP@0.5 and mAP@0.5:0.95."""
    if iou_thresholds is None:
        iou_thresholds = [round(0.5 + 0.05 * i, 2) for i in range(10)]
    aps = []
    for thresh in iou_thresholds:
        prec, rec, _ = compute_precision_recall_curve(
            pred_boxes_list, pred_scores_list, gt_boxes_list, thresh
        )
        aps.append(compute_ap(prec, rec))
    return {"mAP50": aps[0], "mAP50_95": float(np.mean(aps))}


def compute_crown_area_metrics(
    pred_boxes_list: list[torch.Tensor],
    pred_scores_list: list[torch.Tensor],
    gt_boxes_list: list[torch.Tensor],
    iou_thresh: float = 0.5,
    conf_thresh: float = 0.5,
    pixel_size_m: float = 0.6,
) -> dict:
    """Predicted vs. true crown area R² and rRMSE on matched pairs."""
    pred_areas, gt_areas = [], []
    for pred_b, pred_s, gt_b in zip(pred_boxes_list, pred_scores_list, gt_boxes_list):
        keep = pred_s >= conf_thresh
        pb, ps = pred_b[keep], pred_s[keep]
        if pb.numel() == 0 or gt_b.numel() == 0:
            continue
        pb_xyxy = boxes_to_xyxy(pb)
        gt_xyxy  = boxes_to_xyxy(gt_b)
        iou_mat  = box_iou(pb_xyxy, gt_xyxy)
        order = ps.argsort(descending=True)
        gt_matched = torch.zeros(gt_b.shape[0], dtype=torch.bool)
        for i in order:
            best_iou, best_j = iou_mat[i].max(dim=0)
            if best_iou >= iou_thresh and not gt_matched[best_j]:
                gt_matched[best_j] = True
                # Area in m² (boxes are normalized fractions of 224-pixel tile)
                pred_w, pred_h = pb[i, 2].item(), pb[i, 3].item()
                gt_w,   gt_h   = gt_b[best_j, 2].item(), gt_b[best_j, 3].item()
                scale = 224 * pixel_size_m
                pred_areas.append(pred_w * pred_h * scale ** 2)
                gt_areas.append(gt_w   * gt_h   * scale ** 2)

    if len(pred_areas) < 2:
        return {"crown_r2": float("nan"), "crown_rrmse": float("nan"), "n_matched": len(pred_areas)}

    pred_arr = np.array(pred_areas)
    gt_arr   = np.array(gt_areas)
    ss_res = float(((gt_arr - pred_arr) ** 2).sum())
    ss_tot = float(((gt_arr - gt_arr.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / max(1e-9, ss_tot)
    rrmse = float(np.sqrt((ss_res / len(gt_arr))) / gt_arr.mean())
    return {"crown_r2": r2, "crown_rrmse": rrmse, "n_matched": len(pred_areas)}


def aggregate_to_hectare(
    pred_boxes_list: list[torch.Tensor],
    pred_scores_list: list[torch.Tensor],
    gt_boxes_list: list[torch.Tensor],
    tile_latlons: list[tuple[float, float]],
    conf_thresh: float = 0.5,
    tile_size_m: float = 134.4,   # 224 * 0.6
    cell_size_m: float = 100.0,
) -> dict:
    """Per-100m² cell count MAE and RMSE (trees/ha)."""
    from collections import defaultdict
    pred_cell: dict = defaultdict(int)
    gt_cell:   dict = defaultdict(int)

    for (lat, lon), pred_b, pred_s, gt_b in zip(
        tile_latlons, pred_boxes_list, pred_scores_list, gt_boxes_list
    ):
        cell = (round(lat * 1000) // 10, round(lon * 1000) // 10)
        n_pred = int((pred_s >= conf_thresh).sum().item())
        pred_cell[cell] += n_pred
        gt_cell[cell]   += gt_b.shape[0]

    all_cells = set(pred_cell) | set(gt_cell)
    ha_factor = (cell_size_m ** 2) / 10000.0
    pred_ha = np.array([pred_cell[c] / ha_factor for c in all_cells])
    gt_ha   = np.array([gt_cell[c]   / ha_factor for c in all_cells])
    errors  = pred_ha - gt_ha
    return {
        "ha_mae":  float(np.abs(errors).mean()),
        "ha_rmse": float(np.sqrt((errors ** 2).mean())),
        "n_cells": len(all_cells),
    }
