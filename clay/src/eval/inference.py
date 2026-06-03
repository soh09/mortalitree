"""Inference: tile → filtered boxes, confidence threshold sweep."""
from typing import Optional
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.naip_dataset import NAIPTileDataset, naip_collate_fn
from ..model.detector import TreeDetector
from .metrics import compute_f1_at_iou, compute_precision_recall_curve


def predict_tile(
    model: TreeDetector,
    batch: dict,
    conf_thresh: float = 0.5,
    device: str = "cpu",
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Run model on a batch dict and return filtered boxes and scores per tile.
    Returns:
        boxes_list:  list of (N_kept, 4) arrays, (cx,cy,w,h) in [0,1]
        scores_list: list of (N_kept,) arrays
    """
    model.eval()
    batch_gpu = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
        if k not in ("boxes", "exhaustive", "tile_paths")
    }
    with torch.no_grad():
        cls_logits, pred_boxes = model(batch_gpu)

    B = cls_logits.shape[0]
    boxes_list, scores_list = [], []
    for i in range(B):
        scores = cls_logits[i].sigmoid().cpu()
        boxes  = pred_boxes[i].cpu()
        keep   = scores >= conf_thresh
        boxes_list.append(boxes[keep].numpy())
        scores_list.append(scores[keep].numpy())
    return boxes_list, scores_list


def threshold_sweep(
    model: TreeDetector,
    annotations_path: str,
    norm_stats_path: Optional[str] = None,
    device: str = "cpu",
    batch_size: int = 8,
    thresholds: Optional[list[float]] = None,
    iou_thresh: float = 0.5,
    num_workers: int = 4,
) -> dict:
    """
    Sweep confidence thresholds and return precision-recall at each threshold,
    plus the best-F1 operating point.
    """
    from ..data.naip_dataset import NAIPTileDataset, naip_collate_fn

    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 17).tolist()

    dataset = NAIPTileDataset(annotations_path, norm_stats_path, augment=False)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         collate_fn=naip_collate_fn, num_workers=num_workers)

    # Collect all raw predictions once
    all_pred_boxes, all_pred_scores, all_gt_boxes = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch_gpu = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
                if k not in ("boxes", "exhaustive", "tile_paths")
            }
            cls_logits, pred_boxes = model(batch_gpu)
            for i in range(cls_logits.shape[0]):
                all_pred_boxes.append(pred_boxes[i].cpu())
                all_pred_scores.append(cls_logits[i].sigmoid().cpu())
                all_gt_boxes.append(batch["boxes"][i])

    # PR curve
    prec, rec, thresholds_pr = compute_precision_recall_curve(
        all_pred_boxes, all_pred_scores, all_gt_boxes, iou_thresh
    )

    # Per-threshold F1
    results = []
    best_f1, best_thresh = 0.0, 0.5
    for t in thresholds:
        f1 = compute_f1_at_iou(
            all_pred_boxes, all_pred_scores, all_gt_boxes,
            iou_thresh=iou_thresh, conf_thresh=t,
        )
        results.append({"threshold": t, "f1": f1})
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    return {
        "sweep": results,
        "best_f1": best_f1,
        "best_threshold": best_thresh,
        "precision_curve": prec,
        "recall_curve": rec,
        "threshold_curve": thresholds_pr,
    }
