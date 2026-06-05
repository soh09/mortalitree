"""Stitch quartered-tile predictions back to the parent (256) frame for
parent-level metrics — a head-to-head comparison with the un-quartered baseline.

The model runs on small (e.g. 128) quarter tiles, but evaluation happens on the
reassembled 256 patch: each quarter's predicted boxes are mapped into the parent
frame, NMS de-dups any crown caught by two adjacent quarters, and the result is
scored against the full 256-frame ground truth (NeonPatchDataset.parent_gt) with
the standard metrics + the stitched Hungarian matching loss. Works on a
non-quartered dataset too (one sample per parent, identity mapping), so the same
call evaluates the 256 baseline.

Pass `indices` to score a subset of the dataset's samples (e.g. the val split, or
a fixed train subset) — only the parents present in that subset are scored.
"""
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.ops import box_convert, nms

from ..data.deepforest_dataset import deepforest_collate_fn
from ..train.losses import matching_loss
from .metrics import (
    compute_count_metrics,
    compute_crown_area_metrics,
    compute_f1_at_iou,
    compute_map,
)


def stitch_collate_fn(batch: list) -> dict:
    """deepforest_collate_fn + the per-sample `parent`/`crop` needed to stitch."""
    out = deepforest_collate_fn(batch)
    out["parent"] = [b["parent"] for b in batch]
    out["crop"] = [b["crop"] for b in batch]
    return out


@torch.no_grad()
def stitch_eval(
    model,
    dataset,
    indices=None,
    device: str = "cuda",
    batch_size: int = 16,
    num_workers: int = 4,
    conf_thresh: float = 0.5,
    nms_iou: float = 0.6,
    lam_cls: float = 1.0,
    lam_l1: float = 5.0,
    lam_giou: float = 2.0,
) -> dict:
    """Run `model` over the tiles, stitch predictions to the parent 256 frame,
    and return parent-level count / F1 / mAP / crown-area metrics + the stitched
    matching loss. `indices` restricts evaluation to a subset of dataset.samples."""
    data = Subset(dataset, indices) if indices is not None else dataset
    loader = DataLoader(
        data, batch_size=batch_size, shuffle=False,
        collate_fn=stitch_collate_fn, num_workers=num_workers,
    )
    source = dataset.source_size
    model.eval()

    # Accumulate predictions per parent, mapped to 256-normalized cxcywh. Keep raw
    # logits (for the matching loss); scores = sigmoid(logits) are used for NMS.
    pred_by_parent: dict[str, list] = defaultdict(list)   # parent -> [(boxes, logits)]
    skip = ("boxes", "exhaustive", "parent", "crop")
    for batch in loader:
        batch_gpu = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items() if k not in skip
        }
        cls_logits, pred_boxes = model(batch_gpu)
        logits = cls_logits.cpu()
        boxes = pred_boxes.cpu()                            # (B, Q, 4) cxcywh in [0,1] over tile
        for i in range(boxes.shape[0]):
            off_x, off_y, size = batch["crop"][i]
            b = boxes[i].clone()
            b[:, 0] = (off_x + b[:, 0] * size) / source     # tile-norm -> parent(256)-norm
            b[:, 1] = (off_y + b[:, 1] * size) / source
            b[:, 2] = b[:, 2] * size / source
            b[:, 3] = b[:, 3] * size / source
            pred_by_parent[batch["parent"][i]].append((b, logits[i]))

    # Per parent: concat quarter predictions, NMS in the parent frame, gather GT.
    pred_boxes_list, pred_scores_list, gt_boxes_list = [], [], []
    stitch_loss, n_loss = 0.0, 0
    for p in sorted(pred_by_parent):
        bs = torch.cat([c[0] for c in pred_by_parent[p]], dim=0)
        lg = torch.cat([c[1] for c in pred_by_parent[p]], dim=0)
        keep = nms(box_convert(bs, "cxcywh", "xyxy"), lg.sigmoid(), nms_iou)
        bs, lg = bs[keep], lg[keep]
        gt = torch.from_numpy(dataset.parent_gt[p])
        pred_boxes_list.append(bs)
        pred_scores_list.append(lg.sigmoid())
        gt_boxes_list.append(gt)
        if bs.shape[0] > 0:
            stitch_loss += float(matching_loss(lg, bs, gt, True, lam_cls, lam_l1, lam_giou))
            n_loss += 1

    return {
        "n_parents": len(pred_boxes_list),
        "stitch_loss": stitch_loss / max(1, n_loss),
        "count": compute_count_metrics(pred_boxes_list, pred_scores_list, gt_boxes_list, conf_thresh),
        "f1_at_0.5": compute_f1_at_iou(pred_boxes_list, pred_scores_list, gt_boxes_list, 0.5, conf_thresh),
        "map": compute_map(pred_boxes_list, pred_scores_list, gt_boxes_list),
        "crown_area": compute_crown_area_metrics(
            pred_boxes_list, pred_scores_list, gt_boxes_list,
            conf_thresh=conf_thresh, tile_size_px=source,
        ),
    }
