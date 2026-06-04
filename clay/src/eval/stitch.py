"""Stitch quartered-tile predictions back to the parent (256) frame for
parent-level metrics — a head-to-head comparison with the un-quartered baseline.

The model runs on small (e.g. 128) quarter tiles, but evaluation happens on the
reassembled 256 patch: each quarter's predicted boxes are mapped into the parent
frame, NMS de-dups any crown caught by two adjacent quarters, and the result is
scored against the full 256-frame ground truth (NeonPatchDataset.parent_gt) with
the standard metrics. This works on a non-quartered dataset too (one sample per
parent, identity mapping), so the same call evaluates the 256 baseline.

Build the eval dataset with `augment=False` and `max_trees=0` (keep every quarter,
including dense ones — the Q cap undercounting them is a real, measured limitation).
"""
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_convert, nms

from ..data.deepforest_dataset import deepforest_collate_fn
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
    device: str = "cuda",
    batch_size: int = 16,
    num_workers: int = 4,
    conf_thresh: float = 0.5,
    nms_iou: float = 0.6,
) -> dict:
    """Run `model` over every tile in `dataset`, stitch predictions to the parent
    256 frame, and return parent-level count / F1 / mAP / crown-area metrics."""
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=stitch_collate_fn, num_workers=num_workers,
    )
    source = dataset.source_size
    model.eval()

    # Accumulate predictions per parent, mapped to 256-normalized cxcywh.
    pred_by_parent: dict[str, list] = defaultdict(list)   # parent -> [(boxes, scores)]
    skip = ("boxes", "exhaustive", "parent", "crop")
    for batch in loader:
        batch_gpu = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items() if k not in skip
        }
        cls_logits, pred_boxes = model(batch_gpu)
        scores = cls_logits.sigmoid().cpu()
        boxes = pred_boxes.cpu()                            # (B, Q, 4) cxcywh in [0,1] over tile
        for i in range(boxes.shape[0]):
            off_x, off_y, size = batch["crop"][i]
            b = boxes[i].clone()
            b[:, 0] = (off_x + b[:, 0] * size) / source     # tile-norm -> parent(256)-norm
            b[:, 1] = (off_y + b[:, 1] * size) / source
            b[:, 2] = b[:, 2] * size / source
            b[:, 3] = b[:, 3] * size / source
            pred_by_parent[batch["parent"][i]].append((b, scores[i]))

    # Per parent: concat quarter predictions, NMS in the parent frame, gather GT.
    parents = sorted({s["parent"] for s in dataset.samples})
    pred_boxes_list, pred_scores_list, gt_boxes_list = [], [], []
    for p in parents:
        chunks = pred_by_parent.get(p, [])
        if chunks:
            bs = torch.cat([c[0] for c in chunks], dim=0)
            ss = torch.cat([c[1] for c in chunks], dim=0)
            keep = nms(box_convert(bs, "cxcywh", "xyxy"), ss, nms_iou)
            bs, ss = bs[keep], ss[keep]
        else:
            bs, ss = torch.zeros((0, 4)), torch.zeros((0,))
        pred_boxes_list.append(bs)
        pred_scores_list.append(ss)
        gt_boxes_list.append(torch.from_numpy(dataset.parent_gt[p]))

    return {
        "n_parents": len(parents),
        "count": compute_count_metrics(pred_boxes_list, pred_scores_list, gt_boxes_list, conf_thresh),
        "f1_at_0.5": compute_f1_at_iou(pred_boxes_list, pred_scores_list, gt_boxes_list, 0.5, conf_thresh),
        "map": compute_map(pred_boxes_list, pred_scores_list, gt_boxes_list),
        "crown_area": compute_crown_area_metrics(
            pred_boxes_list, pred_scores_list, gt_boxes_list,
            conf_thresh=conf_thresh, tile_size_px=source,
        ),
    }
