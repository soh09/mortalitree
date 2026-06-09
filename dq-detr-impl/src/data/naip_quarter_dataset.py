"""Quartered NAIP dataset + stitched parent-level eval for the patch-quartering model.

A 256 parent tile is split into a 2x2 grid of 128px quarters. Training treats each
quarter as an independent sample (the detector runs on 128px tiles, per-quarter GT,
per-quarter matching loss). Evaluation stitches the quarter predictions back to the
256 parent frame (map each quarter's boxes by its crop offset, NMS the overlap) and
scores against the full parent GT — a fair parent-level comparison with the
un-quartered detectors.
"""
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_convert, nms

from .augmentation import NAIPAugmentation
from .clay_meta import encode_latlon, encode_time
from .naip_dataset import NAIP_WAVELENGTHS, NAIP_GSD, load_naip_normalization_stats
from ..eval.metrics import compute_detection_metrics


def _quarter_offsets(parent: int, quarter: int):
    """Top-left (ox, oy) of each quarter in parent-pixel units."""
    steps = range(0, parent, quarter)
    return [(ox, oy) for oy in steps for ox in steps]


def _boxes_into_quarter(boxes_256, ox, oy, parent, quarter):
    """Parent-256-normalized cxcywh -> quarter-128-normalized, dropping out-of-quarter centers."""
    if len(boxes_256) == 0:
        return np.zeros((0, 4), np.float32)
    cx = boxes_256[:, 0] * parent
    cy = boxes_256[:, 1] * parent
    w = boxes_256[:, 2] * parent
    h = boxes_256[:, 3] * parent
    qcx = (cx - ox) / quarter
    qcy = (cy - oy) / quarter
    qw = w / quarter
    qh = h / quarter
    inside = (qcx >= 0) & (qcx <= 1) & (qcy >= 0) & (qcy <= 1)
    out = np.stack([qcx, qcy, qw, qh], axis=1).astype(np.float32)[inside]
    x1 = np.clip(out[:, 0] - out[:, 2] / 2, 0, 1)
    y1 = np.clip(out[:, 1] - out[:, 3] / 2, 0, 1)
    x2 = np.clip(out[:, 0] + out[:, 2] / 2, 0, 1)
    y2 = np.clip(out[:, 1] + out[:, 3] / 2, 0, 1)
    return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1).astype(np.float32)


class QuarteredNAIPDataset:
    """One sample per 128px quarter of each 256 parent tile.

    Exposes ``parent_gt`` (parent_id -> 256-normalized GT) and ``source_size`` for
    the stitched parent-level eval, mirroring the patch-quartering repo's dataset.
    """

    def __init__(self, annotations_path, norm_stats_path=None, augment=True,
                 parent_size=256, quarter_size=128):
        with open(annotations_path) as f:
            items = json.load(f)
        self.parent_size = parent_size
        self.quarter_size = quarter_size
        self.augment = NAIPAugmentation() if augment else None
        if norm_stats_path is not None:
            self.mean, self.std = load_naip_normalization_stats(norm_stats_path)
        else:
            self.mean = np.array([110.16, 115.41, 98.15, 139.04], dtype=np.float32)
            self.std = np.array([47.23, 39.82, 35.43, 49.86], dtype=np.float32)

        self.parent_gt = {}
        self.samples = []
        for it in items:
            pid = it["tile_path"]
            boxes = np.asarray(it.get("boxes", []), dtype=np.float32).reshape(-1, 4)
            self.parent_gt[pid] = boxes
            for ox, oy in _quarter_offsets(parent_size, quarter_size):
                self.samples.append({
                    "parent": pid,
                    "crop": (ox, oy, quarter_size),
                    "boxes": _boxes_into_quarter(boxes, ox, oy, parent_size, quarter_size),
                    "lat": float(it.get("lat", 37.0)),
                    "lon": float(it.get("lon", -119.0)),
                    "date": it.get("acquisition_date", "2020-06-01"),
                    "exhaustive": bool(it.get("exhaustive", True)),
                })

    def __len__(self):
        return len(self.samples)

    @property
    def parent_ids(self):
        # Deduplicated, order-preserving list of parent tile paths.
        seen, out = set(), []
        for s in self.samples:
            if s["parent"] not in seen:
                seen.add(s["parent"]); out.append(s["parent"])
        return out

    def _load_quarter(self, parent_path, ox, oy):
        import rasterio
        q = self.quarter_size
        with rasterio.open(parent_path) as src:
            data = src.read(indexes=[1, 2, 3, 4],
                            window=rasterio.windows.Window(ox, oy, q, q)).astype(np.float32)
        if data.shape[1:] != (q, q):
            pad = np.zeros((4, q, q), np.float32)
            pad[:, : data.shape[1], : data.shape[2]] = data
            data = pad
        return data

    def __getitem__(self, idx):
        s = self.samples[idx]
        ox, oy, _ = s["crop"]
        pixels = torch.from_numpy(self._load_quarter(s["parent"], ox, oy))   # (4,q,q) DN
        boxes = torch.from_numpy(s["boxes"])
        if self.augment is not None:
            pixels, boxes = self.augment(pixels, boxes)
        mean = torch.tensor(self.mean).view(4, 1, 1)
        std = torch.tensor(self.std).view(4, 1, 1)
        pixels = (pixels - mean) / std
        from datetime import datetime
        week = datetime.strptime(s["date"], "%Y-%m-%d").isocalendar()[1]
        return {
            "pixels": pixels,
            "wavelengths": NAIP_WAVELENGTHS,
            "gsd": torch.tensor([NAIP_GSD], dtype=torch.float32),
            "time": encode_time(week, 12.0),
            "latlon": encode_latlon(s["lat"], s["lon"]),
            "boxes": boxes,
            "exhaustive": s["exhaustive"],
            "parent": s["parent"],
            "crop": s["crop"],
        }


def quarter_collate_fn(batch):
    return {
        "pixels": torch.stack([b["pixels"] for b in batch]),
        "wavelengths": batch[0]["wavelengths"],
        "gsd": torch.cat([b["gsd"] for b in batch]),
        "time": torch.stack([b["time"] for b in batch]),
        "latlon": torch.stack([b["latlon"] for b in batch]),
        "boxes": [b["boxes"] for b in batch],
        "exhaustive": [b["exhaustive"] for b in batch],
        "parent": [b["parent"] for b in batch],
        "crop": [b["crop"] for b in batch],
    }


@torch.no_grad()
def stitch_predict(model, dataset, device, nms_iou=0.6, batch_size=16, num_workers=4):
    """Run `model` over all quarters and stitch to parent-256 predictions.

    Returns (pred_boxes_list, pred_scores_list, gt_boxes_list) aligned by parent —
    ready for compute_detection_metrics or visualization.
    """
    from collections import defaultdict
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=quarter_collate_fn, num_workers=num_workers)
    source = dataset.parent_size
    by_parent = defaultdict(list)   # parent -> [(boxes_256, scores)]
    skip = ("boxes", "exhaustive", "parent", "crop")
    model.eval()
    for batch in loader:
        gpu = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
               for k, v in batch.items() if k not in skip}
        out = model(gpu)
        cls_logits = out["cls_logits"] if isinstance(out, dict) else out[0]
        pred_boxes = out["pred_boxes"] if isinstance(out, dict) else out[1]
        scores = cls_logits.sigmoid().cpu()
        boxes = pred_boxes.cpu()
        for i in range(boxes.shape[0]):
            ox, oy, size = batch["crop"][i]
            b = boxes[i].clone()
            b[:, 0] = (ox + b[:, 0] * size) / source
            b[:, 1] = (oy + b[:, 1] * size) / source
            b[:, 2] = b[:, 2] * size / source
            b[:, 3] = b[:, 3] * size / source
            by_parent[batch["parent"][i]].append((b, scores[i]))

    pred_boxes_list, pred_scores_list, gt_boxes_list = [], [], []
    for pid in dataset.parent_ids:
        if pid not in by_parent:
            continue
        bs = torch.cat([c[0] for c in by_parent[pid]], dim=0)
        ss = torch.cat([c[1] for c in by_parent[pid]], dim=0)
        if bs.shape[0] > 0:
            keep = nms(box_convert(bs, "cxcywh", "xyxy"), ss, nms_iou)
            bs, ss = bs[keep], ss[keep]
        pred_boxes_list.append(bs)
        pred_scores_list.append(ss)
        gt_boxes_list.append(torch.from_numpy(dataset.parent_gt[pid]))
    return pred_boxes_list, pred_scores_list, gt_boxes_list


def stitch_metrics(model, dataset, device, conf_thresh=0.5, iou_thresh=0.5,
                   nms_iou=0.6, batch_size=16, num_workers=4):
    pb, ps, gt = stitch_predict(model, dataset, device, nms_iou, batch_size, num_workers)
    return compute_detection_metrics(pb, ps, gt, iou_thresh, conf_thresh)


def _parent_rgb(parent_path, size):
    """Percentile-stretched (size,size,3) uint8 RGB of a parent tile (for viz)."""
    import rasterio
    with rasterio.open(parent_path) as src:
        rgb = src.read(indexes=[1, 2, 3]).astype(np.float32)
    out = np.zeros_like(rgb)
    for i in range(3):
        lo, hi = np.percentile(rgb[i], 2), np.percentile(rgb[i], 98)
        if hi > lo:
            out[i] = np.clip((rgb[i] - lo) / (hi - lo) * 255, 0, 255)
    return out.astype(np.uint8).transpose(1, 2, 0)


@torch.no_grad()
def make_quarter_panels(model, dataset, device, n_parents=5, conf_thresh=0.5, nms_iou=0.6):
    """Stitched parent-level prediction panels: predicted boxes (green) over GT
    (blue) on the 256 RGB. Returns a list of (256,256,3) uint8 images."""
    from ..eval.visualize import draw_boxes
    pb, ps, gt = stitch_predict(model, dataset, device, nms_iou,
                                batch_size=16, num_workers=0)
    ids = dataset.parent_ids[:n_parents]
    panels = []
    for j, pid in enumerate(ids):
        if j >= len(pb):
            break
        rgb = _parent_rgb(pid, dataset.parent_size)
        keep = ps[j] >= conf_thresh
        boxes = pb[j][keep].numpy()
        gt_b = gt[j].numpy() if gt[j].numel() else None
        panels.append(draw_boxes(rgb, boxes, gt_boxes=gt_b))
    return panels
