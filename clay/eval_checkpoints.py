"""Evaluate Stage-B tree-detector checkpoints on the good-annotated NAIP tiles.

For every checkpoint in --checkpoints-dir (e.g. q150_final.pt, q500_final.pt):
  * parse the query count Q from the leading "q<NNN>" of the filename,
  * build the TreeDetector with that Q and load the checkpoint,
  * filter the good-annotated labels to tiles with <= Q boxes (a model with Q
    object queries cannot exhaustively detect a tile holding more than Q trees),
  * compute detection metrics (precision/recall/F1, mAP, count error), and
  * save N qualitative sample panels (RGB + GT in blue + predictions in green).

Comparison mode (--compare):
  * filter to tiles with < --compare-max-ann annotations (default 150),
  * for N sample tiles, render one figure per tile with four panels:
        RGB + ground truth | q150 preds | q500 preds | DeepForest preds.

The good-tile GeoTIFFs are native-resolution NAIP (~404x408, 4-band R,G,B,NIR)
at 0.6 m/px. To match the model's Stage-B training distribution — 256px patches
at ~0.6 m/px — each tile is **center-cropped** to a 256x256 native window
(no resampling, GSD preserved). The label CSV boxes are in full-tile-normalized
space (see single_tiles_flat/geojson_to_labels.py); they are remapped into the
crop frame and boxes whose center falls outside the crop are dropped.

Example:
  python eval_checkpoints.py --split prefire --n-samples 6
  python eval_checkpoints.py --split prefire --compare --n-samples 5
"""
import argparse
import importlib
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from src.data.clay_meta import encode_latlon, encode_time  # noqa: E402
from src.eval.metrics import (  # noqa: E402
    compute_count_metrics,
    compute_map,
    compute_precision_recall_curve,
    compute_ap,
    match_predictions_to_gt,
)

REPO = HERE.parent  # .../mortalitree/mortalitree
QUARTER = 128       # patch-quartering: 256 parent -> 2x2 grid of 128px quarters
NAIP_WAVELENGTHS = torch.tensor([0.665, 0.560, 0.493, 0.842])
NAIP_MEAN = np.array([110.16, 115.41, 98.15, 139.04], dtype=np.float32)
NAIP_STD = np.array([47.23, 39.82, 35.43, 49.86], dtype=np.float32)
NAIP_GSD = 1.0
TILE = 256

# Clay v1.5 large encoder args (the released v1.5.0 checkpoint; patch size 8).
CLAY_LARGE = dict(dim=1024, depth=24, heads=16, dim_head=64, mlp_ratio=4)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def _remap_boxes_to_crop(boxes_full, x0, y0, W, H, C):
    """Map full-tile-normalized cxcywh boxes into a [x0:x0+C, y0:y0+C] crop.

    Boxes whose center lands outside the crop are dropped. Returns crop-normalized
    cxcywh in [0,1] (extent clipped to the crop edge).
    """
    if len(boxes_full) == 0:
        return np.zeros((0, 4), np.float32)
    cx = boxes_full[:, 0] * W
    cy = boxes_full[:, 1] * H
    w = boxes_full[:, 2] * W
    h = boxes_full[:, 3] * H
    ncx = (cx - x0) / C
    ncy = (cy - y0) / C
    nw = w / C
    nh = h / C
    inside = (ncx >= 0) & (ncx <= 1) & (ncy >= 0) & (ncy <= 1)
    out = np.stack([ncx, ncy, nw, nh], axis=1).astype(np.float32)[inside]
    # Clip box extent so corners stay within the crop for IoU/area sanity.
    x1 = np.clip(out[:, 0] - out[:, 2] / 2, 0, 1)
    y1 = np.clip(out[:, 1] - out[:, 3] / 2, 0, 1)
    x2 = np.clip(out[:, 0] + out[:, 2] / 2, 0, 1)
    y2 = np.clip(out[:, 1] + out[:, 3] / 2, 0, 1)
    return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1).astype(np.float32)


class GoodTileDataset:
    """One sample per tile from a {split}_labels.csv + good/{split}_img/*.tif.

    Each native-resolution (0.6 m/px) tile is center-cropped to ``crop_size``
    (default 256) with no resampling, so the model sees imagery at the same GSD
    and pixel scale as its Stage-B training patches. GT boxes are remapped into
    the crop frame. ``max_boxes`` filters on the number of boxes *inside the crop*
    (the model's Q queries cap detections within its actual input window).
    """

    def __init__(self, labels_csv, img_dir, max_boxes=None, crop_size=TILE,
                 source_tag=""):
        import pandas as pd
        import rasterio

        self.img_dir = Path(img_dir)
        self.crop_size = crop_size
        df = pd.read_csv(labels_csv)

        # Full-tile-normalized cxcywh (label px are in a 256px full-tile space;
        # /256 gives the resolution-independent [0,1] frame the native tif shares).
        ts = 256.0
        cx = (df["xmin"] + df["xmax"]).to_numpy(np.float32) / 2.0 / ts
        cy = (df["ymin"] + df["ymax"]).to_numpy(np.float32) / 2.0 / ts
        w = (df["xmax"] - df["xmin"]).to_numpy(np.float32) / ts
        h = (df["ymax"] - df["ymin"]).to_numpy(np.float32) / ts
        df = df.assign(_cx=cx, _cy=cy, _w=w, _h=h)

        C = crop_size
        self.items = []
        for imgname, g in df.groupby("imgname", sort=False):
            tif = self.img_dir / f"{imgname}.tif"
            if not tif.exists():
                continue
            with rasterio.open(tif) as src:           # metadata only (no pixels)
                W, H = src.width, src.height
            x0 = max(0, (W - C) // 2)
            y0 = max(0, (H - C) // 2)
            boxes_full = np.stack(
                [g["_cx"].to_numpy(), g["_cy"].to_numpy(),
                 g["_w"].to_numpy(), g["_h"].to_numpy()], axis=1
            ).astype(np.float32)
            boxes = _remap_boxes_to_crop(boxes_full, x0, y0, W, H, C)
            if max_boxes is not None and len(boxes) > max_boxes:
                continue
            r0 = g.iloc[0]
            self.items.append({
                "imgname": f"{source_tag}{imgname}",
                "tile_path": str(tif),
                "boxes": boxes,
                "crop": (x0, y0),
                "lat": float(r0["lat"]),
                "lon": float(r0["lon"]),
                "week": 26,  # NAIP acquisition month unknown -> mid-year proxy
            })

    def __len__(self):
        return len(self.items)

    def _load_crop(self, path, x0, y0):
        """Read the native 4-band tile and return the (4, C, C) raw-DN crop."""
        import rasterio
        C = self.crop_size
        with rasterio.open(path) as src:
            crop = src.read(
                indexes=[1, 2, 3, 4],
                window=rasterio.windows.Window(x0, y0, C, C),
            ).astype(np.float32)
        # Pad if a tile were smaller than the crop (not expected for these tiles).
        if crop.shape[1] != C or crop.shape[2] != C:
            padded = np.zeros((4, C, C), np.float32)
            padded[:, : crop.shape[1], : crop.shape[2]] = crop
            crop = padded
        return crop

    def __getitem__(self, idx):
        item = self.items[idx]
        x0, y0 = item["crop"]
        raw = self._load_crop(item["tile_path"], x0, y0)       # (4, C, C) DN
        mean = NAIP_MEAN.reshape(4, 1, 1)
        std = NAIP_STD.reshape(4, 1, 1)
        pixels = torch.from_numpy((raw - mean) / std)
        return {
            "pixels": pixels,
            "wavelengths": NAIP_WAVELENGTHS,
            "gsd": torch.tensor([NAIP_GSD], dtype=torch.float32),
            "time": encode_time(item["week"], 12.0),
            "latlon": encode_latlon(item["lat"], item["lon"]),
            "boxes": torch.from_numpy(item["boxes"]),
            "imgname": item["imgname"],
            "tile_path": item["tile_path"],
            "rgb": display_rgb(raw),                            # (C, C, 3) uint8 stretched
            "rgb_raw": raw[:3].astype(np.uint8).transpose(1, 2, 0),  # (C,C,3) DN for DeepForest
        }


def display_rgb(raw_dn):
    """Percentile-stretch the raw R,G,B DN bands to an (H, W, 3) uint8 image."""
    rgb = raw_dn[:3].astype(np.float32)
    out = np.zeros_like(rgb)
    for i in range(3):
        lo, hi = np.percentile(rgb[i], 2), np.percentile(rgb[i], 98)
        if hi > lo:
            out[i] = np.clip((rgb[i] - lo) / (hi - lo) * 255, 0, 255)
    return out.astype(np.uint8).transpose(1, 2, 0)


def collate(items):
    return {
        "pixels": torch.stack([b["pixels"] for b in items]),
        "wavelengths": items[0]["wavelengths"],
        "gsd": torch.cat([b["gsd"] for b in items]),
        "time": torch.stack([b["time"] for b in items]),
        "latlon": torch.stack([b["latlon"] for b in items]),
        "boxes": [b["boxes"] for b in items],
        "imgnames": [b["imgname"] for b in items],
        "rgbs": [b["rgb"] for b in items],
        "rgb_raws": [b["rgb_raw"] for b in items],
    }


def split_sources(split, labels_csv, img_dir):
    """Resolve a split to a list of (labels_csv, img_dir, tag) sources.

    'all' combines prefire + postfire; a single split yields one source.
    """
    splits = ["prefire", "postfire"] if split == "all" else [split]
    sources = []
    for s in splits:
        csv = labels_csv or REPO / f"single_tiles_flat/{s}_labels.csv"
        img = img_dir or REPO / f"single_tiles_flat/good/{s}_img"
        tag = f"{s}/" if split == "all" else ""
        sources.append((csv, img, tag))
    return sources


def build_dataset(split, labels_csv, img_dir, max_boxes, crop_size=TILE):
    """Build a (possibly combined) GoodTileDataset for the split.

    For 'all', tiles from both prefire and postfire are merged into one dataset
    so metrics are computed over the union. Per-source tiles keep their own
    absolute paths, so the merge is just a concatenation of item lists.
    """
    ds = None
    for csv, img, tag in split_sources(split, labels_csv, img_dir):
        d = GoodTileDataset(csv, img, max_boxes=max_boxes, crop_size=crop_size,
                            source_tag=tag)
        if ds is None:
            ds = d
        else:
            ds.items.extend(d.items)
    return ds


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def parse_q(checkpoint_path):
    """Parse Q from a 'q<NNN>...' filename, e.g. q150_final.pt -> 150."""
    stem = Path(checkpoint_path).stem
    head = stem.split("_")[0]
    if not (head.startswith("q") and head[1:].isdigit()):
        raise ValueError(f"Cannot parse Q from checkpoint name '{stem}' "
                         f"(expected leading 'q<NNN>').")
    return int(head[1:])


def q_checkpoints(ckpt_dir):
    """The Clay fixed-query detector checkpoints (q<NNN>_*.pt) in a directory.

    Skips other artifacts that may share the folder (dq-detr_final.pt, clay.ckpt,
    clay_ckpt.pt), which are not fixed-query TreeDetector checkpoints.
    """
    return [p for p in sorted(Path(ckpt_dir).glob("*.pt"))
            if re.match(r"^q\d+", p.stem)]


def build_model(num_queries, checkpoint_path, device):
    """Build a TreeDetector with the given Q and load full weights from ckpt.

    The Stage-B checkpoint already contains the Clay encoder weights, so we build
    the encoder architecture directly (no separate clay-v1.5.ckpt needed) and let
    the checkpoint's state_dict populate encoder + neck + head.
    """
    from claymodel.model import Encoder
    from src.model.detector import TreeDetector

    encoder = Encoder(mask_ratio=0.0, patch_size=8, shuffle=False, **CLAY_LARGE)
    model = TreeDetector(
        encoder,
        neck_in_channels=1024,
        neck_out_channels=128,
        num_queries=num_queries,
        hidden=128,
        n_heads=4,
        n_decoder_layers=3,
        dropout=0.1,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    crit = [k for k in missing if "encoder.encoder" in k or "head." in k or "neck." in k]
    if crit:
        raise RuntimeError(f"Checkpoint load missing critical weights: {crit[:5]} ...")
    model.eval().to(device)
    return model


@torch.no_grad()
def run_inference(model, dataset, device, batch_size=4):
    """Run the model over the whole dataset.

    Returns dict of parallel lists: pred_boxes, pred_scores (sorted desc),
    gt_boxes, imgnames, rgbs, latlons.
    """
    pred_boxes, pred_scores, gt_boxes = [], [], []
    imgnames, rgbs = [], []
    items = [dataset[i] for i in range(len(dataset))]
    for start in range(0, len(items), batch_size):
        chunk = items[start:start + batch_size]
        batch = collate(chunk)
        gpu = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
               for k, v in batch.items()
               if k in ("pixels", "wavelengths", "gsd", "time", "latlon")}
        cls_logits, boxes = model(gpu)
        scores = cls_logits.sigmoid().cpu()
        boxes = boxes.cpu()
        for i, it in enumerate(chunk):
            pred_boxes.append(boxes[i])
            pred_scores.append(scores[i])
            gt_boxes.append(it["boxes"])
            imgnames.append(it["imgname"])
            rgbs.append(it["rgb"])
    return {
        "pred_boxes": pred_boxes, "pred_scores": pred_scores,
        "gt_boxes": gt_boxes, "imgnames": imgnames, "rgbs": rgbs,
    }


# --------------------------------------------------------------------------- #
# DQ-DETR model (separate implementation in dq-detr-impl/)
# --------------------------------------------------------------------------- #
_DQDETR_PKG = None


def _load_dqdetr_pkg():
    """Import dq-detr-impl/src as an aliased package ('dqdetr_src').

    It is imported under a distinct name so its `src` package does not collide
    with this repo's already-imported `src`. Submodules use relative imports, so
    they resolve within the alias regardless of its name.
    """
    global _DQDETR_PKG
    if _DQDETR_PKG is not None:
        return _DQDETR_PKG
    root = REPO / "dq-detr-impl" / "src"
    if not root.exists():
        raise FileNotFoundError(f"dq-detr-impl source not found at {root}")
    pkg = "dqdetr_src"
    spec = importlib.util.spec_from_file_location(
        pkg, str(root / "__init__.py"),
        submodule_search_locations=[str(root)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[pkg] = module
    spec.loader.exec_module(module)
    _DQDETR_PKG = (
        importlib.import_module(f"{pkg}.model.detector"),
        importlib.import_module(f"{pkg}.model.clay_loader"),
    )
    return _DQDETR_PKG


def build_dqdetr_model(dqdetr_ckpt, clay_ckpt, config_path, device,
                       enable_cgfe=True):
    """Build the DQ-DETR TreeDetector: Clay encoder (from clay.ckpt) + the
    from-scratch neck/CCM/CGFE/DQS head weights (from dq-detr_final.pt).

    The DQ-DETR checkpoint holds no encoder weights, so the encoder is loaded
    separately from the full Clay v1.5 checkpoint.
    """
    import yaml
    detector_mod, clay_loader_mod = _load_dqdetr_pkg()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    m = cfg["model"]

    encoder = clay_loader_mod.load_clay_encoder(str(clay_ckpt))
    model = detector_mod.TreeDetector(
        encoder,
        neck_in_channels=m["neck_in_channels"],
        neck_out_channels=m["neck_out_channels"],
        hidden=m["hidden"],
        n_heads=m["n_heads"],
        n_decoder_layers=m["n_decoder_layers"],
        dropout=m["dropout"],
        dynamic_query_list=tuple(m.get("dynamic_query_list", (200, 400, 600))),
        ccm_cls_num=m.get("ccm_cls_num", 3),
        ccm_params=tuple(m.get("ccm_params", (100, 300))),
        anchor_size=m.get("anchor_size", 0.05),
        enable_cgfe=enable_cgfe,   # CGFE was on at the end of training (phase 2)
    )
    ck = torch.load(dqdetr_ckpt, map_location="cpu", weights_only=False)
    state = ck["model"] if "model" in ck else ck
    missing, _ = model.load_state_dict(state, strict=False)
    crit = [k for k in missing
            if any(k.startswith(p) for p in ("head.", "ccm.", "cgfe.", "neck."))]
    if crit:
        raise RuntimeError(f"DQ-DETR load missing critical weights: {crit[:5]} ...")
    model.eval().to(device)
    return model


@torch.no_grad()
def dqdetr_predict(model, items, device, conf_thresh):
    """Per-tile DQ-DETR predictions -> list of (N_kept, 4) normalized cxcywh.

    Run one tile at a time: Dynamic Query Selection picks the query count from the
    per-image predicted count category, so per-tile inference avoids batching
    tiles of different densities into one (wider) query tensor.
    """
    out = []
    for it in items:
        batch = collate([it])
        gpu = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
               for k, v in batch.items()
               if k in ("pixels", "wavelengths", "gsd", "time", "latlon")}
        res = model(gpu)
        scores = res["cls_logits"][0].sigmoid().cpu()
        boxes = res["pred_boxes"][0].cpu()
        keep = scores >= conf_thresh
        out.append(boxes[keep].numpy())
    return out


@torch.no_grad()
def run_inference_dqdetr(model, dataset, device):
    """Run DQ-DETR over the dataset, returning the same dict shape as
    run_inference (unfiltered per-tile boxes + sigmoid scores) so the standard
    evaluate()/save_sample_panels() consume it unchanged. One tile at a time
    because DQS sizes the query count per tile.
    """
    pred_boxes, pred_scores, gt_boxes, imgnames, rgbs = [], [], [], [], []
    for i in range(len(dataset)):
        it = dataset[i]
        batch = collate([it])
        gpu = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
               for k, v in batch.items()
               if k in ("pixels", "wavelengths", "gsd", "time", "latlon")}
        res = model(gpu)
        pred_boxes.append(res["pred_boxes"][0].cpu())
        pred_scores.append(res["cls_logits"][0].sigmoid().cpu())
        gt_boxes.append(it["boxes"])
        imgnames.append(it["imgname"])
        rgbs.append(it["rgb"])
    return {
        "pred_boxes": pred_boxes, "pred_scores": pred_scores,
        "gt_boxes": gt_boxes, "imgnames": imgnames, "rgbs": rgbs,
    }


# --------------------------------------------------------------------------- #
# Patch-quartering model (same fixed-query detector, run on 128px quarters)
# --------------------------------------------------------------------------- #
def build_quarter_model(checkpoint_path, device):
    """Build the quartering detector and load quarter_final.pt.

    Architecturally identical to the fixed-query model (clay TreeDetector); the
    quartering is purely an inference-time tiling. num_queries is read off the
    checkpoint's query embedding so 150/500-query checkpoints both work.
    """
    from claymodel.model import Encoder
    from src.model.detector import TreeDetector

    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ck["model"] if "model" in ck else ck
    num_queries = state["head.queries.weight"].shape[0]
    encoder = Encoder(mask_ratio=0.0, patch_size=8, shuffle=False, **CLAY_LARGE)
    model = TreeDetector(
        encoder, neck_in_channels=1024, neck_out_channels=128,
        num_queries=num_queries, hidden=128, n_heads=4, n_decoder_layers=3,
        dropout=0.1,
    )
    missing, _ = model.load_state_dict(state, strict=False)
    crit = [k for k in missing
            if "encoder.encoder" in k or k.startswith(("neck.", "head."))]
    if crit:
        raise RuntimeError(f"Quarter checkpoint missing critical weights: {crit[:5]} ...")
    model.eval().to(device)
    return model, num_queries


@torch.no_grad()
def quarter_stitch(model, item, device, nms_iou=0.6):
    """Split a 256 tile into 4x128 quarters, run the model on each, map the
    predictions back to the parent 256 frame and NMS-merge them.

    Returns (boxes_256, scores) — normalized cxcywh over the 256 frame, NMS'd but
    not yet confidence-filtered (so the caller can sweep thresholds).
    """
    from torchvision.ops import box_convert, nms

    pixels = item["pixels"]                      # (4, 256, 256) normalized
    quarters, offs = [], []
    for oy in (0, QUARTER):
        for ox in (0, QUARTER):
            quarters.append(pixels[:, oy:oy + QUARTER, ox:ox + QUARTER])
            offs.append((ox, oy))
    px = torch.stack(quarters).to(device)        # (4, 4, 128, 128)
    n = px.shape[0]
    batch = {
        "pixels": px,
        "wavelengths": item["wavelengths"],
        "gsd": item["gsd"].repeat(n).to(device),
        "time": item["time"].unsqueeze(0).repeat(n, 1).to(device),
        "latlon": item["latlon"].unsqueeze(0).repeat(n, 1).to(device),
    }
    cls_logits, boxes = model(batch)             # (4, Q), (4, Q, 4) tile-norm
    all_b, all_s = [], []
    for i, (ox, oy) in enumerate(offs):
        b = boxes[i].cpu().clone()
        b[:, 0] = (ox + b[:, 0] * QUARTER) / 256.0
        b[:, 1] = (oy + b[:, 1] * QUARTER) / 256.0
        b[:, 2] = b[:, 2] * QUARTER / 256.0
        b[:, 3] = b[:, 3] * QUARTER / 256.0
        all_b.append(b)
        all_s.append(cls_logits[i].sigmoid().cpu())
    bs = torch.cat(all_b, dim=0)
    ss = torch.cat(all_s, dim=0)
    if bs.shape[0] > 0:
        keep = nms(box_convert(bs, "cxcywh", "xyxy"), ss, nms_iou)
        bs, ss = bs[keep], ss[keep]
    return bs, ss


def run_inference_quarter(model, dataset, device, nms_iou=0.6):
    """Quartered+stitched inference over the dataset; same dict shape as
    run_inference so evaluate()/save_sample_panels() consume it unchanged."""
    pred_boxes, pred_scores, gt_boxes, imgnames, rgbs = [], [], [], [], []
    for i in range(len(dataset)):
        it = dataset[i]
        bs, ss = quarter_stitch(model, it, device, nms_iou)
        pred_boxes.append(bs)
        pred_scores.append(ss)
        gt_boxes.append(it["boxes"])
        imgnames.append(it["imgname"])
        rgbs.append(it["rgb"])
    return {
        "pred_boxes": pred_boxes, "pred_scores": pred_scores,
        "gt_boxes": gt_boxes, "imgnames": imgnames, "rgbs": rgbs,
    }


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def precision_recall_f1(pred_boxes, pred_scores, gt_boxes, iou_thresh, conf_thresh):
    """Aggregate precision/recall/F1 at fixed conf + IoU across all tiles."""
    tp = fp = fn = 0
    for pb, ps, gb in zip(pred_boxes, pred_scores, gt_boxes):
        keep = ps >= conf_thresh
        flags, _ = match_predictions_to_gt(pb[keep], ps[keep], gb, iou_thresh)
        tp += sum(flags)
        fp += sum(1 - int(t) for t in flags)
        fn += max(0, gb.shape[0] - sum(flags))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {"precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def best_threshold(pred_boxes, pred_scores, gt_boxes, iou_thresh,
                   thresholds=None):
    """Sweep confidence thresholds; return the one maximizing F1."""
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    best = {"f1": -1.0}
    for t in thresholds:
        pr = precision_recall_f1(pred_boxes, pred_scores, gt_boxes, iou_thresh, t)
        if pr["f1"] > best["f1"]:
            best = {**pr, "threshold": float(t)}
    return best


def evaluate(res, iou_thresh, conf_thresh):
    pb, ps, gb = res["pred_boxes"], res["pred_scores"], res["gt_boxes"]
    pr = precision_recall_f1(pb, ps, gb, iou_thresh, conf_thresh)
    swept = best_threshold(pb, ps, gb, iou_thresh)
    mp = compute_map(pb, ps, gb)
    counts = compute_count_metrics(pb, ps, gb, conf_thresh)
    prec_c, rec_c, _ = compute_precision_recall_curve(pb, ps, gb, iou_thresh)
    return {
        "n_tiles": len(gb),
        "n_gt_boxes": int(sum(g.shape[0] for g in gb)),
        "conf_thresh": conf_thresh,
        "iou_thresh": iou_thresh,
        "precision": pr["precision"], "recall": pr["recall"], "f1": pr["f1"],
        "tp": pr["tp"], "fp": pr["fp"], "fn": pr["fn"],
        "best_f1": swept["f1"], "best_threshold": swept["threshold"],
        "best_precision": swept["precision"], "best_recall": swept["recall"],
        "mAP50": mp["mAP50"], "mAP50_95": mp["mAP50_95"],
        "ap50": compute_ap(prec_c, rec_c),
        "count_mae": counts["mae"], "count_rmse": counts["rmse"],
        "count_r2": counts["r2"],
    }


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #
def draw(ax, rgb, gt=None, preds=None, scores=None,
         gt_color="#1f77ff", pred_color="#00ff44", title=""):
    import matplotlib.patches as mpatches
    H, W = rgb.shape[:2]
    ax.imshow(rgb)
    ax.set_title(title, fontsize=9)
    ax.axis("off")

    def rect(box, color, lw=0.8):
        cx, cy, w, h = box
        ax.add_patch(mpatches.Rectangle(
            ((cx - w / 2) * W, (cy - h / 2) * H), w * W, h * H,
            fill=False, edgecolor=color, linewidth=lw))

    if gt is not None:
        for b in gt:
            rect(b, gt_color)
    if preds is not None:
        for b in preds:
            rect(b, pred_color, lw=1.0)


def filter_by_conf(boxes, scores, conf):
    keep = scores >= conf
    return boxes[keep].numpy(), scores[keep].numpy()


def save_sample_panels(res, out_dir, tag, conf_thresh, n_samples, seed=0):
    """One row per sample tile: [GT] | [predictions over GT]."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    n = min(n_samples, len(res["gt_boxes"]))
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(res["gt_boxes"]), size=n, replace=False)

    fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n))
    if n == 1:
        axes = axes[None, :]
    for row, i in enumerate(idxs):
        rgb = res["rgbs"][i]
        gt = res["gt_boxes"][i].numpy()
        pb, ps = filter_by_conf(res["pred_boxes"][i], res["pred_scores"][i], conf_thresh)
        name = res["imgnames"][i]
        draw(axes[row, 0], rgb, gt=gt,
             title=f"{name}  GT={len(gt)}")
        draw(axes[row, 1], rgb, gt=gt, preds=pb,
             title=f"{name}  pred={len(pb)} @conf>={conf_thresh}")
    fig.suptitle(f"{tag}: ground truth (blue) vs predictions (green)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    path = out_dir / f"{tag}_samples.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {n} sample panels -> {path}")


# --------------------------------------------------------------------------- #
# DeepForest
# --------------------------------------------------------------------------- #
def load_deepforest():
    from deepforest import main
    m = main.deepforest()
    m.load_model(model_name="weecology/deepforest-tree", revision="main")
    return m


def deepforest_boxes(model, rgb_hwc, patch_size=256, patch_overlap=0.30,
                     score_thresh=0.15):
    """Run DeepForest on a native-res RGB crop (H,W,3 uint8); return normalized cxcywh."""
    import tempfile
    import rasterio
    from rasterio.transform import from_origin

    H, W = rgb_hwc.shape[:2]
    chw = rgb_hwc.transpose(2, 0, 1)  # (3, H, W)
    profile = dict(driver="GTiff", height=H, width=W, count=3, dtype="uint8",
                   transform=from_origin(0, H, 1, 1), crs="EPSG:32610")
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=True) as tmp:
        with rasterio.open(tmp.name, "w", **profile) as dst:
            dst.write(chw)
        ps = min(patch_size, H, W)
        preds = model.predict_tile(path=tmp.name, patch_size=ps,
                                    patch_overlap=patch_overlap)
    if preds is None or len(preds) == 0:
        return np.zeros((0, 4), np.float32)
    preds = preds[preds["score"] >= score_thresh]
    if len(preds) == 0:
        return np.zeros((0, 4), np.float32)
    if "xmin" in preds.columns:
        xmin, ymin = preds["xmin"].to_numpy(), preds["ymin"].to_numpy()
        xmax, ymax = preds["xmax"].to_numpy(), preds["ymax"].to_numpy()
    else:  # GeoDataFrame: derive from geometry bounds
        b = preds.geometry.bounds
        xmin, ymin, xmax, ymax = (b["minx"].to_numpy(), b["miny"].to_numpy(),
                                  b["maxx"].to_numpy(), b["maxy"].to_numpy())
    cx = (xmin + xmax) / 2.0 / W
    cy = (ymin + ymax) / 2.0 / H
    w = (xmax - xmin) / W
    h = (ymax - ymin) / H
    return np.stack([cx, cy, w, h], axis=1).astype(np.float32)


# --------------------------------------------------------------------------- #
# Comparison mode
# --------------------------------------------------------------------------- #
def comparison_mode(args, device):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ckpt_dir = Path(args.checkpoints_dir)
    ckpts = {parse_q(p): p for p in q_checkpoints(ckpt_dir)}
    if not ckpts:
        raise SystemExit(f"No q<NNN>_*.pt checkpoints in {ckpt_dir}")

    # Filter to tiles with < compare-max-ann annotations (combined for 'all').
    ds = build_dataset(args.split, args.labels_csv, args.img_dir,
                       max_boxes=args.compare_max_ann - 1)
    if len(ds) == 0:
        raise SystemExit("No tiles after annotation filter.")
    print(f"[compare] {len(ds)} tiles with < {args.compare_max_ann} annotations")

    rng = np.random.default_rng(args.seed)
    n = min(args.n_samples, len(ds))
    sample_idx = rng.choice(len(ds), size=n, replace=False)
    samples = [ds[i] for i in sample_idx]

    # Per-checkpoint predictions on the sample tiles (sorted Q for stable order).
    model_preds = {}
    for q in sorted(ckpts):
        print(f"[compare] running q{q} ...")
        model = build_model(q, ckpts[q], device)
        per_tile = []
        with torch.no_grad():
            for it in samples:
                batch = collate([it])
                gpu = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                       for k, v in batch.items()
                       if k in ("pixels", "wavelengths", "gsd", "time", "latlon")}
                cls_logits, boxes = model(gpu)
                pb, _ = filter_by_conf(boxes[0].cpu(),
                                       cls_logits[0].sigmoid().cpu(), args.conf_thresh)
                per_tile.append(pb)
        model_preds[q] = per_tile
        del model
        if device == "mps":
            torch.mps.empty_cache()

    # DQ-DETR (separate impl; Clay encoder from clay.ckpt + dq-detr_final.pt head).
    dq_preds = None
    if not args.no_dqdetr:
        dq_ckpt = Path(args.dqdetr_checkpoint)
        clay_ckpt = Path(args.clay_ckpt)
        if not dq_ckpt.exists():
            print(f"[compare] DQ-DETR checkpoint {dq_ckpt} not found — skipping column.")
        elif not clay_ckpt.exists():
            print(f"[compare] Clay encoder ckpt {clay_ckpt} not found — skipping DQ-DETR.")
        else:
            try:
                print(f"[compare] running DQ-DETR (encoder from {clay_ckpt.name}) ...")
                dq_model = build_dqdetr_model(
                    dq_ckpt, clay_ckpt, args.dqdetr_config, device,
                    enable_cgfe=not args.dqdetr_cgfe_off)
                dq_preds = dqdetr_predict(dq_model, samples, device, args.conf_thresh)
                del dq_model
                if device == "mps":
                    torch.mps.empty_cache()
            except Exception as e:
                print(f"[compare] DQ-DETR failed ({type(e).__name__}: {e}) — skipping column.")
                dq_preds = None

    # Patch-quartering (same fixed-query weights, run per-quarter + stitched).
    quarter_preds = None
    if not args.no_quarter:
        qckpt = Path(args.quarter_checkpoint)
        if not qckpt.exists():
            print(f"[compare] quarter checkpoint {qckpt} not found — skipping column.")
        else:
            try:
                print("[compare] running quartering model ...")
                qmodel, _ = build_quarter_model(qckpt, device)
                quarter_preds = []
                for it in samples:
                    bs, ss = quarter_stitch(qmodel, it, device, args.quarter_nms_iou)
                    quarter_preds.append(bs[ss >= args.conf_thresh].numpy())
                del qmodel
                if device == "mps":
                    torch.mps.empty_cache()
            except Exception as e:
                print(f"[compare] quarter failed ({type(e).__name__}: {e}) — skipping column.")
                quarter_preds = None

    print("[compare] running DeepForest ...")
    df_model = load_deepforest()
    df_preds = [deepforest_boxes(df_model, it["rgb_raw"],
                                 score_thresh=args.deepforest_score)
                for it in samples]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    qs = sorted(ckpts)
    # columns: RGB, GT, [each q], [DQ-DETR], [quarter], DeepForest
    has_dq = dq_preds is not None
    has_quarter = quarter_preds is not None
    ncol = 2 + len(qs) + (1 if has_dq else 0) + (1 if has_quarter else 0) + 1
    fig, axes = plt.subplots(n, ncol, figsize=(3.2 * ncol, 3.4 * n))
    if n == 1:
        axes = axes[None, :]
    for row, it in enumerate(samples):
        rgb = it["rgb"]
        gt = it["boxes"].numpy()
        name = it["imgname"]
        draw(axes[row, 0], rgb, title=f"{name}\nRGB")
        draw(axes[row, 1], rgb, gt=gt, title=f"Ground truth ({len(gt)})")
        col = 2
        for q in qs:
            draw(axes[row, col], rgb, gt=gt, preds=model_preds[q][row],
                 title=f"q{q} pred ({len(model_preds[q][row])})")
            col += 1
        if has_dq:
            draw(axes[row, col], rgb, gt=gt, preds=dq_preds[row],
                 pred_color="#ff44dd",
                 title=f"DQ-DETR ({len(dq_preds[row])})")
            col += 1
        if has_quarter:
            draw(axes[row, col], rgb, gt=gt, preds=quarter_preds[row],
                 pred_color="#00ccff",
                 title=f"quarter ({len(quarter_preds[row])})")
            col += 1
        draw(axes[row, col], rgb, gt=gt, preds=df_preds[row],
             pred_color="#ffaa00", title=f"DeepForest ({len(df_preds[row])})")
    dq_legend = " vs DQ-DETR (magenta)" if has_dq else ""
    q_legend = " vs quarter (cyan)" if has_quarter else ""
    fig.suptitle(
        f"{args.split}: GT (blue) vs Clay checkpoints (green){dq_legend}{q_legend} "
        f"vs DeepForest (orange)  [conf>={args.conf_thresh}]", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    path = out_dir / f"compare_{args.split}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare] saved -> {path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def pick_device(requested):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def per_checkpoint_mode(args, device):
    import json

    ckpt_dir = Path(args.checkpoints_dir)
    ckpts = q_checkpoints(ckpt_dir)
    if not ckpts:
        raise SystemExit(f"No q<NNN>_*.pt checkpoints in {ckpt_dir}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for ckpt in ckpts:
        q = parse_q(ckpt)
        print(f"\n=== {ckpt.name}  (Q={q}, split={args.split}) ===")
        ds = build_dataset(args.split, args.labels_csv, args.img_dir, max_boxes=q)
        print(f"  {len(ds)} tiles with <= {q} labels")
        if len(ds) == 0:
            print("  (no tiles after filter, skipping)")
            continue
        model = build_model(q, ckpt, device)
        res = run_inference(model, ds, device, batch_size=args.batch_size)
        metrics = evaluate(res, args.iou_thresh, args.conf_thresh)
        for k, v in metrics.items():
            print(f"    {k:16s} {v:.4f}" if isinstance(v, float) else f"    {k:16s} {v}")
        save_sample_panels(res, out_dir, f"q{q}_{args.split}",
                           args.conf_thresh, args.n_samples, seed=args.seed)
        summary[ckpt.name] = metrics
        del model
        if device == "mps":
            torch.mps.empty_cache()

    # DQ-DETR row (separate impl; encoder from clay.ckpt). Filtered to tiles with
    # <= max(dynamic_query_list) labels — the densest tile DQS can fully cover.
    if not args.no_dqdetr:
        dq_ckpt, clay_ckpt = Path(args.dqdetr_checkpoint), Path(args.clay_ckpt)
        if not (dq_ckpt.exists() and clay_ckpt.exists()):
            print(f"\n[dqdetr] {dq_ckpt.name} or {clay_ckpt.name} missing — skipping DQ-DETR row.")
        else:
            import yaml
            with open(args.dqdetr_config) as f:
                qmax = max(yaml.safe_load(f)["model"].get(
                    "dynamic_query_list", (200, 400, 600)))
            print(f"\n=== {dq_ckpt.name}  (DQ-DETR, max_q={qmax}, split={args.split}) ===")
            ds = build_dataset(args.split, args.labels_csv, args.img_dir, max_boxes=qmax)
            print(f"  {len(ds)} tiles with <= {qmax} labels")
            if len(ds) > 0:
                try:
                    model = build_dqdetr_model(
                        dq_ckpt, clay_ckpt, args.dqdetr_config, device,
                        enable_cgfe=not args.dqdetr_cgfe_off)
                    res = run_inference_dqdetr(model, ds, device)
                    metrics = evaluate(res, args.iou_thresh, args.conf_thresh)
                    for k, v in metrics.items():
                        print(f"    {k:16s} {v:.4f}" if isinstance(v, float)
                              else f"    {k:16s} {v}")
                    save_sample_panels(res, out_dir, f"dqdetr_{args.split}",
                                       args.conf_thresh, args.n_samples, seed=args.seed)
                    summary[dq_ckpt.name] = metrics
                    del model
                    if device == "mps":
                        torch.mps.empty_cache()
                except Exception as e:
                    print(f"  DQ-DETR eval failed ({type(e).__name__}: {e}) — skipping row.")

    # Patch-quartering row: each 256 tile is split into 4x128 quarters, the model
    # runs per-quarter, predictions are stitched (mapped to 256 + NMS) and scored
    # against the parent GT. Filtered to <= 4*num_queries labels (total capacity).
    if not args.no_quarter:
        qckpt = Path(args.quarter_checkpoint)
        if not qckpt.exists():
            print(f"\n[quarter] {qckpt.name} not found — skipping quartering row "
                  f"(upload it to {qckpt.parent}).")
        else:
            try:
                model, nq = build_quarter_model(qckpt, device)
                cap = 4 * nq
                print(f"\n=== {qckpt.name}  (quarter, q/tile={nq}, split={args.split}) ===")
                ds = build_dataset(args.split, args.labels_csv, args.img_dir, max_boxes=cap)
                print(f"  {len(ds)} tiles with <= {cap} labels")
                if len(ds) > 0:
                    res = run_inference_quarter(model, ds, device, args.quarter_nms_iou)
                    metrics = evaluate(res, args.iou_thresh, args.conf_thresh)
                    for k, v in metrics.items():
                        print(f"    {k:16s} {v:.4f}" if isinstance(v, float)
                              else f"    {k:16s} {v}")
                    save_sample_panels(res, out_dir, f"quarter_{args.split}",
                                       args.conf_thresh, args.n_samples, seed=args.seed)
                    summary[qckpt.name] = metrics
                del model
                if device == "mps":
                    torch.mps.empty_cache()
            except Exception as e:
                print(f"  quarter eval failed ({type(e).__name__}: {e}) — skipping row.")

    with open(out_dir / f"metrics_{args.split}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote metrics -> {out_dir / f'metrics_{args.split}.json'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints-dir", default=str(HERE / "checkpoints"))
    ap.add_argument("--split", choices=["prefire", "postfire", "all"], default="prefire",
                    help="'all' merges prefire + postfire tiles into one eval set")
    ap.add_argument("--labels-csv", default=None,
                    help="override; default single_tiles_flat/{split}_labels.csv")
    ap.add_argument("--img-dir", default=None,
                    help="override; default single_tiles_flat/good/{split}_img")
    ap.add_argument("--conf-thresh", type=float, default=0.5)
    ap.add_argument("--iou-thresh", type=float, default=0.5)
    ap.add_argument("--n-samples", type=int, default=6,
                    help="number of tiles to visualize")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--out-dir", default=str(HERE / "eval_out"))
    ap.add_argument("--seed", type=int, default=0)
    # comparison mode
    ap.add_argument("--compare", action="store_true",
                    help="render RGB|GT|q150|q500|DeepForest panels")
    ap.add_argument("--compare-max-ann", type=int, default=150,
                    help="keep tiles with strictly fewer annotations than this")
    ap.add_argument("--deepforest-score", type=float, default=0.15)
    # DQ-DETR comparison model (separate impl in dq-detr-impl/)
    ap.add_argument("--dqdetr-checkpoint", default=str(HERE / "checkpoints/dq-detr_final.pt"))
    ap.add_argument("--clay-ckpt", default=str(HERE / "checkpoints/clay.ckpt"),
                    help="full Clay v1.5 checkpoint supplying the DQ-DETR encoder weights")
    ap.add_argument("--dqdetr-config",
                    default=str(REPO / "dq-detr-impl/configs/stage_b.yaml"))
    ap.add_argument("--dqdetr-cgfe-off", action="store_true",
                    help="disable CGFE at inference (default: on, matching end-of-training)")
    ap.add_argument("--no-dqdetr", action="store_true",
                    help="skip the DQ-DETR column in --compare")
    # Patch-quartering model
    ap.add_argument("--quarter-checkpoint", default=str(HERE / "checkpoints/quarter_final.pt"))
    ap.add_argument("--quarter-nms-iou", type=float, default=0.6,
                    help="NMS IoU for merging stitched quarter predictions")
    ap.add_argument("--no-quarter", action="store_true",
                    help="skip the patch-quartering row/column")
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"device: {device}")

    if args.split == "all" and (args.labels_csv or args.img_dir):
        raise SystemExit("--labels-csv/--img-dir cannot be combined with --split all "
                         "(paths are per-split); run a single split to override.")

    if args.compare:
        comparison_mode(args, device)
    else:
        per_checkpoint_mode(args, device)


if __name__ == "__main__":
    main()
