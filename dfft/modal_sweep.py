"""
Tier-0 inference-time sweep for the fine-tuned DeepForest checkpoint.

No retraining. Selection happens on VAL; TEST is touched only once at the end
to report the chosen config (and an IoU decomposition diagnostic).

Grid: transform.min_size x nms_thresh x score_thresh, scored at IoU 0.5/0.25/0.1.
Inference runs once per min_size with permissive thresholds; score/NMS are swept
in post-processing.

    modal run dfft/modal_sweep.py
"""

import json
from pathlib import Path

import modal

HERE = Path(__file__).parent
REPO = HERE.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("deepforest==2.1.0", "rasterio", "geopandas", "shapely", "pandas", "numpy", "pillow")
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_dir(REPO / "single_tiles_flat" / "stage_c_data", remote_path="/data/stage_c_data")
)

app = modal.App("deepforest-sweep", image=image)
ckpt_vol = modal.Volume.from_name("dfft-checkpoints", create_if_missing=True)

DATA, WORK, CKPT = "/data/stage_c_data", "/work/deepforest_data", "/checkpoints"

MIN_SIZES = [800, 1000, 1300, 1600]
NMS_THRESHS = [0.05, 0.15, 0.30, 0.50]
SCORE_THRESHS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
IOUS = [0.5, 0.25, 0.1]
REFERENCE = {"min_size": 800, "nms": 0.05, "score": 0.1}  # what produced the reported numbers


def _prep():
    import numpy as np, pandas as pd, rasterio
    from PIL import Image
    img_out = Path(WORK) / "images"; img_out.mkdir(parents=True, exist_ok=True)
    for tif in sorted((Path(DATA) / "tiles").glob("*.tif")):
        out = img_out / (tif.stem + ".png")
        if out.exists():
            continue
        with rasterio.open(tif) as src:
            rgb = src.read([1, 2, 3]).astype(np.float32)
        for i in range(3):
            lo, hi = np.percentile(rgb[i], 2), np.percentile(rgb[i], 98)
            if hi > lo:
                rgb[i] = (rgb[i] - lo) / (hi - lo)
            rgb[i] = np.clip(rgb[i], 0, 1)
        Image.fromarray((rgb * 255).astype(np.uint8).transpose(1, 2, 0)).save(out)
    for split in ["train", "val", "test"]:
        items = json.loads((Path(DATA) / f"{split}.json").read_text())
        rows = []
        for it in items:
            png = str(img_out / f"{Path(it['tile_path']).stem}.png")
            for cx, cy, w, h in it["boxes"]:
                rows.append({"image_path": png,
                             "xmin": max(0, int((cx - w/2)*256)), "ymin": max(0, int((cy - h/2)*256)),
                             "xmax": min(256, int((cx + w/2)*256)), "ymax": min(256, int((cy + h/2)*256)),
                             "label": "Tree"})
        pd.DataFrame(rows).to_csv(Path(WORK) / f"{split}.csv", index=False)


def _iou_matrix(a, b):
    import numpy as np
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def _greedy_match(preds, gt, iou_thr):
    """preds sorted by score desc. One-to-one greedy IoU matching -> (tp, fp, fn)."""
    import numpy as np
    tp = 0
    M = _iou_matrix(preds, gt)
    taken = np.zeros(len(gt), bool)
    for i in range(len(preds)):
        if M.shape[1] == 0:
            break
        order = np.argsort(-M[i])
        for j in order:
            if M[i, j] < iou_thr:
                break
            if not taken[j]:
                taken[j] = True; tp += 1; break
    return tp, len(preds) - tp, len(gt) - tp


def _score_grid(raw, gt_by_img, nms_t, score_t, iou_thr):
    """raw: {img: (boxes Nx4, scores N)} already at one min_size."""
    import numpy as np, torch
    from torchvision.ops import nms
    TP = FP = FN = 0; n_pred = 0; abs_err = []
    for img, (boxes, scores) in raw.items():
        gt = gt_by_img.get(img, np.zeros((0, 4)))
        keep = scores >= score_t
        b, s = boxes[keep], scores[keep]
        if len(b):
            k = nms(torch.tensor(b, dtype=torch.float32), torch.tensor(s, dtype=torch.float32), nms_t).numpy()
            b, s = b[k], s[k]
            o = np.argsort(-s); b = b[o]
        tp, fp, fn = _greedy_match(b, gt, iou_thr)
        TP += tp; FP += fp; FN += fn; n_pred += len(b)
        abs_err.append(abs(len(b) - len(gt)))
    p = TP / max(TP + FP, 1); r = TP / max(TP + FN, 1)
    return {"precision": round(p, 4), "recall": round(r, 4),
            "f1": round(2*p*r / max(p + r, 1e-9), 4),
            "tp": TP, "fp": FP, "fn": FN, "n_pred": n_pred,
            "count_mae": round(float(np.mean(abs_err)), 2)}


def _collect(model, split, min_size):
    """One inference pass with permissive thresholds -> raw boxes/scores per image."""
    import numpy as np, pandas as pd
    net = model.model
    net.transform.min_size = (min_size,)
    net.transform.max_size = max(1333, min_size * 2)
    net.score_thresh = 0.01; net.nms_thresh = 0.99
    net.detections_per_img = 3000; net.topk_candidates = 4000
    model.config["score_thresh"] = 0.01; model.config["nms_thresh"] = 0.99

    csv = str(Path(WORK) / f"{split}.csv")
    preds = model.predict_file(csv_file=csv, root_dir="/")
    preds["image_path"] = preds["image_path"].apply(lambda p: Path(p).name)
    out = {}
    gt = pd.read_csv(csv); gt["image_path"] = gt["image_path"].apply(lambda p: Path(p).name)
    for img in gt["image_path"].unique():
        d = preds[preds["image_path"] == img]
        out[img] = (d[["xmin", "ymin", "xmax", "ymax"]].to_numpy(float),
                    d["score"].to_numpy(float)) if len(d) else (np.zeros((0, 4)), np.zeros(0))
    return out


def _gt_by_img(split, min_gt=0):
    import numpy as np, pandas as pd
    gt = pd.read_csv(Path(WORK) / f"{split}.csv")
    gt["image_path"] = gt["image_path"].apply(lambda p: Path(p).name)
    out = {}
    for img, d in gt.groupby("image_path"):
        if len(d) >= min_gt:
            out[img] = d[["xmin", "ymin", "xmax", "ymax"]].to_numpy(float)
    return out


@app.function(cpu=8.0, memory=16384, timeout=60*60*5, volumes={CKPT: ckpt_vol})
def sweep():
    from deepforest import main as df_main
    _prep()
    model = df_main.deepforest.load_from_checkpoint(str(Path(CKPT) / "deepforest_finetuned.pt"))
    model.eval()

    val_gt = _gt_by_img("val")
    print(f"val: {len(val_gt)} tiles, {sum(len(v) for v in val_gt.values())} boxes")

    results = []
    for ms in MIN_SIZES:
        print(f"\n[inference] min_size={ms}")
        raw = _collect(model, "val", ms)
        print(f"  raw boxes: {sum(len(b) for b, _ in raw.values())}")
        for nt in NMS_THRESHS:
            for st in SCORE_THRESHS:
                for iou in IOUS:
                    m = _score_grid(raw, val_gt, nt, st, iou)
                    results.append({"min_size": ms, "nms": nt, "score": st, "iou": iou, **m})
        best = max([r for r in results if r["min_size"] == ms and r["iou"] == 0.5], key=lambda r: r["f1"])
        print(f"  best@IoU0.5: F1={best['f1']} (nms={best['nms']}, score={best['score']}) "
              f"P={best['precision']} R={best['recall']}")

    at50 = [r for r in results if r["iou"] == 0.5]
    best = max(at50, key=lambda r: r["f1"])
    ref = next(r for r in at50 if r["min_size"] == REFERENCE["min_size"]
               and r["nms"] == REFERENCE["nms"] and r["score"] == REFERENCE["score"])
    print(f"\n=== VAL best @IoU0.5: {best}")
    print(f"=== VAL reference:    {ref}")

    # TEST: reference vs val-selected config, plus IoU decomposition. Report convention (min_gt=5).
    test_gt = _gt_by_img("test", min_gt=5)
    print(f"\ntest (min_gt=5): {len(test_gt)} tiles, {sum(len(v) for v in test_gt.values())} boxes")
    test_out = {}
    for tag, cfg in [("reference", REFERENCE),
                     ("val_selected", {"min_size": best["min_size"], "nms": best["nms"], "score": best["score"]})]:
        raw_t = _collect(model, "test", cfg["min_size"])
        test_out[tag] = {"config": cfg,
                         "by_iou": {str(i): _score_grid(raw_t, test_gt, cfg["nms"], cfg["score"], i) for i in IOUS}}
        print(f"  {tag} {cfg}")
        for i in IOUS:
            m = test_out[tag]["by_iou"][str(i)]
            print(f"    IoU {i}: P={m['precision']} R={m['recall']} F1={m['f1']}")

    payload = {"val_grid": results, "val_best": best, "val_reference": ref, "test": test_out,
               "note": "selection on val; test evaluated once with min_gt=5 (report convention)"}
    (Path(CKPT) / "sweep.json").write_text(json.dumps(payload, indent=2))
    ckpt_vol.commit()
    return {"val_best": best, "val_reference": ref, "test": test_out}


@app.local_entrypoint()
def main():
    print(json.dumps(sweep.remote(), indent=2))
