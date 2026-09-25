"""
Re-derive the fine-tuned DeepForest baseline on Modal.

The original run (branch df-ft, commit f5b0cbe) was done locally and the
checkpoint `dfft/runs/deepforest_finetuned.pt` was gitignored and lost. This
ports `prep_data.py` + `train.py` + `eval.py` into a single Modal job so the
weights live on a volume instead of someone's laptop.

Usage:
    modal run dfft/modal_train.py                 # prep + train + eval
    modal run dfft/modal_train.py --epochs 30

Outputs (volume `dfft-checkpoints`):
    /deepforest_finetuned.pt   fine-tuned weights
    /metrics.json              baseline vs finetuned, both threshold conventions
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
    # 28 MB of 256x256 NAIP tiles + split JSONs — small enough to ship in the image.
    .add_local_dir(REPO / "single_tiles_flat" / "stage_c_data", remote_path="/data/stage_c_data")
)

app = modal.App("deepforest-finetune", image=image)
ckpt_vol = modal.Volume.from_name("dfft-checkpoints", create_if_missing=True)

DATA = "/data/stage_c_data"
WORK = "/work/deepforest_data"
CKPT = "/checkpoints"

IOU_THRESHOLD = 0.5
# eval.py defaults to score 0.3; export_stats.py (which produced the report's
# Table 1) uses 0.1 plus a MIN_GT=5 tile filter. Report both.
EVAL_CONVENTIONS = [
    {"name": "eval.py (score=0.3, all tiles)", "score_thresh": 0.3, "min_gt": 0},
    {"name": "export_stats.py (score=0.1, min_gt=5)", "score_thresh": 0.1, "min_gt": 5},
]


def _prep():
    """prep_data.py, rewritten against the Modal paths. TIFF -> RGB PNG + CSVs."""
    import numpy as np
    import pandas as pd
    import rasterio
    from PIL import Image

    img_out = Path(WORK) / "images"
    img_out.mkdir(parents=True, exist_ok=True)
    tiles = Path(DATA) / "tiles"

    for tif in sorted(tiles.glob("*.tif")):
        out_png = img_out / (tif.stem + ".png")
        if out_png.exists():
            continue
        with rasterio.open(tif) as src:
            rgb = src.read([1, 2, 3]).astype(np.float32)
        for i in range(3):
            lo, hi = np.percentile(rgb[i], 2), np.percentile(rgb[i], 98)
            if hi > lo:
                rgb[i] = (rgb[i] - lo) / (hi - lo)
            rgb[i] = np.clip(rgb[i], 0, 1)
        Image.fromarray((rgb * 255).astype(np.uint8).transpose(1, 2, 0)).save(out_png)

    counts = {}
    for split in ["train", "val", "test"]:
        items = json.loads((Path(DATA) / f"{split}.json").read_text())
        rows = []
        for item in items:
            stem = Path(item["tile_path"]).stem
            png = str(img_out / f"{stem}.png")
            for cx, cy, w, h in item["boxes"]:
                # normalized cx,cy,w,h -> pixel xyxy on a 256x256 tile
                rows.append({
                    "image_path": png,
                    "xmin": max(0, int((cx - w / 2) * 256)),
                    "ymin": max(0, int((cy - h / 2) * 256)),
                    "xmax": min(256, int((cx + w / 2) * 256)),
                    "ymax": min(256, int((cy + h / 2) * 256)),
                    "label": "Tree",
                })
        df = pd.DataFrame(rows)
        df.to_csv(Path(WORK) / f"{split}.csv", index=False)
        counts[split] = (len(items), len(df))
        print(f"  {split}: {len(items)} tiles, {len(df)} boxes")
    return counts


def _score(model, score_thresh, min_gt):
    """Box P/R/F1 at IoU 0.5 + count MAE/RMSE on the test split."""
    import numpy as np
    import pandas as pd
    from deepforest import evaluate

    model.config["score_thresh"] = score_thresh
    model.model.score_thresh = score_thresh  # config dict doesn't reach the live model

    test_csv = str(Path(WORK) / "test.csv")
    gt = pd.read_csv(test_csv)
    preds = model.predict_file(csv_file=test_csv, root_dir="/")

    preds = preds.copy()
    preds["image_path"] = preds["image_path"].apply(lambda p: Path(p).name)
    preds["label"] = "Tree"
    gt["image_path"] = gt["image_path"].apply(lambda p: Path(p).name)

    if min_gt > 0:
        keep = gt.groupby("image_path").size()
        keep = set(keep[keep >= min_gt].index)
        gt = gt[gt["image_path"].isin(keep)]
        preds = preds[preds["image_path"].isin(keep)]

    res = evaluate.evaluate_boxes(predictions=preds, ground_df=gt, iou_threshold=IOU_THRESHOLD)
    p, r = res["box_precision"], res["box_recall"]
    f1 = 2 * p * r / max(p + r, 1e-6)

    gt_c = gt.groupby("image_path").size()
    pr_c = preds.groupby("image_path").size() if len(preds) else pd.Series(dtype=int)
    idx = gt_c.index.union(pr_c.index)
    err = (pr_c.reindex(idx, fill_value=0) - gt_c.reindex(idx, fill_value=0)).abs()

    return {
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f1), 4),
        "count_mae": round(float(err.mean()), 2),
        "count_rmse": round(float(np.sqrt((err ** 2).mean())), 2),
        "n_pred": int(len(preds)),
        "n_gt": int(len(gt)),
        "n_tiles": int(gt["image_path"].nunique()),
    }


@app.function(cpu=8.0, memory=16384, timeout=60 * 60 * 5, volumes={CKPT: ckpt_vol})
def finetune(epochs: int = 15, lr: float = 0.001, batch_size: int = 4, seed: int = 42):
    import pytorch_lightning as pl
    from deepforest import main as df_main

    pl.seed_everything(seed, workers=True)  # original train.py had no seed

    print("=== Prep ===")
    counts = _prep()

    print("\n=== Baseline (pretrained NEON, no finetuning) ===")
    base = df_main.deepforest()
    base.load_model()
    baseline = {c["name"]: _score(base, c["score_thresh"], c["min_gt"]) for c in EVAL_CONVENTIONS}
    for k, v in baseline.items():
        print(f"  {k}: P={v['precision']} R={v['recall']} F1={v['f1']} MAE={v['count_mae']}")

    print(f"\n=== Fine-tuning: {epochs} epochs, lr={lr}, bs={batch_size}, seed={seed} ===")
    model = df_main.deepforest()
    model.load_model()
    model.config["train"]["csv_file"] = str(Path(WORK) / "train.csv")
    model.config["train"]["root_dir"] = "/"
    model.config["validation"]["csv_file"] = str(Path(WORK) / "val.csv")
    model.config["validation"]["root_dir"] = "/"
    model.config["train"]["epochs"] = epochs
    model.config["train"]["lr"] = lr
    model.config["batch_size"] = batch_size
    model.config["score_thresh"] = 0.1
    model.model.score_thresh = 0.1

    model.create_trainer(logger=None)
    model.trainer.fit(model)

    out = Path(CKPT) / "deepforest_finetuned.pt"
    model.save_model(str(out))
    ckpt_vol.commit()
    print(f"Saved {out} ({out.stat().st_size / 1e6:.1f} MB)")

    print("\n=== Fine-tuned ===")
    finetuned = {c["name"]: _score(model, c["score_thresh"], c["min_gt"]) for c in EVAL_CONVENTIONS}
    for k, v in finetuned.items():
        print(f"  {k}: P={v['precision']} R={v['recall']} F1={v['f1']} MAE={v['count_mae']}")

    metrics = {
        "config": {"epochs": epochs, "lr": lr, "batch_size": batch_size, "seed": seed,
                   "iou_threshold": IOU_THRESHOLD, "deepforest": "2.1.0"},
        "data": {k: {"tiles": v[0], "boxes": v[1]} for k, v in counts.items()},
        "baseline": baseline,
        "finetuned": finetuned,
        "report_table1_target": {"precision": 0.229, "recall": 0.208, "f1": 0.204},
    }
    (Path(CKPT) / "metrics.json").write_text(json.dumps(metrics, indent=2))
    ckpt_vol.commit()
    return metrics


@app.local_entrypoint()
def main(epochs: int = 15, lr: float = 0.001, batch_size: int = 4, seed: int = 42):
    m = finetune.remote(epochs=epochs, lr=lr, batch_size=batch_size, seed=seed)
    print("\n" + "=" * 72)
    print(json.dumps(m, indent=2))
