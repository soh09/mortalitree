"""
Compute per-image and aggregate precision/recall/F1 + count stats
for baseline and fine-tuned models. Outputs:
  runs/stats/per_image.csv   — one row per (model, image)
  runs/stats/aggregate.csv   — one row per model

Usage:
    python dfft/export_stats.py
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from deepforest import evaluate, main as df_main
from deepforest.evaluate import IoU

DATA_DIR = Path(__file__).parent.parent / "single_tiles_flat" / "deepforest_data"
RUNS_DIR = Path(__file__).parent / "runs"
OUT_DIR = RUNS_DIR / "stats"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IOU_THRESHOLD = 0.5
SCORE_THRESH = 0.1
MIN_GT = 5  # drop tiles with fewer than this many GT boxes (sparsely annotated)

MODELS = {
    "baseline": None,                              # pretrained NEON, no checkpoint
    "finetuned": str(RUNS_DIR / "deepforest_finetuned.pt"),
}


def load_model(ckpt_path):
    if ckpt_path is None:
        m = df_main.deepforest()
        m.load_model()
    else:
        m = df_main.deepforest.load_from_checkpoint(ckpt_path)
    m.config["score_thresh"] = SCORE_THRESH
    m.model.score_thresh = SCORE_THRESH
    return m


def to_gdf(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Convert xmin/ymin/xmax/ymax df to GeoDataFrame with box geometry."""
    from shapely.geometry import box
    df = df.copy()
    df["geometry"] = df.apply(
        lambda r: box(r.xmin, r.ymin, r.xmax, r.ymax), axis=1
    )
    return gpd.GeoDataFrame(df, geometry="geometry")


def per_image_stats(preds: pd.DataFrame, gt: pd.DataFrame, iou_threshold: float) -> pd.DataFrame:
    rows = []
    for img in gt["image_path"].unique():
        gt_img = gt[gt["image_path"] == img].copy()
        pred_img = preds[preds["image_path"] == img].copy() if len(preds) else pd.DataFrame()

        gt_count = len(gt_img)
        pred_count = len(pred_img)

        if pred_count == 0:
            rows.append({
                "image": img, "gt_count": gt_count, "pred_count": 0,
                "count_error": -gt_count, "precision": float("nan"),
                "recall": 0.0, "f1": 0.0,
            })
            continue

        # Convert to GeoDataFrame for IoU matching
        gt_gdf = to_gdf(gt_img.reset_index(drop=True))
        pred_gdf = to_gdf(pred_img.reset_index(drop=True))
        gt_gdf["image_path"] = img
        pred_gdf["image_path"] = img

        matched = IoU.match_polygons(gt_gdf, pred_gdf)
        tp = (matched["IoU"] >= iou_threshold).sum()

        precision = tp / pred_count if pred_count > 0 else float("nan")
        recall = tp / gt_count if gt_count > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        rows.append({
            "image": img,
            "gt_count": gt_count,
            "pred_count": pred_count,
            "count_error": pred_count - gt_count,
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
        })

    return pd.DataFrame(rows)


def main():
    test_csv = DATA_DIR / "test.csv"
    gt_raw = pd.read_csv(test_csv)
    gt_raw["image_path"] = gt_raw["image_path"].apply(lambda p: Path(p).name)

    # Drop sparsely annotated tiles
    gt_counts = gt_raw.groupby("image_path").size()
    sparse = gt_counts[gt_counts < MIN_GT].index.tolist()
    if sparse:
        print(f"Dropping {len(sparse)} sparse tiles (gt < {MIN_GT}): {sparse}")
    gt_raw = gt_raw[~gt_raw["image_path"].isin(sparse)]

    all_per_image = []
    agg_rows = []

    for model_name, ckpt in MODELS.items():
        print(f"\n=== {model_name} ===")
        model = load_model(ckpt)

        raw_preds = model.predict_file(csv_file=str(test_csv), root_dir="/")
        raw_preds = raw_preds.copy()
        raw_preds["image_path"] = raw_preds["image_path"].apply(lambda p: Path(p).name)
        raw_preds["label"] = "Tree"

        df = per_image_stats(raw_preds, gt_raw, IOU_THRESHOLD)
        df.insert(0, "model", model_name)
        all_per_image.append(df)

        # Print per-image table
        print(df.to_string(index=False))

        # Aggregate
        mae = df["count_error"].abs().mean()
        rmse = np.sqrt((df["count_error"] ** 2).mean())
        precision = df["precision"].mean(skipna=True)
        recall = df["recall"].mean()
        f1 = df["f1"].mean()
        agg_rows.append({
            "model": model_name,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "count_mae": round(mae, 3),
            "count_rmse": round(rmse, 3),
            "n_images": len(df),
            "total_gt": int(df["gt_count"].sum()),
            "total_pred": int(df["pred_count"].sum()),
        })

    per_image_df = pd.concat(all_per_image, ignore_index=True)
    agg_df = pd.DataFrame(agg_rows)

    per_img_path = OUT_DIR / "per_image.csv"
    agg_path = OUT_DIR / "aggregate.csv"
    per_image_df.to_csv(per_img_path, index=False)
    agg_df.to_csv(agg_path, index=False)

    print(f"\n=== Aggregate ===")
    print(agg_df.to_string(index=False))
    print(f"\nSaved:\n  {per_img_path}\n  {agg_path}")


if __name__ == "__main__":
    main()
