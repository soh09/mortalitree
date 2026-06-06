"""
Evaluate a fine-tuned (or pretrained) DeepForest model on the test split.

Reports:
  - Box-level precision, recall, F1 at IoU=0.5
  - Count MAE and RMSE per tile

Usage:
    python dfft/eval.py                            # pretrained baseline (no finetuning)
    python dfft/eval.py --model dfft/runs/deepforest_finetuned.pt
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from deepforest import main as df_main
from deepforest import evaluate

DATA_DIR = Path(__file__).parent.parent / "single_tiles_flat" / "deepforest_data"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default=None,
                   help="Path to fine-tuned .pt checkpoint. Omit to use pretrained release.")
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--score-threshold", type=float, default=0.3)
    return p.parse_args()


def align_formats(preds: pd.DataFrame, gt: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    predict_file returns bare filenames in image_path; gt has full absolute paths.
    Also, pred labels are int (class index), gt labels are strings.
    Normalize both to bare filename + string label so evaluate_boxes can match them.
    """
    preds = preds.copy()
    gt = gt.copy()
    preds["image_path"] = preds["image_path"].apply(lambda p: Path(p).name)
    gt["image_path"] = gt["image_path"].apply(lambda p: Path(p).name)
    # DeepForest evaluate_boxes requires labels to match; convert pred int → "Tree"
    preds["label"] = "Tree"
    return preds, gt


def count_metrics(pred_df: pd.DataFrame, gt_df: pd.DataFrame) -> dict:
    gt_counts = gt_df.groupby("image_path").size().rename("gt")
    pred_counts = (
        pred_df.groupby("image_path").size().rename("pred")
        if len(pred_df) else pd.Series(dtype=int, name="pred")
    )
    all_images = gt_counts.index.union(pred_counts.index)
    gt_counts = gt_counts.reindex(all_images, fill_value=0)
    pred_counts = pred_counts.reindex(all_images, fill_value=0)
    errors = (pred_counts - gt_counts).abs()
    return {
        "count_mae": round(float(errors.mean()), 3),
        "count_rmse": round(float(np.sqrt((errors ** 2).mean())), 3),
    }


def main():
    args = parse_args()

    if args.model:
        model = df_main.deepforest.load_from_checkpoint(args.model)
        print(f"Loaded fine-tuned model from {args.model}")
    else:
        model = df_main.deepforest()
        model.load_model()  # pretrained NEON weights (weecology/deepforest-tree)
        print("Using pretrained NEON release model (no fine-tuning)")

    model.config["score_thresh"] = args.score_threshold
    model.model.score_thresh = args.score_threshold  # config dict doesn't reach the live model

    test_csv = DATA_DIR / "test.csv"
    gt_df = pd.read_csv(test_csv)

    print(f"\nEvaluating on {test_csv} ...")
    raw_preds = model.predict_file(csv_file=str(test_csv), root_dir="/")

    preds, gt_norm = align_formats(raw_preds, gt_df)

    # Box metrics
    box_results = evaluate.evaluate_boxes(
        predictions=preds,
        ground_df=gt_norm,
        iou_threshold=args.iou_threshold,
    )
    precision = box_results["box_precision"]
    recall = box_results["box_recall"]
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)

    print(f"\n--- Box metrics (IoU={args.iou_threshold}, score_thresh={args.score_threshold}) ---")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1:        {f1:.3f}")
    print(f"  Total predictions: {len(preds)}  GT boxes: {len(gt_norm)}")

    # Count metrics (use normalized image_path so groupby keys match)
    cnt = count_metrics(preds, gt_norm)
    print(f"\n--- Count metrics ---")
    print(f"  MAE:  {cnt['count_mae']}")
    print(f"  RMSE: {cnt['count_rmse']}")


if __name__ == "__main__":
    main()
