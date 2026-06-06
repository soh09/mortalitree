"""
Visualize model predictions vs GT on N test images without running training.

Usage:
    python dfft/viz_predictions.py                          # pretrained baseline
    python dfft/viz_predictions.py --model runs/deepforest_finetuned.pt
    python dfft/viz_predictions.py --score-thresh 0.05      # lower threshold to see weak detections
"""

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw
from deepforest import main as df_main

DATA_DIR = Path(__file__).parent.parent / "single_tiles_flat" / "deepforest_data"


def draw_boxes(img, boxes, color, width=2):
    draw = ImageDraw.Draw(img)
    for _, row in boxes.iterrows():
        draw.rectangle([row.xmin, row.ymin, row.xmax, row.ymax], outline=color, width=width)
    return img


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--score-thresh", type=float, default=0.1)
    p.add_argument("--out-dir", type=str, default=str(Path(__file__).parent / "viz_out"))
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.model:
        model = df_main.deepforest.load_from_checkpoint(args.model)
        print(f"Loaded {args.model}")
    else:
        model = df_main.deepforest()
        model.load_model()
        print("Using pretrained NEON model")
    model.config["score_thresh"] = args.score_thresh
    model.model.score_thresh = args.score_thresh

    gt = pd.read_csv(DATA_DIR / f"{args.split}.csv")
    images = gt["image_path"].unique()[:args.n]

    for img_path in images:
        preds = model.predict_image(path=img_path)
        n_pred = len(preds) if preds is not None else 0
        n_gt = len(gt[gt["image_path"] == img_path])
        print(f"  {Path(img_path).name}: gt={n_gt}  pred={n_pred}")

        img = Image.open(img_path).convert("RGB")
        img = draw_boxes(img, gt[gt["image_path"] == img_path], color="lime")
        if preds is not None and len(preds):
            img = draw_boxes(img, preds, color="red")

        # Scale up 4x for visibility (256→1024)
        img = img.resize((img.width * 4, img.height * 4), Image.NEAREST)

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(img)
        ax.axis("off")
        legend_handles = [
            mpatches.Patch(facecolor="lime",  edgecolor="lime",  label="Ground truth"),
            mpatches.Patch(facecolor="red",   edgecolor="red",   label="Prediction"),
        ]
        ax.legend(handles=legend_handles, loc="lower right", fontsize=22,
                  framealpha=0.85, edgecolor="white", facecolor="#111111",
                  labelcolor="white", handlelength=1.5, borderpad=0.8)
        plt.tight_layout(pad=0)
        fig.savefig(out_dir / Path(img_path).name, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"\nSaved to {out_dir}  (green=GT, red=predicted)")


if __name__ == "__main__":
    main()
