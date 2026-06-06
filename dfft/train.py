"""
Fine-tune pretrained DeepForest on the stage_c NAIP tree detection data.

Usage:
    python dfft/train.py [--epochs 15] [--lr 0.001] [--batch-size 4] [--out-dir dfft/runs/]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from PIL import Image, ImageDraw
from deepforest import main as df_main

DATA_DIR = Path(__file__).parent.parent / "single_tiles_flat" / "deepforest_data"
N_VIZ = 5  # images to visualize per epoch


def draw_boxes(img: Image.Image, boxes: pd.DataFrame, color: str) -> Image.Image:
    draw = ImageDraw.Draw(img)
    for _, row in boxes.iterrows():
        draw.rectangle([row.xmin, row.ymin, row.xmax, row.ymax], outline=color, width=2)
    return img


class VizCallback(pl.Callback):
    """At the end of each epoch save N val images with GT (green) and predicted (red) boxes."""

    def __init__(self, model, val_csv: Path, out_dir: Path, n: int = N_VIZ):
        super().__init__()
        self.df_model = model
        self.out_dir = out_dir
        self.n = n

        gt = pd.read_csv(val_csv)
        # Pick n unique images
        self.sample_images = gt["image_path"].unique()[:n].tolist()
        self.gt_by_img = {p: gt[gt["image_path"] == p] for p in self.sample_images}

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        epoch_dir = self.out_dir / f"epoch_{epoch:03d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        # predict_image loads images as CPU tensors; move model to CPU for the viz pass
        device = next(self.df_model.model.parameters()).device
        self.df_model.model.to("cpu")
        self.df_model.model.eval()
        for img_path in self.sample_images:
            preds = self.df_model.predict_image(path=img_path)

            img = Image.open(img_path).convert("RGB")
            # Draw GT boxes in green
            img = draw_boxes(img, self.gt_by_img[img_path], color="lime")
            # Draw predicted boxes in red (empty df → no-op)
            if preds is not None and len(preds):
                img = draw_boxes(img, preds, color="red")

            stem = Path(img_path).stem
            img.save(epoch_dir / f"{stem}.png")

        self.df_model.model.to(device)  # restore original device for training
        print(f"  [viz] epoch {epoch} → {epoch_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--score-thresh", type=float, default=0.1,
                   help="Detection score threshold (lower = more boxes shown in viz)")
    p.add_argument("--out-dir", type=str, default=str(Path(__file__).parent / "runs"))
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = df_main.deepforest()
    model.load_model()  # load NEON pretrained weights (weecology/deepforest-tree)

    train_csv = DATA_DIR / "train.csv"
    val_csv = DATA_DIR / "val.csv"

    model.config["train"]["csv_file"] = str(train_csv)
    model.config["train"]["root_dir"] = "/"
    model.config["validation"]["csv_file"] = str(val_csv)
    model.config["validation"]["root_dir"] = "/"
    model.config["train"]["epochs"] = args.epochs
    model.config["train"]["lr"] = args.lr
    model.config["batch_size"] = args.batch_size
    model.config["score_thresh"] = args.score_thresh
    model.model.score_thresh = args.score_thresh  # config dict doesn't reach the live model

    viz_cb = VizCallback(model=model, val_csv=val_csv, out_dir=out_dir / "viz")

    print(f"Training for {args.epochs} epochs, lr={args.lr}, batch_size={args.batch_size}, score_thresh={args.score_thresh}")
    model.create_trainer(logger=None, callbacks=[viz_cb])
    model.trainer.fit(model)

    ckpt_path = out_dir / "deepforest_finetuned.pt"
    model.save_model(str(ckpt_path))
    print(f"Model saved to {ckpt_path}")


if __name__ == "__main__":
    main()
