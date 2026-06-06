"""
Plot training and validation metrics from the latest Lightning CSV log.

Usage:
    python dfft/plot_metrics.py                        # latest version
    python dfft/plot_metrics.py --version 2            # specific version
    python dfft/plot_metrics.py --out dfft/metrics.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

LOG_DIR = Path(__file__).parent / "lightning_logs"


def latest_metrics_csv(version: int | None) -> Path:
    if version is not None:
        return LOG_DIR / f"version_{version}" / "metrics.csv"
    versions = sorted(LOG_DIR.glob("version_*/metrics.csv"),
                      key=lambda p: int(p.parent.name.split("_")[1]))
    if not versions:
        raise FileNotFoundError(f"No metrics.csv found under {LOG_DIR}")
    return versions[-1]


def load_epoch_metrics(csv_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    # Rows with an epoch value are end-of-epoch summaries; step-only rows are mid-epoch
    epoch_rows = raw.dropna(subset=["epoch"])
    # Aggregate: one row per epoch, taking the last non-null value per column
    by_epoch = (
        epoch_rows.groupby("epoch")
        .last()
        .reset_index()
    )
    return by_epoch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--version", type=int, default=None)
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    csv_path = latest_metrics_csv(args.version)
    print(f"Reading {csv_path}")
    df = load_epoch_metrics(csv_path)
    print(df[["epoch", "train_loss_epoch", "val_loss",
              "train_bbox_regression_epoch", "val_bbox_regression",
              "train_classification_epoch", "val_classification"]].to_string(index=False))

    out_dir = Path(args.out) if args.out else Path(__file__).parent / "runs" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = df["epoch"]

    plots = [
        ("total_loss",       "Total Loss",            "train_loss_epoch",             "val_loss"),
        ("bbox_regression",  "BBox Regression Loss",  "train_bbox_regression_epoch",  "val_bbox_regression"),
        ("classification",   "Classification Loss",   "train_classification_epoch",   "val_classification"),
    ]

    for slug, title, train_col, val_col in plots:
        fig, ax = plt.subplots(figsize=(7, 4))
        if train_col in df:
            ax.plot(epochs, df[train_col], label="train", marker="o", markersize=3)
        if val_col in df:
            ax.plot(epochs, df[val_col], label="val", marker="o", markersize=3)
        ax.set_title(title)
        ax.set_xlabel("Epoch", fontsize=13)
        ax.set_ylabel("Loss", fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = out_dir / f"{slug}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
