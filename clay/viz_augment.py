"""Visualize that augmentation transforms boxes consistently with the image.

Renders the original tile plus each geometric augmentation (rot90 k1/k2/k3,
hflip, vflip) and a couple of full random-pipeline samples, with boxes drawn.
If the boxes track the image content in every panel, augmentation is correct.

Usage:
    python viz_augment.py                          # synthetic demo (no data needed)
    python viz_augment.py --tile PATCH.tif --labels labels.csv   # a real 4-band patch
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import sys
sys.path.insert(0, str(Path(__file__).parent))
from src.data.augmentation import rotate90, hflip, vflip, NAIPAugmentation


def synthetic_tile(S=256):
    """A 4-band tile with an asymmetric background (so orientation is obvious)
    plus a few bright 'crowns', and matching cxcywh-normalized boxes."""
    img = torch.zeros(4, S, S)
    # diagonal gradient background -> any rotation/flip is visually distinct
    yy, xx = torch.meshgrid(torch.linspace(0, 1, S), torch.linspace(0, 1, S), indexing="ij")
    img[:3] = (0.25 * xx + 0.55 * yy).unsqueeze(0)         # RGB ramp
    img[3] = yy                                             # NIR
    boxes = []
    # asymmetric crown positions (cx, cy, w, h), normalized
    crowns = [(0.20, 0.15, 0.12, 0.08), (0.75, 0.30, 0.10, 0.10),
              (0.40, 0.65, 0.16, 0.10), (0.85, 0.85, 0.08, 0.12)]
    for cx, cy, w, h in crowns:
        x1, x2 = int((cx - w / 2) * S), int((cx + w / 2) * S)
        y1, y2 = int((cy - h / 2) * S), int((cy + h / 2) * S)
        img[0, y1:y2, x1:x2] = 1.0      # bright red crown marker
        img[1, y1:y2, x1:x2] = 0.9
        boxes.append([cx, cy, w, h])
    return img, torch.tensor(boxes)


def load_real_tile(tile_path, labels_csv, tile_size=256):
    import rasterio
    import pandas as pd
    with rasterio.open(tile_path) as src:
        data = src.read().astype(np.float32)              # (4, H, W) DN
    img = torch.from_numpy(data)
    img = (img - img.amin()) / (img.amax() - img.amin() + 1e-6)  # display scale
    stem = Path(tile_path).stem
    df = pd.read_csv(labels_csv)
    sub = df[df["imgname"] == stem]
    ts = float(tile_size)
    boxes = torch.tensor([
        [(r.xmin + r.xmax) / 2 / ts, (r.ymin + r.ymax) / 2 / ts,
         (r.xmax - r.xmin) / ts, (r.ymax - r.ymin) / ts]
        for r in sub.itertuples()
    ], dtype=torch.float32) if len(sub) else torch.zeros((0, 4))
    print(f"{len(boxes)} boxes for {stem}")
    return img, boxes


def rgb_for_display(img):
    rgb = img[:3].numpy().transpose(1, 2, 0)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)
    return np.clip(rgb, 0, 1)


def draw(ax, img, boxes, title):
    S = img.shape[-1]
    ax.imshow(rgb_for_display(img))
    for cx, cy, w, h in boxes.tolist():
        ax.add_patch(mpatches.Rectangle(
            ((cx - w / 2) * S, (cy - h / 2) * S), w * S, h * S,
            fill=False, edgecolor="lime", linewidth=1.5,
        ))
    ax.set_title(title, fontsize=9); ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", default=None)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--out", default="augment_check.png")
    args = ap.parse_args()

    if args.tile:
        img, boxes = load_real_tile(args.tile, args.labels)
    else:
        img, boxes = synthetic_tile()

    panels = [("original", img, boxes)]
    for k in (1, 2, 3):
        panels.append((f"rot90 k={k}", *rotate90(img, boxes, k)))
    panels.append(("hflip", *hflip(img, boxes)))
    panels.append(("vflip", *vflip(img, boxes)))
    # two full random-pipeline samples (geometry only, so colors stay comparable)
    aug = NAIPAugmentation(brightness=0.0, contrast=0.0, per_band_gain=0.0)
    for i in range(2):
        panels.append((f"random aug #{i+1}", *aug(img.clone(), boxes.clone())))

    n = len(panels)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (title, pim, pb) in zip(axes, panels):
        draw(ax, pim, pb, title)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Augmentation check — boxes (green) should track image content", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
