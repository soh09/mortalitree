"""Render a 4-band NEON/NAIP patch GeoTIFF (R,G,B,NIR) to a viewable PNG.

Shows RGB and NIR side by side, and overlays boxes if you pass --labels.

Usage:
    python view_patch.py YELL_2019_534000_4972000_r5_c1.tif
    python view_patch.py PATCH.tif --labels labels.csv     # overlay GT boxes
    python view_patch.py PATCH.tif --out look.png
"""
import argparse
from pathlib import Path

import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tif")
    ap.add_argument("--labels", default=None, help="labels.csv to overlay boxes from")
    ap.add_argument("--out", default=None, help="output PNG path")
    args = ap.parse_args()

    with rasterio.open(args.tif) as src:
        data = src.read()                       # (C, H, W) uint8
    C = data.shape[0]
    rgb = np.transpose(data[:3], (1, 2, 0))     # (H, W, 3)
    nir = data[3] if C >= 4 else None

    n = 2 if nir is not None else 1
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    axes = np.atleast_1d(axes)
    axes[0].imshow(rgb); axes[0].set_title("RGB"); axes[0].axis("off")
    if nir is not None:
        axes[1].imshow(nir, cmap="gray"); axes[1].set_title("NIR"); axes[1].axis("off")

    if args.labels:
        import pandas as pd
        df = pd.read_csv(args.labels)
        stem = Path(args.tif).stem
        sub = df[df["imgname"] == stem]
        for r in sub.itertuples():
            axes[0].add_patch(mpatches.Rectangle(
                (r.xmin, r.ymin), r.xmax - r.xmin, r.ymax - r.ymin,
                fill=False, edgecolor="lime", linewidth=1,
            ))
        axes[0].set_title(f"RGB + {len(sub)} boxes")
        print(f"{len(sub)} boxes for {stem}")

    out = args.out or str(Path(args.tif).with_suffix(".png"))
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print("wrote", out)


if __name__ == "__main__":
    main()
