"""Visualize 10 TEAK patches: RGB | RGB+boxes | NIR, saved to teak_patches_visual/.
Also produces a combined grid: teak_patches_visual/grid_rgb_nir.png"""
from __future__ import annotations

import random
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

PATCHES_DIR = Path("dataset_dev/teak_patches/images")
LABELS_CSV  = Path("dataset_dev/teak_patches/labels.csv")
OUT_DIR     = Path("dataset_dev/teak_patches_visual")
N = 10
SEED = 42

OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(LABELS_CSV)
# pick 10 patches that each have at least a few boxes
counts = df.groupby("imgname").size()
candidates = counts[counts >= 5].index.tolist()
random.seed(SEED)
chosen = random.sample(candidates, min(N, len(candidates)))

for imgname in chosen:
    tif = PATCHES_DIR / f"{imgname}.tif"
    with rasterio.open(tif) as src:
        data = src.read()          # (4, 256, 256)  R G B NIR

    rgb = np.stack([data[0], data[1], data[2]], axis=-1)   # (256,256,3) uint8
    nir = data[3]                                           # (256,256)   uint8

    boxes = df[df["imgname"] == imgname][["xmin","ymin","xmax","ymax"]].values

    site_year = "_".join(imgname.split("_")[:2])  # e.g. TEAK_2019

    def _save(arr, cmap, suffix, title):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(arr, cmap=cmap)
        ax.set_title(f"{site_year} — {title}", fontsize=9)
        ax.axis("off")
        fig.savefig(OUT_DIR / f"{imgname}_{suffix}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    _save(rgb, None, "rgb", "RGB")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(rgb)
    for (x1, y1, x2, y2) in boxes:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x1, y1), x2 - x1, y2 - y1,
            boxstyle="square,pad=0",
            linewidth=0.8, edgecolor="lime", facecolor="none",
        ))
    ax.set_title(f"{site_year} — RGB + {len(boxes)} boxes", fontsize=9)
    ax.axis("off")
    fig.savefig(OUT_DIR / f"{imgname}_boxes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    _save(nir, "gray", "nir", "NIR (band 4)")
    print(f"saved {imgname}  ({len(boxes)} boxes)")

print(f"\ndone — {len(chosen)} images in {OUT_DIR}/")

# --- per-patch side-by-side: RGB | NIR ---
for imgname in chosen:
    tif = PATCHES_DIR / f"{imgname}.tif"
    with rasterio.open(tif) as src:
        data = src.read()
    rgb = np.stack([data[0], data[1], data[2]], axis=-1)
    nir = data[3]
    site_year = "_".join(imgname.split("_")[:2])

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    fig.subplots_adjust(wspace=0.02)

    axes[0].imshow(rgb)
    axes[0].set_title("RGB", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(nir, cmap="gray")
    axes[1].set_title("NIR", fontsize=10)
    axes[1].axis("off")

    out = OUT_DIR / f"{imgname}_rgb_nir.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out.name}")
