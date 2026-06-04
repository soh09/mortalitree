"""Plot the RGB image of the patch with the most labels, with boxes overlaid.

Usage: python plot_top_patch.py [--rank N]   (N=1 = most labels, N=2 = second most, etc.)
"""
import argparse
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

LABELS_CSV = "eval_set/labels.csv"

parser = argparse.ArgumentParser()
parser.add_argument("--rank", type=int, default=1, help="1=most labels, 2=second most, ...")
args = parser.parse_args()

df = pd.read_csv(LABELS_CSV)
counts = df.groupby("imgname").size().sort_values(ascending=False)
imgname = counts.index[args.rank - 1]
patch_df = df[df.imgname == imgname]
print(f"Patch: {imgname}  ({len(patch_df)} labels)")

tif_path = f"eval_set/patches/{imgname}.tif"
with rasterio.open(tif_path) as src:
    arr = src.read()          # shape (4, 256, 256)  bands: R, G, B, NIR

rgb = np.stack([arr[0], arr[1], arr[2]], axis=-1)   # H x W x 3

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(rgb)

for _, row in patch_df.iterrows():
    rect = mpatches.Rectangle(
        (row.xmin, row.ymin),
        row.xmax - row.xmin,
        row.ymax - row.ymin,
        linewidth=0.8,
        edgecolor="lime",
        facecolor="none",
    )
    ax.add_patch(rect)

ax.set_title(f"{imgname}\n{len(patch_df)} labels", fontsize=11)
ax.axis("off")

out = f"{imgname}_labels.png"
plt.tight_layout()
plt.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved → {out}")
plt.show()
