"""NDVI filtering visualization for TEAK_2019 tile 313000_4092000.

Outputs (dataset_dev/ndvi_filter_visual/):
  histogram.png          — per-box NDVI distribution with threshold line
  patch_r{r}_c{c}.png   — 10 patches with green=kept / red=removed overlays
"""
from __future__ import annotations
from pathlib import Path

import h5py
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from scipy.ndimage import zoom

# --- paths ---
DATASET_DEV = Path("dataset_dev")
HSI_PATH    = DATASET_DEV / "hyperspectral" / "NEON_D17_TEAK_DP3_313000_4092000_reflectance.h5"
RGB_PATH    = DATASET_DEV / "rgb" / "2019_TEAK_4_313000_4092000_image.tif"
LABEL_CSV   = DATASET_DEV / "2020weinstein_labels" / "TEAK_2019.csv"
OUT_DIR     = DATASET_DEV / "ndvi_filter_visual"
OUT_DIR.mkdir(exist_ok=True)

EAST, NORTH    = 313000, 4092000
GEO_INDEX      = f"{EAST}_{NORTH}"
NDVI_THRESHOLD = 0.65
TARGET_GSD     = 0.6
PATCH_PX       = 256
PATCH_M        = PATCH_PX * TARGET_GSD   # 153.6 m
NIR_LO, NIR_HI = 835.0, 920.0
RED_LO, RED_HI = 620.0, 700.0
N_PATCHES      = 10

# ── 1. Load HSI → compute NDVI ────────────────────────────────────────────────
print("loading HSI …")
with h5py.File(HSI_PATH, "r") as f:
    site_key = list(f.keys())[0]
    refl     = f[site_key]["Reflectance"]
    wl       = refl["Metadata/Spectral_Data/Wavelength"][:]
    map_info = refl["Metadata/Coordinate_System/Map_Info"][()].decode()
    data_ds  = refl["Reflectance_Data"]
    scale    = float(data_ds.attrs.get("Scale_Factor", 10000.0))
    nodata   = float(data_ds.attrs.get("Data_Ignore_Value", -9999.0))

    def band_slice(lo, hi):
        idx = np.where((wl >= lo) & (wl <= hi))[0]
        return slice(int(idx[0]), int(idx[-1]) + 1)

    nir_sl = band_slice(NIR_LO, NIR_HI)
    red_sl = band_slice(RED_LO, RED_HI)
    nir_stack = data_ds[:, :, nir_sl].astype(np.float32)
    red_stack = data_ds[:, :, red_sl].astype(np.float32)

nir_stack[nir_stack == nodata] = np.nan
red_stack[red_stack == nodata] = np.nan
nir_map = np.nanmean(nir_stack, axis=2) / scale
red_map = np.nanmean(red_stack, axis=2) / scale
ndvi_map = (nir_map - red_map) / (nir_map + red_map + 1e-8)

parts    = [p.strip() for p in map_info.split(",")]
x_origin = float(parts[3])
y_origin = float(parts[4])
px_size  = float(parts[5])
H_hsi, W_hsi = ndvi_map.shape
print(f"NDVI map: {ndvi_map.shape}, range=[{np.nanmin(ndvi_map):.3f}, {np.nanmax(ndvi_map):.3f}]")

def utm_to_hsi(x, y):
    return (x - x_origin) / px_size, (y_origin - y) / px_size

# ── 2. Load labels for this tile ─────────────────────────────────────────────
print("loading labels …")
labels = pd.read_csv(
    LABEL_CSV,
    usecols=["left", "bottom", "right", "top", "score", "geo_index"],
)
tile_labels = labels[labels["geo_index"] == GEO_INDEX].reset_index(drop=True)
print(f"{len(tile_labels)} crowns for {GEO_INDEX}")

# ── 3. Compute per-box NDVI ───────────────────────────────────────────────────
print("computing per-box NDVI …")
box_ndvi = np.full(len(tile_labels), np.nan, dtype=np.float32)
for i, r in enumerate(tile_labels.itertuples(index=False)):
    c0_f, r0_f = utm_to_hsi(r.left, r.top)
    c1_f, r1_f = utm_to_hsi(r.right, r.bottom)
    c0 = max(int(np.floor(min(c0_f, c1_f))), 0)
    c1 = min(int(np.ceil(max(c0_f, c1_f))), W_hsi)
    r0 = max(int(np.floor(min(r0_f, r1_f))), 0)
    r1 = min(int(np.ceil(max(r0_f, r1_f))), H_hsi)
    if c1 > c0 and r1 > r0:
        patch = ndvi_map[r0:r1, c0:c1]
        if np.isfinite(patch).any():
            box_ndvi[i] = float(np.nanmean(patch))

tile_labels = tile_labels.assign(ndvi=box_ndvi)
alive = tile_labels["ndvi"] >= NDVI_THRESHOLD
dead  = tile_labels["ndvi"] <  NDVI_THRESHOLD
print(f"alive={alive.sum()}  dead={dead.sum()}  nan={tile_labels['ndvi'].isna().sum()}")

# ── 4. Histogram ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
vals = tile_labels["ndvi"].dropna()
bins = np.linspace(vals.min(), vals.max(), 50)
ax.hist(vals, bins=bins, color="steelblue", edgecolor="white", linewidth=0.3)
ax.axvline(NDVI_THRESHOLD, color="red", ls="--", lw=1.5, label=f"threshold = {NDVI_THRESHOLD}")
ax.set_xlabel("mean NDVI per bounding box")
ax.set_ylabel("count")
ax.set_title("NDVI-filtering on TEAK_2019")
ax.legend()
plt.tight_layout()
fig.savefig(OUT_DIR / "histogram.png", dpi=150)
plt.close(fig)
print("saved histogram.png")

# ── 5. Patch visualizations ───────────────────────────────────────────────────
# Build 6x6 grid, pick patches that have both alive and dead boxes.
with rasterio.open(RGB_PATH) as src:
    rgb_transform = src.transform
    rgb_bounds    = src.bounds

tile_left  = float(rgb_bounds.left)
tile_top   = float(rgb_bounds.top)
PATCHES_PER_SIDE = 6

candidates = []
for r in range(PATCHES_PER_SIDE):
    for c in range(PATCHES_PER_SIDE):
        p_left = tile_left + c * PATCH_M
        p_top  = tile_top  - r * PATCH_M
        p_right  = p_left + PATCH_M
        p_bottom = p_top  - PATCH_M

        cx = (tile_labels["left"] + tile_labels["right"]) / 2
        cy = (tile_labels["bottom"] + tile_labels["top"]) / 2
        inside = (cx >= p_left) & (cx < p_right) & (cy >= p_bottom) & (cy < p_top)
        patch_boxes = tile_labels[inside].copy()
        n_alive = (patch_boxes["ndvi"] >= NDVI_THRESHOLD).sum()
        n_dead  = (patch_boxes["ndvi"] <  NDVI_THRESHOLD).sum()
        if n_alive > 0 or n_dead > 0:
            candidates.append((r, c, p_left, p_top, patch_boxes, n_alive, n_dead))

# prefer patches with both alive and dead visible
candidates.sort(key=lambda x: min(x[5], x[6]), reverse=True)
chosen = candidates[:N_PATCHES]
print(f"rendering {len(chosen)} patches …")

for r, c, p_left, p_top, patch_boxes, n_alive, n_dead in chosen:
    p_right  = p_left + PATCH_M
    p_bottom = p_top  - PATCH_M

    with rasterio.open(RGB_PATH) as src:
        inv = ~src.transform
        col0, row0 = inv * (p_left, p_top)
        col1, row1 = inv * (p_right, p_bottom)
        col0, col1 = int(min(col0, col1)), int(max(col0, col1))
        row0, row1 = int(min(row0, row1)), int(max(row0, row1))
        win = rasterio.windows.Window(col0, row0, col1 - col0, row1 - row0)
        rgb = src.read([1, 2, 3], window=win,
                       out_shape=(3, PATCH_PX, PATCH_PX),
                       resampling=Resampling.average)
        win_transform = src.window_transform(win)

    rgb = np.transpose(rgb, (1, 2, 0))
    inv_win = ~win_transform

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(rgb)

    scale_x = PATCH_PX / (col1 - col0)
    scale_y = PATCH_PX / (row1 - row0)
    for row in patch_boxes.itertuples(index=False):
        # inv_win gives window-relative pixel coords (0..window_width)
        x0, y0 = inv_win * (row.left, row.top)
        x1, y1 = inv_win * (row.right, row.bottom)
        x0, x1 = x0 * scale_x, x1 * scale_x
        y0, y1 = y0 * scale_y, y1 * scale_y
        if np.isnan(row.ndvi):
            color = "yellow"
        elif row.ndvi < NDVI_THRESHOLD:
            color = "red"
        else:
            color = "limegreen"
        ax.add_patch(mpatches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            fill=False, edgecolor=color, linewidth=1.0,
        ))

    legend_elems = [
        mpatches.Patch(edgecolor="limegreen", facecolor="none", label="kept (alive)"),
        mpatches.Patch(edgecolor="red",       facecolor="none", label="removed (dead)"),
    ]
    ax.legend(handles=legend_elems, loc="upper right", fontsize=7,
              framealpha=0.7, handlelength=1.2)
    ax.set_title("NDVI-filtering on TEAK_2019", fontsize=9)
    ax.axis("off")

    plt.tight_layout()
    out = OUT_DIR / f"patch_r{r}_c{c}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}  (alive={n_alive} dead={n_dead})")

print(f"\ndone — outputs in {OUT_DIR}/")
