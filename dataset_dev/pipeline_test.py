"""Local pipeline test for Stage-B tile preparation.

For every paired (RGB camera GeoTIFF, HSI .h5) source tile in dataset_dev/,
this script:

  1. Computes per-bbox NDVI from HSI (mean over 835-920 nm vs 620-700 nm)
     and drops boxes with NDVI < 0.3 (dead/burned trees).
  2. Builds 256x256 patches at 0.6 m GSD (= 153.6 m x 153.6 m on the ground)
     by area-averaging RGB from 0.1 m and bilinear-upsampling HSI NIR from 1 m.
  3. Writes each 4-channel patch as a GeoTIFF and one labels.csv with
     pixel-space boxes for every patch.

Output:
    patches/
        TEAK/TEAK_2019_{east}_{north}_r{row}_c{col}.tif      (4 bands R,G,B,NIR; 256x256)
        SOAP/...
        YELL/...
        labels_TEAK.csv                                      (per-site, written this run)
        labels_SOAP.csv                                      (per-site, from a prior run)
        labels.csv                                           (concat of all labels_*.csv)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from scipy.ndimage import zoom
from tqdm import tqdm

HERE = Path(__file__).parent
RGB_DIR = HERE / "rgb"
HSI_DIR = HERE / "hyperspectral"
LABEL_DIR = HERE / "2020weinstein_labels"
OUT_DIR = HERE / "patches"

SITE = "TEAK"
YEAR = "2019"

TARGET_GSD = 0.6                              # m / px
PATCH_PX = 256                                 # tile side in pixels
PATCH_M = PATCH_PX * TARGET_GSD                # 153.6 m
SRC_TILE_M = 1000                              # NEON 1 km tile
PATCHES_PER_SIDE = int(SRC_TILE_M // PATCH_M)  # 6

NIR_LO, NIR_HI = 835.0, 920.0
RED_LO, RED_HI = 620.0, 700.0
NDVI_DEAD = 0.6

UTM_RE = re.compile(r"_(\d{6})_(\d{7})_")


def _band_slice(waves: np.ndarray, lo: float, hi: float) -> slice:
    idx = np.where((waves >= lo) & (waves <= hi))[0]
    return slice(int(idx[0]), int(idx[-1]) + 1)


def load_hsi(hsi_path: Path):
    """Return (nir_1m, ndvi_1m, x_origin, y_origin, px_size_m)."""
    with h5py.File(hsi_path, "r") as f:
        site_key = list(f.keys())[0]
        refl = f[site_key]["Reflectance"]
        waves = refl["Metadata/Spectral_Data/Wavelength"][:]
        map_info = refl["Metadata/Coordinate_System/Map_Info"][()].decode()
        data = refl["Reflectance_Data"]
        scale = float(data.attrs.get("Scale_Factor", 10000.0))
        nodata = float(data.attrs.get("Data_Ignore_Value", -9999.0))
        nir_sl = _band_slice(waves, NIR_LO, NIR_HI)
        red_sl = _band_slice(waves, RED_LO, RED_HI)
        nir_stack = data[:, :, nir_sl].astype(np.float32)
        red_stack = data[:, :, red_sl].astype(np.float32)
    nir_stack[nir_stack == nodata] = np.nan
    red_stack[red_stack == nodata] = np.nan
    nir = np.nanmean(nir_stack, axis=2) / scale
    red = np.nanmean(red_stack, axis=2) / scale
    ndvi = (nir - red) / (nir + red + 1e-8)
    parts = [p.strip() for p in map_info.split(",")]
    return nir, ndvi, float(parts[3]), float(parts[4]), float(parts[5])


def per_box_ndvi(
    labels: pd.DataFrame, ndvi: np.ndarray, x0: float, y0: float, px: float
) -> np.ndarray:
    H, W = ndvi.shape
    out = np.full(len(labels), np.nan, dtype=np.float32)
    for i, r in enumerate(labels.itertuples(index=False)):
        c0f = (r.left  - x0) / px
        c1f = (r.right - x0) / px
        r0f = (y0 - r.top)    / px
        r1f = (y0 - r.bottom) / px
        cmin, cmax = max(0, int(np.floor(min(c0f, c1f)))), min(W, int(np.ceil(max(c0f, c1f))))
        rmin, rmax = max(0, int(np.floor(min(r0f, r1f)))), min(H, int(np.ceil(max(r0f, r1f))))
        if cmax > cmin and rmax > rmin:
            patch = ndvi[rmin:rmax, cmin:cmax]
            if np.isfinite(patch).any():
                out[i] = float(np.nanmean(patch))
    return out


def make_patch_4band(
    rgb_src: rasterio.io.DatasetReader,
    nir_full: np.ndarray,
    x_hsi_origin: float, y_hsi_origin: float, px_hsi: float,
    patch_left: float, patch_top: float,
) -> np.ndarray:
    """Build a (4, PATCH_PX, PATCH_PX) uint8 array at TARGET_GSD."""
    patch_right = patch_left + PATCH_M
    patch_bottom = patch_top - PATCH_M

    # --- RGB camera: area-average from 0.1 m to 0.6 m ---
    cl, rt = (~rgb_src.transform) * (patch_left, patch_top)
    cr, rb = (~rgb_src.transform) * (patch_right, patch_bottom)
    c0, c1 = int(min(cl, cr)), int(max(cl, cr))
    r0, r1 = int(min(rt, rb)), int(max(rt, rb))
    win = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)
    rgb = rgb_src.read(
        [1, 2, 3], window=win,
        out_shape=(3, PATCH_PX, PATCH_PX),
        resampling=Resampling.average,
    )

    # --- HSI NIR: crop at 1 m, bilinear upsample to 0.6 m ---
    c0f = (patch_left  - x_hsi_origin) / px_hsi
    c1f = (patch_right - x_hsi_origin) / px_hsi
    r0f = (y_hsi_origin - patch_top)    / px_hsi
    r1f = (y_hsi_origin - patch_bottom) / px_hsi
    cmin, cmax = int(np.floor(min(c0f, c1f))), int(np.ceil(max(c0f, c1f)))
    rmin, rmax = int(np.floor(min(r0f, r1f))), int(np.ceil(max(r0f, r1f)))
    nir_crop = nir_full[rmin:rmax, cmin:cmax]
    nir_up = zoom(
        nir_crop,
        zoom=(PATCH_PX / nir_crop.shape[0], PATCH_PX / nir_crop.shape[1]),
        order=1,
    )
    nir_q = np.clip(nir_up * 255.0, 0, 255).astype(np.uint8)
    return np.stack([rgb[0], rgb[1], rgb[2], nir_q], axis=0)


def process_source_tile(
    rgb_path: Path, hsi_path: Path, alive_labels: pd.DataFrame,
    source_east: int, source_north: int, out_img_dir: Path, crs,
) -> list[dict]:
    """Tile one 1 km source tile into PATCHES_PER_SIDE^2 patches."""
    nir, _ndvi, x0, y0, px = load_hsi(hsi_path)

    src_left = source_east
    src_top  = source_north + SRC_TILE_M

    cx_box = (alive_labels["left"]  + alive_labels["right"]).to_numpy() / 2
    cy_box = (alive_labels["bottom"] + alive_labels["top"]).to_numpy()  / 2

    rows: list[dict] = []
    with rasterio.open(rgb_path) as rgb_src:
        for r in range(PATCHES_PER_SIDE):
            for c in range(PATCHES_PER_SIDE):
                patch_left = src_left + c * PATCH_M
                patch_top  = src_top  - r * PATCH_M
                patch_right  = patch_left + PATCH_M
                patch_bottom = patch_top  - PATCH_M

                imgname = f"{SITE}_{YEAR}_{source_east}_{source_north}_r{r}_c{c}"
                patch = make_patch_4band(
                    rgb_src, nir, x0, y0, px, patch_left, patch_top,
                )
                tr = from_origin(patch_left, patch_top, TARGET_GSD, TARGET_GSD)
                with rasterio.open(
                    out_img_dir / f"{imgname}.tif", "w",
                    driver="GTiff", height=PATCH_PX, width=PATCH_PX, count=4,
                    dtype=patch.dtype, crs=crs, transform=tr,
                    compress="deflate",
                ) as dst:
                    dst.write(patch)

                # Labels whose centers fall inside this patch.
                inside = (
                    (cx_box >= patch_left)  & (cx_box < patch_right) &
                    (cy_box >  patch_bottom) & (cy_box <= patch_top)
                )
                if not inside.any():
                    continue
                sub = alive_labels.iloc[inside]
                for s in sub.itertuples(index=False):
                    xmin = (s.left  - patch_left) / TARGET_GSD
                    xmax = (s.right - patch_left) / TARGET_GSD
                    ymin = (patch_top - s.top)    / TARGET_GSD
                    ymax = (patch_top - s.bottom) / TARGET_GSD
                    # clamp into patch
                    xmin = max(0.0, xmin); ymin = max(0.0, ymin)
                    xmax = min(float(PATCH_PX), xmax); ymax = min(float(PATCH_PX), ymax)
                    if xmax - xmin < 1.0 or ymax - ymin < 1.0:
                        continue
                    rows.append({
                        "site": SITE,
                        "imgname": imgname,
                        "xmin": round(xmin, 2),
                        "ymin": round(ymin, 2),
                        "xmax": round(xmax, 2),
                        "ymax": round(ymax, 2),
                        "score": float(s.score),
                        "ndvi": float(s.ndvi),
                        "patch_left_utm": patch_left,
                        "patch_top_utm": patch_top,
                        "source_geo_index": f"{source_east}_{source_north}",
                    })
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img_dir = OUT_DIR / SITE                  # patches/{SITE}/...tif
    img_dir.mkdir(exist_ok=True)

    csv = LABEL_DIR / f"{SITE}_{YEAR}.csv"
    print(f"reading {csv} ...")
    labels = pd.read_csv(
        csv,
        usecols=["left", "bottom", "right", "top", "score", "geo_index"],
    )

    rgb_paths = sorted(RGB_DIR.glob(f"{YEAR}_{SITE}_*_image.tif"))
    print(f"found {len(rgb_paths)} RGB tiles in {RGB_DIR}")

    all_rows: list[dict] = []
    total_kept = total_dropped = 0
    for rgb_path in tqdm(rgb_paths, desc="source tiles"):
        m = UTM_RE.search(rgb_path.name)
        if not m:
            continue
        east, north = int(m.group(1)), int(m.group(2))
        geo_index = f"{east}_{north}"

        hsi_path = next(HSI_DIR.glob(f"NEON_*_{SITE}_DP3_{east}_{north}_reflectance.h5"), None)
        if hsi_path is None:
            print(f"  skip {geo_index}: no HSI", file=sys.stderr)
            continue

        labels_for_tile = labels[labels["geo_index"] == geo_index]
        if labels_for_tile.empty:
            print(f"  skip {geo_index}: no labels", file=sys.stderr)
            continue

        _nir, ndvi, x0, y0, px = load_hsi(hsi_path)
        labels_for_tile = labels_for_tile.copy()
        labels_for_tile["ndvi"] = per_box_ndvi(labels_for_tile, ndvi, x0, y0, px)
        alive = labels_for_tile[labels_for_tile["ndvi"] >= NDVI_DEAD]
        dropped = len(labels_for_tile) - len(alive)
        total_kept += len(alive); total_dropped += dropped

        with rasterio.open(rgb_path) as rgb_src:
            crs = rgb_src.crs
        all_rows.extend(process_source_tile(
            rgb_path, hsi_path, alive, east, north, img_dir, crs,
        ))

    # Per-site CSV: patches/labels_{SITE}.csv
    site_csv = OUT_DIR / f"labels_{SITE}.csv"
    pd.DataFrame(all_rows).to_csv(site_csv, index=False)
    print(
        f"\n[{SITE}] {len(list(img_dir.glob('*.tif')))} patches, "
        f"{len(all_rows)} boxes "
        f"(kept {total_kept} alive, dropped {total_dropped} dead/nan) "
        f"-> {site_csv.name}"
    )

    # Concat every labels_*.csv into one combined patches/labels.csv.
    per_site = sorted(OUT_DIR.glob("labels_*.csv"))
    combined = pd.concat([pd.read_csv(p) for p in per_site], ignore_index=True)
    combined_csv = OUT_DIR / "labels.csv"
    combined.to_csv(combined_csv, index=False)
    print(
        f"combined {len(per_site)} site CSV(s) "
        f"({[p.stem.replace('labels_', '') for p in per_site]}) "
        f"-> {combined_csv} ({len(combined)} rows)"
    )


if __name__ == "__main__":
    main()
