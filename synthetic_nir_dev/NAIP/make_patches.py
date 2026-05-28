"""
Slice raw NAIP COGs into 256x256 paired RGB+NIR patches at several effective
resolutions. Lower-res patches are built by reading a larger native window and
resampling (Resampling.average) down to 256x256, so each scale gets distinct
ground coverage but identical pixel dims.

Output layout:
  patches/<res>m/rgb/<scene_id>__y<row>_x<col>.tif    (3-band uint8)
  patches/<res>m/nir/<scene_id>__y<row>_x<col>.tif    (1-band uint8)
  manifest.csv

Requires: rasterio, numpy, tqdm
"""

import csv
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.windows import Window
from tqdm import tqdm

ROOT = Path(__file__).parent
RAW_DIR = ROOT / "raw" / "2022"
PATCHES_DIR = ROOT / "patches"
MANIFEST_PATH = ROOT / "manifest.csv"

PATCH_SIZE = 256
STRIDE = 128
RESOLUTIONS_M = [0.6, 1.0, 1.5, 2.0]
NATIVE_GSD_M = 0.6

NODATA_THRESH = 0.05  # skip patch if >5% of pixels are border fill
MIN_NDVI = 0.1        # skip patch if mean NDVI below this (drops water/rock)


def patch_transform(src_transform: Affine, col: int, row: int,
                    src_window_px: int, out_size: int) -> Affine:
    """Transform for a downsampled patch that originated from a native window."""
    scale = src_window_px / out_size
    return src_transform * Affine.translation(col, row) * Affine.scale(scale, scale)


def write_geotiff(path: Path, array: np.ndarray, transform: Affine, crs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bands, h, w = array.shape
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=h, width=w, count=bands,
        dtype=array.dtype,
        crs=crs,
        transform=transform,
        compress="deflate",
        predictor=2,
        tiled=True,
        blockxsize=256, blockysize=256,
    ) as dst:
        dst.write(array)


def process_tile(tile_path: Path, writer: csv.DictWriter) -> dict[float, int]:
    """Process one NAIP scene at all resolutions. Returns {res_m: kept_patch_count}."""
    scene_id = tile_path.stem
    counts: dict[float, int] = {r: 0 for r in RESOLUTIONS_M}

    with rasterio.open(tile_path) as src:
        if src.count < 4:
            print(f"  skip {scene_id}: only {src.count} bands (need 4)")
            return counts

        for r_m in RESOLUTIONS_M:
            src_window_px = int(round(PATCH_SIZE * r_m / NATIVE_GSD_M))
            src_stride_px = int(round(STRIDE * r_m / NATIVE_GSD_M))
            ground_extent_m = PATCH_SIZE * r_m

            res_dir_name = f"{r_m:g}m"
            rgb_dir = PATCHES_DIR / res_dir_name / "rgb"
            nir_dir = PATCHES_DIR / res_dir_name / "nir"

            max_col = src.width - src_window_px
            max_row = src.height - src_window_px
            if max_col < 0 or max_row < 0:
                continue

            for row in range(0, max_row + 1, src_stride_px):
                for col in range(0, max_col + 1, src_stride_px):
                    window = Window(col, row, src_window_px, src_window_px)
                    arr = src.read(
                        indexes=[1, 2, 3, 4],
                        window=window,
                        out_shape=(4, PATCH_SIZE, PATCH_SIZE),
                        resampling=Resampling.average,
                    )

                    rgb = arr[:3]
                    nir = arr[3:4]

                    # nodata: NAIP fills border with 0 across all RGB bands
                    nodata_mask = np.all(rgb == 0, axis=0)
                    nodata_frac = float(nodata_mask.mean())
                    if nodata_frac > NODATA_THRESH:
                        continue

                    red_f = rgb[0].astype(np.float32)
                    nir_f = nir[0].astype(np.float32)
                    ndvi = (nir_f - red_f) / (nir_f + red_f + 1e-6)
                    mean_ndvi = float(ndvi.mean())
                    if mean_ndvi < MIN_NDVI:
                        continue

                    tfm = patch_transform(src.transform, col, row,
                                          src_window_px, PATCH_SIZE)
                    name = f"{scene_id}__y{row}_x{col}.tif"
                    rgb_path = rgb_dir / name
                    nir_path = nir_dir / name
                    write_geotiff(rgb_path, rgb, tfm, src.crs)
                    write_geotiff(nir_path, nir, tfm, src.crs)

                    utm_x_min, utm_y_min = tfm * (0, PATCH_SIZE)
                    writer.writerow({
                        "patch_id": f"{scene_id}__{res_dir_name}__y{row}_x{col}",
                        "scene_id": scene_id,
                        "resolution_m": r_m,
                        "rgb_path": str(rgb_path.relative_to(ROOT)),
                        "nir_path": str(nir_path.relative_to(ROOT)),
                        "row": row,
                        "col": col,
                        "utm_x_min": utm_x_min,
                        "utm_y_min": utm_y_min,
                        "ground_extent_m": ground_extent_m,
                        "nodata_frac": round(nodata_frac, 4),
                        "mean_ndvi": round(mean_ndvi, 4),
                    })
                    counts[r_m] += 1
    return counts


def main():
    tiles = sorted(RAW_DIR.glob("*.tif"))
    if not tiles:
        raise SystemExit(f"No tiles found in {RAW_DIR}. Run download_naip.py first.")

    PATCHES_DIR.mkdir(exist_ok=True)
    fieldnames = [
        "patch_id", "scene_id", "resolution_m",
        "rgb_path", "nir_path",
        "row", "col", "utm_x_min", "utm_y_min",
        "ground_extent_m", "nodata_frac", "mean_ndvi",
    ]

    totals: dict[float, int] = {r: 0 for r in RESOLUTIONS_M}
    with open(MANIFEST_PATH, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for tile in tqdm(tiles, desc="tiles"):
            counts = process_tile(tile, writer)
            for r, n in counts.items():
                totals[r] += n
            tqdm.write(f"{tile.name}: " +
                       ", ".join(f"{r:g}m={n}" for r, n in counts.items()))

    print("\nDone. Patch counts per resolution:")
    for r in RESOLUTIONS_M:
        print(f"  {r:g}m: {totals[r]}")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
