"""
Slice raw NAIP COGs into 256x256 paired RGB+NIR patches at several effective
resolutions. Lower-res patches are built by reading a larger native window and
resampling (Resampling.average) down to 256x256, so each scale gets distinct
ground coverage but identical pixel dims.

Output layout (sharded by scene_id to keep leaf-dir file counts manageable):
  patches/<res>m/rgb/<scene_id>/y<row>_x<col>.png    (3-channel uint8)
  patches/<res>m/nir/<scene_id>/y<row>_x<col>.png    (1-channel uint8)
  manifest.csv  (per-patch geographic metadata: UTM origin, ground extent, etc.)

Requires: rasterio, numpy, tqdm, pillow
"""

import csv
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.windows import Window
from tqdm import tqdm

ROOT = Path(__file__).parent
RAW_DIR = ROOT / "raw" / "2022"
PATCHES_DIR = ROOT / "patches"
MANIFEST_PATH = ROOT / "manifest.csv"
# Sidecar written by download_naip.py; mapping scene_id -> aoi.
TILE_AOI_PATH = RAW_DIR / "tile_aoi.csv"

PATCH_SIZE = 256
STRIDE = 256
RESOLUTIONS_M = [0.6, 1.0, 1.5, 2.0]
NATIVE_GSD_M = 0.6

NODATA_THRESH = 0.05  # skip patch if >5% of pixels are border fill
# NDVI is computed per patch and recorded in manifest.csv (column: mean_ndvi)
# so downstream code can filter on it without re-running this script.


def load_scene_to_aoi() -> dict[str, str]:
    """Read the sidecar written by download_naip.py. Empty dict if missing —
    callers will write blank AOIs and the user can run add_aoi_to_manifest.py."""
    if not TILE_AOI_PATH.exists():
        print(f"WARN: no AOI sidecar at {TILE_AOI_PATH}; manifest's aoi "
              f"column will be blank. Run add_aoi_to_manifest.py to backfill.")
        return {}
    with open(TILE_AOI_PATH, newline="") as fh:
        return {row["scene_id"]: row["aoi"] for row in csv.DictReader(fh)}


def patch_origin_utm(src_transform: Affine, col: int, row: int,
                     src_window_px: int) -> tuple[float, float]:
    """UTM (x_min, y_min) of the patch's lower-left corner in the source raster."""
    tfm = src_transform * Affine.translation(col, row)
    return tfm * (0, src_window_px)


def write_png_rgb(path: Path, arr: np.ndarray) -> None:
    # arr: (3, H, W) uint8 -> PIL expects (H, W, 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(np.transpose(arr, (1, 2, 0)), mode="RGB")
    img.save(path, format="PNG", optimize=False, compress_level=6)


def write_png_gray(path: Path, arr: np.ndarray) -> None:
    # arr: (H, W) uint8
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(arr, mode="L")
    img.save(path, format="PNG", optimize=False, compress_level=6)


def count_windows(tile_path: Path) -> int:
    """Theoretical max window count across all resolutions for one tile (header-only)."""
    with rasterio.open(tile_path) as src:
        if src.count < 4:
            return 0
        total = 0
        for r_m in RESOLUTIONS_M:
            src_window_px = int(round(PATCH_SIZE * r_m / NATIVE_GSD_M))
            src_stride_px = int(round(STRIDE * r_m / NATIVE_GSD_M))
            max_col = src.width - src_window_px
            max_row = src.height - src_window_px
            if max_col < 0 or max_row < 0:
                continue
            n_rows = len(range(0, max_row + 1, src_stride_px))
            n_cols = len(range(0, max_col + 1, src_stride_px))
            total += n_rows * n_cols
        return total


def process_tile(tile_path: Path, writer: csv.DictWriter,
                 scene_to_aoi: dict[str, str], pbar) -> dict[float, int]:
    """Process one NAIP scene at all resolutions. Returns {res_m: kept_patch_count}."""
    scene_id = tile_path.stem
    aoi = scene_to_aoi.get(scene_id, "")
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
            rgb_dir = PATCHES_DIR / res_dir_name / "rgb" / scene_id
            nir_dir = PATCHES_DIR / res_dir_name / "nir" / scene_id

            max_col = src.width - src_window_px
            max_row = src.height - src_window_px
            if max_col < 0 or max_row < 0:
                continue

            for row in range(0, max_row + 1, src_stride_px):
                for col in range(0, max_col + 1, src_stride_px):
                    pbar.update(1)
                    window = Window(col, row, src_window_px, src_window_px)
                    arr = src.read(
                        indexes=[1, 2, 3, 4],
                        window=window,
                        out_shape=(4, PATCH_SIZE, PATCH_SIZE),
                        resampling=Resampling.average,
                    )

                    rgb = arr[:3]
                    nir = arr[3]

                    # nodata: NAIP fills border with 0 across all RGB bands
                    nodata_mask = np.all(rgb == 0, axis=0)
                    nodata_frac = float(nodata_mask.mean())
                    if nodata_frac > NODATA_THRESH:
                        pbar.skipped_nodata += 1
                        pbar.set_postfix(saved=pbar.saved,
                                         drop_nodata=pbar.skipped_nodata,
                                         refresh=False)
                        continue

                    red_f = rgb[0].astype(np.float32)
                    nir_f = nir.astype(np.float32)
                    ndvi = (nir_f - red_f) / (nir_f + red_f + 1e-6)
                    mean_ndvi = float(ndvi.mean())

                    name = f"y{row}_x{col}.png"
                    rgb_path = rgb_dir / name
                    nir_path = nir_dir / name
                    write_png_rgb(rgb_path, rgb)
                    write_png_gray(nir_path, nir)

                    utm_x_min, utm_y_min = patch_origin_utm(
                        src.transform, col, row, src_window_px)
                    writer.writerow({
                        "patch_id": f"{scene_id}__{res_dir_name}__y{row}_x{col}",
                        "scene_id": scene_id,
                        "aoi": aoi,
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
                    pbar.saved += 1
                    pbar.set_postfix(saved=pbar.saved,
                                     drop_nodata=pbar.skipped_nodata,
                                     refresh=False)
    return counts


def main():
    tiles = sorted(RAW_DIR.glob("*.tif"))
    if not tiles:
        raise SystemExit(f"No tiles found in {RAW_DIR}. Run download_naip.py first.")

    PATCHES_DIR.mkdir(exist_ok=True)
    fieldnames = [
        "patch_id", "scene_id", "aoi", "resolution_m",
        "rgb_path", "nir_path",
        "row", "col", "utm_x_min", "utm_y_min",
        "ground_extent_m", "nodata_frac", "mean_ndvi",
    ]
    scene_to_aoi = load_scene_to_aoi()

    print("Counting expected windows ...")
    total_windows = sum(count_windows(t) for t in tiles)
    print(f"  {total_windows} windows across {len(tiles)} tiles "
          f"x {len(RESOLUTIONS_M)} resolutions\n")

    totals: dict[float, int] = {r: 0 for r in RESOLUTIONS_M}
    with open(MANIFEST_PATH, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        with tqdm(total=total_windows, desc="patches", unit="win") as pbar:
            pbar.saved = 0
            pbar.skipped_nodata = 0
            for tile in tiles:
                counts = process_tile(tile, writer, scene_to_aoi, pbar)
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
