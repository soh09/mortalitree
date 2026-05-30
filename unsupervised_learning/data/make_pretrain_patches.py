"""
Slice raw NAIP COGs into 4-channel (R,G,B,NIR) 224x224 uint8 patches for
Stage A MAE/SimMIM pretraining.

Scans every year subdir under raw/ (e.g. raw/2020/, raw/2022/) so a single
tile run covers all downloaded years. Per-year AOI sidecars are merged.

Defaults: PATCH_SIZE=224, STRIDE=224 (non-overlapping). Stride==size keeps
the tile pool diverse-but-not-redundant — overlap is augmentation-flavored,
not new information, and the model already does rot/flip aug at every step.

Output layout:
  patches/<scene_id>/y<row>_x<col>.npy   shape (4, 224, 224) uint8
  manifest.csv                           index (scene_id, year, aoi, ...)

Requires: rasterio, numpy, tqdm
"""

import csv
import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

DATA_ROOT = Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parent.parent))
RAW_ROOT = DATA_ROOT / "raw"
PATCHES_DIR = DATA_ROOT / "patches"
MANIFEST_PATH = DATA_ROOT / "manifest.csv"

PATCH_SIZE = 224
STRIDE = int(os.environ.get("PATCH_STRIDE", "224"))  # non-overlapping by default
NODATA_THRESH = 0.05  # skip if >5% of pixels are border fill


def load_scene_to_aoi() -> dict[str, str]:
    """Merge AOI sidecars from every raw/<year>/ subdir."""
    scene_to_aoi: dict[str, str] = {}
    sidecars = sorted(RAW_ROOT.glob("*/tile_aoi.csv"))
    if not sidecars:
        print(f"WARN: no tile_aoi.csv under {RAW_ROOT}; aoi column will be blank.")
        return scene_to_aoi
    for sc in sidecars:
        with open(sc, newline="") as fh:
            for row in csv.DictReader(fh):
                # If a scene_id appears in multiple year sidecars (shouldn't,
                # since NAIP item IDs are date-stamped) the first wins.
                scene_to_aoi.setdefault(row["scene_id"], row["aoi"])
    print(f"Loaded AOI mapping for {len(scene_to_aoi)} scenes from {len(sidecars)} sidecars")
    return scene_to_aoi


def count_windows(tile_path: Path) -> int:
    with rasterio.open(tile_path) as src:
        if src.count < 4:
            return 0
        max_col = src.width - PATCH_SIZE
        max_row = src.height - PATCH_SIZE
        if max_col < 0 or max_row < 0:
            return 0
        n_rows = len(range(0, max_row + 1, STRIDE))
        n_cols = len(range(0, max_col + 1, STRIDE))
        return n_rows * n_cols


def process_tile(tile_path: Path, year: str, writer: csv.DictWriter,
                 scene_to_aoi: dict[str, str], pbar) -> int:
    scene_id = tile_path.stem
    aoi = scene_to_aoi.get(scene_id, "")
    saved = 0

    with rasterio.open(tile_path) as src:
        if src.count < 4:
            print(f"  skip {scene_id}: only {src.count} bands")
            return 0

        out_dir = PATCHES_DIR / scene_id
        out_dir.mkdir(parents=True, exist_ok=True)

        max_col = src.width - PATCH_SIZE
        max_row = src.height - PATCH_SIZE
        if max_col < 0 or max_row < 0:
            return 0

        for row in range(0, max_row + 1, STRIDE):
            for col in range(0, max_col + 1, STRIDE):
                pbar.update(1)
                window = Window(col, row, PATCH_SIZE, PATCH_SIZE)
                arr = src.read(indexes=[1, 2, 3, 4], window=window)  # (4, H, W) uint8

                rgb = arr[:3]
                nodata_mask = np.all(rgb == 0, axis=0)
                nodata_frac = float(nodata_mask.mean())
                if nodata_frac > NODATA_THRESH:
                    pbar.skipped_nodata += 1
                    pbar.set_postfix(saved=pbar.saved,
                                     drop_nodata=pbar.skipped_nodata,
                                     refresh=False)
                    continue

                out_path = out_dir / f"y{row}_x{col}.npy"
                np.save(out_path, arr.astype(np.uint8))

                writer.writerow({
                    "patch_id": f"{scene_id}__y{row}_x{col}",
                    "scene_id": scene_id,
                    "year": year,
                    "aoi": aoi,
                    "rel_path": str(out_path.relative_to(DATA_ROOT)),
                    "row": row,
                    "col": col,
                    "nodata_frac": round(nodata_frac, 4),
                })
                saved += 1
                pbar.saved += 1
                pbar.set_postfix(saved=pbar.saved,
                                 drop_nodata=pbar.skipped_nodata,
                                 refresh=False)
    return saved


def main():
    # year subdir name is the inferred year for every tile inside it.
    tiles: list[tuple[Path, str]] = []
    for year_dir in sorted(RAW_ROOT.iterdir() if RAW_ROOT.exists() else []):
        if not year_dir.is_dir():
            continue
        for tif in sorted(year_dir.glob("*.tif")):
            tiles.append((tif, year_dir.name))
    if not tiles:
        raise SystemExit(f"No tiles in {RAW_ROOT}/*/. Run download_naip.py first.")

    PATCHES_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "patch_id", "scene_id", "year", "aoi",
        "rel_path", "row", "col", "nodata_frac",
    ]
    scene_to_aoi = load_scene_to_aoi()

    print(f"Tiling {len(tiles)} scenes across "
          f"{len({y for _, y in tiles})} years at stride={STRIDE}")
    print("Counting expected windows ...")
    total_windows = sum(count_windows(t) for t, _ in tiles)
    print(f"  {total_windows} windows\n")

    total_saved = 0
    with open(MANIFEST_PATH, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        with tqdm(total=total_windows, desc="patches", unit="win") as pbar:
            pbar.saved = 0
            pbar.skipped_nodata = 0
            for tile, year in tiles:
                kept = process_tile(tile, year, writer, scene_to_aoi, pbar)
                total_saved += kept
                tqdm.write(f"[{year}] {tile.name}: {kept} patches")

    print(f"\nDone. Saved {total_saved} patches to {PATCHES_DIR}")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
