"""
Convert stage_c_data (4-band NAIP TIFFs + JSON annotations) into DeepForest-ready format:
  - RGB PNGs in single_tiles_flat/deepforest_data/images/
  - train.csv / val.csv / test.csv in single_tiles_flat/deepforest_data/

CSV columns: image_path, xmin, ymin, xmax, ymax, label
Boxes are converted from normalized (cx, cy, w, h) → pixel (xmin, ymin, xmax, ymax).
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from PIL import Image

TILES_DIR = Path(__file__).parent.parent / "single_tiles_flat" / "stage_c_data" / "tiles"
SPLITS_DIR = Path(__file__).parent.parent / "single_tiles_flat" / "stage_c_data"
OUT_DIR = Path(__file__).parent.parent / "single_tiles_flat" / "deepforest_data"
IMG_OUT = OUT_DIR / "images"
IMG_OUT.mkdir(parents=True, exist_ok=True)


def tif_stem_to_path(tile_path_str: str) -> Path:
    """Map JSON tile_path (absolute /data/... path) to actual local file."""
    stem = Path(tile_path_str).name  # e.g. prefire_17_21008_50917.tif
    return TILES_DIR / stem


def convert_tif_to_png(tif_path: Path, out_path: Path) -> tuple[int, int]:
    """Read R,G,B bands from 4-band NAIP TIFF, scale to uint8, save as PNG."""
    with rasterio.open(tif_path) as src:
        # Bands 1,2,3 = R,G,B (1-indexed in rasterio)
        rgb = src.read([1, 2, 3]).astype(np.float32)  # (3, H, W)
        H, W = rgb.shape[1], rgb.shape[2]

    # Per-band percentile stretch to uint8 for visibility
    for i in range(3):
        lo, hi = np.percentile(rgb[i], 2), np.percentile(rgb[i], 98)
        if hi > lo:
            rgb[i] = (rgb[i] - lo) / (hi - lo)
        rgb[i] = np.clip(rgb[i], 0, 1)

    img = (rgb * 255).astype(np.uint8).transpose(1, 2, 0)  # (H, W, 3)
    Image.fromarray(img).save(out_path)
    return H, W


def boxes_to_df(items: list[dict], split: str) -> pd.DataFrame:
    rows = []
    for item in items:
        tif_path = tif_stem_to_path(item["tile_path"])
        png_name = tif_path.stem + ".png"
        png_path = IMG_OUT / png_name
        H, W = 256, 256  # all tiles are 256x256

        for cx, cy, w, h in item["boxes"]:
            xmin = int((cx - w / 2) * W)
            ymin = int((cy - h / 2) * H)
            xmax = int((cx + w / 2) * W)
            ymax = int((cy + h / 2) * H)
            # Clamp to tile bounds
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(W, xmax), min(H, ymax)
            rows.append({
                "image_path": str(png_path),
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "label": "Tree",
            })
    return pd.DataFrame(rows)


def main():
    # Convert all TIFFs to PNGs
    tif_files = list(TILES_DIR.glob("*.tif"))
    print(f"Converting {len(tif_files)} TIFFs to RGB PNGs...")
    for tif in tif_files:
        out_png = IMG_OUT / (tif.stem + ".png")
        if not out_png.exists():
            convert_tif_to_png(tif, out_png)
    print(f"  Done. PNGs in {IMG_OUT}")

    # Build and save annotation CSVs
    for split in ["train", "val", "test"]:
        json_path = SPLITS_DIR / f"{split}.json"
        with open(json_path) as f:
            items = json.load(f)
        df = boxes_to_df(items, split)
        out_csv = OUT_DIR / f"{split}.csv"
        df.to_csv(out_csv, index=False)
        n_tiles = len(items)
        n_boxes = len(df)
        print(f"  {split}: {n_tiles} tiles, {n_boxes} boxes → {out_csv}")


if __name__ == "__main__":
    main()
