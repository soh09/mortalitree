"""
Download NAIP imagery for each annotated ZXY tile.

- prefire/  *.geojson  → 2020 NAIP  → prefire_img/{stem}.tif + prefire_img/{stem}.png
- postfire/ *.geojson  → 2022 NAIP  → postfire_img/{stem}.tif + postfire_img/{stem}.png

Tile filenames are "{z}_{x}_{y}.geojson" in standard Web Mercator (XYZ / Slippy Map) convention.
NAIP COGs are fetched from Microsoft Planetary Computer STAC.
Output GeoTIFFs are 4-band (R,G,B,NIR) in the COG's native CRS at native NAIP resolution.
Output PNGs are RGB 8-bit (NIR band dropped) resampled to 256×256.
"""

import math
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import planetary_computer as pc
import rasterio
import rasterio.transform
import rasterio.warp
from PIL import Image
from pystac_client import Client

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────

BASE = Path(__file__).parent
DIRS = {
    "prefire":  {"src": BASE / "prefire",  "dst": BASE / "prefire_img",  "year": "2020"},
    "postfire": {"src": BASE / "postfire", "dst": BASE / "postfire_img", "year": "2022"},
}

MPC_URL    = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "naip"
PNG_SIZE   = 256          # output PNG side length in pixels
MAX_WORKERS = 4

# ── Geometry helpers ──────────────────────────────────────────────────────────

def tile_to_bbox(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    """Return (west, south, east, north) in WGS-84 for a Slippy Map tile."""
    n = 2 ** z
    lon_w = x       / n * 360.0 - 180.0
    lon_e = (x + 1) / n * 360.0 - 180.0
    lat_n = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 *  y      / n))))
    lat_s = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return lon_w, lat_s, lon_e, lat_n


# ── STAC helpers ──────────────────────────────────────────────────────────────

def find_naip_item(catalog: Client, bbox: tuple, year: str):
    """Return best (largest overlap) signed NAIP item for the bbox in given year."""
    results = list(catalog.search(
        collections=[COLLECTION],
        bbox=list(bbox),
        datetime=f"{year}-01-01/{year}-12-31",
    ).items())
    if not results:
        return None
    # Prefer the item whose centroid is closest to the tile centre — simple proxy
    # for "most coverage" when multiple items exist.
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    def dist(item):
        b = item.bbox  # [west, south, east, north]
        return abs((b[0] + b[2]) / 2 - cx) + abs((b[1] + b[3]) / 2 - cy)
    best = min(results, key=dist)
    return pc.sign(best)


# ── Per-tile download ─────────────────────────────────────────────────────────

def process_tile(stem: str, dst_dir: Path, year: str, catalog: Client) -> str:
    """Download one tile; returns a short status string."""
    tif_path = dst_dir / f"{stem}.tif"
    png_path = dst_dir / f"{stem}.png"
    if tif_path.exists() and png_path.exists():
        return f"{stem}: skip (exists)"

    parts = stem.split("_")
    z, x, y = int(parts[0]), int(parts[1]), int(parts[2])
    bbox = tile_to_bbox(z, x, y)   # (west, south, east, north) WGS-84

    item = find_naip_item(catalog, bbox, year)
    if item is None:
        return f"{stem}: NO NAIP FOUND for {year}"

    url = item.assets["image"].href

    try:
        with rasterio.open(url) as src:
            # Project tile bbox corners into the COG's native CRS
            xs = [bbox[0], bbox[2]]
            ys = [bbox[1], bbox[3]]
            xs_native, ys_native = rasterio.warp.transform(
                "EPSG:4326", src.crs, xs, ys
            )
            west_n  = min(xs_native)
            east_n  = max(xs_native)
            south_n = min(ys_native)
            north_n = max(ys_native)

            window = rasterio.windows.from_bounds(
                west_n, south_n, east_n, north_n,
                transform=src.transform,
            )
            # Clamp window to valid raster extent
            window = window.intersection(
                rasterio.windows.Window(0, 0, src.width, src.height)
            )
            if window.width < 1 or window.height < 1:
                return f"{stem}: tile bbox outside raster extent"

            data = src.read(window=window)          # (bands, rows, cols)
            win_transform = src.window_transform(window)

            # ── Save GeoTIFF ────────────────────────────────────────────────
            if not tif_path.exists():
                meta = src.meta.copy()
                meta.update({
                    "height": data.shape[1],
                    "width":  data.shape[2],
                    "transform": win_transform,
                })
                with rasterio.open(tif_path, "w", **meta) as dst:
                    dst.write(data)

    except Exception as e:
        return f"{stem}: ERROR reading COG — {e}"

    # ── Save PNG (RGB only, 256×256) ────────────────────────────────────────
    if not png_path.exists():
        rgb = data[:3].astype(np.float32)   # R, G, B bands
        # Percentile stretch per band for display
        for i in range(3):
            lo, hi = np.percentile(rgb[i], 2), np.percentile(rgb[i], 98)
            if hi > lo:
                rgb[i] = np.clip((rgb[i] - lo) / (hi - lo) * 255, 0, 255)
            else:
                rgb[i] = np.zeros_like(rgb[i])
        rgb_uint8 = rgb.astype(np.uint8).transpose(1, 2, 0)  # (H, W, 3)
        img = Image.fromarray(rgb_uint8, mode="RGB").resize(
            (PNG_SIZE, PNG_SIZE), Image.LANCZOS
        )
        img.save(png_path)

    return f"{stem}: done"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    catalog = Client.open(MPC_URL)

    for split, cfg in DIRS.items():
        src_dir: Path = cfg["src"]
        dst_dir: Path = cfg["dst"]
        year: str     = cfg["year"]

        dst_dir.mkdir(exist_ok=True)

        stems = sorted(p.stem for p in src_dir.glob("*.geojson"))
        if not stems:
            print(f"[{split}] no geojson files found, skipping")
            continue

        print(f"\n[{split}]  {len(stems)} tiles  →  {dst_dir}  (NAIP {year})")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(process_tile, stem, dst_dir, year, catalog): stem
                for stem in stems
            }
            for fut in as_completed(futures):
                print(f"  {fut.result()}")

    print("\ndone.")


if __name__ == "__main__":
    main()
