"""Sanity-check overlays: render every z17 NAIP tile owned by each labeler's
crown annotations, with the crowns drawn on top.

Imagery is year-matched per phase via Microsoft Planetary Computer NAIP:
  prefire  -> 2018 NAIP   (CA has no 2019 NAIP; even-year cadence)
  postfire -> 2020 NAIP

For each labels/*.geojson:
  - sunet id = filename.split('_')[0]   (pre/post share one dir)
  - phase    = 'prefire' / 'postfire'   (from filename)
  - each crown is assigned to ONE z17 tile via its lon/lat bbox center
  - for every tile that owns >=1 crown: fetch that year's NAIP, overlay its
    crowns, save a titled PNG. Per-tile counts sum to the file's crown total.

Output: tiles/<sunet>/<sunet>_<phase>_z17_<x>_<y>_n<count>.png
NAIP tiles are cached under tiles/_naip_cache/ so re-runs don't re-download.
"""
import math
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import geopandas as gpd
import requests
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds as transform_from_bounds
from PIL import Image

LABEL_DIR = Path("labels")
OUT_DIR = Path("tiles")
CACHE_DIR = OUT_DIR / "_naip_cache"
ZOOM = 17
TILE = 256
R = 6378137.0
ORIGIN = math.pi * R

PHASE_YEAR = {"prefire": 2020, "postfire": 2022}

STAC_SEARCH = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS_SIGN = "https://planetarycomputer.microsoft.com/api/sas/v1/sign"


# ---- slippy-tile math (web mercator) ---------------------------------------
def lonlat_to_tile(lon, lat, z):
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    return x, y


def tile_to_merc_bounds(x, y, z):
    """slippy tile -> EPSG:3857 (west, south, east, north) in meters."""
    n = 2 ** z
    west = x / n * 2 * ORIGIN - ORIGIN
    east = (x + 1) / n * 2 * ORIGIN - ORIGIN
    north = ORIGIN - y / n * 2 * ORIGIN
    south = ORIGIN - (y + 1) / n * 2 * ORIGIN
    return west, south, east, north


def tile_to_lonlat_bounds(x, y, z):
    """slippy tile -> lon/lat (west, south, east, north)."""
    n = 2 ** z
    lon = lambda xt: xt / n * 360.0 - 180.0
    lat = lambda yt: math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * yt / n))))
    return lon(x), lat(y + 1), lon(x + 1), lat(y)


# ---- Planetary Computer NAIP (year-matched, cached) -------------------------
_signed = {}  # href -> (signed_href, ts)


def sign(href):
    now = time.time()
    cached = _signed.get(href)
    if cached and now - cached[1] < 3000:  # SAS tokens last ~1h; refresh early
        return cached[0]
    s = requests.get(SAS_SIGN, params={"href": href}, timeout=30).json()["href"]
    _signed[href] = (s, now)
    return s


def search_naip(bbox_lonlat, year):
    body = {"collections": ["naip"], "bbox": list(bbox_lonlat),
            "datetime": f"{year}-01-01/{year}-12-31", "limit": 10}
    r = requests.post(STAC_SEARCH, json=body, timeout=60)
    r.raise_for_status()
    return r.json().get("features", [])


def fetch_naip_tile(x, y, z, year):
    """256x256 RGB NAIP tile (EPSG:3857) for the given year, slippy-aligned.

    Reprojects each intersecting NAIP quad into the fixed tile grid and fills
    holes from later quads, so tiles straddling two quads render fully.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{year}_{z}_{x}_{y}.jpg"
    if cache.exists():
        return Image.open(cache).convert("RGB")

    mw, ms, me, mn = tile_to_merc_bounds(x, y, z)
    dst_transform = transform_from_bounds(mw, ms, me, mn, TILE, TILE)
    out = np.zeros((3, TILE, TILE), dtype=np.uint8)
    filled = np.zeros((TILE, TILE), dtype=bool)

    for ft in search_naip(tile_to_lonlat_bounds(x, y, z), year):
        href = sign(ft["assets"]["image"]["href"])
        tmp = np.zeros((3, TILE, TILE), dtype=np.uint8)
        with rasterio.open("/vsicurl/" + href) as src:
            reproject(
                source=rasterio.band(src, [1, 2, 3]),
                destination=tmp,
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=dst_transform, dst_crs="EPSG:3857",
                resampling=Resampling.bilinear,
            )
        valid = (tmp.sum(axis=0) > 0) & (~filled)
        out[:, valid] = tmp[:, valid]
        filled |= valid
        if filled.all():
            break

    img = Image.fromarray(np.transpose(out, (1, 2, 0)), "RGB")
    img.save(cache, "JPEG", quality=90)
    return img


def merc_to_px(mx, my, west, north, span_x, span_y):
    return (mx - west) / span_x * TILE, (north - my) / span_y * TILE


# ---- per-file processing ----------------------------------------------------
def phase_of(name):
    low = name.lower()
    if "prefire" in low:
        return "prefire"
    if "postfire" in low:
        return "postfire"
    return "unknown"


def process_file(path):
    sunet = path.stem.split("_")[0]
    phase = phase_of(path.stem)
    year = PHASE_YEAR.get(phase)
    if year is None:
        print(f"skip {path.name}: cannot infer phase/year")
        return
    out_dir = OUT_DIR / sunet
    out_dir.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(path)
    n_total = len(gdf)
    gdf = gdf[~(gdf.geometry.isna() | gdf.geometry.is_empty)].reset_index(drop=True)
    n_valid = len(gdf)

    # assign each crown to one z17 tile by its lon/lat bbox center
    b = gdf.geometry.bounds
    gdf_m = gdf.to_crs(3857).reset_index(drop=True)
    by_tile = defaultdict(list)
    for idx, r in enumerate(b.itertuples()):
        tx, ty = lonlat_to_tile((r.minx + r.maxx) / 2, (r.miny + r.maxy) / 2, ZOOM)
        by_tile[(tx, ty)].append(idx)

    print(f"\n{path.name}: sunet={sunet} phase={phase} year={year}  "
          f"crowns={n_valid} (dropped {n_total - n_valid})  tiles={len(by_tile)}")

    for i, ((tx, ty), idxs) in enumerate(sorted(by_tile.items()), 1):
        west, south, east, north = tile_to_merc_bounds(tx, ty, ZOOM)
        span_x, span_y = east - west, north - south
        count = len(idxs)

        img = fetch_naip_tile(tx, ty, ZOOM, year)
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(np.asarray(img), extent=[0, TILE, TILE, 0])
        for geom in gdf_m.geometry.iloc[idxs]:
            polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
            for poly in polys:
                xs, ys = poly.exterior.xy
                px = [merc_to_px(x, y, west, north, span_x, span_y)
                      for x, y in zip(xs, ys)]
                ax.add_patch(mpatches.Polygon(px, closed=True, fill=False,
                                              edgecolor="cyan", linewidth=1.2))
        ax.set_xlim(0, TILE)
        ax.set_ylim(TILE, 0)
        ax.set_title(f"{sunet} {phase} {year}  z{ZOOM} x={tx} y={ty}  trees={count}",
                     fontsize=11)
        ax.axis("off")
        fig.tight_layout()
        out = out_dir / f"{sunet}_{phase}_z{ZOOM}_{tx}_{ty}_n{count}.png"
        fig.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"  [{i}/{len(by_tile)}] {out.name}")


def main():
    files = sorted(LABEL_DIR.glob("*.geojson"))
    print(f"{len(files)} label files -> {OUT_DIR}/  (prefire={PHASE_YEAR['prefire']}, "
          f"postfire={PHASE_YEAR['postfire']})")
    for f in files:
        process_file(f)
    print("\ndone.")


if __name__ == "__main__":
    main()
