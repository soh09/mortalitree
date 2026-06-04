"""Build the eval set from the manually-vetted 'good images'.

For every PNG in 'good images/' (named <sunet>_<phase>_z17_<x>_<y>_n<count>.png):
  - write a 4-band R,G,B,NIR uint8 GeoTIFF for that z17 tile to eval_set/patches/
    (NAIP, year-matched: prefire->2018, postfire->2020, EPSG:3857, 256x256)
  - emit one box row per crown whose lon/lat bbox center falls in that tile
    (same center-assignment used to make the PNGs, so counts line up)

The label CSV matches the Stage-B pretraining schema (clay/src/data/neon_dataset.py):
  site, imgname, xmin, ymin, xmax, ymax, score, ndvi, lat, lon,
  acquisition_yyyymm, patch_left_utm, patch_top_utm, source_geo_index
Boxes are pixel xyxy in [0, 256], y from the top. For these mercator tiles the
patch_left/top columns hold EPSG:3857 meters (no UTM here); score is 1.0 (human GT).

Output: eval_set/patches/<imgname>.tif  +  eval_set/labels.csv
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds as transform_from_bounds

import overlay_tiles as ot  # tile math, STAC search, SAS signing

GOOD_DIR = Path("good images")
LABEL_DIR = Path("labels")
EVAL_DIR = Path("eval_set")
PATCH_DIR = EVAL_DIR / "patches"
ZOOM = ot.ZOOM
TILE = ot.TILE
PHASE_YEAR = ot.PHASE_YEAR

NAME_RE = re.compile(r"(?P<sunet>.+?)_(?P<phase>prefire|postfire)_z17_"
                     r"(?P<x>\d+)_(?P<y>\d+)_n(?P<n>\d+)\.png")


def geojson_lookup():
    """(sunet, phase) -> geojson path."""
    look = {}
    for g in LABEL_DIR.glob("*.geojson"):
        look[(g.stem.split("_")[0], ot.phase_of(g.stem))] = g
    return look


_gdf_cache = {}


def load_crowns(path):
    """Cached: (gdf_4326_valid, gdf_3857) with invalid/empty geometries dropped."""
    if path not in _gdf_cache:
        gdf = gpd.read_file(path)
        gdf = gdf[~(gdf.geometry.isna() | gdf.geometry.is_empty)].reset_index(drop=True)
        _gdf_cache[path] = (gdf, gdf.to_crs(3857).reset_index(drop=True))
    return _gdf_cache[path]


def fetch_naip_4band(x, y, z, year):
    """4-band (R,G,B,NIR) uint8 tile in EPSG:3857, slippy-aligned, hole-filled
    across quads. Returns (array[4,256,256], dst_transform, acquisition_yyyymm)."""
    mw, ms, me, mn = ot.tile_to_merc_bounds(x, y, z)
    dst_transform = transform_from_bounds(mw, ms, me, mn, TILE, TILE)
    out = np.zeros((4, TILE, TILE), dtype=np.uint8)
    filled = np.zeros((TILE, TILE), dtype=bool)
    acq = ""
    for ft in ot.search_naip(ot.tile_to_lonlat_bounds(x, y, z), year):
        if not acq:
            acq = ft["properties"]["datetime"][:7].replace("-", "")  # yyyymm
        href = ot.sign(ft["assets"]["image"]["href"])
        tmp = np.zeros((4, TILE, TILE), dtype=np.uint8)
        with rasterio.open("/vsicurl/" + href) as src:
            reproject(
                source=rasterio.band(src, [1, 2, 3, 4]),
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
    return out, dst_transform, acq


def write_tif(path, arr, transform):
    with rasterio.open(
        path, "w", driver="GTiff", height=TILE, width=TILE, count=4,
        dtype="uint8", crs="EPSG:3857", transform=transform, compress="deflate",
    ) as dst:
        dst.write(arr)
        for i, name in enumerate(["red", "green", "blue", "nir"], start=1):
            dst.set_band_description(i, name)


def box_ndvi(arr, x0, y0, x1, y1):
    """Mean NDVI inside a pixel box from band 1 (red) and band 4 (nir)."""
    xi0, yi0 = int(np.floor(x0)), int(np.floor(y0))
    xi1, yi1 = int(np.ceil(x1)), int(np.ceil(y1))
    red = arr[0, yi0:yi1, xi0:xi1].astype(np.float32)
    nir = arr[3, yi0:yi1, xi0:xi1].astype(np.float32)
    if red.size == 0:
        return 0.0
    denom = nir + red
    ndvi = np.where(denom > 0, (nir - red) / denom, 0.0)
    return round(float(ndvi.mean()), 4)


def main():
    PATCH_DIR.mkdir(parents=True, exist_ok=True)
    look = geojson_lookup()

    pngs = sorted(p for p in GOOD_DIR.iterdir() if p.suffix == ".png")
    print(f"{len(pngs)} good images -> {EVAL_DIR}/")

    rows = []
    for i, png in enumerate(pngs, 1):
        m = NAME_RE.match(png.name)
        if not m:
            print(f"  skip (unparsed): {png.name}")
            continue
        sunet, phase = m["sunet"], m["phase"]
        tx, ty = int(m["x"]), int(m["y"])
        year = PHASE_YEAR[phase]
        imgname = f"{sunet}_{phase}_z{ZOOM}_{tx}_{ty}"

        gdf4326, gdf3857 = load_crowns(look[(sunet, phase)])

        # crowns assigned to this tile by lon/lat bbox center (matches PNG counts)
        b = gdf4326.geometry.bounds
        idxs = [j for j, r in enumerate(b.itertuples())
                if ot.lonlat_to_tile((r.minx + r.maxx) / 2, (r.miny + r.maxy) / 2, ZOOM)
                == (tx, ty)]

        arr, transform, acq = fetch_naip_4band(tx, ty, ZOOM, year)
        write_tif(PATCH_DIR / f"{imgname}.tif", arr, transform)

        west, south, east, north = ot.tile_to_merc_bounds(tx, ty, ZOOM)
        span_x, span_y = east - west, north - south
        lw, ls, le, ln = ot.tile_to_lonlat_bounds(tx, ty, ZOOM)
        clat, clon = (ls + ln) / 2, (lw + le) / 2

        n_box = 0
        for geom in gdf3857.geometry.iloc[idxs]:
            mnx, mny, mxx, mxy = geom.bounds
            xmin = (mnx - west) / span_x * TILE
            xmax = (mxx - west) / span_x * TILE
            ymin = (north - mxy) / span_y * TILE   # top edge
            ymax = (north - mny) / span_y * TILE   # bottom edge
            xmin = max(0.0, xmin); ymin = max(0.0, ymin)
            xmax = min(float(TILE), xmax); ymax = min(float(TILE), ymax)
            if xmax - xmin < 1.0 or ymax - ymin < 1.0:
                continue
            rows.append({
                "site": sunet,
                "imgname": imgname,
                "xmin": round(xmin, 2), "ymin": round(ymin, 2),
                "xmax": round(xmax, 2), "ymax": round(ymax, 2),
                "score": 1.0,
                "ndvi": box_ndvi(arr, xmin, ymin, xmax, ymax),
                "lat": round(clat, 6), "lon": round(clon, 6),
                "acquisition_yyyymm": acq,
                "patch_left_utm": round(west, 2),   # EPSG:3857 meters here
                "patch_top_utm": round(north, 2),
                "source_geo_index": f"{tx}_{ty}",
            })
            n_box += 1
        print(f"  [{i}/{len(pngs)}] {imgname}.tif  boxes={n_box} (expected ~{m['n']})")

    cols = ["site", "imgname", "xmin", "ymin", "xmax", "ymax", "score", "ndvi",
            "lat", "lon", "acquisition_yyyymm", "patch_left_utm", "patch_top_utm",
            "source_geo_index"]
    df = pd.DataFrame(rows, columns=cols)
    out_csv = EVAL_DIR / "labels.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}: {len(df)} boxes across {df['imgname'].nunique()} patches")


if __name__ == "__main__":
    main()
