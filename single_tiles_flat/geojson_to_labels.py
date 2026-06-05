"""
Convert per-tile annotation geojsons into a combined labels CSV in the style of
clay/labels.csv.

Two groups:
  good/prefire/*.geojson   -> prefire_labels.csv   (NAIP 2020)
  postfire/*.geojson       -> postfire_labels.csv  (NAIP 2022)

Tile filenames are "{z}_{x}_{y}.geojson" in Web Mercator (slippy / XYZ) convention.
Bounding boxes are emitted in 256x256 pixel space (matching the 256px PNG tiles and
the value range of clay/labels.csv).

Columns: site, imgname, xmin, ymin, xmax, ymax, score, ndvi, lat, lon,
         acquisition_yyyymm, patch_left_utm, patch_top_utm, source_geo_index
"""

import csv
import json
import math
from pathlib import Path

from pyproj import Transformer

BASE = Path(__file__).parent
GROUPS = {
    "prefire":  {"src": BASE / "good" / "prefire", "out": BASE / "prefire_labels.csv",  "year": "2020"},
    "postfire": {"src": BASE / "good" / "postfire", "out": BASE / "postfire_labels.csv", "year": "2022"},
}

COLUMNS = [
    "site", "imgname", "xmin", "ymin", "xmax", "ymax", "score", "ndvi",
    "lat", "lon", "acquisition_yyyymm", "patch_left_utm", "patch_top_utm",
    "source_geo_index",
]
TILE_PX = 256  # output pixel space


def lonlat_to_pixel(lon, lat, z, x, y, size=TILE_PX):
    """Map a WGS-84 point to pixel coords within slippy tile (z, x, y)."""
    n = 2 ** z
    px = ((lon + 180.0) / 360.0 * n - x) * size
    lat_rad = math.radians(lat)
    gy = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    py = (gy - y) * size
    return px, py


def tile_center_lonlat(z, x, y):
    n = 2 ** z
    lon = (x + 0.5) / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 0.5) / n))))
    return lon, lat


def tile_nw_lonlat(z, x, y):
    """North-west (top-left) corner of the tile in WGS-84."""
    n = 2 ** z
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lon, lat


def utm_epsg(lon, lat):
    zone = int((lon + 180.0) / 6.0) + 1
    return (32600 if lat >= 0 else 32700) + zone


def iter_coords(geom):
    """Yield (lon, lat) for every vertex of a (Multi)Polygon/(Multi)LineString/Point."""
    t = geom.get("type")
    coords = geom.get("coordinates")
    if t == "Point":
        yield coords[0], coords[1]
    elif t in ("MultiPoint", "LineString"):
        for c in coords:
            yield c[0], c[1]
    elif t in ("Polygon", "MultiLineString"):
        for ring in coords:
            for c in ring:
                yield c[0], c[1]
    elif t == "MultiPolygon":
        for poly in coords:
            for ring in poly:
                for c in ring:
                    yield c[0], c[1]


def process_group(src_dir, out_path, year):
    rows = []
    files = sorted(src_dir.glob("*.geojson"))
    skipped_features = 0
    for fp in files:
        stem = fp.stem  # "17_21008_50937" (note: some have a trailing _a)
        parts = stem.split("_")
        z, x, y = int(parts[0]), int(parts[1]), int(parts[2])

        clon, clat = tile_center_lonlat(z, x, y)
        nw_lon, nw_lat = tile_nw_lonlat(z, x, y)
        epsg = utm_epsg(nw_lon, nw_lat)
        to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        left_utm, top_utm = to_utm.transform(nw_lon, nw_lat)

        with open(fp) as f:
            gj = json.load(f)
        feats = gj.get("features", [])
        site = ""
        if feats:
            site = feats[0].get("properties", {}).get("FIRE_SLUG", "") or ""

        for feat in feats:
            geom = feat.get("geometry")
            if not geom:
                continue
            xs, ys = [], []
            for lon, lat in iter_coords(geom):
                px, py = lonlat_to_pixel(lon, lat, z, x, y)
                xs.append(px)
                ys.append(py)
            if not xs:
                continue
            xmin = max(0.0, min(xs))
            ymin = max(0.0, min(ys))
            xmax = min(float(TILE_PX), max(xs))
            ymax = min(float(TILE_PX), max(ys))
            if xmax - xmin < 0.5 or ymax - ymin < 0.5:  # degenerate / fully outside
                skipped_features += 1
                continue
            rows.append({
                "site": site,
                "imgname": stem,
                "xmin": round(xmin, 2),
                "ymin": round(ymin, 2),
                "xmax": round(xmax, 2),
                "ymax": round(ymax, 2),
                "score": 1.0,
                "ndvi": "",
                "lat": round(clat, 6),
                "lon": round(clon, 6),
                "acquisition_yyyymm": year,
                "patch_left_utm": round(left_utm, 1),
                "patch_top_utm": round(top_utm, 1),
                "source_geo_index": f"{x}_{y}",
            })

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"{out_path.name}: {len(files)} tiles, {len(rows)} boxes "
          f"({skipped_features} degenerate skipped)")


def main():
    for name, cfg in GROUPS.items():
        process_group(cfg["src"], cfg["out"], cfg["year"])


if __name__ == "__main__":
    main()
