#!/usr/bin/env python3
"""
List the CHM-NAIP chips in one Allred et al. (2025) tar bin that overlap a fire
perimeter from CAL FIRE's public perimeter layer.

Writes:
  OUT_DIR/perimeter.geojson   the perimeter as downloaded (WGS84)
  OUT_DIR/fire_chips.json     [{key, lidar, naip, inside}] -- `inside` is True when
                              the 256 m chip lies entirely within the perimeter

    python baselines/select_fire_chips.py NAIP_TAR OUT_DIR --name-like CZU \
        [--layer https://.../2020_California_Fire_Perimeters_View/FeatureServer/0]
"""
import argparse, json, re, tarfile, urllib.parse, urllib.request
from pathlib import Path

from pyproj import Transformer
from shapely.geometry import box, shape
from shapely.ops import transform

LAYER_2020 = ("https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/arcgis/rest/services/"
              "2020_California_Fire_Perimeters_View/FeatureServer/0")
NAME = re.compile(r"(\d+)_(\d+)_(\d+)_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.tif$")


def fetch_perimeter(layer, name_like):
    q = urllib.parse.urlencode({"where": f"UPPER(FIRE_NAME) LIKE '%{name_like.upper()}%'",
                                "outFields": "FIRE_NAME,GIS_ACRES,ALARM_DATE,CONT_DATE",
                                "outSR": "4326", "f": "geojson"})
    gj = json.load(urllib.request.urlopen(f"{layer}/query?{q}", timeout=120))
    if len(gj.get("features", [])) != 1:
        raise SystemExit(f"expected 1 perimeter matching '{name_like}', got {len(gj.get('features', []))}")
    return gj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("naip_tar"); ap.add_argument("out")
    ap.add_argument("--name-like", required=True)
    ap.add_argument("--layer", default=LAYER_2020)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    gj = fetch_perimeter(a.layer, a.name_like)
    (out / "perimeter.geojson").write_text(json.dumps(gj))
    props = gj["features"][0]["properties"]
    print(f"perimeter: {props['FIRE_NAME']}  {props['GIS_ACRES']:.2f} acres")

    names = [m.name for m in tarfile.open(a.naip_tar).getmembers() if m.name.endswith(".tif")]
    zone = NAME.search(names[0]).group(1)
    # filename coordinates are WGS84 UTM (EPSG:326ZZ), per the dataset README
    to_utm = Transformer.from_crs(4326, int(f"326{zone}"), always_xy=True).transform
    per = transform(to_utm, shape(gj["features"][0]["geometry"]))

    rows = []
    for n in names:
        m = NAME.search(n)
        x, y = int(m.group(2)), int(m.group(3))
        fp = box(x - 128, y - 128, x + 128, y + 128)
        if fp.intersects(per):
            rows.append({"key": f"{m.group(1)}_{x}_{y}_{m.group(4)}", "lidar": m.group(4),
                         "naip": m.group(5), "inside": per.contains(fp)})
    (out / "fire_chips.json").write_text(json.dumps(rows))
    print(f"{len(names)} chips in tar; {len(rows)} overlap the perimeter "
          f"({sum(r['inside'] for r in rows)} fully inside) -> {out/'fire_chips.json'}")


if __name__ == "__main__":
    main()
