"""Shared paths and helpers for the 3DEP-pretraining experiment (see README.md)."""
import json, sys, urllib.parse, urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
TILES = REPO / "single_tiles_flat" / "stage_c_data"          # 108 hand-labelled NAIP tiles
sys.path.insert(0, str(REPO / "baselines"))

BIN_URL = "https://rangeland.ntsg.umt.edu/data/rap/chm-naip"
PERIMETERS = ("https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/arcgis/rest/services/"
              "2020_California_Fire_Perimeters_View/FeatureServer/0")
FIRES = {  # short name -> CAL FIRE FIRE_NAME (largest feature with that exact name)
    "czu": "CZU LIGHTNING COMPLEX", "scu": "SCU LIGHTNING COMPLEX", "north": "NORTH COMPLEX",
    "creek": "CREEK", "castle": "CASTLE",
}
TILE_PX = 256            # hand-labelled tiles: 256 x 256 px of native 0.6 m NAIP
NAIP_RES = 0.6


def fetch_perimeter(fire_name):
    """(shapely geometry in WGS84, alarm date) for the largest 2020 perimeter named fire_name."""
    from shapely.geometry import shape
    q = urllib.parse.urlencode({"where": f"FIRE_NAME = '{fire_name}'", "outFields": "FIRE_NAME,GIS_ACRES,ALARM_DATE",
                                "outSR": "4326", "f": "geojson"})
    gj = json.load(urllib.request.urlopen(f"{PERIMETERS}/query?{q}", timeout=120))
    f = max(gj["features"], key=lambda f: f["properties"]["GIS_ACRES"] or 0)
    alarm = datetime.fromtimestamp(f["properties"]["ALARM_DATE"] / 1000, timezone.utc).date()
    return shape(f["geometry"]), alarm


def stretch_to_rgb8(bands):
    """Per-image 2-98 percentile stretch of bands 1-3 -> HxWx3 uint8. Same as dfft/modal_train._prep,
    so pretraining windows and hand-labelled tiles reach the model identically."""
    rgb = bands[:3].astype(np.float32)
    for i in range(3):
        lo, hi = np.percentile(rgb[i], 2), np.percentile(rgb[i], 98)
        if hi > lo:
            rgb[i] = (rgb[i] - lo) / (hi - lo)
        rgb[i] = np.clip(rgb[i], 0, 1)
    return (rgb * 255).astype(np.uint8).transpose(1, 2, 0)


def split_items(split):
    return json.loads((TILES / f"{split}.json").read_text())


def tile_footprints(splits=("val", "test")):
    """[(shapely box in the tile's CRS, EPSG int)] for the hand-labelled tiles in `splits`."""
    import rasterio
    from shapely.geometry import box
    out = []
    for s in splits:
        for it in split_items(s):
            with rasterio.open(TILES / "tiles" / Path(it["tile_path"]).name) as r:
                out.append((box(*r.bounds), r.crs.to_epsg()))
    return out
