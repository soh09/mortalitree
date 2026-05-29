"""
Build a scene-level train/val split for Stage A pretraining.

The split MUST operate on whole source .tif scenes — never on individual
patches. Patches cropped from the same orthomosaic share crowns, shadows,
and radiometry, so a patch-level random split leaks nearly identical
content across the val boundary and makes val loss meaningless. The
one-line rule is: split the files, then tile — never the other way around.

Strategy, within each AOI:
  1. Read the geographic centroid (lon, lat) of every scene from the
     source .tif's bounds.
  2. Project centroids onto the AOI's longer geographic axis (longitude
     if the AOI is wider east-west, latitude otherwise).
  3. Sort scenes along that axis; the last ceil(VAL_FRACTION * N) become
     val (the AOI's spatial edge). The rest are train.
  4. Optionally drop BUFFER_SCENES scenes immediately adjacent to the
     val partition (mark them 'reserved') so train and val are
     separated by at least one extra scene of physical distance.

This gives val coverage stratified across every biome AND geographic
separation between train and val within each biome.

Reserved scenes — excluded from the pretraining pool entirely:
  Drop a JSON file at $DATA_ROOT/reserved_scenes.json with shape
    {"scenes": ["m_3711714_se_11_060_...", ...],
     "aois":   ["some_aoi_name", ...]}
  Any scene_id listed in `scenes` OR whose AOI is listed in `aois` is
  reserved. Use this for any scene overlapping the supervised detection
  test set so test imagery never touches the encoder during pretraining.

Output: $DATA_ROOT/splits.csv with columns
  scene_id, aoi, split, lon_centroid, lat_centroid
where split ∈ {"train", "val", "reserved"}.

Env vars:
  VAL_FRACTION       fraction of scenes per AOI for val (default 0.10)
  BUFFER_SCENES      buffer scenes between train and val per AOI (default 0)
  RESERVED_SCENES    comma-separated scene_ids to reserve
  RESERVED_AOIS      comma-separated AOI names to reserve

Requires: rasterio, pandas
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pandas as pd
import rasterio
from rasterio.warp import transform_bounds

DATA_ROOT = Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parent.parent))
MANIFEST_PATH = DATA_ROOT / "manifest.csv"
SPLITS_PATH = DATA_ROOT / "splits.csv"
RESERVED_JSON = DATA_ROOT / "reserved_scenes.json"
RAW_ROOT = DATA_ROOT / "raw"

VAL_FRACTION = float(os.environ.get("VAL_FRACTION", "0.10"))
BUFFER_SCENES = int(os.environ.get("BUFFER_SCENES", "0"))


def find_tif_for_scene(scene_id: str) -> Path | None:
    """Search every raw/<year>/ subdir for the source .tif."""
    if not RAW_ROOT.exists():
        return None
    for year_dir in RAW_ROOT.iterdir():
        if not year_dir.is_dir():
            continue
        candidate = year_dir / f"{scene_id}.tif"
        if candidate.exists():
            return candidate
    return None


def scene_centroid_lonlat(tif_path: Path) -> tuple[float, float]:
    """Centroid of a scene's bounding box, reprojected to WGS-84 lon/lat."""
    with rasterio.open(tif_path) as src:
        bounds = src.bounds  # (left, bottom, right, top) in src CRS (UTM)
        lon_l, lat_b, lon_r, lat_t = transform_bounds(
            src.crs, "EPSG:4326", *bounds, densify_pts=21
        )
    return ((lon_l + lon_r) / 2.0, (lat_b + lat_t) / 2.0)


def load_reserved() -> tuple[set[str], set[str]]:
    """Union reserved scenes/AOIs from JSON file and env vars."""
    scenes: set[str] = set()
    aois: set[str] = set()
    if RESERVED_JSON.exists():
        data = json.loads(RESERVED_JSON.read_text())
        scenes.update(data.get("scenes", []))
        aois.update(data.get("aois", []))
    env_scenes = os.environ.get("RESERVED_SCENES", "").strip()
    if env_scenes:
        scenes.update(s.strip() for s in env_scenes.split(",") if s.strip())
    env_aois = os.environ.get("RESERVED_AOIS", "").strip()
    if env_aois:
        aois.update(a.strip() for a in env_aois.split(",") if a.strip())
    return scenes, aois


def split_aoi(scenes_df: pd.DataFrame) -> pd.DataFrame:
    """Assign train/val/reserved within one AOI. Expects columns
    scene_id, lon_centroid, lat_centroid."""
    n = len(scenes_df)
    n_val = max(1, math.ceil(n * VAL_FRACTION)) if n > 1 else 0

    lon_rng = scenes_df["lon_centroid"].max() - scenes_df["lon_centroid"].min()
    lat_rng = scenes_df["lat_centroid"].max() - scenes_df["lat_centroid"].min()
    axis_col = "lon_centroid" if lon_rng >= lat_rng else "lat_centroid"

    sorted_df = scenes_df.sort_values(axis_col).reset_index(drop=True)
    splits = ["train"] * n
    if n_val:
        # val = the AOI's spatial edge along the longer axis
        for i in range(n - n_val, n):
            splits[i] = "val"
        if BUFFER_SCENES > 0:
            start = max(0, n - n_val - BUFFER_SCENES)
            stop = n - n_val
            for i in range(start, stop):
                splits[i] = "reserved"
    sorted_df["split"] = splits
    return sorted_df


def main() -> None:
    if not MANIFEST_PATH.exists():
        raise SystemExit(f"No manifest at {MANIFEST_PATH}; run tile step first.")

    print(f"Reading manifest: {MANIFEST_PATH}")
    manifest = pd.read_csv(MANIFEST_PATH)
    scenes = (manifest[["scene_id", "aoi"]]
              .drop_duplicates()
              .reset_index(drop=True))
    print(f"  {len(scenes)} unique scenes across {scenes['aoi'].nunique()} AOIs")

    reserved_scenes, reserved_aois = load_reserved()
    if reserved_scenes or reserved_aois:
        print(f"reserved input: {len(reserved_scenes)} scenes, "
              f"{len(reserved_aois)} AOIs")
    print(f"VAL_FRACTION={VAL_FRACTION}  BUFFER_SCENES={BUFFER_SCENES}")

    print("Reading scene centroids from raw .tifs ...")
    centroids: list[tuple[float, float] | None] = []
    for sid in scenes["scene_id"]:
        tif = find_tif_for_scene(sid)
        if tif is None:
            print(f"  WARN: no .tif for {sid}; marking reserved")
            centroids.append(None)
            continue
        try:
            centroids.append(scene_centroid_lonlat(tif))
        except Exception as e:
            print(f"  WARN: could not read centroid for {sid}: {e}")
            centroids.append(None)
    scenes["lon_centroid"] = [c[0] if c else None for c in centroids]
    scenes["lat_centroid"] = [c[1] if c else None for c in centroids]

    scenes["split"] = "train"
    is_reserved = (
        scenes["scene_id"].isin(reserved_scenes)
        | scenes["aoi"].isin(reserved_aois)
        | scenes["lon_centroid"].isna()
    )
    scenes.loc[is_reserved, "split"] = "reserved"

    available = scenes[scenes["split"] == "train"]
    parts = [scenes[scenes["split"] != "train"]]
    for _, group in available.groupby("aoi"):
        parts.append(split_aoi(group))
    out = pd.concat(parts).sort_values("scene_id").reset_index(drop=True)

    out.to_csv(SPLITS_PATH, index=False)
    print(f"\nWrote {SPLITS_PATH}")
    print("Split counts:")
    print(out["split"].value_counts().to_string())
    print("\nPer-AOI breakdown:")
    print(pd.crosstab(out["aoi"], out["split"], margins=True))

    # Patch-count sanity check: how many patches actually land in each split
    enriched = manifest.merge(out[["scene_id", "split"]], on="scene_id")
    print("\nPatch counts per split:")
    print(enriched["split"].value_counts().to_string())


if __name__ == "__main__":
    main()
