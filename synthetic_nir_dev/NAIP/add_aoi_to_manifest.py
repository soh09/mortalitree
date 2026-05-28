"""
One-off: re-query MPC STAC for every CA forest AOI, build a scene_id -> aoi
map, and add an `aoi` column to manifest.csv. Use after a download was done
with an older download_naip.py that didn't record AOI. Idempotent — safe to
re-run; the aoi column is just overwritten.

When a scene matches multiple AOIs, the first AOI in CA_FOREST_BBOXES wins,
mirroring the dedup order in download_naip.collect_items.
"""

from pathlib import Path

import pandas as pd
from pystac_client import Client

from aois import CA_FOREST_BBOXES
from download_naip import MPC_URL, YEAR, search_aoi

ROOT = Path(__file__).parent
MANIFEST_PATH = ROOT / "manifest.csv"


def build_scene_to_aoi() -> dict[str, str]:
    catalog = Client.open(MPC_URL)
    scene_to_aoi: dict[str, str] = {}
    for aoi in CA_FOREST_BBOXES:
        items = search_aoi(catalog, aoi["bbox"], YEAR)
        # setdefault => first AOI to claim a scene_id wins (matches download_naip)
        for it in items:
            scene_to_aoi.setdefault(it.id, aoi["name"])
        print(f"  {aoi['name']}: {len(items)} items")
    return scene_to_aoi


def main():
    if not MANIFEST_PATH.exists():
        raise SystemExit(f"No manifest at {MANIFEST_PATH}")

    print(f"Re-querying STAC for {len(CA_FOREST_BBOXES)} AOIs ...")
    scene_to_aoi = build_scene_to_aoi()
    print(f"\nMapped {len(scene_to_aoi)} unique scenes to AOIs.\n")

    df = pd.read_csv(MANIFEST_PATH)
    df["aoi"] = df["scene_id"].map(scene_to_aoi)

    unmapped = df.loc[df["aoi"].isna(), "scene_id"].unique()
    if len(unmapped):
        print(f"WARN: {len(unmapped)} scene(s) in manifest had no AOI match:")
        for sid in unmapped:
            print(f"  - {sid}")

    # Place aoi right after scene_id for readability.
    cols = list(df.columns)
    cols.remove("aoi")
    cols.insert(cols.index("scene_id") + 1, "aoi")
    df = df[cols]

    df.to_csv(MANIFEST_PATH, index=False)
    print(f"\nWrote {MANIFEST_PATH}")
    print("Per-AOI patch counts:")
    print(df["aoi"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
