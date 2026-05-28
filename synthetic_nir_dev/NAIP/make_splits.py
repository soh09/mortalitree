"""
Generate scene-level train/val/test splits, stratified by AOI.

For each AOI, scenes are sorted by scene_id and the first N_TEST_PER_AOI go
to test, the next N_VAL_PER_AOI go to val, and the rest to train. This
guarantees every biome is represented in every split.

Writes splits.csv with columns scene_id,split. Reads manifest.csv as the
source of which scenes exist; the manifest must have the `aoi` column —
run add_aoi_to_manifest.py first if not.

Sort-based assignment is stable as long as the set of downloaded tiles
doesn't change. If you add more tiles later, scenes may shift between
splits — re-run and commit the new splits.csv if so.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
MANIFEST_PATH = ROOT / "manifest.csv"
SPLITS_PATH = ROOT / "splits.csv"

N_TEST_PER_AOI = 1
N_VAL_PER_AOI = 1


def main():
    if not MANIFEST_PATH.exists():
        raise SystemExit(f"No manifest at {MANIFEST_PATH}")

    df = pd.read_csv(MANIFEST_PATH)
    if "aoi" not in df.columns:
        raise SystemExit(
            "manifest.csv has no 'aoi' column. Run add_aoi_to_manifest.py first."
        )

    # One row per scene, sorted for deterministic assignment.
    scene_df = (
        df[["scene_id", "aoi"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["aoi", "scene_id"])
        .reset_index(drop=True)
    )

    rows: list[tuple[str, str]] = []
    for aoi, group in scene_df.groupby("aoi"):
        scenes = list(group["scene_id"])
        test = scenes[:N_TEST_PER_AOI]
        val = scenes[N_TEST_PER_AOI : N_TEST_PER_AOI + N_VAL_PER_AOI]
        train = scenes[N_TEST_PER_AOI + N_VAL_PER_AOI :]
        if not train:
            print(f"WARN: aoi={aoi} has only {len(scenes)} scene(s); "
                  f"no train coverage from this biome.")
        rows.extend((s, "test") for s in test)
        rows.extend((s, "val") for s in val)
        rows.extend((s, "train") for s in train)

    splits = pd.DataFrame(rows, columns=["scene_id", "split"])
    splits.to_csv(SPLITS_PATH, index=False)

    print(f"\nWrote {SPLITS_PATH} ({len(splits)} scenes)\n")
    print("Overall split counts:")
    print(splits["split"].value_counts().to_string())
    print("\nPer-AOI breakdown:")
    annotated = splits.merge(scene_df, on="scene_id")
    print(pd.crosstab(annotated["aoi"], annotated["split"], margins=True))


if __name__ == "__main__":
    main()
