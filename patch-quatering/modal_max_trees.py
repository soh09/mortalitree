"""Count the max (and distribution of) trees per 256x256 patch, per site.

Reads the labels.csv produced by dataset_dev/modal_pipeline.py from the `mot`
volume and reports boxes-per-patch statistics for each site (TEAK, SOAP, YELL)
plus an overall block, to set `num_queries` (Q) for the detection head.
Spec gotcha #12: Q should be >= max trees per tile, with headroom.

Note: patches with zero boxes don't appear in labels.csv, which is fine — we
want the maximum, and that's necessarily among patches that have boxes.

Usage:
    modal run modal_max_trees.py
    modal run modal_max_trees.py --labels-csv /data/patches/labels.csv
"""
from __future__ import annotations

import modal

VOLUME_NAME = "mot"
image = modal.Image.debian_slim(python_version="3.11").pip_install("pandas", "numpy")
app = modal.App("clay-max-trees")
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(image=image, volumes={"/data": vol}, timeout=600)
def count(labels_csv: str = "/data/patches/labels.csv") -> dict:
    import numpy as np
    import pandas as pd

    def _stats(per_patch: "pd.Series") -> dict:
        """Boxes-per-patch summary for one group of patches."""
        counts = per_patch.to_numpy()
        return {
            "n_patches": int(per_patch.shape[0]),
            "total_boxes": int(counts.sum()),
            "mean": float(counts.mean()),
            "p50": float(np.percentile(counts, 50)),
            "p90": float(np.percentile(counts, 90)),
            "p95": float(np.percentile(counts, 95)),
            "p99": float(np.percentile(counts, 99)),
            "max": int(counts.max()),
            "max_imgname": str(per_patch.idxmax()),
        }

    df = pd.read_csv(labels_csv)

    overall = _stats(df.groupby("imgname").size())

    per_site: dict[str, dict] = {}
    if "site" in df.columns:
        for site, sub in df.groupby("site"):
            per_site[str(site)] = _stats(sub.groupby("imgname").size())

    return {"labels_csv": labels_csv, "overall": overall, "per_site": per_site}


def _print_block(title: str, s: dict) -> None:
    import math
    print("-" * 60)
    print(f"  {title}")
    print("-" * 60)
    print(f"  labeled patches : {s['n_patches']:>8,}    total boxes : {s['total_boxes']:>9,}")
    print(f"  mean / patch    : {s['mean']:>8.1f}")
    print(f"  p50 / p90 / p95 / p99 : "
          f"{s['p50']:.0f} / {s['p90']:.0f} / {s['p95']:.0f} / {s['p99']:.0f}")
    print(f"  MAX / patch     : {s['max']:>8,}   ({s['max_imgname']})")
    # Suggested Q from p99 (robust) and from max (hard ceiling), rounded to /4.
    q_p99 = int(math.ceil(s["p99"] * 1.2 / 4) * 4)
    q_max = int(math.ceil(s["max"] * 1.2 / 4) * 4)
    print(f"  suggested Q     : {q_p99}  (p99 x1.2)   |   {q_max}  (max x1.2)")


@app.local_entrypoint()
def main(labels_csv: str = "/data/patches/labels.csv"):
    r = count.remote(labels_csv)

    print("\n" + "=" * 60)
    print(f"  Boxes-per-patch report  ({r['labels_csv']})")
    print("=" * 60)

    for site in sorted(r["per_site"]):
        _print_block(f"SITE: {site}", r["per_site"][site])

    _print_block("OVERALL (all sites)", r["overall"])
    print("=" * 60)
    print("  Q must be >= max-per-patch to avoid dropping GT (spec gotcha #12),")
    print("  but max is often a lone outlier — prefer the p99-based Q and filter")
    print("  pathologically dense patches in modal_pipeline.py instead.")
    print("  Set model.num_queries in configs/stage_b.yaml & stage_c.yaml.")
    print("=" * 60 + "\n")
