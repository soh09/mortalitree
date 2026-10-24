#!/usr/bin/env python3
"""Step 4: table of test results per arm, mean +/- sd over seeds.

    python lidar_pretrain/summarize.py WORK [RUNS_DIR]
"""
import json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

NAMES = {"A": "prebuilt -> fine-tune (current)", "B": "prebuilt -> LiDAR pretrain -> fine-tune",
         "C": "prebuilt -> LiDAR pretrain only",
         "ClayA": "Clay DQ-DETR: Stage B -> Stage C (current)",
         "ClayB": "Clay DQ-DETR: Stage B -> LiDAR stage -> Stage C"}
COLS = [("test_all_tiles", "iou0.4", "f1", "F1@0.4"), ("test_all_tiles", "iou0.5", "f1", "F1@0.5"),
        ("test_all_tiles", "iou0.25", "f1", "F1@0.25"), ("test_all_tiles", None, "count_mae", "MAE"),
        ("test_min5", "iou0.5", "f1", "F1@0.5 min5"), ("test_fixed_ref_min5", "iou0.5", "f1", "F1@0.5 ref")]


def main(work, runs="runs"):
    vals = defaultdict(lambda: defaultdict(list))
    for f in sorted(Path(work, runs).glob("seed*/*/metrics.json")):
        m = json.loads(f.read_text()); arm = m["arm"]
        for sec, iou, k, name in COLS:
            r = m["metrics"][sec]
            vals[arm][name].append(r[iou][k] if iou else r[k])
    print("| arm | " + " | ".join(c[3] for c in COLS) + " | seeds |")
    print("|---" * (len(COLS) + 2) + "|")
    for arm in sorted(vals):
        cells = [f"{np.mean(v):.3f} ± {np.std(v):.3f}" if len(v) > 1 else f"{v[0]:.3f}" for v in vals[arm].values()]
        print(f"| {arm} {NAMES.get(arm, '')} | " + " | ".join(cells) + f" | {len(next(iter(vals[arm].values())))} |")


if __name__ == "__main__":
    main(*sys.argv[1:3])
