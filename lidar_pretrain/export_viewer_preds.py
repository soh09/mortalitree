#!/usr/bin/env python3
"""
Export one run's predictions as an overlay for the tile viewer
(single_tiles_flat/make_tile_viewer.py reads stage_c_data/predictions_extra_*.json).

Picks the seed with the best val F1 (never test), applies that run's val-selected NMS
(score threshold is left to the viewer's slider, which starts at the val-selected value),
and writes normalized x1,y1,x2,y2,score boxes for every tile that has predictions.

    python lidar_pretrain/export_viewer_preds.py RUNS_DIR ARM KEY "LABEL" [--out STAGE_C_DATA_DIR]
    # RUNS_DIR: a local copy of e.g. lidar-pretrain-work:/work/runs_clay_v2 (seed*/ARM/ inside)
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd


def nms(b, s, thr):
    if thr >= 1.0:
        return np.argsort(-s)
    order = np.argsort(-s); keep = []
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    while len(order):
        i = order[0]; keep.append(i); r = order[1:]
        if not len(r):
            break
        w = np.clip(np.minimum(b[i, 2], b[r, 2]) - np.maximum(b[i, 0], b[r, 0]), 0, None)
        h = np.clip(np.minimum(b[i, 3], b[r, 3]) - np.maximum(b[i, 1], b[r, 1]), 0, None)
        iou = w * h / np.maximum(area[i] + area[r] - w * h, 1e-9)
        order = r[iou <= thr]
    return np.array(keep, int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs"); ap.add_argument("arm"); ap.add_argument("key"); ap.add_argument("label")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "single_tiles_flat" / "stage_c_data"))
    a = ap.parse_args()
    runs = sorted(Path(a.runs).glob(f"seed*/{a.arm}/metrics.json"))
    if not runs:
        raise SystemExit(f"no {a.arm} runs under {a.runs}")
    recs = [(json.loads(f.read_text()), f.parent) for f in runs]
    rec, d = max(recs, key=lambda r: r[0]["metrics"]["selected_on_val"]["val_f1_iou0.4"])
    sel, test = rec["metrics"]["selected_on_val"], rec["metrics"]["test_all_tiles"]
    print("val F1@0.4 by seed: " + ", ".join(f"{p.parent.name} {r['metrics']['selected_on_val']['val_f1_iou0.4']:.3f}"
                                             for r, p in recs) + f" -> using {d.parent.name}")
    out = {}
    for f in ("raw_predictions_val.csv", "raw_predictions_test.csv"):
        df = pd.read_csv(d / f); df = df[df.score >= 0.05]
        for img, g in df.groupby("image_path"):
            b = g[["xmin", "ymin", "xmax", "ymax"]].to_numpy(float); s = g.score.to_numpy(float)
            k = nms(b, s, sel["nms"]); k = k[np.argsort(-s[k])]
            out[Path(img).stem] = [[round(x / 256, 4) for x in b[i]] + [round(float(s[i]), 3)] for i in k]
    cfg = {"key": a.key, "label": a.label, "run": f"{Path(a.runs).name}/{d.parent.name}/{a.arm}",
           "chosen_checkpoint": rec.get("chosen_checkpoint"), "val_selected_score": sel["score"], "nms": sel["nms"],
           "val_f1_iou0.4": round(sel["val_f1_iou0.4"], 4), "test_f1_iou0.4": test["iou0.4"]["f1"],
           "test_count_mae": test["count_mae"]}
    dst = Path(a.out) / f"predictions_extra_{a.key}.json"
    dst.write_text(json.dumps({"preds": out, "config": cfg}))
    print(f"{len(out)} tiles, {sum(len(v) for v in out.values())} boxes -> {dst}\n{cfg}")


if __name__ == "__main__":
    main()
