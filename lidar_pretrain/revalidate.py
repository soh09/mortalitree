#!/usr/bin/env python3
"""
Re-run validation for a new val/test split without a GPU, from saved predictions.

For every run in WORK/RUNS_DIR/seed*/{arm}/:
  - Clay runs keep predictions for every saved Stage C checkpoint (each 5 epochs, best, final)
    on all 108 tiles (raw_predictions_all_ckpts.csv.gz): the checkpoint, score threshold and
    NMS are re-chosen on the NEW val tiles (best F1 at IoU 0.4), then the NEW test tiles are
    scored once.
  - DeepForest runs keep predictions on the old val + test tiles only (raw_predictions_{val,test}.csv),
    so a new split must draw from those tiles; threshold and NMS are re-chosen, the model is fixed.
Tiles the model was trained on (the original train.json) are refused as val/test.

    python lidar_pretrain/revalidate.py WORK RUNS_DIR [--val NEW_VAL.json] [--test NEW_TEST.json]

NEW_VAL / NEW_TEST: a list of tile records like single_tiles_flat/stage_c_data/val.json, or a plain
JSON list of tile names (e.g. "prefire_17_22358_51378"). Defaults: the current val.json / test.json.
Writes WORK/revalidate_{RUNS_DIR}.csv and prints mean ± sd per arm.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

import common as C
import train as T

NMSS = T.NMSS + [1.0]


def tile_names(path):
    items = json.loads(Path(path).read_text())
    return sorted(Path(x["tile_path"]).stem if isinstance(x, dict) else Path(x).stem for x in items)


def all_gt():
    """{tile.png: (N, 4) pixel xyxy}, built exactly as train.prep_labelled does."""
    out = {}
    for split in ("train", "val", "test"):
        for it in C.split_items(split):
            b = [[max(0, int((cx - w / 2) * 256)), max(0, int((cy - h / 2) * 256)),
                  min(256, int((cx + w / 2) * 256)), min(256, int((cy + h / 2) * 256))] for cx, cy, w, h in it["boxes"]]
            out[f"{Path(it['tile_path']).stem}.png"] = np.array(b, float).reshape(-1, 4)
    return out


def load_preds(run_dir):
    """{ckpt_tag: {tile.png: (boxes, scores)}} and the set of tiles that have predictions."""
    f = run_dir / "raw_predictions_all_ckpts.csv.gz"
    if f.exists():
        d = pd.read_csv(f)
        tiles = None                                   # all tiles were predicted
    else:
        d = pd.concat([pd.read_csv(run_dir / f"raw_predictions_{s}.csv") for s in ("val", "test")
                       if (run_dir / f"raw_predictions_{s}.csv").exists()])
        d["ckpt"] = "final"
        tiles = set(d.image_path.unique())
    out = {tag: {i: (x[["xmin", "ymin", "xmax", "ymax"]].to_numpy(float), x.score.to_numpy(float))
                 for i, x in g.groupby("image_path")} for tag, g in d.groupby("ckpt")}
    return out, tiles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work"); ap.add_argument("runs")
    ap.add_argument("--val", default=str(C.TILES / "val.json")); ap.add_argument("--test", default=str(C.TILES / "test.json"))
    a = ap.parse_args()
    val, test = [f"{t}.png" for t in tile_names(a.val)], [f"{t}.png" for t in tile_names(a.test)]
    train = {f"{Path(x['tile_path']).stem}.png" for x in C.split_items("train")}
    bad = (set(val) | set(test)) & train
    if bad:
        sys.exit(f"refusing: {len(bad)} new val/test tiles were training tiles, e.g. {sorted(bad)[:3]}")
    if set(val) & set(test):
        sys.exit("refusing: val and test overlap")
    gt = all_gt()
    empty = (np.zeros((0, 4)), np.zeros(0))
    rows = []
    for run_dir in sorted(Path(a.work, a.runs).glob("seed*/*/")):
        if not (run_dir / "metrics.json").exists():
            continue
        arm, seed = json.loads((run_dir / "metrics.json").read_text())["arm"], run_dir.parent.name
        preds, have = load_preds(run_dir)
        if have is not None and not (set(val) | set(test)) <= have:
            print(f"skip {seed}/{arm}: no predictions for some new tiles (only old val/test tiles were predicted)")
            continue
        vg, tg = {t: gt[t] for t in val}, {t: gt[t] for t in test}
        best = None
        for tag, p in preds.items():
            pv = {t: p.get(t, empty) for t in val}
            for st in T.SCORES:
                for nt in NMSS:
                    f1 = T.score(pv, vg, st, nt, [0.4])["iou0.4"]["f1"]
                    if best is None or f1 > best[0]:
                        best = (f1, tag, st, nt)
        f1, tag, st, nt = best
        r = T.score({t: preds[tag].get(t, empty) for t in test}, tg, st, nt)
        rows.append({"seed": seed, "arm": arm, "ckpt": tag, "score": st, "nms": nt, "val_F1@0.4": f1,
                     **{f"F1@{i}": r[f"iou{i}"]["f1"] for i in T.IOUS}, "MAE": r["count_mae"],
                     "val_tiles": len(val), "test_tiles": len(test)})
    df = pd.DataFrame(rows)
    out = Path(a.work, f"revalidate_{a.runs}.csv"); df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print("\n| arm | F1@0.25 | F1@0.4 | F1@0.5 | MAE | seeds |\n|---|---|---|---|---|---|")
    for arm, g in df.groupby("arm"):
        print(f"| {arm} | " + " | ".join(f"{g[c].mean():.3f} ± {g[c].std(ddof=0):.3f}"
                                        for c in ("F1@0.25", "F1@0.4", "F1@0.5", "MAE")) + f" | {len(g)} |")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
