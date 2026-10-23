#!/usr/bin/env python3
"""
Re-score saved predictions on pre-fire (2020 NAIP) vs post-fire (2022 NAIP) test tiles.
LiDAR pretraining uses pre-fire imagery only, so pre-fire tiles are where it should help.

Two threshold choices, both from val only:
  all_val      the score/NMS already selected on all 18 val tiles (as in metrics.json)
  period_val   re-selected on the val tiles of the same period (9 pre-fire, 9 post-fire)

    python lidar_pretrain/rescore_by_period.py WORK [RUNS_DIR ...]   # WORK/RUNS_DIR/seed*/{A,B,C}/raw_predictions_*.csv
Several RUNS_DIRs are pooled (e.g. `runs runs_full`, to compare B/C from the full set against A).
"""
import json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import train as T


def raw_from_csv(path, images):
    d = pd.read_csv(path)
    out = {img: (np.zeros((0, 4)), np.zeros(0)) for img in images}
    for img, g in d.groupby("image_path"):
        out[img] = (g[["xmin", "ymin", "xmax", "ymax"]].to_numpy(float), g["score"].to_numpy(float))
    return out


def main(work, *runs_dirs):
    work = Path(work); lab = work / "labelled_rescore"; T.prep_labelled(lab)
    runs_dirs = runs_dirs or ("runs",)
    gts = {s: T.gt_by_img(lab / f"{s}.csv") for s in ("val", "test")}
    imgs = {s: sorted(pd.read_csv(lab / f"{s}.csv").image_path.unique()) for s in ("val", "test")}
    period = lambda img: "pre" if img.startswith("prefire") else "post"
    sub = lambda g, p: {k: v for k, v in g.items() if period(k) == p}

    rows = []
    files = [(rd, mf) for rd in runs_dirs for mf in sorted(work.glob(f"{rd}/seed*/*/metrics.json"))]
    for rd, mf in files:
        m = json.loads(mf.read_text()); seed = mf.parent.parent.name
        arm = m["arm"] if rd == runs_dirs[0] or len(runs_dirs) == 1 else f"{m['arm']}[{rd}]"
        raw = {s: raw_from_csv(mf.parent / f"raw_predictions_{s}.csv", imgs[s]) for s in ("val", "test")}
        sel = m["metrics"]["selected_on_val"]
        for p in ("pre", "post"):
            vg, tg = sub(gts["val"], p), sub(gts["test"], p)
            grid = [(st, nt, T.score(raw["val"], vg, st, nt, [0.4])["iou0.4"]["f1"]) for st in T.SCORES for nt in T.NMSS]
            pst, pnt, _ = max(grid, key=lambda g: g[2])
            for how, (st, nt) in (("all_val", (sel["score"], sel["nms"])), ("period_val", (pst, pnt))):
                r = T.score(raw["test"], tg, st, nt)
                rows.append({"seed": seed, "arm": arm, "runs": rd, "period": p, "thresholds": how, "score": st, "nms": nt,
                             "tiles": r["n_tiles"], "crowns": int(sum(len(v) for v in tg.values())),
                             **{f"F1@{i}": r[f"iou{i}"]["f1"] for i in T.IOUS}, "MAE": r["count_mae"]})
    df = pd.DataFrame(rows); df.to_csv(work / f"rescore_by_period_{'_'.join(runs_dirs)}.csv", index=False)

    for how in ("all_val", "period_val"):
        print(f"\n## thresholds: {how}")
        for p in ("pre", "post"):
            d = df[(df.thresholds == how) & (df.period == p)]
            print(f"\n{p}-fire test: {d.tiles.iloc[0]} tiles, {d.crowns.iloc[0]} crowns")
            print("| arm | F1@0.25 | F1@0.4 | F1@0.5 | MAE |\n|---|---|---|---|---|")
            for arm, g in d.groupby("arm"):
                print(f"| {arm} | " + " | ".join(f"{g[c].mean():.3f} ± {g[c].std(ddof=0):.3f}"
                                                for c in ("F1@0.25", "F1@0.4", "F1@0.5", "MAE")) + " |")
            a = d[d.arm == "A"].set_index("seed")
            for barm in sorted(x for x in d.arm.unique() if x.startswith("B")):
                b = d[d.arm == barm].set_index("seed")
                diff = (b[["F1@0.25", "F1@0.4", "F1@0.5", "MAE"]] - a[["F1@0.25", "F1@0.4", "F1@0.5", "MAE"]]).dropna()
                print(f"{barm} − A per seed: " + "; ".join(f"{s}: " + ", ".join(f"{c} {v:+.3f}" for c, v in r.items())
                                                         for s, r in diff.iterrows()))
    print(f"\n-> {work/('rescore_by_period_' + '_'.join(runs_dirs) + '.csv')}")


if __name__ == "__main__":
    main(*sys.argv[1:])
