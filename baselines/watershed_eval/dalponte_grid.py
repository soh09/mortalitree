#!/usr/bin/env python3
"""
Dalponte & Coomes (2016) region growing, ported line-for-line from lidR's
C_dalponte2016 (src/C_dalponte2016.cpp), in place of the marker watershed, with the
same 80/20 site-stratified split and selection rule as watershed_grid.py.

Growth rule (lidR): passes over the image; each labelled pixel offers its 4
neighbours to its crown, which takes a neighbour if
    h > th_tree,  h > th_seed x seed height,  h > th_cr x crown's running mean height,
    h <= 1.05 x seed height,  |dx| and |dy| to the seed < max_cr (px),
    and the neighbour was unlabelled at the start of the pass.
Writes go to a copy (last writer in scan order wins); the running sums update
immediately; border pixels never grow. Note max_cr is a square half-width, so
lidR's default 10 px allows crowns up to ~19 m across at 1 m.

Seeds are the pipeline's treetops (variable-window local maxima on the smoothed CHM,
one seed pixel per maximum). Crowns with top <= min_top are dropped, as before.

The plot CHMs are only 40 x 40 px and lidR never grows from border pixels, so a seed
on the plot edge (common: trees cut by the boundary peak there) stays one pixel. To
match lidR on a full-size raster, the CHM is edge-padded by 1 px before growing and
the result cropped back.

    python baselines/watershed_eval/dalponte_grid.py NEONTREE_DIR OUT_DIR [--seed 0]
"""
import argparse, itertools, json, sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import neontree_eval as E
import watershed_grid as G

GRID = {
    "smooth": ["median3", "gauss0.5"],
    "th_seed": [0.45, 0.55, 0.65],
    "th_cr": [0.45, 0.55, 0.65, 0.75],
    "max_cr": [4, 6, 8, 10],
    "th_tree": [2.0, 3.0],            # lidR default 2; the watershed's mask is > 3
}
LIDR_DEFAULT = {"smooth": "median3", "th_seed": 0.45, "th_cr": 0.55, "max_cr": 10, "th_tree": 2.0}
RADIUS, MIN_TOP = 2.0, 5.0


def seeds(sm, radius=RADIUS, min_top=MIN_TOP):
    """Treetops exactly as watershed_grid.segment finds them, reduced to one pixel each."""
    ws = np.where(sm < 32, 4.0, np.where(sm <= 80, sm / 8.0, 10.0))
    rad = np.where(sm < 32, radius, np.maximum(1, np.round(np.round(ws) / 2)))
    tops = np.zeros(sm.shape, bool)
    for r in np.unique(rad):
        mx = ndi.maximum_filter(sm, footprint=G.disk(r), mode="nearest")
        tops |= (rad == r) & (sm >= mx) & (sm > min_top)
    lab, n = ndi.label(tops, structure=np.ones((3, 3)))
    out = np.zeros(sm.shape, np.int32)
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        ys, xs = np.nonzero(lab[sl] == i)
        out[ys[0] + sl[0].start, xs[0] + sl[1].start] = i
    return out


def dalponte(img, seed_img, th_seed, th_cr, th_tree, dist):
    """lidR C_dalponte2016, same loop order and update semantics."""
    nrow, ncol = img.shape
    I = img.tolist(); region = seed_img.tolist(); temp = [row[:] for row in region]
    seed, s_h, n_px = {}, {}, {}
    for i in range(nrow):
        for j in range(ncol):
            v = region[i][j]
            if v:
                seed[v] = (i, j); s_h[v] = I[i][j]; n_px[v] = 1
    grown = True
    while grown:
        grown = False
        for r in range(1, nrow - 1):
            for k in range(1, ncol - 1):
                cid = region[r][k]
                if not cid:
                    continue
                sx, sy = seed[cid]; hs = I[sx][sy]; mh = s_h[cid] / n_px[cid]
                for x, y in ((r - 1, k), (r, k - 1), (r, k + 1), (r + 1, k)):
                    z = I[x][y]
                    if (z > th_tree and z > hs * th_seed and z > mh * th_cr and z <= hs * 1.05
                            and abs(sx - x) < dist and abs(sy - y) < dist and region[x][y] == 0):
                        temp[x][y] = cid; n_px[cid] += 1; s_h[cid] += I[x][y]; grown = True
        region = [row[:] for row in temp]
    return np.array(region, np.int32)


def segment(chm, smooth_how, th_seed, th_cr, max_cr, th_tree, min_top=MIN_TOP):
    sm = G.smooth(chm, smooth_how)
    lab = dalponte(np.pad(sm, 1, mode="edge"), np.pad(seeds(sm, min_top=min_top), 1),
                   th_seed, th_cr, th_tree, max_cr)[1:-1, 1:-1]
    out = []
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None:
            continue
        m = lab[sl] == i
        top = float(sm[sl][m].max())
        if top <= min_top:
            continue
        ys, xs = np.nonzero(m)
        out.append(((ys.min() + sl[0].start, xs.min() + sl[1].start,
                     ys.max() + sl[0].start + 1, xs.max() + sl[1].start + 1), top))
    return out


def run(cfg, plots, meta):
    preds, extra = {}, {}
    for p, (chm, ct, rt, H, W, px) in plots.items():
        preds[p], extra[p] = G.to_rgb_boxes(segment(chm, cfg["smooth"], cfg["th_seed"], cfg["th_cr"], cfg["max_cr"], cfg["th_tree"]),
                                            ct, rt, H, W, px)
    return E.evaluate(meta, preds, extra), preds, extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("neontree"); ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--write-best", help="also write the selected config's crowns here, in neontree_watershed's format")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    orig, cache = E.chm_max_in_boxes, {}
    E.chm_max_in_boxes = lambda c, r, g: cache.setdefault(c, orig(c, r, g))

    plots, meta = G.load_plots(Path(a.neontree))
    train, val = G.split(meta, a.seed)
    print(f"{len(plots)} plots -> train {len(train)} / val {len(val)}")

    refs = {"watershed_current": G.BASELINE,
            "watershed_tuned": {"smooth": "gauss0.5", "radius": 2.0, "rel": 0.65, "max_cr": 10.0, "min_top": 5.0}}
    ref_rows = {}
    for name, cfg in refs.items():
        res = G.run(cfg, plots, meta)
        ref_rows[name] = {"cfg": cfg, "train": G.score(res, train), "val": G.score(res, val)}

    keys = list(GRID); rows = []
    for vals in itertools.product(*GRID.values()):
        cfg = dict(zip(keys, vals))
        res, preds, extra = run(cfg, plots, meta)
        rows.append({"cfg": cfg, "train": G.score(res, train), "_res": res, "_pe": (preds, extra)})
        print(f"  {cfg}  train F1 {rows[-1]['train']['F1']:.3f}", flush=True)
    rows.sort(key=lambda r: -r["train"]["F1"])
    for r in rows:
        r["val"] = G.score(r["_res"], val)
    best = rows[0]
    lidr = next(r for r in rows if r["cfg"] == LIDR_DEFAULT)

    def site_scores(res):
        return {s: G.score(res, ps) for s in ("SJER", "SOAP", "TEAK")
                if (ps := [p for p in val if meta[p]["site"] == s])}

    report = {"seed": a.seed, "refs": ref_rows,
              "lidr_default": {k: lidr[k] for k in ("cfg", "train", "val")} | {"val_ca": site_scores(lidr["_res"])},
              "best": {k: best[k] for k in ("cfg", "train", "val")} | {"val_ca": site_scores(best["_res"])},
              "all": [{k: r[k] for k in ("cfg", "train", "val")} for r in rows]}
    (out / "dalponte_results.json").write_text(json.dumps(report, indent=1, default=float))

    f = lambda s: (f"P {s['P']:.2f} R {s['R']:.2f} F1 {s['F1']:.3f} | match {s['labels']['match']:.0%} "
                   f"merged {s['labels']['merged']:.0%} split {s['labels']['over_seg']:.0%} | "
                   f"FP {s['predictions']['false_pos']:.0%} | area x{s['area_ratio_p50']} | {s['n_pred']} crowns")
    print("\n== top 10 by train F1 ==")
    for r in rows[:10]:
        print(f"  {r['cfg']}  train {r['train']['F1']:.3f}  val {r['val']['F1']:.3f}")
    for name, r in list(ref_rows.items()) + [("dalponte_lidr_default", lidr), ("dalponte_best", best)]:
        print(f"\n{name} {r['cfg']}\n  train: {f(r['train'])}\n  val:   {f(r['val'])}")
    print("\nval CA sites (F1): " + "; ".join(
        f"{s} lidR-default {v['F1']:.3f} best {site_scores(best['_res'])[s]['F1']:.3f}"
        for s, v in site_scores(lidr["_res"]).items()))
    vr = np.array([[r["train"]["F1"], r["val"]["F1"]] for r in rows])
    print(f"train-val F1 correlation over grid: {np.corrcoef(vr.T)[0, 1]:.2f}; "
          f"best's val rank: {1 + sorted(vr[:, 1], reverse=True).index(best['val']['F1'])}/{len(rows)}")

    if a.write_best:
        import csv
        d = Path(a.write_best); d.mkdir(parents=True, exist_ok=True)
        preds, extra = best["_pe"]
        with open(d / "predictions_5m.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["plot", "tier", "site", "xmin", "ymin", "xmax", "ymax", "height_m", "area_m2", "edge"])
            for p, bs in preds.items():
                for b, (h, e) in zip(bs, extra[p]):
                    w.writerow([p, meta[p]["tier"], meta[p]["site"], *b, h, -1, e])
        (d / "plots.json").write_text(json.dumps(meta, indent=1))
        (d / "config.json").write_text(json.dumps(best["cfg"]))
        print(f"wrote {d}")


if __name__ == "__main__":
    main()
