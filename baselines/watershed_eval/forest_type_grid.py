#!/usr/bin/env python3
"""
Watershed vs Dalponte 2016 vs Silva 2016 on the NeonTreeEvaluation plots, tuned
once for all forests ("global") and separately per forest type ("per-type"), on the
same 80/20 site-stratified split as watershed_grid.py. Selection is always on train
IoU-F1 only; val IoU scores and field-stem scores (neontree_stems.py) are reported.

Silva 2016 is ported from lidR's silva2016 (R/algorithm-its.R): every CHM pixel goes
to its nearest treetop; it is kept if  h >= exclusion x the cell's max height  and
distance to the treetop <= max_cr_factor x that height. (lidR's docs call the factor
a crown *diameter* fraction; the code applies it as a radius, as here.) Weinstein et
al. 2020 used 0.9 for deciduous and 0.2 for coniferous forests, on a 0.5 m CHM.
All three methods share the pipeline's treetops.

    python baselines/watershed_eval/forest_type_grid.py NEONTREE_DIR FIELD_RDA OUT_DIR [--seed 0]
"""
import argparse, itertools, json, sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import neontree_eval as E
import neontree_stems as S
import watershed_grid as G
import dalponte_grid as D

FOREST = {  # NEON site -> forest type
    "SJER": "oak_woodland", "CLBJ": "oak_woodland",
    "TEAK": "conifer", "SOAP": "conifer", "NIWO": "conifer", "WREF": "conifer", "ABBY": "conifer", "BONA": "conifer",
    "BART": "deciduous", "BLAN": "deciduous", "DELA": "deciduous", "HARV": "deciduous", "LENO": "deciduous",
    "MLBS": "deciduous", "SCBI": "deciduous", "SERC": "deciduous", "UNDE": "deciduous",
    "DSNY": "southern_pine", "JERC": "southern_pine", "OSBS": "southern_pine", "TALL": "southern_pine",
}
SILVA_GRID = {"smooth": ["median3", "gauss0.5"], "max_cr_factor": [0.1, 0.15, 0.2, 0.3, 0.45, 0.6, 0.9],
              "exclusion": [0.3, 0.45, 0.6]}
DALPONTE_GRID = {k: v for k, v in D.GRID.items() if k != "th_tree"}   # th_tree made no difference


def silva(sm, seed_img, max_cr_factor, exclusion):
    ys, xs = np.nonzero(seed_img)
    if not len(ys):
        return np.zeros(sm.shape, np.int32)
    ids = seed_img[ys, xs]
    yy, xx = np.indices(sm.shape)
    d, k = cKDTree(np.c_[ys, xs]).query(np.c_[yy.ravel(), xx.ravel()])
    lab = ids[k].reshape(sm.shape); d = d.reshape(sm.shape)
    hmax = np.zeros(ids.max() + 1); hmax[ids] = ndi.maximum(sm, lab, index=ids)
    h = hmax[lab]
    return np.where((sm >= exclusion * h) & (d <= max_cr_factor * h), lab, 0)


def boxes_from_labels(lab, sm, min_top=D.MIN_TOP):
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


def segment(method, chm, cfg):
    if method == "watershed":
        return G.segment(chm, cfg["smooth"], cfg["radius"], cfg["rel"], cfg["max_cr"], cfg["min_top"])
    if method == "dalponte":
        return D.segment(chm, cfg["smooth"], cfg["th_seed"], cfg["th_cr"], cfg["max_cr"], 2.0)
    sm = G.smooth(chm, cfg["smooth"])
    return boxes_from_labels(silva(sm, D.seeds(sm), cfg["max_cr_factor"], cfg["exclusion"]), sm)


def configs():
    for method, grid in (("watershed", G.GRID), ("dalponte", DALPONTE_GRID), ("silva", SILVA_GRID)):
        for vals in itertools.product(*grid.values()):
            yield method, dict(zip(grid, vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("neontree"); ap.add_argument("field"); ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    orig, cache = E.chm_max_in_boxes, {}
    E.chm_max_in_boxes = lambda c, r, g: cache.setdefault(c, orig(c, r, g))

    plots, meta = G.load_plots(Path(a.neontree))
    train, val = G.split(meta, a.seed)
    ftype = {p: FOREST.get(m["site"], "other") for p, m in meta.items()}
    types = sorted(set(ftype.values()) - {"other"})
    stems = {k: S.plot_stems(S.load_field(a.field, k), meta) for k in (1, 2)}
    labels = {p: E.read_gt(m["xml"]).tolist() for p, m in meta.items()}
    sub = lambda ps, t=None: [p for p in ps if t is None or ftype[p] == t]

    print(f"{len(plots)} plots, train {len(train)} / val {len(val)}; stem plots: "
          f"{len(stems[1])} (>=1 survey), {len(stems[2])} (>=2 surveys)")
    for t in types:
        print(f"  {t}: train {len(sub(train, t))} / val {len(sub(val, t))}, stem plots {len(sub(stems[1], t))}")

    rows = []
    for method, cfg in configs():
        preds, extra = {}, {}
        for p, (chm, ct, rt, H, W, px) in plots.items():
            preds[p], extra[p] = G.to_rgb_boxes(segment(method, chm, cfg), ct, rt, H, W, px)
        res = E.evaluate(meta, preds, extra)
        rows.append({"method": method, "cfg": cfg, "res": res, "preds": preds,
                     "train": {t: G.score(res, sub(train, t))["F1"] for t in [None] + types}})
        print(f"  {method} {cfg} train F1 {rows[-1]['train'][None]:.3f}", flush=True)

    def report(r, ps):
        s = G.score(r["res"], ps)
        return {"F1": s["F1"], "P": s["P"], "R": s["R"], "merged": round(s["labels"]["merged"], 3),
                "n_pred": s["n_pred"], "n_gt": s["n_gt"]}

    def stem_report(preds, ps):
        return {k: S.summarise(preds, stems[k], ps) for k in (1, 2)}

    pick = lambda cands, t: max(cands, key=lambda r: r["train"][t])
    methods = ["watershed", "dalponte", "silva", "any"]
    cand = {m: [r for r in rows if m == "any" or r["method"] == m] for m in methods}
    summary = {}
    for m in methods:
        g = pick(cand[m], None)
        per = {t: pick(cand[m], t) for t in types}
        # pooled val: every val plot scored with its own forest type's choice
        pooled_res = {p: (per[ftype[p]] if ftype[p] in per else g)["res"][p] for p in val}
        pooled_preds = {p: (per[ftype[p]] if ftype[p] in per else g)["preds"][p] for p in meta}
        s = {"global_cfg": [g["method"], g["cfg"]], "per_type_cfg": {t: [r["method"], r["cfg"]] for t, r in per.items()},
             "val_all": {"global": report(g, val), "per_type": {k: v for k, v in G.score(pooled_res, val).items() if k in ("F1", "P", "R")}},
             "by_type": {}, "by_site": {},
             "stems_all": {"global": stem_report(g["preds"], list(meta)), "per_type": stem_report(pooled_preds, list(meta))}}
        for t in types:
            vp = sub(val, t)
            s["by_type"][t] = {"val_global": report(g, vp) if vp else None, "val_per_type": report(per[t], vp) if vp else None,
                               "stems_global": stem_report(g["preds"], sub(list(meta), t)),
                               "stems_per_type": stem_report(per[t]["preds"], sub(list(meta), t))}
        for site in ("SJER", "TEAK"):
            vp = [p for p in val if meta[p]["site"] == site]
            s["by_site"][site] = {"global": report(g, vp), "per_type": report(per[FOREST[site]], vp)}
        summary[m] = s
    summary["human_labels_stems"] = {"all": stem_report(labels, list(meta)),
                                     **{t: stem_report(labels, sub(list(meta), t)) for t in types}}
    summary["split"] = {t: [len(sub(train, t)), len(sub(val, t))] for t in types}
    (out / "forest_type_results.json").write_text(json.dumps(summary, indent=1, default=float))

    f = lambda r: f"F1 {r['F1']:.3f}" if r else "   -    "
    st = lambda r: (f"stem R {r[1]['recall_1to1']:.2f} merged {r[1]['merged']:.2f}" if r and r[1] else "no stems")
    for m in methods:
        s = summary[m]
        print(f"\n===== {m} =====\nglobal choice: {s['global_cfg']}")
        print(f"val all plots: global {f(s['val_all']['global'])}  per-type {f(s['val_all']['per_type'])}")
        for t in types:
            b = s["by_type"][t]
            print(f"  {t:14s} ({s['per_type_cfg'][t][0]} {s['per_type_cfg'][t][1]})\n"
                  f"     val: global {f(b['val_global'])} -> per-type {f(b['val_per_type'])} | "
                  f"stems: global {st(b['stems_global'])} -> per-type {st(b['stems_per_type'])}")
        for site, b in s["by_site"].items():
            print(f"  val {site}: global {f(b['global'])} -> per-type {f(b['per_type'])}")
        print(f"  stems all plots: global {st(s['stems_all']['global'])} -> per-type {st(s['stems_all']['per_type'])}")
    h = summary["human_labels_stems"]
    print("\nhuman labels vs stems: all " + st(h["all"]) + "; " + "; ".join(f"{t} {st(h[t])}" for t in types))
    print(f"wrote {out/'forest_type_results.json'}")


if __name__ == "__main__":
    main()
