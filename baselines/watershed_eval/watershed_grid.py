#!/usr/bin/env python3
"""
Small grid search over the watershed's parameters on the NeonTreeEvaluation
evaluation plots, with a site-stratified 80/20 split so the chosen parameters are
scored on plots they were not tuned on.

Parameters (current pipeline value in brackets):
  smooth     CHM smoothing before treetops + watershed: none | median3 | gauss0.5 | gauss1  [median3]
  radius     treetop window radius (px = m) for trees under 32 m: 1 | 1.5 | 2               [2]
             (the pipeline's "4 m diameter" becomes radius round(4/2) = 2; 3 m would also give 2)
  rel        trim each crown to pixels >= rel x its top height; 0 = keep the 3 m floor only [0]
             (Dalponte 2016's th_seed: 0.45 = lidR default / Dixon 2023, 0.65 = Hemming-Schroeder 2023)
  max_cr     cap crown diameter (m): keep pixels within max_cr/2 of the treetop; 0 = no cap [0]
             (Dalponte 2016's max_cr, lidR default 10 px = 10 m at 1 m)
  min_top    minimum treetop height, m: 3 | 5                                                [5]

Selection: highest F1 of the per-image-averaged precision and recall (IoU >= 0.4,
neontree_eval's protocol) on the 80% split. The 20% split is scored once, at the end.

    python baselines/watershed_eval/watershed_grid.py NEONTREE_DIR OUT_DIR [--seed 0]
"""
import argparse, itertools, json, sys
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage as ndi
from skimage.measure import regionprops
from skimage.segmentation import watershed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import neontree_eval as E
import neontree_watershed as WS

GRID = {
    "smooth": ["none", "median3", "gauss0.5", "gauss1"],
    "radius": [1.0, 1.5, 2.0],
    "rel": [0.0, 0.45, 0.65],
    "max_cr": [0.0, 10.0],
    "min_top": [3.0, 5.0],
}
BASELINE = {"smooth": "median3", "radius": 2.0, "rel": 0.0, "max_cr": 0.0, "min_top": 5.0}


def disk(r):
    n = int(np.ceil(r))
    y, x = np.ogrid[-n:n + 1, -n:n + 1]
    return x * x + y * y <= r * r


def smooth(chm, how):
    if how == "none":
        return chm
    if how == "median3":
        return ndi.median_filter(chm, size=3)
    return ndi.gaussian_filter(chm, sigma=float(how[5:]))


def segment(chm, smooth_how, radius, rel, max_cr, min_top):
    """crowns_from_chm with the four knobs exposed; BASELINE reproduces it exactly."""
    sm = smooth(chm, smooth_how)
    ws = np.where(sm < 32, 4.0, np.where(sm <= 80, sm / 8.0, 10.0))
    d = np.round(ws)
    rad = np.where(sm < 32, radius, np.maximum(1, np.round(d / 2)))
    tops = np.zeros(sm.shape, bool)
    for r in np.unique(rad):
        mx = ndi.maximum_filter(sm, footprint=disk(r), mode="nearest")
        tops |= (rad == r) & (sm >= mx) & (sm > min_top)
    markers, n = ndi.label(tops, structure=np.ones((3, 3)))
    if n == 0:
        return []
    seg = watershed(-sm, markers, mask=sm > 3)
    out = []
    for reg in regionprops(seg, intensity_image=sm):
        top = float(reg.max_intensity)
        if reg.area < 1 or top <= min_top:
            continue
        if rel > 0 or max_cr > 0:
            c = reg.coords; h = sm[c[:, 0], c[:, 1]]
            keep = np.ones(len(c), bool)
            if rel > 0:
                keep &= h >= rel * top
            if max_cr > 0:
                t = c[np.argmax(h)]
                keep &= np.hypot(c[:, 0] - t[0], c[:, 1] - t[1]) <= max_cr / 2
            c = c[keep]
            bbox = (c[:, 0].min(), c[:, 1].min(), c[:, 0].max() + 1, c[:, 1].max() + 1)
        else:
            bbox = reg.bbox
        out.append((bbox, top))
    return out


def to_rgb_boxes(crowns, ct, rt, H, W, px):
    """Same CHM-pixel -> RGB-pixel conversion and filtering as neontree_watershed.main."""
    edge_px = WS.EDGE_M / px; inv = ~rt; boxes, extra = [], []
    for (y0, x0, y1, x1), top in crowns:
        u0, v0 = inv * (ct * (x0, y0)); u1, v1 = inv * (ct * (x1, y1))
        xa, xb = sorted((u0, u1)); ya, yb = sorted((v0, v1))
        cx, cy = (xa + xb) / 2, (ya + yb) / 2
        if not (0 <= cx < W and 0 <= cy < H):
            continue
        xa, ya, xb, yb = max(xa, 0), max(ya, 0), min(xb, W), min(yb, H)
        edge = xa < edge_px or ya < edge_px or xb > W - edge_px or yb > H - edge_px
        boxes.append([round(xa, 1), round(ya, 1), round(xb, 1), round(yb, 1)])
        extra.append((round(top, 1), int(edge)))
    return boxes, extra


def load_plots(root):
    plots, meta = {}, {}
    for plot, tier, xml_p, rgb_p, chm_p in WS.find_plots(root, False):
        chm, ct, ccrs = WS.read_chm(chm_p)
        with rasterio.open(rgb_p) as r:
            rt, rcrs, H, W = r.transform, r.crs, r.height, r.width
        if ccrs != rcrs:
            continue
        cb, rb = rasterio.transform.array_bounds(*chm.shape, ct), rasterio.transform.array_bounds(H, W, rt)
        ov = max(0, min(cb[2], rb[2]) - max(cb[0], rb[0])) * max(0, min(cb[3], rb[3]) - max(cb[1], rb[1]))
        if ov < 0.9 * (rb[2] - rb[0]) * (rb[3] - rb[1]):
            continue
        px = abs(rt.a)
        plots[plot] = (chm, ct, rt, H, W, px)
        meta[plot] = {"tier": tier, "site": WS.site_of(plot), "xml": str(xml_p), "rgb": str(rgb_p),
                      "chm": str(chm_p), "rgb_px_m": px, "rgb_shape": [H, W]}
    return plots, meta


def split(meta, seed):
    rng = np.random.default_rng(seed); train, val = [], []
    by_site = {}
    for p, m in sorted(meta.items()):
        by_site.setdefault(m["site"], []).append(p)
    for site, ps in sorted(by_site.items()):
        ps = list(rng.permutation(ps))
        k = int(round(0.2 * len(ps))) if len(ps) >= 3 else 0
        val += ps[:k]; train += ps[k:]
    return sorted(train), sorted(val)


def score(res, plots_subset):
    s = E.summarise([res[p] for p in plots_subset])
    P, R = s["precision_mean_per_image"] or 0, s["recall_mean_per_image"] or 0
    pp, rp = s["precision_pooled"], s["recall_pooled"]
    f = lambda a, b: round(2 * a * b / max(a + b, 1e-9), 4)
    return {"P": P, "R": R, "F1": f(P, R), "P_pooled": pp, "R_pooled": rp, "F1_pooled": f(pp, rp),
            "labels": s["labels"], "predictions": s["predictions"],
            "area_ratio_p50": s["matched_localisation_p25_p50_p75"]["area_ratio"][1]
            if s["matched_localisation_p25_p50_p75"]["area_ratio"] else None,
            "n_pred": int(sum(res[p]["n_pred"] for p in plots_subset)),
            "n_gt": int(sum(res[p]["n_gt"] for p in plots_subset))}


def run(cfg, plots, meta):
    preds, extra = {}, {}
    for p, (chm, ct, rt, H, W, px) in plots.items():
        preds[p], extra[p] = to_rgb_boxes(segment(chm, cfg["smooth"], cfg["radius"], cfg["rel"], cfg["max_cr"], cfg["min_top"]),
                                          ct, rt, H, W, px)
    return E.evaluate(meta, preds, extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("neontree"); ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    # ground-truth heights don't depend on the config: compute them once per plot
    orig, cache = E.chm_max_in_boxes, {}
    E.chm_max_in_boxes = lambda c, r, g: cache.setdefault(c, orig(c, r, g))

    plots, meta = load_plots(Path(a.neontree))
    train, val = split(meta, a.seed)
    sites = sorted({m["site"] for m in meta.values()})
    print(f"{len(plots)} plots, {len(sites)} sites -> train {len(train)} / val {len(val)} "
          f"(val sites: {len({meta[p]['site'] for p in val})})")

    base_all = score(run(BASELINE, plots, meta), list(plots))
    print(f"baseline on all plots: P {base_all['P']:.2f} / R {base_all['R']:.2f} per image, "
          f"{base_all['n_pred']} crowns  (So's write-up: 0.40 / 0.37, 3,104 crowns)")

    keys = list(GRID); rows = []
    for vals in itertools.product(*GRID.values()):
        cfg = dict(zip(keys, vals))
        res = run(cfg, plots, meta)
        rows.append({"cfg": cfg, "train": score(res, train), "_res": res})
        print(f"  {cfg}  train F1 {rows[-1]['train']['F1']:.3f}", flush=True)
    rows.sort(key=lambda r: -r["train"]["F1"])
    for r in rows:                       # val is computed for reporting only, after selection
        r["val"] = score(r["_res"], val)
    best = rows[0]
    base = next(r for r in rows if r["cfg"] == BASELINE)

    def site_scores(r):
        o = {}
        for s in ("SJER", "SOAP", "TEAK"):
            ps = [p for p in val if meta[p]["site"] == s]
            if ps:
                o[s] = score(r["_res"], ps)
        return o

    report = {"seed": a.seed, "n_train": len(train), "n_val": len(val), "val_plots": val,
              "baseline": {k: base[k] for k in ("cfg", "train", "val")} | {"val_ca": site_scores(base)},
              "best": {k: best[k] for k in ("cfg", "train", "val")} | {"val_ca": site_scores(best)},
              "all": [{k: r[k] for k in ("cfg", "train", "val")} for r in rows]}
    (out / "grid_results.json").write_text(json.dumps(report, indent=1, default=float))

    f = lambda s: (f"P {s['P']:.2f} R {s['R']:.2f} F1 {s['F1']:.3f} | pooled F1 {s['F1_pooled']:.3f} | "
                   f"match {s['labels']['match']:.0%} merged {s['labels']['merged']:.0%} "
                   f"split {s['labels']['over_seg']:.0%} | FP {s['predictions']['false_pos']:.0%} | "
                   f"area x{s['area_ratio_p50']}")
    print("\n== top 10 by train F1 (val shown for stability, not used to pick) ==")
    for r in rows[:10]:
        print(f"  {r['cfg']}  train {r['train']['F1']:.3f}  val {r['val']['F1']:.3f}")
    print(f"\nbaseline {BASELINE}\n  train: {f(base['train'])}\n  val:   {f(base['val'])}")
    print(f"best     {best['cfg']}\n  train: {f(best['train'])}\n  val:   {f(best['val'])}")
    for s, v in site_scores(best).items():
        bv = site_scores(base).get(s)
        print(f"  val {s} ({v['n_gt']} labels): baseline F1 {bv['F1']:.3f} -> best F1 {v['F1']:.3f}")
    print(f"\nwrote {out/'grid_results.json'}")


if __name__ == "__main__":
    main()
