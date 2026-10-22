#!/usr/bin/env python3
"""
Score LiDAR watershed crowns (neontree_watershed.py) against NeonTreeEvaluation human
crown boxes, and break the disagreement down by error type.

  1. Benchmark score: one-to-one matching at IoU >= 0.4 (Hungarian on IoU); precision and
     recall per image, then averaged -- the protocol of Weinstein et al. (2021).
  2. Error taxonomy. Every label and every prediction gets exactly one outcome, assigned
     in this order ("inside" = at least half of the box's area lies within the other):
       label side:  match | over_seg (>= 2 predictions inside it) | merged (it and >= 1
                    other label inside one prediction) | poor (best IoU 0.1-0.4) |
                    miss_<3m / miss_3-5m / miss_>5m (no partner; CHM max inside the box)
       pred side:   match | fragment (inside a label that holds >= 2 predictions) |
                    merge (holds >= 2 labels) | poor | false_pos
  3. Localisation of matches: centre offset east/north (m), area/width/height ratios.

Writes OUT_DIR/metrics_{tag}.json and, with --figures, OUT_DIR/fig_{site}_{tag}.png.

    uv run python baselines/watershed_eval/neontree_eval.py PRED_DIR [--min-top 5] [--figures]
    uv run python baselines/watershed_eval/neontree_eval.py PRED_DIR --sanity
"""
import argparse, csv, json, sys, xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
from neontree_watershed import EDGE_M, read_chm

IOU_MATCH, IOU_POOR, INSIDE = 0.4, 0.1, 0.5
SIZE_BINS = [0, 10, 25, 50, 100, np.inf]          # label crown area, m^2
CA_SITES = ("SJER", "SOAP", "TEAK")
GT_CLASSES = ["match", "over_seg", "merged", "poor", "miss_<3m", "miss_3-5m", "miss_>5m"]
PRED_CLASSES = ["match", "fragment", "merge", "poor", "false_pos"]


def read_gt(xml):
    boxes = [[float(o.findtext(f"bndbox/{k}")) for k in ("xmin", "ymin", "xmax", "ymax")]
             for o in ET.parse(xml).findall("object")]
    return np.array(boxes, float).reshape(-1, 4)


def area(b):
    return np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)


def overlaps(p, g):
    """(intersection, IoU), each shape (len(p), len(g))."""
    ix = np.clip(np.minimum(p[:, None, 2], g[None, :, 2]) - np.maximum(p[:, None, 0], g[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(p[:, None, 3], g[None, :, 3]) - np.maximum(p[:, None, 1], g[None, :, 1]), 0, None)
    inter = ix * iy
    union = area(p)[:, None] + area(g)[None, :] - inter
    return inter, np.where(union > 0, inter / np.maximum(union, 1e-9), 0)


def classify(p, g, gt_height):
    """Label every prediction and label box with one outcome; return labels + matched pairs."""
    M, N = len(p), len(g)
    pl, gl = np.array([""] * M, object), np.array([""] * N, object)
    inter, iou = overlaps(p, g)
    pairs = []
    if M and N:
        r, c = linear_sum_assignment(-iou)
        for i, j in zip(r, c):
            if iou[i, j] >= IOU_MATCH:
                pl[i] = gl[j] = "match"; pairs.append((i, j))
        p_in_g = inter / np.maximum(area(p), 1e-9)[:, None]       # share of pred inside label
        g_in_p = inter / np.maximum(area(g), 1e-9)[None, :]       # share of label inside pred
        n_in_g = (p_in_g >= INSIDE).sum(0)                        # preds inside each label
        n_in_p = (g_in_p >= INSIDE).sum(1)                        # labels inside each pred
        # over-segmentation: one label split into >= 2 predictions
        gl[(gl == "") & (n_in_g >= 2)] = "over_seg"
        for i in np.where(pl == "")[0]:
            js = np.where((p_in_g[i] >= INSIDE) & (n_in_g >= 2))[0]
            if len(js):
                pl[i] = "fragment"
        # under-segmentation: >= 2 labels merged into one prediction
        pl[(pl == "") & (n_in_p >= 2)] = "merge"
        for j in np.where(gl == "")[0]:
            if ((g_in_p[:, j] >= INSIDE) & (n_in_p >= 2)).any():
                gl[j] = "merged"
        # poor delineation: some overlap, not enough to match
        gl[(gl == "") & (iou.max(0) >= IOU_POOR)] = "poor"
        pl[(pl == "") & (iou.max(1) >= IOU_POOR)] = "poor"
    pl[pl == ""] = "false_pos"
    for j in np.where(gl == "")[0]:
        h = gt_height[j]
        gl[j] = "miss_<3m" if h < 3 else "miss_3-5m" if h <= 5 else "miss_>5m"
    return pl, gl, pairs


def chm_max_in_boxes(chm_path, rgb_path, g):
    """Max CHM height inside each label box (boxes in RGB pixels)."""
    chm, ct, _ = read_chm(chm_path)
    with rasterio.open(rgb_path) as r:
        rt = r.transform
    inv = ~ct
    out = np.zeros(len(g))
    for k, (x0, y0, x1, y1) in enumerate(g):
        c0, r0 = inv * (rt * (x0, y0)); c1, r1 = inv * (rt * (x1, y1))
        ca, cb = sorted((c0, c1)); ra, rb = sorted((r0, r1))
        ca, ra = max(int(np.floor(ca)), 0), max(int(np.floor(ra)), 0)
        cb, rb = min(int(np.ceil(cb)), chm.shape[1]), min(int(np.ceil(rb)), chm.shape[0])
        if cb > ca and rb > ra:
            out[k] = chm[ra:rb, ca:cb].max()
    return out


def load(pred_dir, tag):
    meta = json.loads((pred_dir / "plots.json").read_text())
    preds = defaultdict(list); extra = defaultdict(list)
    with open(pred_dir / f"predictions_{tag}.csv") as f:
        for row in csv.DictReader(f):
            preds[row["plot"]].append([float(row[k]) for k in ("xmin", "ymin", "xmax", "ymax")])
            extra[row["plot"]].append((float(row["height_m"]), int(row["edge"])))
    return meta, preds, extra


def evaluate(meta, preds, extra, gt_override=None):
    """Per-plot results: outcome labels, P/R, and per-box attributes for breakdowns."""
    res = {}
    for plot, m in meta.items():
        g = gt_override[plot] if gt_override else read_gt(m["xml"])
        p = np.array(preds.get(plot, []), float).reshape(-1, 4)
        px = m["rgb_px_m"]; H, W = m["rgb_shape"]
        gh = chm_max_in_boxes(m["chm"], m["rgb"], g) if len(g) else np.zeros(0)
        pl, gl, pairs = classify(p, g, gh)
        tp = len(pairs)
        e = EDGE_M / px
        g_edge = (g[:, 0] < e) | (g[:, 1] < e) | (g[:, 2] > W - e) | (g[:, 3] > H - e)
        loc = []
        for i, j in pairs:
            pc, gc = (p[i, :2] + p[i, 2:]) / 2, (g[j, :2] + g[j, 2:]) / 2
            pw, ph = p[i, 2:] - p[i, :2]; gw, gh_ = g[j, 2:] - g[j, :2]
            loc.append([(pc[0] - gc[0]) * px, -(pc[1] - gc[1]) * px,
                        pw * ph / max(gw * gh_, 1e-9), pw / max(gw, 1e-9), ph / max(gh_, 1e-9)])
        ex = extra.get(plot, [])
        res[plot] = {
            "site": m["site"], "tier": m["tier"], "n_gt": len(g), "n_pred": len(p), "tp": tp,
            "precision": tp / len(p) if len(p) else np.nan,
            "recall": tp / len(g) if len(g) else np.nan,
            "gl": gl, "pl": pl, "g_area_m2": area(g) * px * px, "g_edge": g_edge,
            "p_height": np.array([h for h, _ in ex]) if ex else np.zeros(len(p)),
            "p_edge": np.array([bool(x) for _, x in ex]) if ex else np.zeros(len(p), bool),
            "loc": np.array(loc).reshape(-1, 5),
        }
    return res


def shares(labels, classes):
    n = len(labels)
    return {c: round(float((labels == c).sum()) / n, 3) if n else None for c in classes} | {"n": n}


def summarise(rs):
    """Aggregate a list of per-plot results."""
    if not rs:
        return None
    gl = np.concatenate([r["gl"] for r in rs]); pl = np.concatenate([r["pl"] for r in rs])
    loc = np.concatenate([r["loc"] for r in rs])
    q = lambda a: [round(float(x), 2) for x in np.percentile(a, [25, 50, 75])] if len(a) else None
    ph = np.concatenate([r["p_height"] for r in rs]); fp = ph[pl == "false_pos"]
    return {
        "images": len(rs),
        "precision_mean_per_image": round(float(np.nanmean([r["precision"] for r in rs])), 3)
        if any(r["n_pred"] for r in rs) else None,
        "recall_mean_per_image": round(float(np.nanmean([r["recall"] for r in rs])), 3)
        if any(r["n_gt"] for r in rs) else None,
        "precision_pooled": round(sum(r["tp"] for r in rs) / max(1, sum(r["n_pred"] for r in rs)), 3),
        "recall_pooled": round(sum(r["tp"] for r in rs) / max(1, sum(r["n_gt"] for r in rs)), 3),
        "labels": shares(gl, GT_CLASSES),
        "predictions": shares(pl, PRED_CLASSES),
        "matched_localisation_p25_p50_p75": {
            "offset_east_m": q(loc[:, 0]), "offset_north_m": q(loc[:, 1]),
            "offset_dist_m": q(np.hypot(loc[:, 0], loc[:, 1])),
            "area_ratio": q(loc[:, 2]), "width_ratio": q(loc[:, 3]), "height_ratio": q(loc[:, 4]),
        },
        "false_pos_height_m_p25_p50_p75": q(fp),
    }


def label_breakdown(rs, key):
    """Label-side outcome shares split by a per-label attribute (edge, size bin)."""
    gl = np.concatenate([r["gl"] for r in rs])
    k = np.concatenate([key(r) for r in rs])
    return {str(v): shares(gl[k == v], GT_CLASSES) for v in np.unique(k)}


def report(res, tiers):
    out = {}
    for tier in tiers:
        rs = [r for r in res.values() if r["tier"] == tier]
        if not rs:
            continue
        sites = sorted({r["site"] for r in rs})
        size_bin = lambda r: np.array([f"{SIZE_BINS[i]}-{SIZE_BINS[i + 1]}m2" for i in
                                       np.digitize(r["g_area_m2"], SIZE_BINS[1:-1])], object)
        out[tier] = {
            "all": summarise(rs),
            "by_site": {s: summarise([r for r in rs if r["site"] == s]) for s in sites},
            "labels_by_edge": label_breakdown(rs, lambda r: np.where(r["g_edge"], "edge", "interior")),
            "labels_by_size": label_breakdown(rs, size_bin),
        }
    return out


def print_table(out):
    for tier, t in out.items():
        print(f"\n== {tier}")
        print(f"{'site':6} {'imgs':>4} {'P':>5} {'R':>5} | " + " ".join(f"{c:>9}" for c in GT_CLASSES)
              + " | " + " ".join(f"{c:>9}" for c in PRED_CLASSES[1:]) + " | med dE,dN m  area×")
        rows = [("ALL", t["all"])] + list(t["by_site"].items())
        for s, a in rows:
            L, Pr, loc = a["labels"], a["predictions"], a["matched_localisation_p25_p50_p75"]
            med = lambda k: f"{loc[k][1]:+.1f}" if loc[k] else "  -"
            print(f"{s:6} {a['images']:>4} {a['precision_mean_per_image'] or 0:>5.2f} {a['recall_mean_per_image'] or 0:>5.2f} | "
                  + " ".join(f"{(L[c] or 0):>9.2f}" for c in GT_CLASSES) + " | "
                  + " ".join(f"{(Pr[c] or 0):>9.2f}" for c in PRED_CLASSES[1:])
                  + f" | {med('offset_east_m')},{med('offset_north_m')}  {loc['area_ratio'][1] if loc['area_ratio'] else '-'}")


def figures(res, meta, preds, out_dir, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from make_alignment_viewer import height_colour, hillshade

    for site in CA_SITES:
        rs = sorted([(k, r) for k, r in res.items() if r["site"] == site and r["tier"] == "evaluation" and r["n_gt"]],
                    key=lambda kr: np.nan_to_num(2 * kr[1]["tp"] / max(1, kr[1]["n_gt"] + kr[1]["n_pred"])))
        if not rs:
            continue
        pick = [("worst", rs[0]), ("median", rs[len(rs) // 2]), ("best", rs[-1])]
        fig, axes = plt.subplots(3, 2, figsize=(9, 13.5), layout="constrained")
        for row, (lab, (plot, r)) in zip(axes, pick):
            m = meta[plot]
            with rasterio.open(m["rgb"]) as src:
                rgb = src.read([1, 2, 3]).transpose(1, 2, 0); rt = src.transform
            chm, ct, _ = read_chm(m["chm"])
            sm = chm
            hs = height_colour(sm) * (0.35 + 0.65 * hillshade(sm * 1.5))[..., None]
            (x0, y0), (x1, y1) = ~rt * (ct * (0, 0)), ~rt * (ct * (chm.shape[1], chm.shape[0]))
            g = read_gt(m["xml"]); p = np.array(preds.get(plot, []), float).reshape(-1, 4)
            f1 = 2 * r["tp"] / max(1, r["n_gt"] + r["n_pred"])
            for ax, img, ext in ((row[0], rgb, None), (row[1], hs, (x0, x1, y1, y0))):
                ax.imshow(img, extent=ext) if ext else ax.imshow(img)
                for b in g:
                    ax.add_patch(Rectangle(b[:2], *(b[2:] - b[:2]), fill=False, ec="white", lw=1.4))
                for b in p:
                    ax.add_patch(Rectangle(b[:2], *(b[2:] - b[:2]), fill=False, ec="#e87ba4", lw=1.2))
                ax.set_xlim(0, m["rgb_shape"][1]); ax.set_ylim(m["rgb_shape"][0], 0); ax.axis("off")
            row[0].set_title(f"{lab}: {plot}  F1 {f1:.2f}  ({r['n_gt']} labels, {r['n_pred']} crowns)", loc="left", fontsize=9)
        fig.suptitle(f"{site}: human labels (white) vs LiDAR watershed, min top {tag} (pink)", fontsize=11)
        fig.savefig(out_dir / f"fig_{site}_{tag}.png", dpi=110)
        plt.close(fig)
        print(f"  wrote {out_dir / f'fig_{site}_{tag}.png'}")


def sanity(meta):
    """Labels scored against themselves, then against themselves shifted 2 m east."""
    ev = {k: m for k, m in meta.items() if m["tier"] == "evaluation"}
    gts = {k: read_gt(m["xml"]) for k, m in ev.items()}
    for name, dx_m in (("identity", 0.0), ("shift 2 m east", 2.0)):
        preds = {k: (g + np.array([1, 0, 1, 0]) * dx_m / ev[k]["rgb_px_m"]).tolist() for k, g in gts.items()}
        a = summarise(list(evaluate(ev, preds, {}, gts).values()))
        print(f"{name:15} P {a['precision_mean_per_image']}  R {a['recall_mean_per_image']}  "
              f"labels {a['labels']}  median east offset {a['matched_localisation_p25_p50_p75']['offset_east_m']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred_dir")
    ap.add_argument("--min-top", type=float, default=5.0)
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--sanity", action="store_true", help="score the labels against themselves instead")
    a = ap.parse_args()
    d = Path(a.pred_dir); tag = f"{a.min_top:g}m"
    if a.sanity:
        sanity(json.loads((d / "plots.json").read_text())); return
    meta, preds, extra = load(d, tag)
    res = evaluate(meta, preds, extra)
    out = report(res, ("evaluation", "training"))
    (d / f"metrics_{tag}.json").write_text(json.dumps(out, indent=1))
    print_table(out)
    print(f"\nwrote {d / f'metrics_{tag}.json'}")
    if a.figures:
        figures(res, meta, preds, d, tag)


if __name__ == "__main__":
    main()
