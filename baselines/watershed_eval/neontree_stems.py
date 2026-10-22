#!/usr/bin/env python3
"""
Score crown boxes against NEON's field-mapped stems (Woody Vegetation Structure,
DP1.10098.001), as shipped in the NeonTreeEvaluation R package (data/field.rda).

Stem filters follow the package's documentation (evaluate_field_stems):
  itc coordinates present; not a 2014 event; growth form tree/sapling; DBH > 15 cm;
  height > 3 m; between-year height change <= 8 m; measured in > 1 survey;
  and |CHM at the stem - field height| < 4 m, so understory stems are dropped.
(The package code applies only the last two per plot, against the unfiltered table;
the documented filters are stricter and used here.) Each individual's most recent
record is kept. Stems are matched to plots by location, not plot name.

Scores, per plot:
  recall_1to1  one-to-one matching of stems to boxes that contain them, / stems.
               A box holding two stems matches only one, so merging is penalised.
               (The package counts boxes containing >= 1 stem, which can exceed 1
               when boxes overlap.)
  hit          share of stems inside any box (merging not penalised)
  merged       share of stem-containing boxes that contain >= 2 stems
There is no precision: not every tree in a plot is mapped.

    python baselines/watershed_eval/neontree_stems.py FIELD_RDA PRED_DIR [PRED_DIR ...]
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import rasterio
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import neontree_eval as E

FORMS = {"single bole tree", "multi-bole tree", "small tree", "sapling"}


def load_field(rda, min_surveys=2):
    """min_surveys=2 is the package's rule; 1 keeps single-survey sites (TEAK and SOAP
    were surveyed once, in 2015), leaving the CHM-height check to catch bad heights."""
    import pyreadr
    df = next(iter(pyreadr.read_r(str(rda)).values()))
    df = df[df.itcEasting.notna() & ~df.eventID.astype(str).str.contains("2014")
            & df.growthForm.isin(FORMS) & (df.stemDiameter > 15) & (df.height > 3)]
    df = df.sort_values("date")
    jump = df.groupby("individualID", observed=True).height.apply(lambda h: abs(np.diff(h.values).sum()))
    n_ev = df.groupby("individualID", observed=True).eventID.nunique()
    keep = set(jump.index[jump <= 8]) & set(n_ev.index[n_ev >= min_surveys])
    df = df[df.individualID.isin(keep)]
    return df.groupby("individualID", observed=True).tail(1)          # most recent record


def plot_stems(field, meta):
    """{plot: (n, 2) stem positions in RGB pixel coordinates}, for plots with >= 1 stem."""
    out = {}
    for plot, m in meta.items():
        with rasterio.open(m["rgb"]) as r:
            b, inv, epsg = r.bounds, ~r.transform, r.crs.to_epsg()
        f = field[(field.utmZone.astype(str).str[:-1].astype(int) + 32600 == epsg)
                  & field.itcEasting.between(b.left, b.right) & field.itcNorthing.between(b.bottom, b.top)]
        if not len(f):
            continue
        with rasterio.open(m["chm"]) as c:
            chm = np.array([v[0] for v in c.sample(zip(f.itcEasting, f.itcNorthing))], float)
        f = f[np.abs(chm - f.height.values) < 4]
        if len(f):
            out[plot] = np.array([inv * xy for xy in zip(f.itcEasting, f.itcNorthing)])
    return out


def score_plot(boxes, stems):
    b = np.asarray(boxes, float).reshape(-1, 4)
    inside = ((stems[:, None, 0] >= b[None, :, 0]) & (stems[:, None, 0] <= b[None, :, 2])
              & (stems[:, None, 1] >= b[None, :, 1]) & (stems[:, None, 1] <= b[None, :, 3]))
    n = len(stems)
    if not len(b):
        return {"n": n, "recall_1to1": 0.0, "hit": 0.0, "merged": None}
    r, c = linear_sum_assignment(-inside.astype(float))
    per_box = inside.sum(0)
    return {"n": n, "recall_1to1": float(inside[r, c].sum()) / n, "hit": float(inside.any(1).mean()),
            "merged": float((per_box >= 2).sum() / max((per_box >= 1).sum(), 1)),
            "_m": int(inside[r, c].sum()), "_h": int(inside.any(1).sum()),
            "_b1": int((per_box >= 1).sum()), "_b2": int((per_box >= 2).sum())}


def summarise(preds, stems, plots=None):
    plots = [p for p in (plots if plots is not None else stems) if p in stems]
    rs = [score_plot(preds.get(p, []), stems[p]) for p in plots]
    if not rs:
        return None
    n = sum(r["n"] for r in rs)
    mean = lambda k: float(np.mean([r[k] for r in rs if r[k] is not None]))
    return {"plots": len(rs), "stems": n, "recall_1to1": round(mean("recall_1to1"), 3), "hit": round(mean("hit"), 3),
            "merged": round(mean("merged"), 3),
            "recall_1to1_pooled": round(sum(r.get("_m", 0) for r in rs) / n, 3),
            "merged_pooled": round(sum(r.get("_b2", 0) for r in rs) / max(sum(r.get("_b1", 0) for r in rs), 1), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("field"); ap.add_argument("pred_dirs", nargs="+")
    ap.add_argument("--min-surveys", type=int, default=2)
    a = ap.parse_args()
    field = load_field(a.field, a.min_surveys)
    for d in a.pred_dirs:
        meta, preds, _ = E.load(Path(d), "5m")
        stems = plot_stems(field, meta)
        labels = {p: E.read_gt(m["xml"]) for p, m in meta.items()}
        print(Path(d).name, json.dumps(summarise(preds, stems)), " | human labels:", json.dumps(summarise(labels, stems)))


if __name__ == "__main__":
    main()
