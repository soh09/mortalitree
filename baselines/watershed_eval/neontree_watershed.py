#!/usr/bin/env python3
"""
Run our LiDAR watershed (chm_crowns.crowns_from_chm, unchanged) on every annotated
NeonTreeEvaluation plot/tile and write the crowns as boxes in the RGB image's pixel
coordinates -- the same frame as the human annotations.

Inputs are the output of neontree_fetch.py. Writes:
  OUT_DIR/predictions_{min_top}m.csv  plot, tier, site, xmin, ymin, xmax, ymax (RGB px),
                                      height_m, area_m2, edge
  OUT_DIR/plots.json                  per plot: tier, site, rgb/chm paths, rgb pixel size

`edge` marks crowns within EDGE_M of the image border: on 40 m plots, a tree whose top is
outside the crop gets no marker, so edge crowns carry extra segmentation error.

    uv run python baselines/watershed_eval/neontree_watershed.py NEONTREE_DIR OUT_DIR [--min-top 5] [--training]
"""
import argparse, csv, json, sys, xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # baselines/, for chm_crowns
from chm_crowns import crowns_from_chm

EDGE_M = 1.0


def site_of(plot):
    """SJER_059_2018 -> SJER; 2019_SJER_4_251000_4103000_image -> SJER."""
    return next(t for t in plot.split("_") if len(t) == 4 and t.isalpha() and t.isupper())


def find_plots(root, training):
    """[(plot, tier, xml_path, rgb_path, chm_path)] for annotation files whose RGB and CHM both exist."""
    tiers = [("evaluation", root / "evaluation")] + ([("training", root / "training")] if training else [])
    out, missing = [], []
    for xml in sorted((root / "annotations").glob("*.xml")):
        fname = ET.parse(xml).findtext("filename") or f"{xml.stem}.tif"
        stem = Path(fname).stem
        for tier, d in tiers:
            rgb, chm = d / "RGB" / f"{stem}.tif", d / "CHM" / f"{stem}_CHM.tif"
            if rgb.exists():
                (out if chm.exists() else missing).append((stem, tier, xml, rgb, chm))
                break
    if missing:
        print(f"  {len(missing)} annotated images have RGB but no CHM, skipped: "
              + ", ".join(m[0] for m in missing[:5]) + (" ..." if len(missing) > 5 else ""))
    return out


def read_chm(path):
    with rasterio.open(path) as c:
        a = c.read(1).astype(np.float32)
        nod = c.nodata
        tr, crs = c.transform, c.crs
    bad = ~np.isfinite(a) | (a < 0) | ((a == nod) if nod is not None else False)
    return np.where(bad, 0, a), tr, crs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("neontree"); ap.add_argument("out")
    ap.add_argument("--min-top", type=float, default=5.0, help="minimum treetop height, m (pipeline: 5, benchmark: 3)")
    ap.add_argument("--training", action="store_true", help="include the RGB-only training tiles")
    a = ap.parse_args()
    root, out = Path(a.neontree), Path(a.out); out.mkdir(parents=True, exist_ok=True)

    plots = find_plots(root, a.training)
    print(f"{len(plots)} annotated images with CHM "
          f"({sum(p[1] == 'evaluation' for p in plots)} evaluation, {sum(p[1] == 'training' for p in plots)} training)")

    rows, meta = [], {}
    for i, (plot, tier, xml_p, rgb_p, chm_p) in enumerate(plots):
        chm, ct, ccrs = read_chm(chm_p)
        with rasterio.open(rgb_p) as r:
            rt, rcrs, H, W = r.transform, r.crs, r.height, r.width
        if ccrs != rcrs:
            print(f"  skip {plot}: CHM {ccrs} vs RGB {rcrs}"); continue
        # a few benchmark CHMs are truncated (e.g. TALL_043_2019 is 40 x 1 px)
        cb, rb = rasterio.transform.array_bounds(*chm.shape, ct), rasterio.transform.array_bounds(H, W, rt)
        ov = max(0, min(cb[2], rb[2]) - max(cb[0], rb[0])) * max(0, min(cb[3], rb[3]) - max(cb[1], rb[1]))
        if ov < 0.9 * (rb[2] - rb[0]) * (rb[3] - rb[1]):
            print(f"  skip {plot}: CHM {chm.shape} covers {ov / ((rb[2] - rb[0]) * (rb[3] - rb[1])):.0%} of the image"); continue
        px = abs(rt.a); edge_px = EDGE_M / px
        sm, seg, crowns = crowns_from_chm(chm, a.min_top)
        inv = ~rt
        for c in crowns:
            y0, x0, y1, x1 = c.bbox
            u0, v0 = inv * (ct * (x0, y0)); u1, v1 = inv * (ct * (x1, y1))
            xa, xb = sorted((u0, u1)); ya, yb = sorted((v0, v1))
            cx, cy = (xa + xb) / 2, (ya + yb) / 2
            if not (0 <= cx < W and 0 <= cy < H):
                continue                         # CHM extends past the annotated image
            xa, ya, xb, yb = max(xa, 0), max(ya, 0), min(xb, W), min(yb, H)
            edge = xa < edge_px or ya < edge_px or xb > W - edge_px or yb > H - edge_px
            rows.append([plot, tier, site_of(plot), round(xa, 1), round(ya, 1), round(xb, 1), round(yb, 1),
                         round(float(c.max_intensity), 1), int(c.area * abs(ct.a * ct.e)), int(edge)])
        meta[plot] = {"tier": tier, "site": site_of(plot), "xml": str(xml_p), "rgb": str(rgb_p), "chm": str(chm_p),
                      "rgb_px_m": px, "rgb_shape": [H, W]}
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(plots)}")

    tag = f"{a.min_top:g}m"
    with open(out / f"predictions_{tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["plot", "tier", "site", "xmin", "ymin", "xmax", "ymax", "height_m", "area_m2", "edge"])
        w.writerows(rows)
    (out / "plots.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {len(rows)} crowns for {len(meta)} images -> {out / f'predictions_{tag}.csv'}")


if __name__ == "__main__":
    main()
