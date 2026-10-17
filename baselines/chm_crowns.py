#!/usr/bin/env python3
"""
LiDAR crown labels for NAIP, from the CONUS CHM-NAIP pairs (Sci. Data 2025,
https://rangeland.ntsg.umt.edu/data/rap/chm-naip/, CC BY 4.0).

For each 1 m LiDAR CHM chip:
  1. 3x3 median smooth
  2. treetops: variable-window local maxima (Hemming-Schroeder et al. 2023):
     window diameter 4 m below 32 m tall, h/8 for 32-80 m, 10 m above
  3. crowns: marker-controlled watershed on the CHM, restricted to >3 m
  4. keep crowns with top >5 m (trees, not shrubs) and area >=1 m^2
Then, on the paired NAIP chip:
  5. estimate the NAIP-vs-LiDAR offset by cross-correlating the CHM (resampled
     to the NAIP grid) against NAIP NDVI over integer pixel shifts
  6. shift crowns by that offset and emit boxes in NAIP pixel coordinates

Requires scipy<1.12 on this macOS (newer wheels fail to load).

    python baselines/chm_crowns.py CHM_TAR NAIP_TAR OUT_DIR [--n 200] [--min-forest 0.3]
"""
import argparse, io, json, re, tarfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.warp import reproject, Resampling
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.measure import regionprops

MAX_SHIFT_M = 12.0          # search radius for registration; NAIP spec is +/-6 m at 95%


def disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def treetops(chm):
    """Variable-window local maxima; window diameter in metres = pixels at 1 m."""
    tops = np.zeros(chm.shape, bool)
    ws = np.where(chm < 32, 4.0, np.where(chm <= 80, chm / 8.0, 10.0))
    for d in np.unique(np.round(ws)):
        r = max(1, int(round(d / 2)))
        mx = ndi.maximum_filter(chm, footprint=disk(r), mode="nearest")
        sel = (np.round(ws) == d) & (chm >= mx) & (chm > 5)
        tops |= sel
    # collapse flat plateaus to one marker each
    lab, n = ndi.label(tops, structure=np.ones((3, 3)))
    return lab, n


def crowns_from_chm(chm):
    chm = ndi.median_filter(chm, size=3)
    markers, n = treetops(chm)
    if n == 0:
        return chm, np.zeros(chm.shape, int), []
    seg = watershed(-chm, markers, mask=chm > 3)
    keep = []
    for r in regionprops(seg, intensity_image=chm):
        if r.area >= 1 and r.max_intensity > 5:
            keep.append(r)
    return chm, seg, keep


def ndvi(naip):
    r, n = naip[0].astype(np.float32), naip[3].astype(np.float32)
    return (n - r) / np.maximum(n + r, 1e-6)


def best_shift(a, b, max_px):
    """Integer (dy, dx) maximising normalised correlation of a shifted onto b."""
    a = (a - a.mean()) / (a.std() + 1e-6)
    b = (b - b.mean()) / (b.std() + 1e-6)
    best, arg, scores = -9, (0, 0), {}
    H, W = a.shape
    for dy in range(-max_px, max_px + 1):
        for dx in range(-max_px, max_px + 1):
            ys, ye = max(0, dy), min(H, H + dy)
            xs, xe = max(0, dx), min(W, W + dx)
            pa = a[ys - dy:ye - dy, xs - dx:xe - dx]
            pb = b[ys:ye, xs:xe]
            s = float((pa * pb).mean())
            scores[(dy, dx)] = s
            if s > best:
                best, arg = s, (dy, dx)
    zero = scores[(0, 0)]
    return arg, best, zero


def open_member(tf, name):
    return MemoryFile(tf.extractfile(name).read()).open()


NODATA = 65535   # no LiDAR returns, e.g. open water; must never be read as height


def read_chm(tf, name):
    """CHM chip -> (height in metres with no-data set to 0, no-data mask, transform, crs)."""
    with open_member(tf, name) as c:
        raw = c.read(1)
        tr, crs = c.transform, c.crs
    nod = raw == NODATA
    return np.where(nod, 0, raw.astype(np.float32) / 100.0), nod, tr, crs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chm_tar"); ap.add_argument("naip_tar"); ap.add_argument("out")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--min-forest", type=float, default=0.3)
    ap.add_argument("--keys", help="JSON list of {key, ...} to restrict to; extra fields are copied into the output")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    ctf, ntf = tarfile.open(a.chm_tar), tarfile.open(a.naip_tar)
    chms = {Path(m.name).stem: m.name for m in ctf.getmembers() if m.name.endswith(".tif")}
    naips = {}
    for m in ntf.getmembers():
        if m.name.endswith(".tif"):
            stem = Path(m.name).stem
            naips[stem.rsplit("_", 1)[0]] = m.name      # key on the CHM part of the name
    keys = sorted(set(chms) & set(naips))
    print(f"pairs in tar: {len(keys)}")
    extra = {}
    if a.keys:
        extra = {r["key"]: r for r in json.load(open(a.keys))}
        keys = [k for k in keys if k in extra]
        print(f"restricted to {len(keys)} listed chips")

    recs, done, skipped = [], 0, 0
    for k in keys:
        if done >= a.n:
            break
        try:
            chm, nod, c_tr, c_crs = read_chm(ctf, chms[k])
        except Exception:
            skipped += 1; continue          # a few chips in the release are corrupt
        if nod.mean() > 0.05 or (chm > 5).mean() < a.min_forest:
            continue
        try:
            with open_member(ntf, naips[k]) as n:
                naip = n.read()
                n_tr, n_crs, n_shape = n.transform, n.crs, (n.height, n.width)
        except Exception:
            skipped += 1; continue
        if (naip[4] == 0).mean() > 0.05:
            continue

        sm, seg, crowns = crowns_from_chm(chm)

        # CHM onto the NAIP grid, for registration
        chm_on_naip = np.zeros(n_shape, np.float32)
        reproject(sm, chm_on_naip, src_transform=c_tr, src_crs=c_crs,
                  dst_transform=n_tr, dst_crs=n_crs, resampling=Resampling.bilinear)
        px = abs(n_tr.a)
        (dy, dx), corr, corr0 = best_shift(chm_on_naip, ndvi(naip), int(MAX_SHIFT_M / px))

        # crown boxes: CHM pixel -> map coords -> NAIP pixel, then apply shift
        boxes = []
        inv = ~n_tr
        for r in crowns:
            y0, x0, y1, x1 = r.bbox
            X0, Y0 = c_tr * (x0, y0); X1, Y1 = c_tr * (x1, y1)
            u0, v0 = inv * (X0, Y0); u1, v1 = inv * (X1, Y1)
            boxes.append([round(min(u0, u1) + dx, 1), round(min(v0, v1) + dy, 1),
                          round(max(u0, u1) + dx, 1), round(max(v0, v1) + dy, 1),
                          round(float(r.max_intensity), 1)])

        m = re.match(r"(\d+)_(\d+)_(\d+)_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})", Path(naips[k]).stem)
        rec = {"key": k, "lidar_date": m.group(4) if m else None, "naip_date": m.group(5) if m else None,
               "naip_px_m": px, "naip_shape": list(n_shape), "forest_frac": round(float((chm > 5).mean()), 3),
               "n_crowns": len(boxes), "shift_px": [dy, dx], "shift_m": [round(dy * px, 2), round(dx * px, 2)],
               "corr_best": round(corr, 3), "corr_zero": round(corr0, 3), "boxes": boxes}
        rec.update({f: v for f, v in extra.get(k, {}).items() if f not in rec})
        recs.append(rec); done += 1
        np.save(out / f"{k}_naip.npy", naip[:4])
        if done % 25 == 0:
            print(f"  {done} chips")

    (out / "crowns.json").write_text(json.dumps(recs))
    print(f"wrote {len(recs)} chips -> {out/'crowns.json'}  (skipped {skipped} unreadable)")


if __name__ == "__main__":
    main()
