#!/usr/bin/env python3
"""
Step 2: turn selected 3DEP chips into DeepForest pretraining images.

For each chip from select_chips.py (random order, up to --max-chips per fire):
  1. LiDAR crowns with the pipeline's original segmentation
     (baselines/chm_crowns.crowns_from_chm: 3x3 median, variable-window treetops,
     watershed, top > 5 m). Each crown becomes its bounding rectangle.
  2. Pre-fire NAIP from Microsoft Planetary Computer, read only over the chip and
     resampled onto a 0.6 m grid anchored at the chip's corner: the latest NAIP year
     acquired before the fire's alarm date that covers the chip, within
     --max-year-gap years of the LiDAR (Allred et al. pair within +/-2 years).
  3. Four overlapping 256 x 256 px windows (the hand-labelled tile size), each
     stretched to RGB exactly as the hand-labelled tiles are (common.stretch_to_rgb8).
     A crown box is kept if >= 50% of its area falls inside the window.
Crowns are NOT shifted: NAIP-LiDAR misregistration (several metres) is left in,
since the automatic offset estimate did worse than random (REPRODUCE_LIDAR_CROWNS.md).
Chips are skipped if > 5% of the CHM is no-data or < --min-forest of it is > 5 m.

    python lidar_pretrain/build_pretrain.py WORK [--max-chips 400] [--workers 8]

Writes WORK/pretrain/images/*.png, WORK/pretrain/pretrain.csv (DeepForest format,
image_path relative to WORK/pretrain/images) and WORK/pretrain/chips_used.json.
Resumable: finished chips are logged in WORK/pretrain/done.jsonl.
"""
import argparse, json, random, threading, time, urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from PIL import Image
from rasterio.warp import Resampling, reproject, transform_bounds

import common as C
from chm_crowns import NODATA, crowns_from_chm

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token/naip"
N_PX = int(round(256 / C.NAIP_RES))                       # 427 px covers the 256 m chip
OFFSETS = [0, N_PX - C.TILE_PX]                            # 2 x 2 windows, 85 px overlap
_tok = {"t": None, "at": 0.0}; _tok_lock = threading.Lock()


def token():
    with _tok_lock:
        if _tok["t"] is None or time.time() - _tok["at"] > 20 * 60:
            _tok["t"] = json.load(urllib.request.urlopen(SAS, timeout=60))["token"]; _tok["at"] = time.time()
        return _tok["t"]


def post_json(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    for i in range(4):
        try:
            return json.load(urllib.request.urlopen(req, timeout=60))
        except Exception:
            time.sleep(2 ** i)
    raise RuntimeError(f"STAC search failed: {body}")


def prefire_naip(tr, crs, alarm, lidar_year, max_gap):
    """(4 x N x N uint8, naip year, acquisition dates) or None."""
    bounds = rasterio.transform.array_bounds(N_PX, N_PX, tr)
    bbox = transform_bounds(crs, "EPSG:4326", *bounds)
    items = post_json(STAC, {"collections": ["naip"], "bbox": list(bbox), "limit": 50,
                             "datetime": f"2010-01-01T00:00:00Z/{alarm}T00:00:00Z"})["features"]
    by_year = defaultdict(list)
    for it in items:
        if it["properties"]["datetime"][:10] < alarm:
            by_year[int(it["properties"]["naip:year"])].append(it)
    for year in sorted(by_year, reverse=True):
        if abs(year - lidar_year) > max_gap:
            continue
        dst = np.zeros((4, N_PX, N_PX), np.uint8); filled = np.zeros((N_PX, N_PX), bool); dates = set()
        for it in by_year[year]:
            with rasterio.open(it["assets"]["image"]["href"] + "?" + token()) as src:
                if src.count < 4:
                    continue
                tmp = np.zeros((4, N_PX, N_PX), np.uint8)
                reproject(rasterio.band(src, [1, 2, 3, 4]), tmp, src_transform=src.transform, src_crs=src.crs,
                          dst_transform=tr, dst_crs=crs, resampling=Resampling.bilinear, src_nodata=0, dst_nodata=0)
            got = tmp.any(0) & ~filled
            dst[:, got] = tmp[:, got]; filled |= got
            if got.any():
                dates.add(it["properties"]["datetime"][:10])
        if filled.mean() > 0.99:
            return dst, year, sorted(dates)
    return None


def read_chip(path):
    """Same as chm_crowns.read_chm, for an extracted chip: metres, no-data -> 0."""
    with rasterio.open(path) as c:
        raw = c.read(1); tr, crs = c.transform, c.crs
    nod = raw == NODATA
    return np.where(nod, 0, raw.astype(np.float32) / 100.0), nod, tr, crs


def process(chip, work, out_img, min_forest, max_gap):
    chm, nod, c_tr, crs = read_chip(Path(work) / chip["chm"])
    if nod.mean() > 0.05:
        return {"key": chip["key"], "skip": "chm_nodata"}
    if (chm > 5).mean() < min_forest:
        return {"key": chip["key"], "skip": "not_forest"}
    _, _, crowns = crowns_from_chm(chm)
    X0, Y0 = c_tr.c, c_tr.f
    tr = Affine(C.NAIP_RES, 0, X0, 0, -C.NAIP_RES, Y0)
    got = prefire_naip(tr, crs, chip["alarm"], int(chip["lidar"][:4]), max_gap)
    if got is None:
        return {"key": chip["key"], "skip": "no_prefire_naip"}
    naip, year, dates = got
    # crown rectangles in NAIP-grid pixels (CHM is 1 m, north-up, same CRS)
    b = np.array([r.bbox for r in crowns], float).reshape(-1, 4)          # y0, x0, y1, x1 in CHM px
    s = abs(c_tr.a) / C.NAIP_RES
    boxes = np.c_[b[:, 1] * s, b[:, 0] * s, b[:, 3] * s, b[:, 2] * s]       # xmin, ymin, xmax, ymax
    rows, windows = [], 0
    for oy in OFFSETS:
        for ox in OFFSETS:
            win = naip[:, oy:oy + C.TILE_PX, ox:ox + C.TILE_PX]
            if (~win.any(0)).mean() > 0.01:
                continue
            bx = boxes - [ox, oy, ox, oy]
            cl = np.clip(bx, 0, C.TILE_PX)
            area = (bx[:, 2] - bx[:, 0]) * (bx[:, 3] - bx[:, 1])
            carea = (cl[:, 2] - cl[:, 0]) * (cl[:, 3] - cl[:, 1])
            keep = (carea >= 0.5 * area) & (cl[:, 2] - cl[:, 0] >= 2) & (cl[:, 3] - cl[:, 1] >= 2)
            if not keep.any():
                continue
            name = f"{chip['key']}_{ox}_{oy}.png"
            Image.fromarray(C.stretch_to_rgb8(win)).save(out_img / name)
            rows += [{"image_path": name, "xmin": int(x0), "ymin": int(y0), "xmax": int(x1), "ymax": int(y1),
                      "label": "Tree"} for x0, y0, x1, y1 in cl[keep]]
            windows += 1
    return {"key": chip["key"], "fire": chip["fire"], "lidar": chip["lidar"], "naip_year": year,
            "naip_dates": dates, "crowns": len(crowns), "windows": windows, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("--max-chips", type=int, default=400, help="per fire")
    ap.add_argument("--min-forest", type=float, default=0.1)
    ap.add_argument("--max-year-gap", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    work = Path(a.work); out = work / "pretrain"; out_img = out / "images"; out_img.mkdir(parents=True, exist_ok=True)
    chips = json.loads((work / "chips.json").read_text())
    random.Random(a.seed).shuffle(chips)

    log = out / "done.jsonl"
    done = [json.loads(l) for l in log.read_text().splitlines()] if log.exists() else []
    seen = {d["key"] for d in done}
    used = Counter(d["fire"] for d in done if "rows" in d and d["windows"])
    lock = threading.Lock()
    queue = [c for c in chips if c["key"] not in seen]
    print(f"{len(chips)} eligible chips, {len(seen)} already processed; target {a.max_chips} per fire", flush=True)

    with ThreadPoolExecutor(a.workers) as pool, open(log, "a") as fh:
        pending = {}
        it = iter(queue)

        def submit():
            for c in it:
                if used[c["fire"]] + sum(p["fire"] == c["fire"] for p in pending.values()) >= a.max_chips:
                    continue
                pending[pool.submit(process, c, work, out_img, a.min_forest, a.max_year_gap)] = c
                return True
            return False

        for _ in range(a.workers * 2):
            submit()
        n = 0
        while pending:
            fut = next(as_completed(pending))
            c = pending.pop(fut)
            try:
                r = fut.result()
            except Exception as e:
                r = {"key": c["key"], "skip": f"error: {type(e).__name__}: {e}"[:200]}
            with lock:
                fh.write(json.dumps(r) + "\n"); fh.flush(); done.append(r)
                if r.get("windows"):
                    used[c["fire"]] += 1
            n += 1
            if n % 25 == 0:
                print(f"  {n} processed; chips used per fire {dict(used)}", flush=True)
            submit()

    ok = [d for d in done if d.get("windows")]
    rows = [row for d in ok for row in d["rows"]]
    pd.DataFrame(rows).to_csv(out / "pretrain.csv", index=False)
    (out / "chips_used.json").write_text(json.dumps([{k: v for k, v in d.items() if k != "rows"} for d in ok], indent=1))
    skips = Counter(d["skip"].split(":")[0] for d in done if "skip" in d)
    by_fire = defaultdict(lambda: Counter())
    for d in ok:
        by_fire[d["fire"]].update(chips=1, windows=d["windows"], boxes=len(d["rows"]))
    print(f"{len(ok)} chips -> {sum(d['windows'] for d in ok)} images, {len(rows)} boxes; skipped {dict(skips)}")
    for f, v in by_fire.items():
        print(f"  {f}: {dict(v)}; NAIP years {dict(Counter(d['naip_year'] for d in ok if d['fire'] == f))}")
    print(f"-> {out/'pretrain.csv'}")


if __name__ == "__main__":
    main()
