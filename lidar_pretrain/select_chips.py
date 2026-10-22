#!/usr/bin/env python3
"""
Step 1: find the 3DEP CHM chips (Allred et al. 2025) under each 2020 fire.

For each fire: fetch the CAL FIRE perimeter, work out which tar bins (UTM zone x
10 km of northing) it touches, download those bins' CHM tars (0.2-2.2 GB each;
resumable, cached in WORK/chm/), and keep chips that
  - overlap the perimeter,
  - have LiDAR flown before the fire's alarm date,
  - are > EXCLUDE_M from every val/test hand-labelled tile (so pretraining never
    sees LiDAR labels for the trees we test on; Weinstein et al. 2020 used 1 km).
Eligible chips' CHM GeoTIFFs (~100 KB each) are extracted to WORK/chm_chips/, so the
tars can be deleted as they are scanned (--delete-tars; peak disk = one tar).
No NAIP is needed here; pre-fire NAIP is fetched per chip in build_pretrain.py.

    python lidar_pretrain/select_chips.py WORK [--fires czu scu north creek castle] [--delete-tars] [--local-only]

Writes WORK/chips.json (all eligible chips) and WORK/coverage.json (per fire:
bins, chips, LiDAR years) -- the latter answers "does 3DEP cover this fire?".
"""
import argparse, json, re, tarfile, urllib.request
from collections import Counter
from pathlib import Path

from pyproj import Transformer
from shapely.geometry import box
from shapely.ops import transform, unary_union

import common as C

EXCLUDE_M = 1000.0
CHIP = re.compile(r"(\d+)_(\d+)_(\d+)_(\d{4}-\d{2}-\d{2})$")


def download(url, dst):
    """Resumable download (like curl -C -)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    size = int(urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60).headers["Content-Length"])
    have = dst.stat().st_size if dst.exists() else 0
    if have == size:
        return dst
    print(f"  downloading {url} ({size/1e9:.2f} GB, have {have/1e9:.2f})", flush=True)
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(req, timeout=120) as r, open(dst, "ab" if have else "wb") as f:
        while chunk := r.read(1 << 22):
            f.write(chunk)
    assert dst.stat().st_size == size, f"incomplete download {dst}"
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work"); ap.add_argument("--fires", nargs="+", default=list(C.FIRES))
    ap.add_argument("--local-only", action="store_true", help="use only CHM tars already in WORK/chm/")
    ap.add_argument("--delete-tars", action="store_true", help="delete each CHM tar after extracting its chips")
    a = ap.parse_args()
    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    held_out = C.tile_footprints(("val", "test"))

    chips, coverage = [], {}
    for fire in a.fires:
        geom, alarm = C.fetch_perimeter(C.FIRES[fire])
        zone = int((geom.centroid.x + 180) // 6) + 1
        to_utm = Transformer.from_crs(4326, 26900 + zone, always_xy=True).transform
        per = transform(to_utm, geom)
        # held-out tiles in this zone's CRS, buffered
        excl = unary_union([transform(Transformer.from_crs(epsg, 26900 + zone, always_xy=True).transform, b)
                            .buffer(EXCLUDE_M) for b, epsg in held_out])
        x0, y0, x1, y1 = per.bounds
        bins = [b for b in range(int(y0 // 10000), int(y1 // 10000) + 1)
                if per.intersects(box(x0, b * 10000, x1, (b + 1) * 10000))]
        print(f"{fire}: {C.FIRES[fire]}, alarm {alarm}, zone {zone}, bins {bins}", flush=True)
        rows, years, n_excl, n_late = [], Counter(), 0, 0
        for b in bins:
            name = f"{zone}-{b}"
            if a.local_only and not (work / "chm" / f"{name}.tar").exists():
                print(f"  skipping bin {name} (not downloaded)"); continue
            tar = download(f"{C.BIN_URL}/chm/{zone}/{name}.tar", work / "chm" / f"{name}.tar")
            tf = tarfile.open(tar)
            for m in tf.getmembers():
                g = CHIP.match(Path(m.name).stem) if m.name.endswith(".tif") else None
                if not g:
                    continue
                x, y, lidar = int(g.group(2)), int(g.group(3)), g.group(4)
                fp = box(x - 128, y - 128, x + 128, y + 128)
                if not fp.intersects(per):
                    continue
                if lidar >= str(alarm):
                    n_late += 1; continue
                if fp.intersects(excl):
                    n_excl += 1; continue
                years[lidar[:4]] += 1
                dst = work / "chm_chips" / f"{Path(m.name).stem}.tif"
                if not dst.exists():
                    dst.parent.mkdir(exist_ok=True); dst.write_bytes(tf.extractfile(m).read())
                rows.append({"key": Path(m.name).stem, "chm": str(dst.relative_to(work)), "bin": name, "fire": fire,
                             "zone": zone, "x": x, "y": y, "lidar": lidar, "alarm": str(alarm),
                             "inside": per.contains(fp)})
            tf.close()
            if a.delete_tars and not tar.is_symlink():
                tar.unlink(); print(f"  deleted {tar.name}")
        coverage[fire] = {"fire_name": C.FIRES[fire], "alarm": str(alarm), "zone": zone, "bins": bins,
                          "eligible_chips": len(rows), "fully_inside": sum(r["inside"] for r in rows),
                          "excluded_near_val_test": n_excl, "lidar_after_alarm": n_late,
                          "lidar_years": dict(sorted(years.items())),
                          "perimeter_km2": round(per.area / 1e6, 1),
                          "covered_km2": round(len(rows) * 0.256 ** 2, 1)}
        print(f"  {json.dumps(coverage[fire])}", flush=True)
        chips += rows
    (work / "chips.json").write_text(json.dumps(chips))
    (work / "coverage.json").write_text(json.dumps(coverage, indent=1))
    print(f"{len(chips)} eligible chips -> {work/'chips.json'}")


if __name__ == "__main__":
    main()
