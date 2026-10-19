#!/usr/bin/env python3
"""
Fetch post-fire NAIP for each CHM-NAIP chip, on the same pixel grid as the
chip's pre-fire NAIP.

Source: Microsoft Planetary Computer NAIP collection (anonymous SAS signing).
Only each chip's footprint is read from the cloud-optimised GeoTIFFs, then
resampled into the pre-fire chip's transform/shape. The result is placed by the
2022 imagery's *own* georeferencing -- it is not aligned to the LiDAR or to the
pre-fire NAIP, so it carries its own offset.

    uv run python baselines/fetch_postfire_naip.py NAIP_TAR CROWNS_JSON OUT_DIR --year 2022
"""
import argparse, json, tarfile, urllib.request
from pathlib import Path

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.warp import reproject, Resampling, transform_bounds

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token/naip"


def post_json(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=60))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("naip_tar"); ap.add_argument("crowns_json"); ap.add_argument("out")
    ap.add_argument("--year", default="2022")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    token = json.load(urllib.request.urlopen(SAS, timeout=60))["token"]
    ntf = tarfile.open(a.naip_tar)
    nn = {Path(m.name).stem.rsplit("_", 1)[0]: m.name for m in ntf.getmembers() if m.name.endswith(".tif")}
    recs = json.load(open(a.crowns_json)); meta = {}

    for i, r in enumerate(recs):
        k = r["key"]
        with MemoryFile(ntf.extractfile(nn[k]).read()).open() as n:
            tr, crs, H, W = n.transform, n.crs, n.height, n.width
        bounds = rasterio.transform.array_bounds(H, W, tr)
        bbox = transform_bounds(crs, "EPSG:4326", *bounds)
        items = post_json(STAC, {"collections": ["naip"], "bbox": list(bbox), "limit": 20,
                                 "query": {"naip:year": {"eq": a.year}}})["features"]
        dst = np.zeros((4, H, W), np.uint8); filled = np.zeros((H, W), bool); dates = set()
        for it in items:
            href = it["assets"]["image"]["href"] + "?" + token
            with rasterio.open(href) as src:
                tmp = np.zeros((4, H, W), np.uint8)
                reproject(rasterio.band(src, [1, 2, 3, 4]), tmp, src_transform=src.transform, src_crs=src.crs,
                          dst_transform=tr, dst_crs=crs, resampling=Resampling.bilinear, src_nodata=0, dst_nodata=0)
            got = tmp.any(0) & ~filled
            dst[:, got] = tmp[:, got]; filled |= got
            if got.any():
                dates.add(it["properties"]["datetime"][:10])
        cover = float(filled.mean())
        if cover > 0:
            np.save(out / f"{k}_post.npy", dst)
        meta[k] = {"post_dates": sorted(dates), "post_cover": round(cover, 3), "post_items": [it["id"] for it in items]}
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(recs)}")
    (out / "postfire.json").write_text(json.dumps(meta, indent=1))
    full = sum(v["post_cover"] > 0.99 for v in meta.values())
    print(f"{len(meta)} chips: {full} fully covered by {a.year} NAIP, "
          f"{sum(0 < v['post_cover'] <= 0.99 for v in meta.values())} partial, "
          f"{sum(v['post_cover'] == 0 for v in meta.values())} none")


if __name__ == "__main__":
    main()
