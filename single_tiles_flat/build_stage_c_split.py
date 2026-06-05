"""Build the Stage-C train/val/test split from the combined good pre/post tiles.

For every good prefire + postfire tile:
  * center-crop the native 0.6 m/px GeoTIFF (~404x408) to a 256x256 window
    (no resampling; matches the model's Stage-B input scale and is a multiple of
    the encoder patch size 8), writing a 4-band R,G,B,NIR uint8 tile, and
  * remap its labels into the crop frame (boxes whose center leaves the crop are
    dropped), producing normalized (cx,cy,w,h) boxes.

Tiles are split GEOGRAPHICALLY by location (x,y), 70/15/15, with each location's
prefire AND postfire versions kept in the SAME fold (their imagery is the same
ground, so splitting them would leak). Locations are clustered by the large x-gap
between the two fire sub-areas, and within each cluster a contiguous y-band split
keeps train/val/test spatially separated while both clusters feed every fold.

Outputs (under --out-dir, default single_tiles_flat/stage_c_data):
    tiles/{prefire|postfire}_{z}_{x}_{y}.tif      256x256 4-band crops
    train.json  val.json  test.json               NAIPTileDataset annotation lists

The JSON tile_path entries point at the *Modal* location (--modal-root, default
/data/stage_c) so the lists can be uploaded and consumed on Modal directly.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent  # .../mortalitree/mortalitree
CROP = 256
SOURCES = {
    "prefire":  {"img": "good/prefire_img",  "date": "2020-06-01"},
    "postfire": {"img": "good/postfire_img", "date": "2022-06-01"},
}


def remap_boxes_to_crop(boxes_full, x0, y0, W, H, C):
    """Full-tile-normalized cxcywh -> crop-normalized cxcywh, dropping out-of-crop centers."""
    if len(boxes_full) == 0:
        return np.zeros((0, 4), np.float32)
    cx = boxes_full[:, 0] * W
    cy = boxes_full[:, 1] * H
    w = boxes_full[:, 2] * W
    h = boxes_full[:, 3] * H
    ncx = (cx - x0) / C
    ncy = (cy - y0) / C
    nw = w / C
    nh = h / C
    inside = (ncx >= 0) & (ncx <= 1) & (ncy >= 0) & (ncy <= 1)
    out = np.stack([ncx, ncy, nw, nh], axis=1).astype(np.float32)[inside]
    x1 = np.clip(out[:, 0] - out[:, 2] / 2, 0, 1)
    y1 = np.clip(out[:, 1] - out[:, 3] / 2, 0, 1)
    x2 = np.clip(out[:, 0] + out[:, 2] / 2, 0, 1)
    y2 = np.clip(out[:, 1] + out[:, 3] / 2, 0, 1)
    return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1).astype(np.float32)


def assign_splits(locations, ratios, seed):
    """Geographic split: cluster by the largest x-gap, then a contiguous y-band
    70/15/15 split inside each cluster (both clusters feed every fold)."""
    locations = sorted(locations)
    xs = sorted({x for x, _ in locations})
    # Split into two clusters at the largest gap between consecutive x values.
    if len(xs) > 1:
        gi = max(range(len(xs) - 1), key=lambda i: xs[i + 1] - xs[i])
        xthr = (xs[gi] + xs[gi + 1]) / 2.0
    else:
        xthr = xs[0] + 1
    assignment = {}
    for in_cluster in (lambda l: l[0] < xthr, lambda l: l[0] >= xthr):
        clocs = sorted([l for l in locations if in_cluster(l)], key=lambda l: (l[1], l[0]))
        n = len(clocs)
        n_tr = round(n * ratios[0])
        n_val = round(n * ratios[1])
        for i, loc in enumerate(clocs):
            assignment[loc] = "train" if i < n_tr else ("val" if i < n_tr + n_val else "test")
    return assignment


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(REPO / "single_tiles_flat"),
                    help="single_tiles_flat dir holding the label CSVs + good/ images")
    ap.add_argument("--out-dir", default=str(REPO / "single_tiles_flat/stage_c_data"))
    ap.add_argument("--modal-root", default="/data/stage_c",
                    help="path the JSON tile_path entries point to (Modal-side)")
    ap.add_argument("--ratios", default="0.7,0.15,0.15")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import rasterio
    from rasterio.windows import Window

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    tiles_dir = out_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    ratios = tuple(float(x) for x in args.ratios.split(","))

    # 1) Gather every (source, tile) record with its crop-frame boxes + location.
    records = []          # dicts: source, stem, loc, boxes, lat, lon
    locations = set()
    for source, cfg in SOURCES.items():
        csv = root / f"{source}_labels.csv"
        img_dir = root / cfg["img"]
        df = pd.read_csv(csv)
        ts = 256.0
        df = df.assign(
            _cx=(df["xmin"] + df["xmax"]) / 2.0 / ts,
            _cy=(df["ymin"] + df["ymax"]) / 2.0 / ts,
            _w=(df["xmax"] - df["xmin"]) / ts,
            _h=(df["ymax"] - df["ymin"]) / ts,
        )
        for stem, g in df.groupby("imgname", sort=False):
            tif = img_dir / f"{stem}.tif"
            if not tif.exists():
                continue
            z, x, y = stem.split("_")
            loc = (int(x), int(y))
            with rasterio.open(tif) as src:
                W, H = src.width, src.height
            x0 = max(0, (W - CROP) // 2)
            y0 = max(0, (H - CROP) // 2)
            boxes_full = np.stack(
                [g["_cx"].to_numpy(), g["_cy"].to_numpy(),
                 g["_w"].to_numpy(), g["_h"].to_numpy()], axis=1).astype(np.float32)
            boxes = remap_boxes_to_crop(boxes_full, x0, y0, W, H, CROP)
            r0 = g.iloc[0]
            records.append({
                "source": source, "stem": stem, "loc": loc,
                "tif": tif, "crop": (x0, y0), "boxes": boxes,
                "lat": float(r0["lat"]), "lon": float(r0["lon"]),
                "date": cfg["date"],
            })
            locations.add(loc)

    # 2) Geographic split by location (pre+post of a location share a fold).
    assignment = assign_splits(locations, ratios, args.seed)

    # 3) Write the 256x256 crops + per-split annotation lists.
    splits = {"train": [], "val": [], "test": []}
    for rec in records:
        name = f"{rec['source']}_{rec['stem']}"
        x0, y0 = rec["crop"]
        with rasterio.open(rec["tif"]) as src:
            crop = src.read(indexes=[1, 2, 3, 4],
                            window=Window(x0, y0, CROP, CROP))
            profile = src.profile.copy()
        if crop.shape[1:] != (CROP, CROP):
            padded = np.zeros((4, CROP, CROP), dtype=crop.dtype)
            padded[:, : crop.shape[1], : crop.shape[2]] = crop
            crop = padded
        profile.update(width=CROP, height=CROP, count=4)
        out_tif = tiles_dir / f"{name}.tif"
        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(crop)

        split = assignment[rec["loc"]]
        splits[split].append({
            "tile_path": f"{args.modal_root}/tiles/{name}.tif",
            "boxes": rec["boxes"].tolist(),
            "exhaustive": True,
            "lat": rec["lat"], "lon": rec["lon"],
            "acquisition_date": rec["date"],
        })

    for split, items in splits.items():
        with open(out_dir / f"{split}.json", "w") as f:
            json.dump(items, f)
        n_box = sum(len(it["boxes"]) for it in items)
        n_loc = len({(it["lat"], it["lon"]) for it in items})
        print(f"{split:5s}: {len(items):3d} tiles  {n_box:5d} boxes  (~{n_loc} locations)")

    print(f"\nWrote dataset to {out_dir}")
    print("\nUpload to Modal (clay-data volume):")
    print(f"  modal volume put clay-data {tiles_dir}/ /stage_c/tiles/")
    for split in splits:
        print(f"  modal volume put clay-data {out_dir / f'{split}.json'} /stage_c/{split}.json")


if __name__ == "__main__":
    main()
