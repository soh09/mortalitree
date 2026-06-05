"""Modal patch pipeline for Stage-B tile preparation.

Reads NEON paired (HSI, RGB) data from the `mot` volume under
/data/{SITE}/{hyperspectral,rgb}/ and writes:

    /data/patches/{SITE}/{SITE}_{YEAR}_{east}_{north}_r{r}_c{c}.tif   (4 bands R,G,B,NIR; 256x256)
    /data/patches/labels_{SITE}.csv                                    (per-site)
    /data/patches/labels.csv                                           (combined)

For every source tile:
  1. Compute NDVI from HSI (mean over 835-920 nm vs 620-700 nm)
  2. Drop boxes with NDVI < NDVI_DEAD (burned/dead trees)
  3. Carve a 6x6 grid of 256-px patches at 0.6 m GSD (= 153.6 m on the ground)
     by area-averaging RGB from 0.1 m and bilinear-upsampling HSI NIR from 1 m
  4. Save 4-band GeoTIFFs and a per-tile labels chunk

Annotation CSVs are fetched from Zenodo to the volume server-side (never read
locally). Patches with more than --max-trees alive boxes are dropped (default
150) so no tile exceeds the detector's query budget; --max-trees 0 keeps all.

Requires the matching imagery already on the volume (run modal_neon_dl.py first
for the same --sites).

Output goes to /data/{out_tag}/ on the volume (default out_tag="patches"). Use a
distinct tag per config so different thresholds / site sets stay isolated and
reproducible; the shared imagery and label chunks are reused either way.

Usage:
    modal run modal_pipeline.py --sites SERC               # one site, max 150 trees/patch
    modal run modal_pipeline.py --sites SERC,TEAK,SOAP     # several
    modal run modal_pipeline.py --sites SERC --max-trees 200
    modal run modal_pipeline.py --sites SERC --max-trees 0 # no density filter
    modal run modal_pipeline.py --max-trees 500 --out-tag patches_q500
    modal run modal_pipeline.py --sites SERC --dry-run     # plan only
"""
from __future__ import annotations

import re
from pathlib import Path

import modal

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
DEFAULT_SITES = ["BART, BLAN, BONA, CLBJ, DEJU, SJER, SOAP, TALL, TEAK, WREF, YELL"]
YEAR = "2019"
VOLUME_NAME = "mot"
APP_NAME = "momrtalitree-data-preprocessing"

# Weinstein 2020 prediction CSVs in this Zenodo record, named {SITE}_{YEAR}.csv.
ZENODO_CSV_URL = "https://zenodo.org/records/3765872/files/{name}?download=1"
HSI_DPID = "DP3.30006.001"   # NEON spectrometer product (for acquisition month)

TARGET_GSD = 0.6                              # m / px
PATCH_PX = 256                                # tile side in pixels
PATCH_M = PATCH_PX * TARGET_GSD               # 153.6 m
SRC_TILE_M = 1000                             # NEON 1 km tile
PATCHES_PER_SIDE = int(SRC_TILE_M // PATCH_M) # 6

# Quartering: each 256 parent cell is split into N_SPLIT x N_SPLIT sub-tiles,
# aligned to the parent grid so quarters nest exactly into a parent (2 x 128 px
# = 256 px). Keeps trees/tile within the query budget AND recovers dense-forest
# patches a per-256 max_trees filter would drop. See clay/experiment.md.
N_SPLIT = 2
QUARTER_PX = PATCH_PX // N_SPLIT              # 128
QUARTER_M = PATCH_M / N_SPLIT                 # 76.8 m

NIR_LO, NIR_HI = 835.0, 920.0
RED_LO, RED_HI = 620.0, 700.0
NDVI_DEAD = 0.6

# NEON 2019 AOP flight month per site (used for Clay's `time` metadata).
# Update if you add a new site or a different year.
SITE_ACQUISITION_YYYYMM = {
    "TEAK": "2019-06",
    "SOAP": "2019-06",
    "YELL": "2019-07",
    "BART": "2019-07",
    "BLAN": "2019-07",
    "BONA": "2019-07",
    "CLBJ": "2019-07",
    "DEJU": "2019-07",
    "SJER": "2019-07",
    "TALL": "2019-07",
    "WREF": "2019-07"
}

UTM_RE = re.compile(r"_(\d{6})_(\d{7})_")
HERE = Path(__file__).parent

# --------------------------------------------------------------------------- #
# Modal setup
# --------------------------------------------------------------------------- #
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "h5py", "numpy", "pandas", "pyproj", "rasterio", "scipy", "rich", "tqdm",
    "requests",
)
app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


# --------------------------------------------------------------------------- #
# Per-tile worker
# --------------------------------------------------------------------------- #
@app.function(
    image=image,
    volumes={"/data": vol},
    timeout=3600,
    retries=2,
    max_containers=32,
)
def process_tile(
    site: str, year: str, month: str, east: int, north: int,
    max_trees: int = 0, out_root: str = "/data/patches", quarter: bool = False,
) -> dict:
    """Process one 1 km source tile, write patches + label chunk to the volume,
    return a small ack dict (so we never trigger Modal's blob result path).

    Only patches with 1..max_trees alive boxes are written (max_trees=0 disables
    the upper bound). The count is the post-NDVI boxes whose centers fall in the
    patch — the same quantity the density scout reports — so a patch over the
    detector's query budget is dropped before any pixel work is done.

    quarter=True: split each 256 parent cell into N_SPLIT x N_SPLIT sub-tiles
    (128 px) aligned to the parent grid; apply the max_trees budget per *quarter*
    and write 128-px tifs + two label chunks: `quarter_{geo}.csv` (per-quarter
    boxes, 128 frame, with parent/q_row/q_col) for training, and `parent_{geo}.csv`
    (full unclipped 256-frame GT per parent) for stitched parent-level scoring.
    """
    import h5py
    import numpy as np
    import pandas as pd
    import rasterio
    from pyproj import Transformer
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from scipy.ndimage import zoom

    ack = {
        "site": site, "geo": f"{east}_{north}",
        "n_patches": 0, "n_boxes": 0,
        "dropped_patches": 0, "dropped_boxes": 0, "status": "ok",
    }

    rgb_glob = list(Path(f"/data/{site}/rgb").glob(
        f"{year}_{site}_*_{east}_{north}_image.tif"
    ))
    hsi_glob = list(Path(f"/data/{site}/hyperspectral").glob(
        f"NEON_*_{site}_DP3_{east}_{north}_reflectance.h5"
    ))
    if not rgb_glob or not hsi_glob:
        ack["status"] = "missing_pair"; return ack
    rgb_path, hsi_path = rgb_glob[0], hsi_glob[0]

    out_dir = Path(out_root) / site
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir = Path(out_root) / "_chunks" / site
    chunk_dir.mkdir(parents=True, exist_ok=True)

    # --- Load HSI: mean NIR and NDVI at 1 m ----------------------------------
    with h5py.File(hsi_path, "r") as f:
        site_key = list(f.keys())[0]
        refl = f[site_key]["Reflectance"]
        waves = refl["Metadata/Spectral_Data/Wavelength"][:]
        map_info = refl["Metadata/Coordinate_System/Map_Info"][()].decode()
        data = refl["Reflectance_Data"]
        scale = float(data.attrs.get("Scale_Factor", 10000.0))
        nodata = float(data.attrs.get("Data_Ignore_Value", -9999.0))

        def _slice(lo, hi):
            idx = np.where((waves >= lo) & (waves <= hi))[0]
            return slice(int(idx[0]), int(idx[-1]) + 1)

        nir_stack = data[:, :, _slice(NIR_LO, NIR_HI)].astype(np.float32)
        red_stack = data[:, :, _slice(RED_LO, RED_HI)].astype(np.float32)

    nir_stack[nir_stack == nodata] = np.nan
    red_stack[red_stack == nodata] = np.nan
    # All-NaN slices (nodata pixels on partial/edge tiles) make nanmean warn
    # "Mean of empty slice" and return NaN — which is exactly what we want
    # (NaN -> dropped by NDVI checks / zeroed in the NIR patch). Silence the noise.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        nir = np.nanmean(nir_stack, axis=2) / scale
        red = np.nanmean(red_stack, axis=2) / scale
    ndvi = (nir - red) / (nir + red + 1e-8)

    parts = [p.strip() for p in map_info.split(",")]
    x0_hsi, y0_hsi, px_hsi = float(parts[3]), float(parts[4]), float(parts[5])
    H, W = ndvi.shape

    # --- Per-box NDVI and alive filter ---------------------------------------
    # Boxes for this tile come from a per-tile chunk CSV written by prepare_labels
    # (columns: left, bottom, right, top, score), so workers read only MB, not
    # the full multi-GB site CSV.
    chunk_path = Path(f"/data/labels/{site}/{east}_{north}.csv")
    if not chunk_path.exists():
        ack["status"] = "no_labels"; return ack
    labels = pd.read_csv(chunk_path)
    if labels.empty:
        ack["status"] = "no_labels"; return ack
    box_ndvi = np.full(len(labels), np.nan, dtype=np.float32)
    for i, r in enumerate(labels.itertuples(index=False)):
        c0f = (r.left  - x0_hsi) / px_hsi
        c1f = (r.right - x0_hsi) / px_hsi
        r0f = (y0_hsi - r.top)    / px_hsi
        r1f = (y0_hsi - r.bottom) / px_hsi
        cmn = max(0, int(np.floor(min(c0f, c1f))))
        cmx = min(W, int(np.ceil(max(c0f, c1f))))
        rmn = max(0, int(np.floor(min(r0f, r1f))))
        rmx = min(H, int(np.ceil(max(r0f, r1f))))
        if cmx > cmn and rmx > rmn:
            patch = ndvi[rmn:rmx, cmn:cmx]
            if np.isfinite(patch).any():
                box_ndvi[i] = float(np.nanmean(patch))
    labels["ndvi"] = box_ndvi
    alive = labels[labels["ndvi"] >= NDVI_DEAD].reset_index(drop=True)
    if alive.empty:
        ack["status"] = "no_alive"; return ack

    cx_box = (alive["left"]  + alive["right"]).to_numpy() / 2
    cy_box = (alive["bottom"] + alive["top"]).to_numpy()  / 2

    src_left = east
    src_top  = north + SRC_TILE_M
    geo = f"{east}_{north}"
    n_written = 0
    n_dropped = 0          # tiles skipped for exceeding max_trees
    boxes_dropped = 0      # boxes living in those dropped tiles

    with rasterio.open(rgb_path) as rgb_src:
        crs = rgb_src.crs
        # UTM -> WGS84 lat/lon for Clay's metadata. Built once per tile.
        to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

        def render(left, top, right, bottom, out_px):
            """4-band (R,G,B,NIR) uint8 array for a UTM bbox: RGB area-averaged
            from 0.1 m, NIR bilinear-upsampled from 1 m. boundless + offset-padded
            so edge-tile windows aren't stretched (see notes below)."""
            win = rasterio.windows.from_bounds(left, bottom, right, top, rgb_src.transform)
            rgb = rgb_src.read(
                [1, 2, 3], window=win, out_shape=(3, out_px, out_px),
                resampling=Resampling.average, boundless=True, fill_value=0,
            )
            c0f = (left  - x0_hsi) / px_hsi
            c1f = (right - x0_hsi) / px_hsi
            r0f = (y0_hsi - top)    / px_hsi
            r1f = (y0_hsi - bottom) / px_hsi
            cmn = int(np.floor(min(c0f, c1f))); cmx = int(np.ceil(max(c0f, c1f)))
            rmn = int(np.floor(min(r0f, r1f))); rmx = int(np.ceil(max(r0f, r1f)))
            Hh, Ww = nir.shape
            full = np.zeros((max(1, rmx - rmn), max(1, cmx - cmn)), dtype=np.float32)
            cs, ce = max(0, cmn), min(Ww, cmx)
            rs, re2 = max(0, rmn), min(Hh, rmx)
            if ce > cs and re2 > rs:
                full[rs - rmn:re2 - rmn, cs - cmn:ce - cmn] = np.nan_to_num(
                    nir[rs:re2, cs:ce], nan=0.0)
            nir_up = zoom(full, zoom=(out_px / full.shape[0], out_px / full.shape[1]), order=1)
            nir_q = np.clip(nir_up * 255.0, 0, 255).astype(np.uint8)
            return np.stack([rgb[0], rgb[1], rgb[2], nir_q], axis=0)

        def write_tif(arr, left, top, out_px, path):
            tr = from_origin(left, top, TARGET_GSD, TARGET_GSD)
            with rasterio.open(
                path, "w", driver="GTiff", height=out_px, width=out_px, count=4,
                dtype=arr.dtype, crs=crs, transform=tr, compress="deflate",
            ) as dst:
                dst.write(arr)

        def box_rows(mask, ref_left, ref_top, out_px, head, tail):
            """Pixel-frame box dicts for alive boxes selected by `mask`, clamped to
            the [0, out_px] tile and dropping degenerate (<1 px) boxes. `head`/`tail`
            wrap the box fields so column order stays site,imgname,...,box,...,meta."""
            out = []
            for s in alive.iloc[mask].itertuples(index=False):
                xmin = (s.left  - ref_left) / TARGET_GSD
                xmax = (s.right - ref_left) / TARGET_GSD
                ymin = (ref_top - s.top)    / TARGET_GSD
                ymax = (ref_top - s.bottom) / TARGET_GSD
                xmin = max(0.0, xmin); ymin = max(0.0, ymin)
                xmax = min(float(out_px), xmax); ymax = min(float(out_px), ymax)
                if xmax - xmin < 1.0 or ymax - ymin < 1.0:
                    continue
                out.append({
                    **head,
                    "xmin": round(xmin, 2), "ymin": round(ymin, 2),
                    "xmax": round(xmax, 2), "ymax": round(ymax, 2),
                    "score": float(s.score), "ndvi": float(s.ndvi),
                    **tail,
                })
            return out

        def centers_in(left, right, top, bottom):
            return (
                (cx_box >= left) & (cx_box < right) &
                (cy_box > bottom) & (cy_box <= top)
            )

        if not quarter:
            # --- 256 patches (baseline) ---------------------------------------
            rows: list[dict] = []
            for r in range(PATCHES_PER_SIDE):
                for c in range(PATCHES_PER_SIDE):
                    pl = src_left + c * PATCH_M;  pt = src_top - r * PATCH_M
                    pr = pl + PATCH_M;            pb = pt - PATCH_M
                    inside = centers_in(pl, pr, pt, pb)
                    n_inside = int(inside.sum())
                    if n_inside == 0:
                        continue
                    if max_trees and n_inside > max_trees:
                        n_dropped += 1; boxes_dropped += n_inside; continue
                    lon, lat = to_wgs84.transform(pl + PATCH_M / 2, pt - PATCH_M / 2)
                    imgname = f"{site}_{year}_{east}_{north}_r{r}_c{c}"
                    write_tif(render(pl, pt, pr, pb, PATCH_PX), pl, pt, PATCH_PX,
                              out_dir / f"{imgname}.tif")
                    n_written += 1
                    rows.extend(box_rows(
                        inside, pl, pt, PATCH_PX,
                        head={"site": site, "imgname": imgname},
                        tail={"lat": round(float(lat), 6), "lon": round(float(lon), 6),
                              "acquisition_yyyymm": month,
                              "patch_left_utm": pl, "patch_top_utm": pt,
                              "source_geo_index": geo},
                    ))
            if rows:
                pd.DataFrame(rows).to_csv(chunk_dir / f"labels_{geo}.csv", index=False)
            ack["n_boxes"] = len(rows)
        else:
            # --- 128 quarters, 2x2 aligned to each 256 parent cell ------------
            q_rows: list[dict] = []
            p_rows: list[dict] = []   # full 256-frame GT per parent (for stitched eval)
            for r in range(PATCHES_PER_SIDE):
                for c in range(PATCHES_PER_SIDE):
                    pl = src_left + c * PATCH_M;  pt = src_top - r * PATCH_M
                    pr = pl + PATCH_M;            pb = pt - PATCH_M
                    p_inside = centers_in(pl, pr, pt, pb)
                    if int(p_inside.sum()) == 0:
                        continue
                    parent = f"{site}_{year}_{east}_{north}_r{r}_c{c}"
                    # full parent GT — all boxes in the 256 cell, regardless of the
                    # per-quarter cap, so dropped-quarter trees still count at eval.
                    p_rows.extend(box_rows(
                        p_inside, pl, pt, PATCH_PX,
                        head={"site": site, "parent": parent},
                        tail={"source_geo_index": geo},
                    ))
                    for qr in range(N_SPLIT):
                        for qc in range(N_SPLIT):
                            ql = pl + qc * QUARTER_M;  qt = pt - qr * QUARTER_M
                            qrr = ql + QUARTER_M;      qb = qt - QUARTER_M
                            q_inside = centers_in(ql, qrr, qt, qb)
                            n_q = int(q_inside.sum())
                            if n_q == 0:
                                continue
                            if max_trees and n_q > max_trees:
                                n_dropped += 1; boxes_dropped += n_q; continue
                            lon, lat = to_wgs84.transform(ql + QUARTER_M / 2, qt - QUARTER_M / 2)
                            imgname = f"{parent}_q{qr}{qc}"
                            write_tif(render(ql, qt, qrr, qb, QUARTER_PX), ql, qt,
                                      QUARTER_PX, out_dir / f"{imgname}.tif")
                            n_written += 1
                            q_rows.extend(box_rows(
                                q_inside, ql, qt, QUARTER_PX,
                                head={"site": site, "imgname": imgname,
                                      "parent": parent, "q_row": qr, "q_col": qc},
                                tail={"lat": round(float(lat), 6), "lon": round(float(lon), 6),
                                      "acquisition_yyyymm": month,
                                      "patch_left_utm": ql, "patch_top_utm": qt,
                                      "source_geo_index": geo},
                            ))
            if q_rows:
                pd.DataFrame(q_rows).to_csv(chunk_dir / f"quarter_{geo}.csv", index=False)
            if p_rows:
                pd.DataFrame(p_rows).to_csv(chunk_dir / f"parent_{geo}.csv", index=False)
            ack["n_boxes"] = len(q_rows)

    ack["n_patches"] = n_written
    ack["dropped_patches"] = n_dropped
    ack["dropped_boxes"] = boxes_dropped
    return ack


def _concat_chunks(chunks_root, glob_pat: str, out_dir, out_name: str,
                   per_site_name) -> dict:
    """Concat per-tile chunk CSVs matching `glob_pat` under each site dir into
    one combined CSV (`out_name`) and per-site CSVs, returning row counts.

    Streams one chunk at a time, appending to the per-site and combined CSVs on
    disk — never holds the full table in memory. labels_parent.csv is the full
    unfiltered 256 GT (tens of millions of rows), so an in-memory pd.concat of
    everything OOMs; this does not."""
    import pandas as pd

    summary: dict[str, int] = {}
    combined_path = out_dir / out_name
    if combined_path.exists():
        combined_path.unlink()           # fresh combined file (we append below)
    combined_header = True
    total = 0
    for site_dir in sorted(chunks_root.iterdir()):
        if not site_dir.is_dir():
            continue
        chunk_paths = sorted(site_dir.glob(glob_pat))
        if not chunk_paths:
            continue
        site_path = out_dir / per_site_name(site_dir.name)
        if site_path.exists():
            site_path.unlink()
        site_header = True
        site_total = 0
        for p in chunk_paths:
            df = pd.read_csv(p)
            if df.empty:
                continue
            df.to_csv(site_path, mode="a", index=False, header=site_header)
            df.to_csv(combined_path, mode="a", index=False, header=combined_header)
            site_header = False
            combined_header = False
            n = len(df)
            site_total += n
            total += n
        if site_total:
            summary[site_dir.name] = site_total
    if total:
        summary["__combined__"] = total
    return summary


@app.function(image=image, volumes={"/data": vol}, timeout=6000, memory=16384)
def write_csvs(out_root: str = "/data/patches", quarter: bool = False) -> dict:
    """Concatenate per-tile chunk CSVs under {out_root}/_chunks/{SITE}/.

    Baseline: labels_*.csv -> labels_{SITE}.csv + labels.csv.
    Quarter:  quarter_*.csv -> labels_quarter_{SITE}.csv + labels_quarter.csv
              parent_*.csv  -> labels_parent_{SITE}.csv  + labels_parent.csv
    """
    out_dir = Path(out_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks_root = out_dir / "_chunks"
    if not chunks_root.exists():
        return {}

    if not quarter:
        return _concat_chunks(chunks_root, "labels_*.csv", out_dir, "labels.csv",
                              lambda s: f"labels_{s}.csv")

    summary: dict[str, int] = {}
    q = _concat_chunks(chunks_root, "quarter_*.csv", out_dir, "labels_quarter.csv",
                       lambda s: f"labels_quarter_{s}.csv")
    p = _concat_chunks(chunks_root, "parent_*.csv", out_dir, "labels_parent.csv",
                       lambda s: f"labels_parent_{s}.csv")
    summary["quarter_boxes"] = q.get("__combined__", 0)
    summary["parent_boxes"] = p.get("__combined__", 0)
    return summary


@app.function(image=image, volumes={"/data": vol}, timeout=14400, memory=16384,
              cpu=8.0, max_containers=16, retries=2)
def _pack_site(out_root: str, site: str) -> dict:
    """Pack one site's quarter tifs into {out_root}/packed/{SITE}.npy
    (memmapped (N,4,128,128) uint8) + {SITE}_index.csv (imgname,row). Threaded
    reads (I/O-bound). Resumable: skips if {SITE}.done already exists."""
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import pandas as pd
    import rasterio
    from numpy.lib.format import open_memmap

    out_dir = Path(out_root)
    packed_dir = out_dir / "packed"
    packed_dir.mkdir(parents=True, exist_ok=True)
    npy_path = packed_dir / f"{site}.npy"
    done = packed_dir / f"{site}.done"
    idx_path = packed_dir / f"{site}_index.csv"

    # Tile list: prefer the small per-site CSV; fall back to filtering the combined.
    per_site = out_dir / f"labels_quarter_{site}.csv"
    if per_site.exists():
        names = sorted(pd.read_csv(per_site, usecols=["imgname"])["imgname"].unique().tolist())
    else:
        df = pd.read_csv(out_dir / "labels_quarter.csv", usecols=["site", "imgname"])
        names = sorted(df[df["site"] == site]["imgname"].unique().tolist())
    N = len(names)

    if done.exists() and npy_path.exists() and idx_path.exists():
        return {"site": site, "n": N, "status": "skip"}

    arr = open_memmap(npy_path, mode="w+", dtype=np.uint8,
                      shape=(N, 4, QUARTER_PX, QUARTER_PX))

    def _load(i_nm):
        i, nm = i_nm
        with rasterio.open(out_dir / site / f"{nm}.tif") as src:
            arr[i] = src.read(indexes=[1, 2, 3, 4])

    with ThreadPoolExecutor(max_workers=16) as ex:
        for _ in ex.map(_load, enumerate(names)):
            pass
    arr.flush(); del arr
    pd.DataFrame({"imgname": names, "row": range(N)}).to_csv(idx_path, index=False)
    done.touch()
    vol.commit()
    return {"site": site, "n": N, "status": "packed"}


@app.function(image=image, volumes={"/data": vol}, timeout=14400, memory=8192)
def pack_quarters(out_root: str = "/data/patches_quarter") -> dict:
    """Pack the per-quarter 128-px tifs into one memmapped uint8 array per site
    ({out_root}/packed/{SITE}.npy) + packed_index.csv (imgname,site,row), so
    training reads memmap rows instead of opening 100k+ GeoTIFFs.

    Fans out one container per site (parallel), reads only the *existing*
    labels_quarter*.csv + tifs (no re-gen, no re-CSV), and is resumable: sites
    with a `{SITE}.done` marker are skipped.
    """
    import pandas as pd

    out_dir = Path(out_root)
    packed_dir = out_dir / "packed"
    packed_dir.mkdir(parents=True, exist_ok=True)

    # Site list from the per-site quarter CSVs (fall back to the combined one).
    site_files = sorted(out_dir.glob("labels_quarter_*.csv"))
    if site_files:
        sites = [p.name[len("labels_quarter_"):-len(".csv")] for p in site_files]
    elif (out_dir / "labels_quarter.csv").exists():
        sites = sorted(pd.read_csv(out_dir / "labels_quarter.csv",
                                   usecols=["site"])["site"].unique().tolist())
    else:
        return {"error": "no labels_quarter*.csv found — run the pipeline first"}

    summary: dict[str, int] = {}
    for ack in _pack_site.starmap([(out_root, s) for s in sites]):
        summary[ack["site"]] = ack["n"]
        print(f"[pack] {ack['site']}: {ack['status']} ({ack['n']} tiles)", flush=True)

    # Stitch per-site indices into one packed_index.csv (imgname, site, row).
    frames = []
    for s in sites:
        ip = packed_dir / f"{s}_index.csv"
        if ip.exists():
            d = pd.read_csv(ip)
            d["site"] = s
            frames.append(d[["imgname", "site", "row"]])
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined.to_csv(packed_dir / "packed_index.csv", index=False)
        summary["__index_rows__"] = len(combined)
    vol.commit()
    return summary


@app.function(image=image, volumes={"/data": vol}, timeout=7200, memory=16384)
def prepare_labels(site: str, year: str) -> list[str]:
    """Fetch {SITE}_{YEAR}.csv from Zenodo to the volume (skip if present) and
    split it into per-tile chunk CSVs at /data/labels/{SITE}/{geo}.csv, so each
    process_tile worker reads only its tile's boxes instead of the whole
    (multi-GB) site CSV. Returns the unique geo_index list.

    Runs server-side, so the large CSV never has to be downloaded locally.
    """
    import pandas as pd
    import requests

    name = f"{site}_{year}.csv"
    dest = Path("/data/labels") / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        url = ZENODO_CSV_URL.format(name=name)
        tmp = dest.with_suffix(".part")
        with requests.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
        tmp.rename(dest)

    chunk_dir = Path(f"/data/labels/{site}")
    chunk_dir.mkdir(parents=True, exist_ok=True)
    done = chunk_dir / "_chunks_done"
    if done.exists():
        return sorted(p.stem for p in chunk_dir.glob("*.csv"))

    df = pd.read_csv(dest, usecols=["left", "bottom", "right", "top", "score", "geo_index"])
    df["geo_index"] = df["geo_index"].astype(str)
    for geo, sub in df.groupby("geo_index"):
        sub[["left", "bottom", "right", "top", "score"]].to_csv(
            chunk_dir / f"{geo}.csv", index=False,
        )
    done.touch()
    vol.commit()
    return sorted(df["geo_index"].unique().tolist())


def _acquisition_month(site: str, year: str) -> str:
    """Acquisition YYYY-MM for Clay's `time` metadata: a known override if we
    have one, else look it up from the NEON API, else a mid-summer fallback."""
    if site in SITE_ACQUISITION_YYYYMM:
        return SITE_ACQUISITION_YYYYMM[site]
    try:
        import requests
        r = requests.get(f"https://data.neonscience.org/api/v0/products/{HSI_DPID}", timeout=30)
        r.raise_for_status()
        for s in r.json()["data"]["siteCodes"]:
            if s["siteCode"] == site:
                months = sorted(m for m in s["availableMonths"] if m.startswith(year))
                if months:
                    return months[0]
    except Exception:
        pass
    return f"{year}-07"


# --------------------------------------------------------------------------- #
# Local entrypoint
# --------------------------------------------------------------------------- #
@app.local_entrypoint()
def main(
    sites: str = "BART, BLAN, BONA, CLBJ, DEJU, SJER, SOAP, TALL, TEAK, WREF, YELL",
    year: str = "2019",
    dry_run: bool = False,
    max_trees: int = 150,
    out_tag: str = "patches",
    quarter: bool = False,
    pack: bool = False,
):
    """Process every site in --sites (comma-separated NEON codes). Each site's
    CSV is fetched from Zenodo to the volume and chunked server-side, so nothing
    is read locally.

    max_trees: drop patches with more than this many alive boxes (0 = keep all).
    Set it to your detector's num_queries so no patch exceeds the query budget.

    out_tag: output folder name under /data on the volume (default "patches").
    Use a distinct tag per config (e.g. patches_q500) so different thresholds /
    site sets don't overwrite each other; shared imagery + label chunks are
    reused regardless. Point training at /neon/{out_tag}/labels.csv."""
    from rich.console import Console
    from rich.progress import (
        BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
        TextColumn, TimeElapsedColumn, TimeRemainingColumn,
    )
    from rich.table import Table

    console = Console()

    # --- Pack-only: skip gen/CSV, just memmap the existing quarter tifs --------
    if pack:
        if quarter and out_tag == "patches":
            out_tag = "patches_quarter"
        out_root = f"/data/{out_tag}"
        console.rule(f"[bold cyan]Packing quarter tifs in {out_root} -> per-site memmaps")
        summary = pack_quarters.remote(out_root)
        tbl = Table(show_header=True, header_style="bold")
        tbl.add_column("site"); tbl.add_column("tiles")
        for k, v in summary.items():
            tbl.add_row(k, str(v))
        console.print(tbl)
        console.rule("[bold green]pack done")
        return

    site_list = [s.strip().upper() for s in sites.split(",") if s.strip()]
    if not site_list:
        console.print("[red]pass --sites SITE1,SITE2,...  "
                      "(CSVs are auto-fetched from Zenodo to the volume)[/red]")
        return

    def _vol_utms(subdir: str, suffix: str) -> set:
        out: set = set()
        try:
            for entry in vol.iterdir(subdir):
                if not entry.path.endswith(suffix):
                    continue
                m = UTM_RE.search(entry.path.split("/")[-1])
                if m:
                    out.add(f"{m.group(1)}_{m.group(2)}")
        except Exception:
            pass   # subdir not on the volume yet (site not downloaded)
        return out

    # --- Plan: intersect annotated tiles (CSV) with paired tiles (volume) ---
    if quarter and out_tag == "patches":
        out_tag = "patches_quarter"     # don't clobber the 256 baseline output
    out_root = f"/data/{out_tag}"
    unit = "quarter" if quarter else "patch"
    filt = f"max {max_trees} trees/{unit}" if max_trees else "no density filter"
    mode = f"2x2 quarters ({QUARTER_PX}px)" if quarter else f"{PATCH_PX}px patches"
    console.rule(f"[bold cyan]Patch pipeline for {site_list}  [{mode}, {filt}] -> {out_root}")
    jobs: list[tuple] = []
    plan = Table(show_header=True, header_style="bold")
    plan.add_column("site"); plan.add_column("annotated")
    plan.add_column("paired on vol"); plan.add_column("→ tiles")

    for site in site_list:
        console.print(f"[cyan]{site}[/cyan]: fetching CSV + chunking labels on volume ...")
        annotated = set(prepare_labels.remote(site, year))
        month = _acquisition_month(site, year)

        vol_paired = _vol_utms(f"{site}/rgb", ".tif") & _vol_utms(f"{site}/hyperspectral", ".h5")
        target_geo = annotated & vol_paired
        target_utms = sorted({tuple(g.split("_")) for g in target_geo})

        for east, north in target_utms:
            jobs.append((site, year, month, int(east), int(north), max_trees, out_root, quarter))
        plan.add_row(site, str(len(annotated)), str(len(vol_paired)),
                     str(len(target_utms)))

    plan.add_row("[bold]TOTAL", "", "", f"[bold]{len(jobs)}")
    console.print(plan)

    if not jobs:
        console.print("[red]nothing to process[/red]"); return
    if dry_run:
        console.print("[yellow]--dry-run set, exiting[/yellow]"); return

    # --- Fan out across containers ------------------------------------------
    console.rule(f"[bold cyan]Processing {len(jobs)} source tiles")
    acks: list[dict] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("eta"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("tiles", total=len(jobs))
        for ack in process_tile.starmap(jobs, order_outputs=False):
            acks.append(ack)
            progress.update(task, advance=1)

    kept_patches  = sum(a["n_patches"] for a in acks)
    kept_boxes    = sum(a["n_boxes"] for a in acks)
    drop_patches  = sum(a["dropped_patches"] for a in acks)
    drop_boxes    = sum(a["dropped_boxes"] for a in acks)
    by_status: dict[str, int] = {}
    for a in acks:
        by_status[a["status"]] = by_status.get(a["status"], 0) + 1

    tot_patches = kept_patches + drop_patches
    tot_boxes   = kept_boxes + drop_boxes
    pct_p = 100 * kept_patches / tot_patches if tot_patches else 0.0
    pct_b = 100 * kept_boxes / tot_boxes if tot_boxes else 0.0
    cap = f"max {max_trees} trees/{unit}" if max_trees else "no filter"
    console.rule(f"[bold cyan]Kept after filtering ({cap})")
    res = Table(show_header=True, header_style="bold")
    res.add_column(""); res.add_column("kept", justify="right")
    res.add_column("dropped", justify="right"); res.add_column("kept %", justify="right")
    res.add_row(f"{unit}s", f"{kept_patches:,}", f"{drop_patches:,}", f"{pct_p:.1f}%")
    res.add_row("annotations", f"{kept_boxes:,}", f"{drop_boxes:,}", f"{pct_b:.1f}%")
    console.print(res)
    console.print(f"[dim]{len(acks)} tile workers, status: {by_status}[/dim]")

    # --- Concat per-tile chunks into per-site + combined CSVs on the volume --
    console.rule("[bold cyan]Writing CSVs")
    summary = write_csvs.remote(out_root, quarter)
    tbl = Table(show_header=True, header_style="bold")
    tbl.add_column("site"); tbl.add_column("rows")
    for k, v in summary.items():
        tbl.add_row(k, str(v))
    console.print(tbl)
    console.rule("[bold green]done")
