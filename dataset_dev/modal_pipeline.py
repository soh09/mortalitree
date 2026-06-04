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

Usage:
    modal run modal_pipeline.py --sites SERC               # one site, max 150 trees/patch
    modal run modal_pipeline.py --sites SERC,TEAK,SOAP     # several
    modal run modal_pipeline.py --sites SERC --max-trees 200
    modal run modal_pipeline.py --sites SERC --max-trees 0 # no density filter
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
    max_trees: int = 0,
) -> dict:
    """Process one 1 km source tile, write patches + label chunk to the volume,
    return a small ack dict (so we never trigger Modal's blob result path).

    Only patches with 1..max_trees alive boxes are written (max_trees=0 disables
    the upper bound). The count is the post-NDVI boxes whose centers fall in the
    patch — the same quantity the density scout reports — so a patch over the
    detector's query budget is dropped before any pixel work is done.
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

    out_dir = Path(f"/data/patches/{site}")
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir = Path(f"/data/patches/_chunks/{site}")
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
    rows: list[dict] = []
    n_written = 0
    n_dropped = 0          # patches skipped for exceeding max_trees
    boxes_dropped = 0      # boxes living in those dropped patches

    with rasterio.open(rgb_path) as rgb_src:
        crs = rgb_src.crs
        # UTM -> WGS84 lat/lon for Clay's metadata. Built once per tile.
        to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        for r in range(PATCHES_PER_SIDE):
            for c in range(PATCHES_PER_SIDE):
                patch_left   = src_left + c * PATCH_M
                patch_top    = src_top  - r * PATCH_M
                patch_right  = patch_left + PATCH_M
                patch_bottom = patch_top  - PATCH_M

                # --- count boxes whose centers fall inside this patch -------
                # Done first so over-dense / empty patches are skipped before any
                # expensive pixel work (RGB read, NIR upsample, tif write).
                inside = (
                    (cx_box >= patch_left)  & (cx_box < patch_right) &
                    (cy_box >  patch_bottom) & (cy_box <= patch_top)
                )
                n_inside = int(inside.sum())
                if n_inside == 0:
                    continue
                if max_trees and n_inside > max_trees:
                    n_dropped += 1
                    boxes_dropped += n_inside
                    continue

                patch_cx_utm = patch_left + PATCH_M / 2
                patch_cy_utm = patch_top  - PATCH_M / 2
                lon, lat = to_wgs84.transform(patch_cx_utm, patch_cy_utm)

                # --- RGB area-average from 0.1 m to 0.6 m --------------------
                cl, rt = (~rgb_src.transform) * (patch_left, patch_top)
                cr, rb = (~rgb_src.transform) * (patch_right, patch_bottom)
                cc0, cc1 = int(min(cl, cr)), int(max(cl, cr))
                rr0, rr1 = int(min(rt, rb)), int(max(rt, rb))
                win = rasterio.windows.Window(cc0, rr0, cc1 - cc0, rr1 - rr0)
                rgb = rgb_src.read(
                    [1, 2, 3], window=win,
                    out_shape=(3, PATCH_PX, PATCH_PX),
                    resampling=Resampling.average,
                )

                # --- HSI NIR crop + bilinear upsample to 0.6 m ---------------
                c0f = (patch_left  - x0_hsi) / px_hsi
                c1f = (patch_right - x0_hsi) / px_hsi
                r0f = (y0_hsi - patch_top)   / px_hsi
                r1f = (y0_hsi - patch_bottom) / px_hsi
                cmn = int(np.floor(min(c0f, c1f)))
                cmx = int(np.ceil(max(c0f, c1f)))
                rmn = int(np.floor(min(r0f, r1f)))
                rmx = int(np.ceil(max(r0f, r1f)))
                nir_crop = nir[rmn:rmx, cmn:cmx]
                nir_up = zoom(
                    nir_crop,
                    zoom=(PATCH_PX / nir_crop.shape[0],
                          PATCH_PX / nir_crop.shape[1]),
                    order=1,
                )
                nir_q = np.clip(nir_up * 255.0, 0, 255).astype(np.uint8)

                # R, G, B, NIR — matches Clay NAIP order
                patch = np.stack([rgb[0], rgb[1], rgb[2], nir_q], axis=0)

                imgname = f"{site}_{year}_{east}_{north}_r{r}_c{c}"
                tr = from_origin(patch_left, patch_top, TARGET_GSD, TARGET_GSD)
                with rasterio.open(
                    out_dir / f"{imgname}.tif", "w",
                    driver="GTiff", height=PATCH_PX, width=PATCH_PX, count=4,
                    dtype=patch.dtype, crs=crs, transform=tr,
                    compress="deflate",
                ) as dst:
                    dst.write(patch)

                # --- write box labels for this (kept) patch ----------------
                n_written += 1
                sub = alive.iloc[inside]
                for s in sub.itertuples(index=False):
                    xmin = (s.left  - patch_left) / TARGET_GSD
                    xmax = (s.right - patch_left) / TARGET_GSD
                    ymin = (patch_top - s.top)    / TARGET_GSD
                    ymax = (patch_top - s.bottom) / TARGET_GSD
                    xmin = max(0.0, xmin); ymin = max(0.0, ymin)
                    xmax = min(float(PATCH_PX), xmax); ymax = min(float(PATCH_PX), ymax)
                    if xmax - xmin < 1.0 or ymax - ymin < 1.0:
                        continue
                    rows.append({
                        "site": site,
                        "imgname": imgname,
                        "xmin": round(xmin, 2),
                        "ymin": round(ymin, 2),
                        "xmax": round(xmax, 2),
                        "ymax": round(ymax, 2),
                        "score": float(s.score),
                        "ndvi": float(s.ndvi),
                        "lat": round(float(lat), 6),
                        "lon": round(float(lon), 6),
                        "acquisition_yyyymm": month,
                        "patch_left_utm": patch_left,
                        "patch_top_utm": patch_top,
                        "source_geo_index": f"{east}_{north}",
                    })

    if rows:
        pd.DataFrame(rows).to_csv(
            chunk_dir / f"labels_{east}_{north}.csv", index=False,
        )
    ack["n_patches"] = n_written
    ack["n_boxes"] = len(rows)
    ack["dropped_patches"] = n_dropped
    ack["dropped_boxes"] = boxes_dropped
    return ack


@app.function(image=image, volumes={"/data": vol}, timeout=600)
def write_csvs() -> dict:
    """Read per-tile chunks under /data/patches/_chunks/{SITE}/labels_*.csv,
    concatenate into /data/patches/labels_{SITE}.csv and /data/patches/labels.csv.
    """
    import pandas as pd

    out_dir = Path("/data/patches")
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks_root = out_dir / "_chunks"
    if not chunks_root.exists():
        return {}

    summary: dict[str, int] = {}
    per_site_frames = []
    for site_dir in sorted(chunks_root.iterdir()):
        if not site_dir.is_dir():
            continue
        chunk_paths = sorted(site_dir.glob("labels_*.csv"))
        if not chunk_paths:
            continue
        site_df = pd.concat(
            [pd.read_csv(p) for p in chunk_paths], ignore_index=True,
        )
        site_df.to_csv(out_dir / f"labels_{site_dir.name}.csv", index=False)
        summary[site_dir.name] = len(site_df)
        per_site_frames.append(site_df)

    if per_site_frames:
        combined = pd.concat(per_site_frames, ignore_index=True)
        combined.to_csv(out_dir / "labels.csv", index=False)
        summary["__combined__"] = len(combined)
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
):
    """Process every site in --sites (comma-separated NEON codes). Each site's
    CSV is fetched from Zenodo to the volume and chunked server-side, so nothing
    is read locally.

    max_trees: drop patches with more than this many alive boxes (0 = keep all).
    Set it to your detector's num_queries so no patch exceeds the query budget."""
    from rich.console import Console
    from rich.progress import (
        BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
        TextColumn, TimeElapsedColumn, TimeRemainingColumn,
    )
    from rich.table import Table

    console = Console()
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
    filt = f"max {max_trees} trees/patch" if max_trees else "no density filter"
    console.rule(f"[bold cyan]Patch pipeline for {site_list}  ({filt})")
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
            jobs.append((site, year, month, int(east), int(north), max_trees))
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
    cap = f"max {max_trees} trees/patch" if max_trees else "no filter"
    console.rule(f"[bold cyan]Kept after filtering ({cap})")
    res = Table(show_header=True, header_style="bold")
    res.add_column(""); res.add_column("kept", justify="right")
    res.add_column("dropped", justify="right"); res.add_column("kept %", justify="right")
    res.add_row("patches", f"{kept_patches:,}", f"{drop_patches:,}", f"{pct_p:.1f}%")
    res.add_row("annotations", f"{kept_boxes:,}", f"{drop_boxes:,}", f"{pct_b:.1f}%")
    console.print(res)
    console.print(f"[dim]{len(acks)} tile workers, status: {by_status}[/dim]")

    # --- Concat per-tile chunks into per-site + combined CSVs on the volume --
    console.rule("[bold cyan]Writing CSVs")
    summary = write_csvs.remote()
    tbl = Table(show_header=True, header_style="bold")
    tbl.add_column("site"); tbl.add_column("rows")
    for k, v in summary.items():
        tbl.add_row(k, str(v))
    console.print(tbl)
    console.rule("[bold green]done")
