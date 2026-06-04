"""Download NEON paired (HSI, RGB) AOP tiles to a Modal volume.

Restricted to tiles that have Weinstein 2020 annotations (so we don't pull
hundreds of GB of pixels we have no labels for). The annotation CSVs are
fetched from Zenodo straight to the volume server-side, so they never have to
be downloaded locally.

Usage:
    modal run modal_neon_dl.py --sites SERC                 # one site
    modal run modal_neon_dl.py --sites SERC,TEAK,SOAP       # several
    modal run modal_neon_dl.py --sites SERC --dry-run       # just print the plan
    modal run modal_neon_dl.py --sites SERC --max-tiles 3   # smoke-test 3 tiles

Layout on the volume (name: mot):
    /labels/{SITE}_{YEAR}.csv
    /{SITE}/hyperspectral/NEON_*_reflectance.h5
    /{SITE}/rgb/{YEAR}_{SITE}_*_image.tif
"""
from __future__ import annotations

import re
from pathlib import Path

import modal

API = "https://data.neonscience.org/api/v0"
HSI_DPID = "DP3.30006.001"
RGB_DPID = "DP3.30010.001"
UTM_RE = re.compile(r"(\d{6})_(\d{7})")

# Weinstein 2020 prediction CSVs live in this Zenodo record as {SITE}_{YEAR}.csv.
ZENODO_CSV_URL = "https://zenodo.org/records/3765872/files/{name}?download=1"

VOLUME_NAME = "mot"
APP_NAME = "momrtalitree"

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "requests", "pandas", "rich", "tqdm"
)

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _list_site_months(dpid: str, year: str) -> dict[str, list[str]]:
    import requests

    r = requests.get(f"{API}/products/{dpid}", timeout=30)
    r.raise_for_status()
    out: dict[str, list[str]] = {}
    for s in r.json()["data"]["siteCodes"]:
        months = [m for m in s["availableMonths"] if m.startswith(year)]
        if months:
            out[s["siteCode"]] = months
    return out


def _list_tile_files(dpid: str, site: str, year_month: str) -> list[dict]:
    import requests

    r = requests.get(f"{API}/data/{dpid}/{site}/{year_month}", timeout=30)
    r.raise_for_status()
    out: list[dict] = []
    for f in r.json()["data"]["files"]:
        m = UTM_RE.search(f["name"])
        if not m:
            continue
        out.append({
            "name": f["name"],
            "url": f["url"],
            "size": f["size"],
            "utm": (m.group(1), m.group(2)),
        })
    return out


@app.function(
    image=image,
    volumes={"/data": vol},
    timeout=86400,
    retries=3,
    max_containers=32,
)
def download_one(url: str, dest_rel: str, expected_size: int) -> tuple[str, int, bool]:
    """Stream one file to /data/{dest_rel}. Returns (name, bytes, skipped)."""
    import requests

    dest = Path("/data") / dest_rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == expected_size:
        return dest.name, dest.stat().st_size, True
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    tmp.rename(dest)
    return dest.name, dest.stat().st_size, False


@app.function(image=image, volumes={"/data": vol}, timeout=3600, retries=2)
def fetch_csv(site: str, year: str) -> list[str]:
    """Download {SITE}_{YEAR}.csv from Zenodo to the volume (skip if present),
    and return the unique annotated geo_index values. Runs server-side so the
    (large) CSV never has to touch the local machine."""
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
    geo = pd.read_csv(dest, usecols=["geo_index"])["geo_index"].astype(str).unique()
    return sorted(geo.tolist())


@app.local_entrypoint()
def main(
    # sites: str = "BART, BLAN, BONA, CLBJ, DEJU, DELA, HARV, LENO, MLBS, NIWO, OSBS, RMNP, SERC, SJER, SOAP, TALL, TEAK, WREF, YELL",
    sites: str = "BART, BLAN, BONA, CLBJ, DEJU, SJER, SOAP, TALL, TEAK, WREF, YELL",
    year: str = "2019",
    dry_run: bool = False,
    max_tiles: int = 0,
):
    from rich.console import Console

    console = Console()
    site_list = [s.strip().upper() for s in sites.split(",") if s.strip()]
    if not site_list:
        console.print("[red]pass --sites SITE1,SITE2,...  "
                      "(CSVs are auto-fetched from Zenodo to the volume)[/red]")
        return
    for site in site_list:
        _download_site(console, site, year, dry_run, max_tiles)


def _download_site(console, site: str, year: str, dry_run: bool, max_tiles: int):
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.table import Table

    # ---- 1. Annotated tiles from CSV (fetched to volume from Zenodo) -------
    console.rule(f"[bold cyan]NEON {site} {year} — planning")
    console.print("fetching CSV from Zenodo to volume ...")
    annotated = set(fetch_csv.remote(site, year))
    console.print(f"  {len(annotated):>5} annotated tiles in CSV")

    # ---- 2. Available files from NEON -------------------------------------
    console.print(f"querying NEON API ...")
    hsi_sites = _list_site_months(HSI_DPID, year)
    rgb_sites = _list_site_months(RGB_DPID, year)
    if site not in hsi_sites or site not in rgb_sites:
        console.print(f"[red]{site} not available for HSI+RGB in {year}[/red]")
        return

    hsi_month = sorted(hsi_sites[site])[0]
    rgb_month = sorted(rgb_sites[site])[0]
    hsi_files = _list_tile_files(HSI_DPID, site, hsi_month)
    rgb_files = _list_tile_files(RGB_DPID, site, rgb_month)
    hsi_by_utm = {f["utm"]: f for f in hsi_files}
    rgb_by_utm = {f["utm"]: f for f in rgb_files}

    paired = set(hsi_by_utm) & set(rgb_by_utm)
    paired_geo = {f"{u[0]}_{u[1]}" for u in paired}
    target_geo = paired_geo & annotated
    target_utms = sorted({tuple(g.split("_")) for g in target_geo})
    if max_tiles > 0:
        target_utms = target_utms[:max_tiles]

    # ---- 3. Summary table -------------------------------------------------
    hsi_bytes = sum(hsi_by_utm[u]["size"] for u in target_utms)
    rgb_bytes = sum(rgb_by_utm[u]["size"] for u in target_utms)
    tbl = Table(show_header=False, box=None, pad_edge=False)
    tbl.add_row("HSI month",          hsi_month)
    tbl.add_row("RGB month",          rgb_month)
    tbl.add_row("HSI files on NEON",  f"{len(hsi_files):>5}")
    tbl.add_row("RGB files on NEON",  f"{len(rgb_files):>5}")
    tbl.add_row("Paired UTMs",        f"{len(paired):>5}")
    tbl.add_row("Annotated UTMs",     f"{len(annotated):>5}")
    tbl.add_row("[bold]Target UTMs", f"[bold]{len(target_utms):>5}")
    tbl.add_row("HSI volume",         f"{hsi_bytes / 1e9:.1f} GB")
    tbl.add_row("RGB volume",         f"{rgb_bytes / 1e9:.1f} GB")
    tbl.add_row("[bold]Total",        f"[bold]{(hsi_bytes + rgb_bytes) / 1e9:.1f} GB")
    console.print(tbl)

    if not target_utms:
        console.print("[red]nothing to download[/red]")
        return
    if dry_run:
        console.print("[yellow]--dry-run set, exiting[/yellow]")
        return

    # ---- 4. Build job list ------------------------------------------------
    jobs: list[tuple[str, str, int]] = []
    for u in target_utms:
        h = hsi_by_utm[u]
        r = rgb_by_utm[u]
        jobs.append((h["url"], f"{site}/hyperspectral/{h['name']}", h["size"]))
        jobs.append((r["url"], f"{site}/rgb/{r['name']}",            r["size"]))

    total_bytes = sum(j[2] for j in jobs)

    # ---- 5. Parallel download with rich progress --------------------------
    console.rule(f"[bold cyan]Downloading {len(jobs)} files ({total_bytes / 1e9:.1f} GB)")
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TextColumn("{task.fields[gb_done]:.2f}/{task.fields[gb_total]:.2f} GB"),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("eta"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )
    with progress:
        task = progress.add_task(
            "downloading",
            total=len(jobs),
            gb_done=0.0,
            gb_total=total_bytes / 1e9,
        )
        bytes_done = 0
        skipped = 0
        for name, size, was_skipped in download_one.starmap(jobs, order_outputs=False):
            bytes_done += size
            skipped += int(was_skipped)
            progress.update(
                task,
                advance=1,
                gb_done=bytes_done / 1e9,
                description=f"last: {name[:48]}",
            )

    console.rule("[bold green]done")
    console.print(
        f"[green]✓[/green] wrote {bytes_done / 1e9:.2f} GB to volume "
        f"[bold]{VOLUME_NAME}[/bold] under /{site}/  "
        f"([dim]{skipped} files skipped (already present)[/dim])"
    )
