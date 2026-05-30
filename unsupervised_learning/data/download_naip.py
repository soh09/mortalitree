"""
Download NAIP tiles intersecting CA forest AOIs (lots of them, for MAE
pretraining). Source: Microsoft Planetary Computer STAC, 4-band (R,G,B,NIR)
COGs at ~0.6 m GSD.

By default fetches BOTH 2020 and 2022 — pretraining benefits from temporal
diversity (different phenology, different radiometric calibration) and from
pre/post-fire pairs at the burn-perimeter AOIs. Override with NAIP_YEARS env
var if you only want one year (e.g. NAIP_YEARS=2022).

Adapted from synthetic_nir_dev/NAIP/download_naip.py:
  - DATA_ROOT configurable via env var (Modal points it at a Volume)
  - N_PER_AOI default is None (take everything)
  - YEARS is a list, not a single year — output written to raw/<year>/

Requires: pystac-client, planetary-computer, requests
"""

import csv
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import planetary_computer as pc
import requests
from pystac_client import Client

from .aois import CA_FOREST_BBOXES

STOP = threading.Event()

MPC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "naip"
YEARS = os.environ.get("NAIP_YEARS", "2020,2022").split(",")
N_PER_AOI = None  # None = take every scene per AOI
MAX_WORKERS = int(os.environ.get("NAIP_DL_WORKERS", "4"))

DATA_ROOT = Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parent.parent))


def raw_dir(year: str) -> Path:
    return DATA_ROOT / "raw" / year


def aoi_sidecar_path(year: str) -> Path:
    return raw_dir(year) / "tile_aoi.csv"


def search_aoi(catalog: Client, bbox: list[float], year: str) -> list:
    search = catalog.search(
        collections=[COLLECTION],
        bbox=bbox,
        datetime=f"{year}-01-01/{year}-12-31",
    )
    items = list(search.items())
    items.sort(key=lambda it: it.datetime, reverse=True)
    return items


def download_one(item, dest_dir: Path) -> tuple[str, str]:
    dest = dest_dir / f"{item.id}.tif"
    if dest.exists():
        return item.id, f"skip (exists, {dest.stat().st_size / 1e6:.1f} MB)"
    if STOP.is_set():
        return item.id, "cancelled (before start)"
    # Sign inside the worker — MPC signed URLs have ~1h TTL.
    signed = pc.sign(item)
    url = signed.assets["image"].href
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(1 << 16):
                    if STOP.is_set():
                        raise KeyboardInterrupt
                    fh.write(chunk)
        tmp.rename(dest)
    except KeyboardInterrupt:
        if tmp.exists():
            tmp.unlink()
        return item.id, "cancelled (mid-download)"
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        return item.id, f"FAILED: {e}"
    return item.id, f"done ({dest.stat().st_size / 1e6:.1f} MB)"


def collect_items_for_year(catalog: Client, year: str) -> tuple[list, dict[str, str]]:
    print(f"Searching NAIP {year} over {len(CA_FOREST_BBOXES)} AOIs ...")
    seen_ids: set[str] = set()
    queue: list = []
    scene_to_aoi: dict[str, str] = {}
    for aoi in CA_FOREST_BBOXES:
        print(f"\n=== [{year}] {aoi['name']}  bbox={aoi['bbox']} ===")
        items = search_aoi(catalog, aoi["bbox"], year)
        if not items:
            print("  no items found")
            continue
        take = len(items) if N_PER_AOI is None else N_PER_AOI
        print(f"  {len(items)} items match, taking {take}")
        for item in items[:N_PER_AOI]:
            if item.id in seen_ids:
                continue
            seen_ids.add(item.id)
            scene_to_aoi[item.id] = aoi["name"]
            queue.append(item)
    return queue, scene_to_aoi


def write_aoi_sidecar(year: str, scene_to_aoi: dict[str, str]) -> None:
    path = aoi_sidecar_path(year)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scene_id", "aoi"])
        for sid, aoi in scene_to_aoi.items():
            w.writerow([sid, aoi])
    print(f"Wrote AOI sidecar: {path} ({len(scene_to_aoi)} scenes)")


def download_year(year: str) -> None:
    dest = raw_dir(year)
    dest.mkdir(parents=True, exist_ok=True)
    catalog = Client.open(MPC_URL)
    queue, scene_to_aoi = collect_items_for_year(catalog, year)
    write_aoi_sidecar(year, scene_to_aoi)
    print(f"\n[{year}] queued {len(queue)} unique tiles; "
          f"downloading with {MAX_WORKERS} workers\n")

    pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    futures = {pool.submit(download_one, it, dest): it.id for it in queue}
    try:
        for fut in as_completed(futures):
            item_id, status = fut.result()
            print(f"  [{year}/{item_id}] {status}")
    except KeyboardInterrupt:
        print(f"\n[{year}] ^C — cancelling pending downloads")
        STOP.set()
        pool.shutdown(wait=False, cancel_futures=True)
        for fut in as_completed(futures):
            try:
                item_id, status = fut.result()
                print(f"  [{year}/{item_id}] {status}")
            except Exception:
                pass
        raise
    finally:
        pool.shutdown(wait=False)


def main():
    print(f"NAIP years to download: {YEARS}")
    for year in YEARS:
        if STOP.is_set():
            break
        download_year(year.strip())


if __name__ == "__main__":
    main()
