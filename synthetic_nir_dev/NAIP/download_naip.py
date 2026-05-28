"""
Download NAIP 2022 tiles intersecting hardcoded California forest AOIs.

Source: Microsoft Planetary Computer STAC (collection "naip"). Each item's
"image" asset is a 4-band (R,G,B,NIR) COG, typically 0.6 m GSD for CA 2022.
We download the full COG to raw/2022/<item_id>.tif so make_patches.py can
slice it later.

Requires: pystac-client, planetary-computer, requests
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import planetary_computer as pc
import requests
from pystac_client import Client

from aois import CA_FOREST_BBOXES

MPC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "naip"
YEAR = "2022"
N_PER_AOI: int | None = 1  # set to None to download every hit per AOI
MAX_WORKERS = 4

RAW_DIR = Path(__file__).parent / "raw" / YEAR
RAW_DIR.mkdir(parents=True, exist_ok=True)


def search_aoi(catalog: Client, bbox: list[float], year: str) -> list:
    """Return STAC items for one AOI bbox, newest first."""
    search = catalog.search(
        collections=[COLLECTION],
        bbox=bbox,
        datetime=f"{year}-01-01/{year}-12-31",
    )
    items = list(search.items())
    items.sort(key=lambda it: it.datetime, reverse=True)
    return items


def download_one(item, dest_dir: Path) -> tuple[str, str]:
    """Sign + stream one item to disk. Returns (item_id, status_str)."""
    dest = dest_dir / f"{item.id}.tif"
    if dest.exists():
        return item.id, f"skip (exists, {dest.stat().st_size / 1e6:.1f} MB)"
    # Sign inside the worker — MPC signed URLs have ~1h TTL, so signing right
    # before the GET avoids expiry if the queue sits a while.
    signed = pc.sign(item)
    url = signed.assets["image"].href
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(1 << 16):
                    fh.write(chunk)
        tmp.rename(dest)
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        return item.id, f"FAILED: {e}"
    return item.id, f"done ({dest.stat().st_size / 1e6:.1f} MB)"


def collect_items() -> list:
    """Walk all AOIs, return deduped list of items to download."""
    catalog = Client.open(MPC_URL)
    print(f"Searching NAIP {YEAR} over {len(CA_FOREST_BBOXES)} CA forest AOIs ...")

    seen_ids: set[str] = set()
    queue: list = []
    for aoi in CA_FOREST_BBOXES:
        print(f"\n=== {aoi['name']}  bbox={aoi['bbox']} ===")
        items = search_aoi(catalog, aoi["bbox"], YEAR)
        if not items:
            print("  no items found")
            continue
        take = len(items) if N_PER_AOI is None else N_PER_AOI
        print(f"  {len(items)} items match, taking {take}")

        for item in items[:N_PER_AOI]:
            if item.id in seen_ids:
                print(f"  skip (already queued): {item.id}")
                continue
            seen_ids.add(item.id)
            queue.append(item)
    return queue


def main():
    queue = collect_items()
    print(f"\nQueued {len(queue)} unique tiles; downloading with {MAX_WORKERS} workers ...\n")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(download_one, it, RAW_DIR): it.id for it in queue}
        for fut in as_completed(futures):
            item_id, status = fut.result()
            print(f"  [{item_id}] {status}")


if __name__ == "__main__":
    main()
