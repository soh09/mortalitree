"""
Discover NEON AOP availability for 2021 and download paired (hyperspectral, RGB)
tiles for one site. Tiles are matched by UTM origin so each RGB has a NIR partner.

Products:
  DP3.30006.001 — Spectrometer surface directional reflectance (1 m, 426 bands, .h5)
  DP3.30010.001 — High-resolution camera imagery mosaic (0.1 m, .tif)
"""

import re
import sys
from pathlib import Path

import requests

API = "https://data.neonscience.org/api/v0"
HSI_DPID = "DP3.30006.001"
RGB_DPID = "DP3.30010.001"
YEAR = "2021"
N_TILES = 20

HSI_DIR = Path(__file__).parent / "hyperspectral"
RGB_DIR = Path(__file__).parent / "rgb"
HSI_DIR.mkdir(exist_ok=True)
RGB_DIR.mkdir(exist_ok=True)

UTM_RE = re.compile(r"(\d{6})_(\d{7})")


def list_site_months(dpid: str, year: str) -> dict[str, list[str]]:
    """Return {siteCode: [yearMonth, ...]} for months matching `year`."""
    r = requests.get(f"{API}/products/{dpid}", timeout=30)
    r.raise_for_status()
    sites = r.json()["data"]["siteCodes"]
    out = {}
    for s in sites:
        months = [m for m in s["availableMonths"] if m.startswith(year)]
        if months:
            out[s["siteCode"]] = months
    return out


def list_tile_files(dpid: str, site: str, year_month: str) -> list[dict]:
    """Return [{name, url, size, utm}, ...] for one site/month."""
    r = requests.get(f"{API}/data/{dpid}/{site}/{year_month}", timeout=30)
    r.raise_for_status()
    files = r.json()["data"]["files"]
    out = []
    for f in files:
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


def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  skip (exists): {dest.name}")
        return
    print(f"  downloading {dest.name} ...", end=" ", flush=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(1 << 16):
                fh.write(chunk)
        tmp.rename(dest)
    print(f"done ({dest.stat().st_size / 1e6:.1f} MB)")


def main():
    print(f"Discovering {YEAR} availability ...")
    hsi_sites = list_site_months(HSI_DPID, YEAR)
    rgb_sites = list_site_months(RGB_DPID, YEAR)
    shared_sites = sorted(set(hsi_sites) & set(rgb_sites))
    print(f"Sites with BOTH hyperspectral and RGB in {YEAR}: {shared_sites}")
    if not shared_sites:
        sys.exit("No overlapping sites — aborting.")

    # Easily change this slice to grab more sites (e.g. shared_sites[:5] or shared_sites)
    for site in shared_sites[:1]:
        # Pick the first matching month (most sites have one flight per year)
        hsi_month = sorted(hsi_sites[site])[0]
        rgb_month = sorted(rgb_sites[site])[0]
        print(f"\n=== {site} (HSI {hsi_month} / RGB {rgb_month}) ===")

        hsi_files = list_tile_files(HSI_DPID, site, hsi_month)
        rgb_files = list_tile_files(RGB_DPID, site, rgb_month)

        # Index by UTM and keep only tiles that exist in BOTH products
        hsi_by_utm = {f["utm"]: f for f in hsi_files}
        rgb_by_utm = {f["utm"]: f for f in rgb_files}
        paired_utms = sorted(set(hsi_by_utm) & set(rgb_by_utm))
        print(f"Found {len(hsi_by_utm)} HSI tiles, {len(rgb_by_utm)} RGB tiles, "
              f"{len(paired_utms)} paired by UTM.")

        for utm in paired_utms[:N_TILES]:
            print(f"\n[{site} tile {utm[0]}_{utm[1]}]")
            hsi = hsi_by_utm[utm]
            rgb = rgb_by_utm[utm]
            download(hsi["url"], HSI_DIR / hsi["name"])
            download(rgb["url"], RGB_DIR / rgb["name"])


if __name__ == "__main__":
    main()
