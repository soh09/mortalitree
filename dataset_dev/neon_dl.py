"""
Download all paired (hyperspectral, RGB) NEON AOP tiles for one site/year.

Usage:
    python neon_dl.py TEAK 2019

Products:
  DP3.30006.001 — Spectrometer surface directional reflectance (1 m, 426 bands, .h5)
  DP3.30010.001 — High-resolution camera imagery mosaic (0.1 m, .tif)
"""

import argparse
import re
import sys
from pathlib import Path

import requests

API = "https://data.neonscience.org/api/v0"
HSI_DPID = "DP3.30006.001"
RGB_DPID = "DP3.30010.001"

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


def fetch_aop(site: str, year: str) -> None:
    print(f"Looking up {site} in {year} ...")

    hsi_sites = list_site_months(HSI_DPID, year)
    rgb_sites = list_site_months(RGB_DPID, year)
    if site not in hsi_sites:
        sys.exit(f"{site} has no HSI data in {year}")
    if site not in rgb_sites:
        sys.exit(f"{site} has no RGB data in {year}")

    hsi_month = sorted(hsi_sites[site])[0]
    rgb_month = sorted(rgb_sites[site])[0]
    print(f"=== {site} (HSI {hsi_month} / RGB {rgb_month}) ===")

    hsi_files = list_tile_files(HSI_DPID, site, hsi_month)
    rgb_files = list_tile_files(RGB_DPID, site, rgb_month)

    hsi_by_utm = {f["utm"]: f for f in hsi_files}
    rgb_by_utm = {f["utm"]: f for f in rgb_files}
    common = sorted(hsi_by_utm.keys() & rgb_by_utm.keys())
    if not common:
        sys.exit(f"No paired HSI+RGB tiles for {site}/{year}")

    hsi_bytes = sum(hsi_by_utm[u]["size"] for u in common)
    rgb_bytes = sum(rgb_by_utm[u]["size"] for u in common)
    print(
        f"{len(common)} paired tiles — "
        f"HSI {hsi_bytes / 1e9:.1f} GB + RGB {rgb_bytes / 1e9:.1f} GB "
        f"= {(hsi_bytes + rgb_bytes) / 1e9:.1f} GB total"
    )
    hsi_only = sorted(hsi_by_utm.keys() - rgb_by_utm.keys())
    rgb_only = sorted(rgb_by_utm.keys() - hsi_by_utm.keys())
    if hsi_only or rgb_only:
        print(f"  ({len(hsi_only)} HSI-only and {len(rgb_only)} RGB-only tiles skipped)")

    for i, utm in enumerate(common, 1):
        print(f"[{i}/{len(common)}] UTM {utm[0]}_{utm[1]}")
        download(hsi_by_utm[utm]["url"], HSI_DIR / hsi_by_utm[utm]["name"])
        download(rgb_by_utm[utm]["url"], RGB_DIR / rgb_by_utm[utm]["name"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("site", help="NEON site code, e.g. TEAK")
    ap.add_argument("year", help="Year, e.g. 2019")
    args = ap.parse_args()
    fetch_aop(args.site, args.year)


if __name__ == "__main__":
    main()
