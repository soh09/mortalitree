"""
Download a single paired (hyperspectral, RGB) NEON AOP tile by identifier.

Usage:
    python neon_dl.py 2019_BLAN_3_763000_4329000

The identifier format is YEAR_SITE_VISIT_EASTING_NORTHING.

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
TILE_RE = re.compile(r"^(\d{4})_([A-Z]+)_(\d+)_(\d{6})_(\d{7})$")


def parse_tile(s: str) -> tuple[str, str, tuple[str, str]]:
    """Parse '2019_BLAN_3_763000_4329000' -> (year, site, (easting, northing))."""
    m = TILE_RE.match(s.strip())
    if not m:
        raise ValueError(
            f"Bad tile string: {s!r} (expected YEAR_SITE_VISIT_EASTING_NORTHING)"
        )
    year, site, _visit, easting, northing = m.groups()
    return year, site, (easting, northing)


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


def fetch_tile(tile_id: str) -> None:
    year, site, utm = parse_tile(tile_id)
    print(f"Looking up {site} in {year} ...")

    hsi_sites = list_site_months(HSI_DPID, year)
    rgb_sites = list_site_months(RGB_DPID, year)
    if site not in hsi_sites:
        sys.exit(f"{site} has no HSI data in {year}")
    if site not in rgb_sites:
        sys.exit(f"{site} has no RGB data in {year}")

    hsi_month = sorted(hsi_sites[site])[0]
    rgb_month = sorted(rgb_sites[site])[0]
    print(f"=== {site} (HSI {hsi_month} / RGB {rgb_month}) tile {utm[0]}_{utm[1]} ===")

    hsi_files = list_tile_files(HSI_DPID, site, hsi_month)
    rgb_files = list_tile_files(RGB_DPID, site, rgb_month)

    hsi_match = [f for f in hsi_files if f["utm"] == utm]
    rgb_match = [f for f in rgb_files if f["utm"] == utm]
    if not hsi_match:
        sys.exit(f"No HSI tile at UTM {utm[0]}_{utm[1]} for {site}/{hsi_month}")
    if not rgb_match:
        sys.exit(f"No RGB tile at UTM {utm[0]}_{utm[1]} for {site}/{rgb_month}")

    for f in hsi_match:
        download(f["url"], HSI_DIR / f["name"])
    for f in rgb_match:
        download(f["url"], RGB_DIR / f["name"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tile", help="Tile identifier, e.g. 2019_BLAN_3_763000_4329000")
    args = ap.parse_args()
    fetch_tile(args.tile)


if __name__ == "__main__":
    main()
