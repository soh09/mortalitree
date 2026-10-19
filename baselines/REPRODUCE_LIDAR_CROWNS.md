# Reproducing the LiDAR-crown / NAIP results

This rebuilds, from public data:

1. **LiDAR crown labels** on NAIP — individual tree crowns segmented from USGS 3DEP
   canopy-height chips, with boxes in NAIP pixel coordinates (`crowns.json`)
2. **Overlay figures** — `crowns_on_chm.png` and `crowns_hillshade.png`
3. **Two interactive viewers** — a 150-chip forest sample, and every chip inside the
   2020 CZU Lightning Complex with pre- *and* post-fire NAIP

Everything runs on a laptop CPU; nothing needs a GPU or credentials.

## What the data is

- **LiDAR and pre-fire NAIP:** the public CHM–NAIP pairs from Allred, McCord & Morford
  (2025), *Scientific Data*, "Canopy height model and NAIP imagery pairs across CONUS"
  (CC BY 4.0). Each 256 × 256 m chip is a 1 m canopy height model built from **USGS 3DEP**
  LiDAR, paired with the NAIP flight nearest in time. They are **not co-registered** —
  NAIP can be several metres off.
- **Post-fire NAIP:** fetched from Microsoft Planetary Computer.
- **Fire perimeter:** CAL FIRE's public 2020 perimeter layer.

We use one tar "bin" of the dataset: `10-411` (UTM zone 10, northing 4,110–4,120 km),
which covers part of the Santa Cruz Mountains and about a quarter of the CZU fire.

## Scripts (all in `baselines/`)

| Script | Does |
|---|---|
| `chm_crowns.py` | LiDAR crowns per chip (treetop local maxima + watershed), boxes in NAIP pixels, plus a naive NAIP offset estimate |
| `select_fire_chips.py` | Downloads a fire perimeter and lists the chips that overlap it |
| `fetch_postfire_naip.py` | Fetches post-fire NAIP for each chip, on the chip's pre-fire grid |
| `make_alignment_viewer.py` | Builds the HTML viewer |
| `make_crown_figures.py` | Builds the two static figures |

---

## 0. Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/) from the repo-root
`pyproject.toml` / `uv.lock`. Install uv (`brew install uv`, or
`curl -LsSf https://astral.sh/uv/install.sh | sh`), then from the repo root:

```bash
uv sync
```

This installs Python 3.10 (pinned in `.python-version`) if needed and creates `.venv`
with the exact versions in `uv.lock` (numpy 2.2, scipy 1.15, scikit-image 0.25,
rasterio 1.4, …). Run scripts with `uv run python …`, as below, or
`source .venv/bin/activate` and drop the `uv run`.

Run all commands from the repo root. Set a work directory with ~10 GB free:

```bash
export WORK=~/lidar_crowns_work && mkdir -p $WORK
```

## 1. Download one bin of the CHM–NAIP pairs (~7.8 GB)

```bash
B=https://rangeland.ntsg.umt.edu/data/rap/chm-naip
curl -L -C - -o $WORK/chm_10-411.tar  $B/chm/10/10-411.tar
curl -L -C - -o $WORK/naip_10-411.tar $B/naip/10/10-411.tar
```

`-C -` resumes if a download is interrupted. Check the files:

| File | Size (bytes) | MD5 |
|---|---|---|
| `chm_10-411.tar` | 784,732,160 | `390b5bc337225a7c62b0a74f5587d951` |
| `naip_10-411.tar` | 7,020,881,920 | `3f6e1e88810b60ea464148de85bcc5a3` |

```bash
md5 -q $WORK/*.tar        # macOS;  on Linux: md5sum $WORK/*.tar
```

(Server files last modified 19 Nov 2024.)

## 2. Forest sample: crowns, viewer, figures

The first 150 chips, in sorted filename order, that are ≥ 30% covered by trees over 5 m:

```bash
uv run python baselines/chm_crowns.py $WORK/chm_10-411.tar $WORK/naip_10-411.tar $WORK/forest --n 150
```

Expected: `wrote 150 chips`. Output: `$WORK/forest/crowns.json`.

```bash
uv run python baselines/make_alignment_viewer.py $WORK/chm_10-411.tar $WORK/naip_10-411.tar \
       $WORK/forest/crowns.json $WORK/viewer_forest
open $WORK/viewer_forest/index.html           # Linux: xdg-open
```

```bash
uv run python baselines/make_crown_figures.py $WORK/chm_10-411.tar $WORK/naip_10-411.tar \
       $WORK/forest/crowns.json $WORK/figures
```

Writes `crowns_on_chm.png` (10 chips, sparse to dense) and `crowns_hillshade.png`
(4 fixed 100 m crops: LiDAR hillshade | + crowns | NAIP | NAIP + crowns).

## 3. CZU fire: chips, crowns, post-fire NAIP, viewer

**Chips overlapping the perimeter:**

```bash
uv run python baselines/select_fire_chips.py $WORK/naip_10-411.tar $WORK/czu --name-like CZU
```

Expected: `CZU LIGHTNING COMPLEX 86553.46 acres`; `268 overlap the perimeter (222 fully inside)`.
All 268 are pre-fire: LiDAR April 2020 + NAIP May 2020, or LiDAR April 2018 + NAIP August 2018.

**Crowns for those chips** (every chip, regardless of tree cover):

```bash
uv run python baselines/chm_crowns.py $WORK/chm_10-411.tar $WORK/naip_10-411.tar $WORK/czu_crowns \
       --keys $WORK/czu/fire_chips.json --min-forest 0 --n 1000
```

Expected: `wrote 267 chips` (one is dropped for incomplete NAIP), ~125,000 crowns in total.

**Post-fire NAIP (2022):**

```bash
uv run python baselines/fetch_postfire_naip.py $WORK/naip_10-411.tar $WORK/czu_crowns/crowns.json \
       $WORK/czu_post --year 2022
```

Expected: `267 chips: 267 fully covered by 2022 NAIP`, all from the 19 May 2022 flight.
Reads only each chip's footprint over the network, so it's modest in size.

**Viewer:**

```bash
uv run python baselines/make_alignment_viewer.py $WORK/chm_10-411.tar $WORK/naip_10-411.tar \
       $WORK/czu_crowns/crowns.json $WORK/viewer_czu $WORK/czu_post
open $WORK/viewer_czu/index.html
```

Expected: `267 pairs (267 with post-fire NAIP)`; about 75 MB.

## 4. Using the viewer

Click a chip to open it. Rows: pre-fire NAIP, post-fire NAIP, LiDAR hillshade; the
right column adds the LiDAR crown outlines and treetops. All crowns come from the
**pre-fire** LiDAR, so on the post-fire row they mark which trees stood before the fire.

| Input | Action |
|---|---|
| scroll / drag | zoom / pan, linked across panels |
| hover | linked crosshair |
| `1` / `2` | choose which NAIP panel the arrows move (pre / post) |
| arrows (Shift = larger) | nudge crowns 0.6 m (3 m) |
| `S` | save both offsets for this chip |
| `A` | try the automatic offset estimate (pre-fire) |
| `R` / `F` | reset offset / fit view |
| `J` / `K` | previous / next chip |
| `T` / `O` | toggle treetops / outlines |

**Export saved offsets** on the main page downloads them as JSON. Offsets are stored in
the browser, so export to keep them.

## 5. Caveats

- **Alignment is unsolved.** The automatic offset is a CHM-vs-NDVI cross-correlation;
  against a random-shift control it was *worse than random*. Treat it as a guess.
  Hand-set offsets from the viewer are the real ground truth.
- **Crowns are an unvalidated algorithm output.** Known errors: roofs and other tall
  structures (the CHM is really a surface model), the edges of water bodies, and
  over-split short, dense vegetation. No building mask is applied yet.
- **A few chips in the release are corrupt** (zlib errors); `chm_crowns.py` skips them.
- **Live services can drift.** The CAL FIRE perimeter layer can be edited, and when a
  chip straddles two NAIP tiles the post-fire mosaic depends on the order Planetary
  Computer returns them. Small differences are possible; check the counts above.
- The full CZU fire spans bins 409–412; only 411 is used here.
