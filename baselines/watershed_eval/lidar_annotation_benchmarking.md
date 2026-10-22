# Benchmarking LiDAR crown annotations against human labels

**Status:** done for the evaluation and training tiers (2026-10-21). Results below.

## Why

Our NAIP crown labels come from running a watershed segmentation on a LiDAR canopy
height model (CHM) and projecting the crowns onto imagery
(`baselines/chm_crowns.py`, see `baselines/REPRODUCE_LIDAR_CROWNS.md`). The labels can be wrong
in two independent ways:

1. **Segmentation error.** The watershed itself gets crowns wrong: over-splitting one
   tree, merging neighbours, missing trees, creating false crowns (roofs, water edges),
   or drawing boxes that are too big, too small or off-centre.
2. **Registration error.** The LiDAR and the imagery don't line up: a whole-chip shift
   (NAIP is ±6 m), height-dependent tree lean, and time between captures.

Today we can't tell how much of the label noise is which. This benchmark measures
**(1) on its own** by running our watershed where registration is already solved and
comparing it with human crown labels.

## Data: NeonTreeEvaluation

Weinstein et al. (2021), *A benchmark dataset for canopy crown detection and delineation
in co-registered airborne RGB, LiDAR and hyperspectral imagery from NEON*, PLoS Comput
Biol 17(7): e1009180. Zenodo record **5914554** (v0.2.2).

- **Evaluation crowns (high confidence):** ~6,600 boxes on 40 × 40 m plots at 22 NEON
  sites, drawn by looking at RGB, LiDAR and hyperspectral together. Each plot ships with
  its own 1 m CHM (`evaluation/CHM/{plot}_CHM.tif`) and 0.1 m RGB, already co-registered
  by NEON, so **no separate NEON download is needed**.
- **Training crowns (medium confidence):** ~22,000 boxes on large tiles, drawn from RGB
  only and less carefully reviewed. NIWO alone is ~10,000 of them. CHM is included for
  most tiles.
- Boxes are axis-aligned, in RGB pixel coordinates (Pascal VOC XML).
- Benchmark protocol: a prediction matches a label at **IoU ≥ 0.4**, one-to-one;
  precision and recall are computed per image and averaged.
- Reference points from the paper: DeepForest (RGB deep learning) scores recall 79.0 /
  precision 65.9; two human annotators agree on ~70–90 % of crowns depending on the IoU
  threshold. That human agreement is the realistic ceiling.

## Steps

### 1. Fetch (`baselines/watershed_eval/neontree_fetch.py`)
Read the Zenodo zips over HTTP range requests and extract only annotations, evaluation
CHMs, and RGB for annotated plots (`--training` adds training-tile CHM + RGB). This skips
~8 GB of hyperspectral and point clouds. Each extracted file is CRC-checked.

### 2. Predict (`baselines/watershed_eval/neontree_watershed.py`)
Run the unchanged `crowns_from_chm` on every annotated plot's CHM and convert each
crown's bounding box from CHM pixels to map coordinates to RGB pixels using both files'
georeferencing. The CHM grid is snapped to whole metres and the RGB is offset by up to
0.4 m, so the origins can't be assumed equal. Run with two minimum tree heights:

- **5 m:** our pipeline as it is today
- **3 m:** the benchmark's tree definition

Record per crown: top height, area, and whether it touches the plot edge.

### 3. Evaluate (`baselines/watershed_eval/neontree_eval.py`)
- **Benchmark score:** precision and recall at IoU ≥ 0.4, per image then averaged;
  per site and overall. Directly comparable to the paper.
- **Error taxonomy:** every label and every prediction is assigned one outcome.

  | Outcome | Rule |
  |---|---|
  | match | one-to-one pair with IoU ≥ 0.4 |
  | over-segmentation | one label contains ≥ 2 predictions (each ≥ 50 % inside it) |
  | under-segmentation | one prediction contains ≥ 2 labels (each ≥ 50 % inside it) |
  | poor delineation | single best partner, 0.1 ≤ IoU < 0.4 |
  | miss | label with no partner; split by CHM height inside it (< 3, 3–5, > 5 m) |
  | false positive | prediction with no partner |

- **Localisation of matches:** centre offset (east/north, metres) and size ratios
  (area, width, height). A non-zero median offset means the watershed boxes are
  systematically shifted; a ratio away from 1 means they're systematically too big or
  too small.
- **Breakdowns:** by site (highlighting SJER, SOAP and TEAK, the California sites), by
  crown size, edge vs interior crowns, and 3 m vs 5 m.
- **Figures:** best, median and worst plots per California site, with labels and
  predictions on RGB and on the CHM hillshade.

### 4. Training tiles (second tier)
Repeat on the training tiles and report them per site, separately from the evaluation
set, since these labels are RGB-only and noisier.

## Caveats

- **Edge effects.** Plots are only 40 m wide. A tree whose top lies just outside the plot
  has no treetop inside it, so the watershed floods its in-plot part into a neighbour
  (looks like a merge) or drops it. Edge-touching crowns are reported separately.
  Re-running on full NEON tiles with a buffer is deferred.
- **Not purely segmentation error.** Annotators draw the *visible* crown in the image,
  which differs from the 3D LiDAR crown (sunlit side vs shadow side), and NEON's own
  RGB/LiDAR co-registration has small residual distortion. Offsets measured here are
  segmentation + visible-crown bias + a little NEON misregistration.
- **NEON vs 3DEP LiDAR.** NEON point density (~5 pts/m²) and CHM processing differ from
  the 3DEP CHMs used on NAIP, so segmentation quality may not transfer exactly.
- **Boxes, not polygons.** Our crowns are compared as bounding boxes, matching how the
  pipeline emits labels.

## How to run

```bash
W=~/lidar_crowns_work
uv run python baselines/watershed_eval/neontree_fetch.py $W/neontree --training          # ~0.8 GB, ~30 min
for t in 5 3; do
  uv run python baselines/watershed_eval/neontree_watershed.py $W/neontree $W/neontree_ws --min-top $t
  uv run python baselines/watershed_eval/neontree_eval.py $W/neontree_ws --min-top $t --figures
done
uv run python baselines/watershed_eval/neontree_eval.py $W/neontree_ws --sanity               # self-check
# training tier (5 m): add --training to the watershed step, output to a separate dir
```

Expected: 189 annotated evaluation plots with a CHM, 2 skipped as truncated
(`SJER_062_2018`, `TALL_043_2019`), leaving 187 plots and 6,199 labels; 16 training tiles, 21,460 labels.
Sanity check: labels scored against themselves give P = R = 1; shifted 2 m east they give a
median east offset of exactly 2.0 m.

## Results

Two averages are reported. **Per image** is the benchmark's protocol (each plot counts
equally, so SJER's 60 sparse plots weigh a lot). **Pooled** counts every crown equally and
is closer to what a training set built from these labels would look like.

### Overall

| Run | Labels | Crowns found | P / R per image | P / R pooled | Match | Merged | Over-seg | Poor | Miss | False pos.* |
|---|---|---|---|---|---|---|---|---|---|---|
| evaluation, 5 m | 6,199 | 3,104 | 0.40 / 0.37 | 0.29 / 0.14 | 14 % | 77 % | 0 % | 4 % | 4 % | 7 % |
| evaluation, 3 m | 6,199 | 3,269 | 0.39 / 0.39 | 0.29 / 0.15 | 15 % | 77 % | 0 % | 5 % | 3 % | 9 % |
| training, 5 m | 21,460 | 10,531 | 0.28 / 0.18 | 0.32 / 0.15 | 15 % | 76 % | 0 % | 3 % | 5 % | 7 % |

Outcome shares are of labels, except *false positives, which are a share of predictions.
For reference: DeepForest scores 0.66 / 0.79 per image; human annotators agree on ~70–90 %.

### By crown size (evaluation, 5 m; label box area)

| Label crown area | Labels | Match | Merged |
|---|---|---|---|
| < 10 m² (≈ 3 m across) | 3,042 | 2 % | 89 % |
| 10–25 m² | 1,933 | 15 % | 76 % |
| 25–50 m² | 815 | 31 % | 61 % |
| 50–100 m² | 317 | 61 % | 32 % |
| > 100 m² | 92 | 76 % | 16 % |

The training tier shows the same gradient (< 10 m²: 0.5 % match, 94 % merged;
> 100 m²: 84 % match).

### California sites (evaluation, 5 m)

| Site | Plots | P / R per image | Match | Merged | Fragment (preds) | Area ratio |
|---|---|---|---|---|---|---|
| SJER (oak woodland) | 60 | 0.56 / 0.64 | 52 % | 23 % | 19 % | 1.16 |
| TEAK (mixed conifer) | 51 | 0.30 / 0.25 | 13 % | 76 % | 0 % | 1.55 |
| SOAP (mixed conifer) | 2 | 0.17 / 0.06 | 6 % | 87 % | 0 % | 1.80 |

Figures: `fig_{SJER,SOAP,TEAK}_{5m,3m}.png` in the output directory (worst, median and
best plot per site; labels white, watershed pink; RGB and CHM hillshade).

### Localisation of matched crowns (evaluation, 5 m; p25 / median / p75)

| | p25 | median | p75 |
|---|---|---|---|
| offset east (m) | −0.6 | −0.1 | +0.4 |
| offset north (m) | −0.5 | 0.0 | +0.65 |
| centre distance (m) | 0.6 | 1.0 | 1.5 |
| area ratio (watershed / label) | 1.06 | 1.44 | 1.87 |
| width ratio | 0.99 | 1.20 | 1.40 |

## What this says about the watershed

1. **The dominant error is merging, driven by crown size.** The watershed finds about
   half as many crowns as humans label (3,104 vs 6,199). Crowns smaller than ~5 m across
   are almost never resolved, because the treetop window is 4 m wide below 32 m height,
   and the 3×3 median smoothing on a 1 m CHM erases the gap between small neighbouring
   crowns. Large, isolated crowns (SJER oaks, anything > 50 m²) are found well.
2. **Over-segmentation is rare overall but real for big irregular crowns.** It shows up
   mainly at SJER (19 % of predictions are fragments of a sprawling oak).
3. **There is no systematic shift.** Median offsets are ≤ 0.1 m; the ~1 m median centre
   distance is scatter, not bias. On NEON at least, bucket 1 isn't a position problem.
4. **Boxes are too large.** Matched boxes are 1.44× the label area (≈ 1.2× per side):
   the watershed floods each crown down to the 3 m mask, picking up lower branches and
   gaps, and a ragged region's bounding box inflates further.
5. **The height threshold doesn't matter.** 3 m vs 5 m changes recall by ~0.01–0.03.
6. **Edges aren't the problem.** Edge-touching labels match *more* often (27 %) than
   interior ones (9 %), so the 40 m plot size isn't what drives the merging.

Implications for the NAIP labels: in dense forest (TEAK-like, i.e. our CZU chips) most
LiDAR "crowns" are clusters of several trees, and boxes are ~20 % too wide. Fixing
segmentation (smaller or adaptive treetop windows, less smoothing, tighter crown
extents) matters more than fixing registration, until crowns are resolved individually.

## Data issues found

- `SJER_062_2018` and `TALL_043_2019` ship truncated CHMs (40 × 3 and 40 × 1 px);
  `neontree_watershed.py` skips any CHM covering < 90 % of its image.
- 5 annotated evaluation RGBs have no CHM in the release and are skipped.
- Training tile `2019_TOOL_…` is tundra: every label is < 3 m, so the watershed (≥ 3 m)
  finds nothing there. Ignore it.

## Next steps: find what causes the merging

Two parts of `crowns_from_chm` (`baselines/chm_crowns.py`) can each erase small, close
crowns, and this benchmark doesn't yet say how much each one contributes.

**1. The 3 × 3 median smoothing** (`ndi.median_filter(chm, size=3)`). Each 1 m pixel is
replaced by the median of itself and its 8 neighbours. This is there to remove LiDAR
noise: *pits* (a pulse slips through the foliage to the ground, which would cut a crown
in two) and *spikes* (a stray high return, which would become a fake treetop). But the
median can't tell 1–2 px noise from 1–2 px real features, and at 1 m that's the size of
small crowns and the gaps between them:

```
before:   2  10  12  10   4  10  12  10   2     two ~3 m trees, 1 m gap
after:    2  10  10  10  10  10  10  10   2     gap filled, both tops shaved to 10
```

The result is one flat plateau, and `treetops` collapses a plateau into a single marker,
so the watershed grows one crown over both trees.

**2. The treetop window.** Below 32 m height a pixel only counts as a treetop if it's the
highest point in a 4 m-wide disk, so two tops 2–3 m apart can't both survive.

**Ablation.** Expose both as parameters (smoothing size; window diameter below 32 m),
then re-run `neontree_watershed.py` + `neontree_eval.py` on the evaluation tier:

| Run | Median filter | Window < 32 m |
|---|---|---|
| baseline (current) | 3 × 3 | 4 m |
| no smoothing | none | 4 m |
| smaller window | 3 × 3 | 2–3 m |
| both | none | 2–3 m |

Removing smoothing will probably bring pits and spikes back as over-segmentation and
false positives. The setting to pick is the one where *merged* drops more than
*fragment* / *false positive* rise; the error taxonomy shows that trade-off directly.
Report it overall, by crown-size bin, and for TEAK (the site closest to our CZU chips).
Watch the < 10 m² bin in particular, which is currently 2 % matched.

**After that:** check whether the boxes stay ~1.2× too wide per side; if so, tighten
crown extents (e.g. a relative-height cutoff per crown instead of the fixed 3 m mask)
before re-generating the NAIP labels.
