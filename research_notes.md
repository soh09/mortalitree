# MortaliTREE — research notes (Sept–Oct 2026)

Consolidated notes from a long working session: what we measured, which papers and
datasets matter, the pipelines we built, and what is still open. Numbers are from
runs in this repo unless a paper is cited.

**Reading status** is marked on every paper: **full** = read the full text;
**partial** = abstract, search snippets, or another paper's description only. Ask for
the PDF before relying on a *partial* entry's methods.

---

## 1. Where things stand

**Goal.** Count individual trees in 4-band NAIP (0.6 m) before and after California
wildfires, to estimate post-fire tree mortality.

**Best current model.** DeepForest (RetinaNet, RGB only) fine-tuned on 75 hand-labelled
NAIP tiles.

**The core finding.** The model *finds* trees reasonably well but *boxes* them poorly.
Box extent, not detection, is the bottleneck — and much of that is geometric: median
crown is ~12 px, so a couple of pixels of edge error fails IoU 0.5.

**What changes the answer most.** Counting and point-style metrics, not IoU-0.5 F1.

**Main open problem.** Getting more labels. The most promising route — LiDAR-derived
crowns transferred to NAIP — has two problems: NAIP–LiDAR misregistration, which no
published paper has solved at tree level, and the crown segmentation itself. So's NEON
benchmark (§7.2) shows **segmentation is currently the bigger one**: in dense forest the
watershed merges several trees into one crown. Tuning (§7.2.1) cut merging from 74% to 40%
of labels and raised held-out F1 from 0.42 to 0.50; Dalponte and Silva region growing do
not beat it. **But on 3DEP at CZU the tuned crowns looked too small by eye, so the original
watershed stays the default for 3DEP labels.** Against field stems the tuned watershed nearly matches human labels on recall
(0.72 vs 0.75) but merges about twice as often.

---

## 2. Our data

| Item | Value |
|---|---|
| Curated set | `single_tiles_flat/stage_c_data/` — 108 tiles, 256 × 256, 4-band NAIP |
| Splits | train 75 / val 18 / test 15 tiles; 5,200 / 1,008 / 997 crowns |
| Manual filter | 483 student-annotated tiles reviewed, **108 accepted (22%)** — `single_tiles_flat/review_state.json`, tool `review.py` |
| Footprint | 2.36 ha per tile; 255 ha total (2.5 km²) |
| Diversity | 66 distinct ~1 km locations, **68 distinct ground footprints** (40 prefire/postfire pairs), 7 regions, 2 dates (2020, 2022) |
| Crown size | median **12.4 px = 7.4 m**; 86% of test crowns < 16 px |
| Annotator overlap | **None** — every student worked a disjoint area, so inter-annotator agreement can't be measured |
| Quirks | 2 train tiles have zero crowns; 2 test tiles have 1 crown (they dominate percentage metrics) |

Fires: CZU Lightning Complex, SCU Lightning Complex, North Complex, Creek, Castle (all 2020).

Viewing aids: `single_tiles_flat/stage_c_data/viewer/index.html` (all tiles, GT +
model overlays, live score/IoU sliders); `stage_c_data/readable/` (pretty JSON + index).

---

## 3. Results

All test numbers use the 15-tile test split unless noted. Thresholds are selected on
**val**, never test.

### 3.1 DeepForest: pretrained vs fine-tuned

| Convention | Pretrained P / R / F1 | Fine-tuned P / R / F1 |
|---|---|---|
| Report Table 1 | 0.127 / 0.124 / 0.120 | 0.229 / 0.208 / 0.204 |
| Re-derived, `export_stats.py` convention (score 0.1, min_gt 5) | 0.127 / 0.124 / 0.125 | 0.245 / 0.233 / 0.239 |
| IoU 0.4 (DeepForest's own default) | F1 0.220 | **F1 0.361** |

- Report Tables 1 and 2 came from `export_stats.py` (score 0.1, `min_gt=5`), not
  `eval.py` (score 0.3). The re-derived baseline reproduced them exactly.
- The report's F1 column isn't the harmonic mean of its own P/R (0.127/0.124 → 0.1255,
  table says 0.120); likely a mean of per-image F1.
- Report Table 3 vs Table 10 disagree on the Q=150 "before" row — they came from two
  different eval sets (`metrics_all_test.json` vs `metrics_test.json`).
- Published DeepForest: F1 ≈ 0.68–0.72 on the NEON benchmark at **10 cm**. Not
  comparable to our 0.6 m task.

### 3.2 It finds trees; it boxes them badly

| Match criterion | Recall (fine-tuned) |
|---|---|
| IoU 0.5 | 0.21 |
| IoU 0.25 | 0.50 |
| IoU 0.1 | **0.58–0.62** |
| Centre inside GT box (point F1) | F1 **0.544** |
| Centre within 6 m (Ventura's criterion) | F1 **0.642** (Ventura et al. urban: 0.735) |

The "58% of trees detected" figure corresponds to loose-match recall, not IoU 0.5.

**Recall by crown size** (found at IoU 0.1 / well-boxed at IoU 0.5):

| Crown side | n | Found | Boxed |
|---|---|---|---|
| < 8 px (< 4.8 m) | 134 | 41.8% | 1.5% |
| 10–12 px | 228 | 74.6% | 22.8% |
| 15–20 px | 160 | 91.9% | 34.4% |
| ≥ 20 px | 35 | 97.1% | — |

Recall is almost entirely a function of pixels on target — a resolution ceiling.

### 3.3 Counting (what mortality actually needs)

| | Pretrained (thr 0.30) | Fine-tuned (thr 0.25) |
|---|---|---|
| Count MAE (trees/tile) | 32.3 | **20.7** |
| R² | 0.41 | **0.78** |
| Median per-tile accuracy | 48.7% | **73.2%** |
| Aggregate total accuracy | 64.5% | **92.1%** |
| Tiles within 25% of truth | 13% | 33% |

- At score 0.1 the pretrained model has R² **−0.31** (worse than predicting the mean).
- Landscape-level totals are good (92%); individual tiles much less so (73% median).
- Mean percentage error over all 15 tiles is meaningless — a 1-crown tile scores 1700%.

### 3.4 What we tried to improve it

| Change | Result |
|---|---|
| Inference sweep (score × NMS × `min_size`), selected on val | test F1@0.5 0.208 → 0.222; **MAE 27.3 → 19.2**. Score threshold is almost the whole effect; NMS barely matters. |
| `min_size` trained *and* evaluated at 800 / 1000 / 1300 | F1@0.5 0.197 / 0.219 / 0.217; recall rises monotonically with resolution at every IoU. The earlier "1300 collapses" was a train/test mismatch artefact. |
| 4-channel RGB+NIR (stem inflated, NIR init = mean RGB) vs 3-channel control | F1@0.5 0.195 → 0.234, but val 0.190 → 0.188 and IoU 0.25 −0.007. Single seed — **inconclusive**. |
| `reg_aug` (weight decay 1e-4, cosine LR, H+V flip, brightness, early stop) | F1@0.5 0.189 → **0.215** (val agrees, +0.027), but **zero change at IoU 0.4/0.25/0.1** — only nudges boxes over the 0.5 line. |
| Shifted anchor pyramid `(16,20,25)` added; combined config | **Not completed** — lost to Modal preemption. |
| Non-ML veg-index + watershed (NIR−mean(R,G), Otsu, opening, watershed), tuned on train | F1@0.5 0.115, MAE 43.4, R² ≈ 0. Fails in closed canopy; fires on shrubs. |

DeepForest's real anchors (torchvision defaults) start at 32 px; `create_anchor_generator`
in DeepForest is dead code. The internal `min_size=800` resize already upsamples tiles 3.1×.

### 3.5 Learning curve — more tiles of the same kind?

| Tiles | Crowns | Mean F1@0.5 | Mean F1@0.25 | Mean MAE |
|---|---|---|---|---|
| 0 | 0 | 0.129 | 0.404 | 25.4 |
| 10 | 518 / 813 | 0.161 | 0.431 | 27.8 |
| 20 | 1,383 / 1,590 | 0.177 | 0.489 | 19.2 |
| 40 | 2,761 / 3,007 | **0.198** | 0.503 | 21.0 |
| 73 | 5,200 | 0.189 | 0.524 | 20.4 |

DeepForest saturates around 40 tiles. **Caveat:** this measures DeepForest's ceiling at
0.6 m (resolution-limited, see 3.2), not how much information the labels hold, and says
nothing about Clay's from-scratch DETR head. Our 5,200 crowns already exceed DeepForest's
own 2,848 hand labels; we're short on **scenes**, not crowns.

### 3.6 Clay / foundation-model experiments (from the report)

Best: DQ-DETR fine-tuned, F1 0.153. Clay encoder is **frozen** — trainable params are
~3.2M (neck + head) vs ~32M for fine-tuned DeepForest. Likely limits: from-scratch DETR
head with no detection prior, ~80 epochs vs DETR's 500 (quartering run killed at epoch
43), and Experiment C (deep supervision) never run. Focal loss (Experiment B) exists in
`dq-detr-impl` only, bundled with other changes. GPU is blocked on Modal, so no Clay
reruns yet.

---

## 4. Papers

### Directly comparable work

| Paper | Read | Summary | Why it matters |
|---|---|---|---|
| **Dixon et al. 2023**, *RSE* 298:113842 — canopy-scale post-fire mortality from PlanetScope | full | 3 m tree/shrub × dead/alive maps for 15 CA fires (2018–21) incl. all 2020 fires. Labels: pre-fire 3DEP LiDAR crowns, hand-labelled dead/alive on pre/post NAIP by one person; trees dead before the fire labelled non-woody; dead = >90% green loss. PlanetScope chosen for temporal frequency and operational cost. Notes NAIP is "offset by meters" from LiDAR. | Closest to our project; shares CZU, North Complex, Creek. Their crown polygons are "on request". |
| **Cheng et al. 2024**, *Nat Comms* — scattered tree death in California | full (+ code) | 91.4M **dead** trees mapped on 2020 NAIP; EfficientUNet + deep watershed; ~96% vs visual, 83% vs field; 6 m matching buffer for NAIP geolocation error; checkpoint chosen by count MAE. Code confirms **4-band input** (`in_channels=4`). | NAIP-native, but labels are **dead trees only** (live = background) and weights are **not** in the release. |
| **Stovall et al. 2019**, *Nat Comms* — tree height explains drought mortality | full | 1.8M trees at NEON SOAP/TEAK; 2013 NEON LiDAR CHM (0.6 m), lidR local maxima + watershed; NAIP 2009–16 pixels classified (max-likelihood, sunlit/shaded × dead/live); crown dead if >37.5% pixels dead (88% acc). **No NAIP–LiDAR registration.** | Same pipeline as ours; later disputed (below). Validation details in Supp. Figs 1–4 (not read). |
| **Hemming-Schroeder et al. 2023**, *JGR Biogeosciences* — individual tree mortality, Sierra Nevada | partial | Same sites with NEON hyperspectral instead of NAIP; reported 92.1% dead/alive; ~1M polygons for 2013/17/18/19/21 incl. Creek Fire; 45.4% disagreement with Stovall, partly blamed on NAIP geometry. Recipe: local maxima 4 m (<32 m tall) / h/8 / 10 m, Dalponte 2016, trees >5 m. | Free LiDAR-based ground truth at SOAP; our crown recipe. **Request PDF to confirm methods.** |
| **Ventura et al. 2024**, *JAG* — urban tree detection, California NAIP | full (arXiv) | ~96k hand-annotated tree **points** on NAIP 4-band + NDVI, 8 cities, 2016/18/20; HR-SFANet; F 0.735 at 6 m matching. | Largest NAIP-native tree label set; same sensor and state. |
| Open & Transferable NAIP framework, *Remote Sensing* 2026 (urban LA) | partial | U-Net canopy + YOLOv9e trees on NAIP, F1 0.687; misses crowns < ~1.5 m; names LiDAR as future work. | Evidence the LiDAR→NAIP-crowns pipeline is open. |
| **Li et al. 2023**, *PNAS Nexus* — national-scale tree counting/crowns/height (Denmark) | partial | Individual crowns + height from aerial imagery at national scale. | Closest precedent for individual crowns at scale. **Request PDF.** |

### LiDAR → imagery

| Paper | Read | Summary | Why it matters |
|---|---|---|---|
| **Allred, McCord & Morford 2025**, *Sci Data* — CHM–NAIP pairs across CONUS | full (PMC11846907) | 22.8M 1 m CHM chips from 3DEP (2014–23, QL2+) via PDAL/EPT + lidR `pitfree`; each paired with nearest NAIP within ±2 yr, **cropped by footprint only — no co-registration**. CHMs are really normalized DSMs (buildings included). Dense-forest locations lacking ground points dropped. Random (non-spatial) train/val/test. | Our LiDAR source. Public, CC BY 4.0. |
| **Wagner et al. 2024**, *RSE* 305:114099 — sub-meter tree height, California | full | U-Net, NAIP 2020 4-band → height, trained on LiDAR CHMs; MAE 2.9 m. "NAIP and CHM images are **not co-registered**"; "**there is still no available method to register trees**". Masked buildings with Microsoft footprints (2 m buffer); added non-vegetation patches; 42 independent test sites. | Best statement of the registration problem; recipes to copy. |
| NAIP-CHM 0.6 m CONUS, *Sci Data* 2026 | partial | U-Net-predicted canopy height from NAIP (RMSE 2.28 m); 96% of input NAIP from 2022–23. | Aligned to NAIP by construction, but a prediction, not LiDAR; post-dates our fires. |
| LiDAR pseudo-labels + SAM 2, arXiv 2602.13022 (2026) | full | ALS watershed crowns refined with SAM 2 box prompts, trains image models; RGB+NIR F1 0.778 vs RGB 0.758. 5 cm drone imagery, Finland; never applied beyond LiDAR. | SAM 2 box refinement could tighten loose boxes. |
| **Weinstein et al. 2019**, *Remote Sensing* 11:1309 — semi-supervised DeepForest | partial | LiDAR-derived pseudo-labels (434,551) + 2,848 hand labels on 10 cm NEON RGB. | The recipe we're adapting to NAIP. |
| **Weinstein et al. 2020**, *Ecol. Inform.* 56:101061 — cross-site learning | full | 4 NEON sites (SJER, TEAK, NIWO, MLBS). Silva 2016 on a 0.5 m CHM, crown cap 90% of height (deciduous) / 20% (conifer), chosen by eye. LiDAR-only P/R at IoU 0.5 ≈ SJER Silva 0.21/0.43, TEAK Dalponte 0.08/0.20, NIWO ≈ 0. LiDAR errors are mostly merging/splitting, not box extent. Bad pretraining labels hurt (NIWO under-segmentation; a TEAK model beat NIWO's own). One model on all sites ≥ single-site; ~1,000 hand crowns ≈ full performance, mainly improving precision. Hand labels < 3 m removed via CHM; training tiles ≥ 1 km from eval. | Our crowns are ~2× better than their pretraining labels at SJER/TEAK (IoU 0.5: watershed 0.44/0.52, tuned 0.35/0.29 at TEAK); their allometry reproduces in our grid. |
| **Weinstein et al. 2020**, *Methods Ecol. Evol.* 11:1743 — DeepForest package | full | Prebuilt model: 30M LiDAR crowns from 22 sites + 10k hand labels from 6. NEON eval (IoU 0.5) R 0.72 / P 0.64. For 0.3 m imagery a 60–120 m window beat 40 m. Field stems used for recall and under-segmentation. Dead trees only "when visible" at SOAP. | Stem-based scoring (§7.2.1); check whether our NAIP labels include snags. |

### Annotation and data quality

| Paper | Read | Summary |
|---|---|---|
| Papadopoulos et al. 2016, CVPR — "We don't need no bounding-boxes" | partial | Train detectors from yes/no human verification; 6–9× less annotation time. |
| Papadopoulos et al. 2017, ICCV — extreme clicking | partial | 4 clicks per box, ~7 s vs ~35 s, same quality. |
| "Is your training data really ground truth?", *Remote Sensing* 2024 (16:2786) | partial | Manual crown annotation matched only **37%** (plantation) / **10%** (natural forest) of real crowns. |
| Sample labeling & distribution for ITCD, *Remote Sensing* 2022 (14:1561) | partial | Random, distributed sampling reduces required labels; exhaustive labelling unnecessary. |
| CNN ITCD systematic review, *Current Forestry Reports* | partial | NIR/multispectral effect "less clear"; generally modest. |

### Background

DETR (Carion et al. 2020; ~500 epochs on COCO); DQ-DETR (Huang et al. 2024);
Stardist ITCD at 15 cm vs 60 cm (Tong & Zhang 2025, *RSE*); NeonTreeEvaluation
benchmark (Weinstein et al. 2021, *PLOS Comp Bio*); Furniss et al. 2020, *Ecosphere*
(Yosemite plot, Rim Fire mortality).

**Novelty note.** NIR for tree detection is *not* new. The report's §2.4 claim to be
"first in the tree-counting literature to … use the NIR spectral band" should be cut.
The narrower claim — open-weights GFM + individual crown detection + 4-band — is fine.
Unclaimed as far as we found: **unsupervised LiDAR crowns → individual-tree detector on
NAIP → applied where no LiDAR exists.**

---

## 5. Data products

### LiDAR

| Product | What | Coverage / dates | Access | Notes |
|---|---|---|---|---|
| **USGS 3DEP** | National airborne LiDAR for elevation; QL2 ≥ 2 pulses/m² | ~all CONUS; one-off flights, often leaf-off | USGS, AWS (EPT) | No imagery. CZU flown Apr 2018 & Apr 2020. Other four fires **unchecked**. |
| **Allred CHM–NAIP pairs** | 1 m CHM chips from 3DEP + paired NAIP (RGBN + mask) | CONUS, 2014–23 | `https://rangeland.ntsg.umt.edu/data/rap/chm-naip/`, CC BY 4.0 | Tars by UTM zone + 10 km northing (e.g. `10-411`). ~6.5 GB NAIP / 0.75 GB CHM per bin. (An earlier "some chips are corrupt" note was wrong — it was a bug in our tar reader, fixed in `d6e484b`; every chip decodes.) |
| **NEON AOP** | LiDAR + 10 cm RGB + ~426-band hyperspectral, one aircraft | ~80 sites; CA: SJER, SOAP, TEAK. SOAP flown 2013, 2017–19, 2021, 2023, 2024, 2026 | NEON API (`DP3.30015.001` CHM, `DP3.30010.001` RGB) | Internally aligned; **not** aligned to NAIP. Leaf-on. |
| NAIP-CHM | Predicted 0.6 m height | CONUS | Zenodo 20148054, GEE `projects/naip-chm/assets/conus-structure-model` | Predicted, 2022–23. |

### Imagery and ancillary

| Product | Notes |
|---|---|
| NAIP | Microsoft Planetary Computer STAC (`naip`, anonymous SAS token); GEE. Over CZU: 2012, 2014, 2016, 2018 (Aug 4), **2020 (May 27)**, **2022 (May 19)**. NAIP spec: ±6 m to ground at 95%. |
| CAL FIRE perimeters | Public layer `2020_California_Fire_Perimeters_View/FeatureServer/0` (ArcGIS). CZU: 86,553 ac, 16 Aug–24 Sep 2020. Creek: 379,842 ac, 4 Sep–24 Dec 2020. (The NIFC history archive has no "CZU" entry.) |
| Microsoft US Building Footprints | Wagner's building mask (2 m buffer). |
| NEON Vegetation Structure `DP1.10098.001` | Field-mapped stems ≥ 10 cm DBH with live/dead status. |

---

## 6. Candidate training and evaluation datasets

| Dataset | Labels | Aligned to NAIP | Fit | Status |
|---|---|---|---|---|
| **Allred pairs → our crowns** (3DEP) | LiDAR crowns (our segmentation) | ❌ ±6 m + lean | Scale, our fires, pre-fire | In use at CZU; registration unsolved |
| **Dixon crown polygons** | LiDAR crowns + dead/alive (human) | ❌ | Best mortality labels; 3 of our fires | Email request needed |
| **Ventura urban-tree-detection-data** | 95,972 points | ✅ native | Pretraining on real NAIP | Public on GitHub (licence unchecked) |
| **NEON (all sites) + NEON LiDAR crowns** | LiDAR crowns | ✅ (to NEON imagery) | Clean labels, simulated NAIP | Untested — Stage B used DeepForest *predictions* instead |
| **Hemming-Schroeder SOAP/TEAK** | ~1M LiDAR polygons, dead/alive, incl. Creek Fire | ❌ to NAIP | Validation; hold out of training | Zenodo `10.5281/zenodo.7812035`; not opened |
| Stovall SOAP/TEAK | 1.8M trees, dead/alive (contested) | ❌ | Tree locations reliable; dead labels disputed | figshare `10.6084/m9.figshare.7609193.v1` |
| Cheng et al. | Dead trees only (sample of 2,546 in code release) | ✅ native | Dead-tree detector only | Full labels/weights not released |
| NeonTreeEvaluation | ~6,200 high-confidence evaluation crowns + ~21,500 training crowns, 22 sites incl. SJER, SOAP, TEAK; each plot ships its own 1 m CHM and 10 cm RGB, co-registered | ✅ (to NEON) | Benchmark for our segmentation | Zenodo 5914554 (v0.2.2); **used in §7.2** |
| Yosemite Forest Dynamics Plot | Every stem ≥ 1 cm, 25.6 ha, Rim Fire | — | Pre/post counting validation | Access terms unchecked |
| Open Forest Observatory | 150+ drone sites, stem-mapped plots, western US | ❌ | Downsample for extra data | Public STAC |
| MillionTrees | Boxes 197k / polygons 760k / points 256k | ❌ | Pretraining pool | Only TreeBoxes downloaded |
| deadtrees.earth, OAM-TCD | cm-scale, dead wood / crowns | ❌ | Downsample | Public |
| ~~Auto Arborist~~ | — | — | — | Shut down May 2025 |

**Recommendation.** Use 3DEP-derived crowns for scale and domain match; NEON for a clean,
aligned comparison and for validation; Ventura for real-NAIP pretraining. Settle "NEON vs
3DEP for training" with a head-to-head on our hand-labelled test set (NEON-only,
3DEP-only, NEON→3DEP). Keep SOAP out of training if it is the validation site.

---

## 7. Pipelines

### 7.1 LiDAR → crown labels (implemented: `baselines/chm_crowns.py`)

1. CHM from LiDAR (Allred already did this: lidR `pitfree`, 1 m).
2. 3×3 median smooth.
3. Treetops: variable-window local maxima — 4 m below 32 m tall, h/8 for 32–80 m, 10 m above; > 5 m tall.
4. Crowns: marker watershed on the CHM, restricted to > 3 m.
5. Keep crowns with top > 5 m.
6. **Still needed:** mask buildings (Microsoft footprints), drop crowns adjacent to no-data / water, reject long-thin or flat-topped regions (power lines, roofs).
7. Transfer to NAIP (unsolved — see §7.3).

At CZU (bin `10-411`): 267 chips overlap the perimeter (221 inside), 125,019 crowns,
median crown 8.3 m (vs 7.4 m in our hand labels). LiDAR Apr 2020 / NAIP May 2020 (190
chips) or Apr 2018 / Aug 2018 (77). CZU spans bins 409–412; only 411 is downloaded.
Given §7.2, the 125,019 is likely a **substantial undercount** of trees: many "crowns"
are merged clusters, which also explains why they come out larger than hand labels.

### 7.2 How good is the segmentation? (So's NEON benchmark)

`baselines/watershed_eval/` — full write-up in `lidar_annotation_benchmarking.md`.
Runs our unchanged `crowns_from_chm` on the 1 m CHM shipped with each NeonTreeEvaluation
plot and scores it against the human crowns. NEON's RGB and LiDAR are already
co-registered, so this isolates **segmentation error** from registration error.
Protocol matches the benchmark paper: IoU ≥ 0.4, one-to-one, per image then averaged.

| Run | Labels | Crowns found | P / R per image | Matched | Merged |
|---|---|---|---|---|---|
| evaluation, 5 m min height | 6,199 | 3,104 | **0.40 / 0.37** | 14% | **77%** |
| evaluation, 3 m | 6,199 | 3,269 | 0.39 / 0.39 | 15% | 77% |
| training tier, 5 m | 21,460 | 10,531 | 0.28 / 0.18 | 15% | 76% |
| *DeepForest (paper)* | | | *0.66 / 0.79* | | |
| *human–human agreement* | | | *~70–90%* | | |

- **Merging dominates.** 77% of human crowns fall inside a watershed crown that swallowed
  several trees; the watershed finds about half as many crowns as humans label.
- **Driven by crown size:** < 10 m² match 2%, 10–25 m² 15%, 25–50 m² 31%, 50–100 m² 61%,
  > 100 m² 76%.
- **California:** SJER oak woodland 0.56 / 0.64 (52% matched); **TEAK mixed conifer — the
  analogue for CZU — 0.30 / 0.25, 76% merged**; SOAP 0.17 / 0.06 (only 2 plots).
- **No systematic shift:** median centre offset ≤ 0.1 m (≈ 1 m scatter). **Boxes too big:**
  matched boxes are 1.44× the label area (≈ 1.2× per side).
- Height threshold (3 vs 5 m) barely matters; edge-touching crowns are not the cause.
- **Likely causes:** the 3×3 median smoothing fills the 1–2 m gaps between small crowns,
  and the 4 m treetop window can't keep two tops 2–3 m apart. Next: an ablation of both.
- Caveats: annotators draw the *visible* crown, not the 3D one; NEON's point density
  (~5 pts/m²) and CHM processing differ from 3DEP.

### 7.2.1 Tuning the segmentation

Scripts in `baselines/watershed_eval/`: `watershed_grid.py`, `dalponte_grid.py`,
`forest_type_grid.py`, `neontree_stems.py`. **Split:** 80/20 by plot within each site
(seed 0; sites with < 3 plots all in train) → 153 train / 34 val plots, 972 val labels,
22 of 34 val plots at SJER/TEAK. Settings are always chosen on **train** IoU-F1 (F1 of the
per-image mean P/R, IoU ≥ 0.4); val is reported only. Val plots share sites with train, so
this tests new plots, not new sites.

**Watershed grid** (144 configs: smoothing, treetop radius under 32 m, relative-height
trim `rel`, crown-diameter cap `max_cr`, min height):

| | Val P / R | Val F1 | Matched | Merged | Box area vs label |
|---|---|---|---|---|---|
| Current (median 3×3, radius 2, no trim) | 0.44 / 0.41 | 0.421 | 17% | 74% | 1.42× |
| **Tuned** (Gaussian σ 0.5, radius 2, `rel` 0.65, `max_cr` 10 m) | 0.51 / 0.49 | **0.499** | 29% | 40% | 1.22× |

- Val TEAK 0.405 → 0.606; val SJER 0.571 → 0.554. Chosen config ranks 5/144 on val;
  train–val F1 correlation over the grid 0.89.
- **Trimming does most of the work** (F1 0.421 → 0.495 alone) but finds no extra trees.
  Lighter smoothing finds ~19% more crowns (518 → 619 on val), still far short of 972 labels.
- **Smaller treetop windows didn't help**, contrary to the §7.2 hypothesis. Min height
  3 vs 5 m: negligible.

**Region growing instead of watershed** (same treetops):

- **Dalponte 2016**, ported line for line from lidR `C_dalponte2016`. lidR never grows
  from border pixels, which on 40 × 40 px plots left ~¼ of crowns as single edge pixels;
  the CHM is edge-padded 1 px to match lidR on a full raster. `max_cr` is a square
  half-width (lidR default 10 → crowns up to ~19 m). lidR defaults: val F1 0.411; best of
  96 configs 0.419. Train–val correlation over the grid is −0.22, i.e. its knobs other than
  `max_cr` are noise.
- **Silva 2016**, ported from lidR `silva2016`: each pixel goes to its nearest treetop and
  is kept if h ≥ `exclusion` × top and distance ≤ `max_cr_factor` × top (the code uses a
  radius; the docs say diameter). lidR defaults: F1 0.365 on all plots; best global 0.431.
- Why the watershed wins: region growing puts boundaries by distance, roughly halfway
  between treetops; the watershed follows the canopy valleys.

**Separate settings per forest type** (oak woodland SJER/CLBJ; conifer TEAK/SOAP/NIWO/
WREF/ABBY/BONA; deciduous; southern pine). Val F1, global setting → per-type setting:

| | All val | Oak woodland (12) | Conifer (14) | Deciduous (3) | S. pine (5) |
|---|---|---|---|---|---|
| Watershed | 0.499 → **0.519** | 0.554 → 0.620 | 0.502 → 0.498 | 0.410 → 0.410 | 0.386 → 0.386 |
| Dalponte | 0.419 → 0.487 | 0.515 → 0.604 | 0.404 → 0.479 | 0.324 → 0.398 | 0.268 → 0.268 |
| Silva | 0.431 → 0.486 | 0.553 → 0.572 | 0.390 → 0.476 | 0.308 → 0.450 | 0.310 → 0.316 |

- Watershed per-type settings: oak woodland Gaussian σ 1 + `rel` 0.45; conifer radius 1;
  others = global. The gain is all oak woodland; TEAK 0.606 → 0.601.
- Silva's per-type search picked `max_cr_factor` 0.2 for conifer and 0.9 for oak woodland,
  exactly Weinstein et al. 2020's hand-picked values — but still below the watershed.
- Deciduous and southern pine have 3 and 5 val plots: no conclusions there.

**Field stems** (`neontree_stems.py`). NEON Woody Vegetation Structure stems from the
NeonTreeEvaluation R package (`data/field.rda`), filtered as its docs describe (DBH > 15 cm,
height > 3 m, height change ≤ 8 m, |CHM − field height| < 4 m), matched to plots by
location. **Stem recall** = one-to-one stem↔containing-box matches / stems (the package's
version can exceed 1 when boxes overlap); **merged** = share of stem-containing boxes with
≥ 2 stems. No precision (not every tree is mapped), so more boxes can inflate recall.

| All 79 plots, 1,001 stems | Stem recall | Merged |
|---|---|---|
| Human labels | 0.75 | 0.14 |
| Watershed, global tuned | 0.72 | 0.27 |
| Watershed, per type | 0.68 | 0.26 |

- **Oak woodland:** per-type setting raises box F1 but drops stem recall 0.85 → 0.66
  (humans 0.86) — better box shapes, fewer trees. Wrong trade for counting.
- **Conifer** (stems almost all NIWO and BONA): per-type radius 1 raises stem recall
  0.50 → 0.60 (humans 0.61); global setting merges 54% of boxes vs humans' 20%.
- **No California conifer stems.** TEAK and SOAP were surveyed once (2015) and their
  mapped stems fall outside the benchmark images. The package's "> 1 survey" rule is
  dropped to keep SJER (20 plots, 68 stems); with it there are 44 plots and none in CA.

**Decision: keep the original watershed for 3DEP.** On the CZU chips (267 chips; 141,763
tuned vs 125,019 original crowns) the tuned crowns looked **too small** against the trees
in NAIP: the 65%-of-top trim and 5 m radius cap cut each crown down to its upper core.
The NEON gain likely reflects how annotators box the visible crown on 10 cm RGB, which
doesn't carry over. Any future tuning should be checked by eye on 3DEP, not only on the
NEON boxes; a gentler trim (`rel` ≤ 0.45, no cap) is untested there. Caveats: the 34 val plots have now been used for four
comparisons; confirm gains with other split seeds or leave-site-out before relying on them.
Viewer: `make_neontree_viewer.py OUT current=… tuned=… dalponte=…` (labels vs up to several
crown sets, linked RGB/CHM panels, per-plot review export).

### 7.3 NAIP–LiDAR registration

Sources of mismatch:

1. **NAIP positional error** — up to ±6 m; one shift per chip would fix it.
2. **Relief displacement** — NAIP is orthorectified to bare earth, so treetops lean by ≈ height × tan(view angle), away from the frame centre. A 20 m tree at 15° off-nadir shifts ~5 m (illustrative). Different for every tree.
3. **Mosaic seams** — direction can jump within a chip.
4. Shadows, grid differences, and up to 2 years between acquisitions.

**What we tried.** Cross-correlating the CHM against NAIP NDVI over integer shifts (±12 m).
Estimated offsets: median 3.0 m, p90 6.3 m. Control — fraction of LiDAR treetops on
sunlit canopy: as paired 17.6%, **estimated shift 13.2%**, random shift of same size 17.1%;
better than no shift on only 18% of chips. **The estimate was worse than random.** NDVI is
not height, shadows bias it, and no single shift is correct when trees lean.

**Ideas.** (a) Register on ground features — roads, shorelines, clearings. (b) Fit
`displacement = t + h·v` per chip (translation + height × lean vector), 4 parameters —
untested. (c) Use tolerant targets: point heatmaps with a few metres' slack, or
crown-fraction rules like Stovall's 37.5%. (d) Measure the real error at NEON SOAP
against NEON's aligned 10 cm imagery.

**Hand-measured ground truth.** The alignment viewers let you nudge labels and export
offsets; 10–20 saved chips would give the first measured error distribution for this
dataset.

### 7.4 Post-fire imagery (implemented: `baselines/fetch_postfire_naip.py`)

Reads each chip's footprint from Planetary Computer COGs and resamples onto the pre-fire
chip grid. All 267 CZU chips covered by the 19 May 2022 flight (~21 months post-fire).
The post-fire flight carries its own, independent offset.

### 7.5 Human-in-the-loop annotation (design, not built)

Model-proposed labels verified by people (Papadopoulos 2016). Risks specific to us:
anchoring on loose boxes; the model never shows trees it missed (~38% at IoU 0.25 even at
score 0.05); one-tree-at-a-time is slow in dense canopy. Better: pre-annotate at score
0.05 (1.43 candidates per real tree), review per tile, mandatory "add missed trees" pass,
seed gold tiles, double-assign ~20% of tiles to measure agreement, retrain between rounds.
Points instead of boxes remove the anchoring problem.

---

## 8. Code and artefacts

| Path | What |
|---|---|
| `dfft/README.md`, `modal_train.py`, `modal_sweep.py`, `modal_viz.py`, `modal_inspect.py`, `viz_out/` | Committed to `df-ft` (`6b894a7`) |
| `dfft/modal_compare.py`, `modal_predict_all.py`, `modal_minsize_sweep.py`, `modal_nir_finetune.py`, `modal_learning_curve.py`, `modal_tuned.py` | Later experiments — untracked (`dfft/` is gitignored on `main`) |
| `dfft/viz_out/*.json` | sweep, min_size, NIR, learning-curve, counting results |
| `baselines/veg_index.py`, `modal_veg_index.py`, `modal_veg_predict.py` | Non-ML baseline |
| `baselines/chm_crowns.py`, `select_fire_chips.py`, `fetch_postfire_naip.py`, `make_alignment_viewer.py`, `make_crown_figures.py` | LiDAR crown + alignment pipeline — committed to branch `lidar-crowns`, with step-by-step `baselines/REPRODUCE_LIDAR_CROWNS.md` |
| `baselines/watershed_eval/neontree_{fetch,watershed,eval}.py`, `lidar_annotation_benchmarking.md` | So's NeonTreeEvaluation benchmark of the segmentation (§7.2) — on `lidar-crowns` |
| `baselines/watershed_eval/{watershed_grid,dalponte_grid,forest_type_grid,neontree_stems,make_neontree_viewer}.py` | Tuning, Dalponte/Silva ports, per-forest-type selection, field-stem scoring, review viewer (§7.2.1) |
| `pyproject.toml`, `uv.lock`, `.python-version` | Dependencies for `baselines/`, managed with `uv` (`uv run python …`) |
| `baselines/alignment_viewer/`, `baselines/alignment_viewer_czu/` | Alignment viewers (30 MB / 75 MB, untracked) |
| `single_tiles_flat/make_tile_viewer.py`, `stage_c_data/viewer/`, `readable/` | Tile viewer and readable JSON |
| Modal volume `dfft-checkpoints` | `deepforest_finetuned.pt`, `deepforest_ft_ms{800,1000,1300}.pt`, `deepforest_ft_{3,4}ch.pt`, `deepforest_tuned_{baseline,reg_aug}.pt`, `predictions_all.json`, `metrics.json`, `sweep.json` |
| Scratchpad (temporary, may vanish) | CZU CHM/NAIP tars, post-fire chips, figures |

**Environment gotchas.**
- Local scipy ≥ 1.12 won't load on this macOS; pin `scipy<1.12`.
- Reading GeoTIFFs from inside a tar: the `MemoryFile` must stay alive while the dataset is open, since GDAL reads tiles lazily. Use `chm_crowns.open_member` (a context manager since `d6e484b`); the earlier version silently decoded freed memory, which looked like corrupt chips.
- Field stems need `pyreadr` + `pandas` (reads the R package's `field.rda`).
- Modal account has no payment method: GPU rejected, CPU containers get preempted (Lightning swallows SIGTERM as `SystemExit`).
- The vendored `DeepForest/` repo does not match PyPI `deepforest==2.1.0` (no `train.optimizer` block; augmentations take `list[str]`).

---

## 9. Open questions and next steps

1. **Merging: still open** (§7.2.1) — the NEON-tuned watershed (Gaussian σ 0.5, trim to 65% of top, 10 m cap) made crowns too small on 3DEP at CZU, so the original stays the default; Dalponte/Silva don't beat it on NEON either. Still open: confirm with other split seeds / leave-site-out; adopt the conifer per-type setting only if it holds; find California conifer stems (TEAK/SOAP tower plots outside the benchmark images) to check counting there; try a 0.5 m CHM from the benchmark's point clouds. Then test whether training on these crowns helps DeepForest on NAIP before tuning further.
2. **3DEP pre-fire coverage for SCU, North Complex, Creek, Castle** — decides whether the LiDAR route works beyond CZU. USGS publishes project footprints and dates.
3. **Measure registration** — save 10–20 hand offsets in the CZU viewer; check whether tall trees stay offset after short ones align (relief displacement). Less urgent until crowns are resolved individually.
4. **NEON SOAP viewer** — NEON LiDAR crowns vs NAIP 2020/2022, NEON 10 cm as reference; check whether 2024 CA NAIP exists to match NEON 2024.
5. **Fix crown filters** — buildings, water edges, power lines.
6. **Train on points**, not boxes, with metre-scale tolerance.
7. **Finish the anchor arm** and run NIR across 3–5 seeds — after adding checkpoint-resume (or fixing Modal billing).
8. **Requests:** Dixon (crown polygons), Cheng (checkpoint + labels); PDFs for Hemming-Schroeder 2023, Li et al. 2023, Stovall supplementary.
9. **Measure the human ceiling** — triple-annotate ~10 tiles.
