# Does pretraining on 3DEP LiDAR crowns help DeepForest on NAIP?

One question: if DeepForest is first trained on LiDAR-derived crowns drawn on real
pre-fire NAIP, and then fine-tuned on our hand labels, does it beat the model we
have now? Everything else is held fixed.

| Arm | Training | Question it answers |
|---|---|---|
| **A** | DeepForest prebuilt (NEON) → fine-tune on the 75 hand-labelled train tiles | Current model, the baseline |
| **B** | prebuilt → pretrain on 3DEP LiDAR crowns → fine-tune exactly as A | Do LiDAR labels on NAIP add anything? |
| **C** | prebuilt → pretrain on 3DEP LiDAR crowns, no fine-tuning | How good are the LiDAR labels on their own? |

B and C share one pretraining run per seed. Run 3 seeds: earlier differences between
training variants were a few F1 points, about the size of seed noise.

**Scoring.** The 15 hand-labelled test tiles. Score threshold and NMS are chosen on the
18 val tiles (best F1 at IoU 0.4); test is scored once. Reported: P/R/F1 at IoU 0.25,
0.4, 0.5 and count MAE, on all tiles and on tiles with ≥ 5 crowns (the report's
convention), plus a fixed reference setting (score 0.1, NMS 0.05). Raw predictions
are saved so anything can be re-scored later.

## Pipeline

```
select_chips.py  CPU + network   CHM chips under each fire, away from val/test tiles
build_pretrain.py CPU + network  crowns + pre-fire NAIP windows -> pretrain.csv
train.py         GPU             arms A, B, C for one seed
summarize.py     any             results table
```

Steps 1-2 need no GPU and can run on a laptop; copy `WORK/pretrain/` and the repo to
the GPU machine. Or run everything on the GPU machine.

## Setup

```bash
git clone -b lidar-crowns https://github.com/soh09/mortalitree && cd mortalitree
python -m venv .venv && source .venv/bin/activate        # Python 3.10-3.12
pip install -r lidar_pretrain/requirements.txt           # deepforest 2.1.0 pulls in torch
export WORK=/path/with/space/lidar_pretrain_work          # ~4 GB peak with --delete-tars
```

On a CUDA machine, install the CUDA build of torch first if pip picks the CPU one
(`pip install torch --index-url https://download.pytorch.org/whl/cu121`).

## 1. Select chips (CPU, ~1-2 h, mostly download)

```bash
python lidar_pretrain/select_chips.py $WORK --delete-tars
```

Downloads the CHM tar for every bin under the five fires (27 bins, 0.2-2.2 GB each,
~26 GB in total, one at a time), extracts the eligible chips (~100 KB each) to
`$WORK/chm_chips/`, then deletes each tar. Eligible = overlaps the fire perimeter,
LiDAR flown before the fire, and > 1 km from every val/test tile.
`$WORK/coverage.json` reports, per fire, how many chips have pre-fire LiDAR — this
also answers whether 3DEP covers SCU, North Complex, Creek and Castle.

Re-runs skip finished downloads; `--fires czu castle` limits the fires.

## 2. Build the pretraining set (CPU, ~20-40 min)

```bash
python lidar_pretrain/build_pretrain.py $WORK --max-chips 400 --workers 8
```

Per chip (random order, up to 400 per fire): crowns from the pipeline's **original**
watershed (`baselines/chm_crowns.crowns_from_chm`; the NEON-tuned setting made crowns
too small on 3DEP), each crown's bounding rectangle; pre-fire NAIP for the chip from
Microsoft Planetary Computer (latest year before the fire, within 2 years of the
LiDAR); four overlapping 256 × 256 px windows, stretched exactly like the hand-labelled
tiles. Crowns are not shifted to correct NAIP–LiDAR misregistration.

Expect roughly 400 chips × 5 fires × 4 windows ≈ 8,000 images and ~1.5 M boxes
(overlapping windows count a crown more than once). Resumable (`done.jsonl`).

## 3. Train (GPU)

```bash
bash lidar_pretrain/run_all.sh $WORK 0 1 2        # or: python lidar_pretrain/train.py $WORK --seed 0
```

Defaults: pretraining 3 epochs, lr 0.001, batch 8; fine-tuning 15 epochs, lr 0.001,
batch 4 (as `dfft/modal_train.py`). All are flags (`--pretrain-epochs`, …;
`--workers` for data loading). Per seed, outputs go to `$WORK/runs/seed{S}/`:
`pretrained.ckpt` (reused if present), `{A,B,C}/metrics.json`,
`{A,B,C}/raw_predictions_{val,test}.csv`, and Lightning CSV logs.

Smoke test without a GPU (minutes; numbers meaningless):

```bash
python lidar_pretrain/train.py $WORK --finetune-epochs 1 --pretrain-epochs 1 \
       --max-pretrain-images 4 --workers 0 --accelerator cpu --pretrain-batch 2
```

## 4. Summarise

```bash
python lidar_pretrain/summarize.py $WORK
```

Mean ± sd over seeds for each arm. Read B − A: if B doesn't beat A by more than the
seed spread, LiDAR pretraining isn't worth scaling.

## Things to keep in mind

- **Pre-fire only.** LiDAR crowns are pre-fire, so pretraining uses pre-fire NAIP only.
  Half of the hand-labelled tiles (and of the test tiles) are post-fire 2022 imagery.
- **Density mismatch.** LiDAR gives ~190 boxes per window vs ~70 on a hand-labelled
  tile (the watershed counts small and understory crowns humans skip). Arm C will
  likely over-count against the hand labels; fine-tuning (B) should recalibrate.
- **Misregistration is left in.** Boxes can be several metres off the trees in NAIP.
  Weinstein et al. (2019) found poor LiDAR labels still helped pretraining.
- **The test set is small** (15 tiles, 997 crowns) and spatially clustered.
- **Spatial separation.** No pretraining chip is within 1 km of a val/test tile.
  Chips may overlap *train* tiles, which A also trains on.

## Clay arms (`clay_app.py`)

The same question for the Clay v1.5 + DQ-DETR detector (`dq-detr-impl/`, untracked;
found next to the repo or via `$DQDETR_DIR`, used unmodified):

| Arm | Training |
|---|---|
| **ClayA** | existing Stage B checkpoint (`clay-checkpoints:/dq-detr_final.pt`, 4-band NEON) → Stage C on the 75 hand-labelled tiles |
| **ClayB** | Stage B checkpoint → **LiDAR stage** (3DEP crowns on raw 4-band pre-fire NAIP; `stage_b_finetune`'s recipe: flat lr 5e-4, CGFE on, encoder frozen, 10 epochs) → Stage C as ClayA |

- LiDAR windows: `build_pretrain.py --clay-dir` on the same 400-chips-per-fire set, written as
  raw R,G,B,NIR GeoTIFFs + `labels_clay.csv` (the `mot`/`NeonPatchDataset` layout);
  windows with > 600 boxes (DQ-DETR's largest query bin) are left out.
- Stage C: `stage_c(model="dqdetr")` settings — 80 epochs, last 2 encoder blocks unfrozen
  after 30, early stopping on val F1, best checkpoint kept; seeds set explicitly.
- Scoring: as the DeepForest arms, plus "NMS 1.0" (none) in the val-selected grid.
  The fixed reference row is Clay's own setting (conf 0.5, no NMS).

```bash
modal deploy lidar_pretrain/clay_app.py
python lidar_pretrain/clay_app.py --stage build,train      # headless; status in pipeline_status_clay.json
modal run lidar_pretrain/clay_app.py::train_arm --seed 0 --arm ClayB --smoke --data pretrain_clay_test
```

Results: `lidar-pretrain-work:/work/runs_clay/seed{S}/{ClayA,ClayB}/`, `summary_runs_clay.md`;
wandb groups `clay-seed{S}`.
