# Clay Tree Detector

NAIP tree detection using Clay v1.5 + ViTDet neck + DETR-style box head.
See `clay_tree_detection_full_spec.md` for full design rationale.

## File layout

```
src/
  data/
    augmentation.py       — rotate90, flip, brightness/contrast, per-band gain jitter
    naip_dataset.py       — 4-band GeoTIFF loader with metadata extraction
    deepforest_dataset.py — RGB-only Stage B wrapper (3 wavelengths, no NIR)
  model/
    clay_loader.py        — loads Clay v1.5 (large, dim=1024, patch=8), wraps forward → (B, 1024, 28, 28) stride-8 map
    neck.py               — StrippedViTDetNeck with bilinear-init deconv
    head.py               — BoxDetectionHead + SinCos2DPositionalEncoding
    detector.py           — full pipeline module with unfreeze helper
  train/
    losses.py             — Hungarian matching loss (cls + L1 + GIoU), batch version
    stage_b.py            — RGB pretraining loop with early stopping
    stage_c.py            — NAIP finetune loop with frozen → unfreeze schedule
    schedulers.py         — cosine + warmup, separate LR groups for encoder vs neck/head
  eval/
    metrics.py            — F1@IoU, count MAE/RMSE/R², mAP@0.5:0.95, crown area R², per-ha aggregation
    inference.py          — tile → filtered boxes, confidence threshold sweep
    visualize.py          — diagnostic tile plots with GT/pred boxes overlaid
configs/
  stage_b.yaml            — Stage B hyperparameters
  stage_c.yaml            — Stage C hyperparameters
  naip_normalization.yaml — Clay NAIP per-band mean/std (verify from Clay metadata.yaml)
train.py                  — CLI entry point for Stage B / Stage C / both
test_shapes.py            — shape sanity tests (run before training)
```

## Quick start

### 1. Install dependencies

```bash
pip install torch torchvision scipy rasterio pyyaml matplotlib
# Install Clay from the official repo:
pip install git+https://github.com/Clay-foundation/model.git
```

### 2. Verify normalization stats

[done by Evan 6/3] Open `configs/naip_normalization.yaml` and replace the placeholder values with the
actual per-band mean/std from `configs/metadata.yaml` in the Clay repo (R, G, B, NIR order).

### 3. Check Clay encoder block attribute name

In `src/model/clay_loader.py`, `unfreeze_last_n_blocks` looks for `blocks`,
`transformer_blocks`, or `layers`. Confirm which attribute your Clay version uses
and adjust if needed.

### 4. Run shape tests

```bash
python3 test_shapes.py
```

All 7 tests should pass before connecting the Clay encoder.

### 5. Prepare annotation JSON files

Each annotation file is a JSON array of tile records:

```json
[
  {
    "tile_path": "data/naip/tile_001.tif",
    "boxes": [[0.52, 0.48, 0.09, 0.11], ...],
    "exhaustive": true,
    "lat": 37.42,
    "lon": -119.88,
    "acquisition_date": "2020-07-15",
    "hour_of_day": 11.5
  }
]
```

`boxes` are `(cx, cy, w, h)` normalized to `[0, 1]` relative to tile dimensions.
Use separate files for Stage B (DeepForest/NEON RGB tiles) and Stage C (NAIP tiles).
**Split Stage C by geographic region, not randomly.**

### 6. Train

```bash
python3 train.py \
  --clay_checkpoint path/to/clay-v1.5.ckpt \
  --stage both \
  --stage_b_annotations data/deepforest_annotations.json \
  --train_annotations   data/naip_train.json \
  --val_annotations     data/naip_val.json \
  --checkpoint_dir      checkpoints/
```

Use `--stage b` or `--stage c` to run a single stage.
For Stage C only, pass `--stage_b_checkpoint checkpoints/stage_b_best.pt`.

## Training stages

| Stage | Data | Encoder | Trainable params | Purpose |
|-------|------|---------|-----------------|---------|
| B | DeepForest NEON RGB | Frozen | Neck + head (~3M) | Pre-train neck/head on abundant RGB crowns |
| C (epochs 1–30) | NAIP 4-band | Frozen | Neck + head | Adapt to NAIP spectral domain |
| C (epochs 31+) | NAIP 4-band | Last 2 blocks unfrozen | Neck + head + 2 blocks | Fine-tune NIR-aware features |

## Training on Modal

### One-time setup

```bash
pip install modal
modal setup          # authenticate

modal volume create clay-data
modal volume create clay-checkpoints
```

### Upload data and Clay checkpoint

```bash
# Annotation JSONs and tile directories
modal volume put clay-data  data/naip_train.json   /naip_train.json
modal volume put clay-data  data/naip_val.json     /naip_val.json
modal volume put clay-data  data/deepforest.json   /deepforest.json
modal volume put clay-data  tiles/                 /tiles/

# Clay v1.5 checkpoint
modal volume put clay-checkpoints  clay-v1.5.ckpt  /clay-v1.5.ckpt
```

### Run training

```bash
# Both stages in sequence (recommended)
modal run modal_train.py

# Stage B only
modal run modal_train.py::stage_b

# Stage C only (Stage B checkpoint must already be in the volume)
modal run modal_train.py::stage_c
```

Override any default parameter:

```bash
modal run modal_train.py -- --stage c --train_annotations /data/naip_train.json
```

### Download checkpoints

```bash
modal volume get clay-checkpoints stage_c_best.pt  ./checkpoints/stage_c_best.pt
```

---

## Evaluation

After training, run a confidence threshold sweep on the validation set:

```python
from src.eval.inference import threshold_sweep
from src.model.clay_loader import load_clay_encoder
from src.model.detector import TreeDetector

encoder = load_clay_encoder("checkpoints/clay-v1.5.ckpt")
model = TreeDetector(encoder)
model.load_state_dict(torch.load("checkpoints/stage_c_best.pt")["model"])

results = threshold_sweep(model, "data/naip_val.json", device="cuda")
print(f"Best F1={results['best_f1']:.3f} at threshold={results['best_threshold']:.2f}")
```

Key metrics reported (matching spec §7):
- Per-tile count MAE, RMSE, R²
- Box-level precision, recall, F1 at IoU = 0.5
- mAP@0.5 and mAP@0.5:0.95
- Crown area R² and rRMSE (matched pairs at IoU ≥ 0.5)
- Per-hectare count MAE and RMSE (100 m × 100 m cells)
