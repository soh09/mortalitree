# DeepForest baseline (dfft)

Fine-tunes the pretrained DeepForest (`weecology/deepforest-tree`, RetinaNet) on the
hand-annotated NAIP tiles in `single_tiles_flat/stage_c_data/`, as the baseline the
Clay/GFM models are compared against in the report.

## Weights live on Modal, not here

The original local run's checkpoint (`dfft/runs/deepforest_finetuned.pt`) was gitignored
and lost. Everything now goes to the Modal volume **`dfft-checkpoints`**:

```
deepforest_finetuned.pt   fine-tuned weights (257 MB)
metrics.json              baseline vs finetuned, both threshold conventions
sweep.json                inference-time grid (min_size x nms x score)
viz/                      qualitative figures
```

```bash
modal volume get dfft-checkpoints /deepforest_finetuned.pt .
```

## Scripts

| Script | What it does |
|---|---|
| `modal_train.py` | prep + fine-tune + eval in one Modal job; writes weights to the volume |
| `modal_sweep.py` | inference-time sweep (min_size x nms_thresh x score_thresh); selects on val, touches test once |
| `modal_viz.py` | qualitative test-set figures (threshold sweep, IoU localization diagnostic) |
| `modal_inspect.py` | prints the anchor/transform config the prebuilt model actually uses |
| `prep_data.py`, `train.py`, `eval.py`, `export_stats.py` | the original local-only versions |

```bash
modal run dfft/modal_train.py          # ~34 min on CPU (GPU needs a payment method on the account)
modal run dfft/modal_sweep.py
modal run dfft/modal_viz.py
```

## Results

Re-derived 2026-09-24 with `deepforest==2.1.0`, 15 epochs, lr 1e-3, batch 4, seed 42.

| | P | R | F1 | MAE | RMSE |
|---|---|---|---|---|---|
| baseline (pretrained) | 0.127 | 0.124 | 0.125 | 32.5 | 40.61 |
| fine-tuned | 0.245 | 0.233 | 0.239 | 28.42 | 36.42 |

IoU 0.5, score 0.1, `min_gt=5` — the `export_stats.py` convention, which is what the
report's Tables 1 and 2 used (the baseline row reproduces them exactly). Note `eval.py`
defaults to score 0.3 and will *not* reproduce the published numbers.

Caveat: the report's F1 column is not the harmonic mean of its own P/R (0.127/0.124
implies 0.1255, the table says 0.120), most likely a mean of per-image F1.

## Known bottleneck

Localization, not detection. Test recall is 0.21 at IoU 0.5 but **0.50 at IoU 0.25** and
**0.62 at IoU 0.1** — the boxes land on trees but sit too loosely to match. Confidence
thresholding therefore buys very little (+0.014 F1); the counting error is where tuning
helps (MAE 27.3 -> 19.2). See `viz_out/localization.png`.
