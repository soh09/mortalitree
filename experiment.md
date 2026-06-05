# Experiments — Stage B tree detector

Working notes for the experiments we're running on the Clay + ViTDet-neck + DETR
tree detector (full architecture in `clay/clay_tree_detection_full_spec.md`). These
are all **Stage B** (RGB/4-band pretraining of the from-scratch neck + head, Clay
encoder frozen). Branch: `exp/aux-focal-loss`.

## Baseline

Evan is running the **canonical recipe** un-quartered: 256-px tiles, Q≈150 queries,
plain BCE classification, supervision on the final decoder layer only, ~50 epochs.
Everything below is meant to be compared head-to-head against that run on the same
parent-tile ground truth.

## Status at a glance

| # | Experiment | Goal | Status |
|---|---|---|---|
| A | Quarter patching | Keep trees/tile within the query budget *and* recover dense-forest patches | **Implemented** |
| B | Focal classification loss | Fix the query class imbalance (mostly-negative queries) | Planned |
| C | Deep supervision (auxiliary loss on non-output decoder layers) | Speed up DETR convergence under the short (~50 epoch) schedule | Planned |

---

## Motivating problems

1. **Density-filter bias.** The patch pipeline (`dataset_dev/modal_pipeline.py`)
   drops 256 patches with > `max_trees` (=150) crowns. At that cap it kept only
   ~2.5% of annotations / ~26% of patches, and the *kept* set averages ~24
   crowns/ha vs ~323 crowns/ha for what was dropped. So training data is biased
   toward sparse/open stands, while the downstream fire-mortality target is dense
   forest. We must not just raise `max_trees` — see (2).
2. **Q is sensitive.** The reference paper reports performance *degrading* as the
   number of object queries Q grows (degradation reported beyond ~30). So "just
   raise Q to cover dense tiles" is off the table — it's the opposite of what we
   want. (Open question: was the degradation absolute, or relative to objects/tile?
   That decides our target tile size — see Experiment A.)
3. **DETR converges slowly + query imbalance.** Vanilla DETR needs many epochs;
   at a ~50-epoch budget it may be undertrained, and with Q≈150 queries vs ~57
   objects/tile most queries are easy negatives that swamp the BCE loss. Motivates
   Experiments B and C.

---

## Experiment A — Quarter patching  (Implemented)

**Idea.** Instead of raising Q, shrink the *tile* so trees-per-tile stays within a
modest query budget — which simultaneously recovers the dense patches the per-256
`max_trees` filter throws away (a 400-crown 256 region the baseline drops wholesale
becomes four ~100-crown 128 tiles, all kept). The data is **regenerated natively at
128 px** by the pipeline, on a 2×2 grid aligned to the 256 parent grid so quarters
nest exactly into a parent (essential for stitching + head-to-head).

> ⚠️ Earlier we briefly quartered the *already-filtered* 256 `labels.csv` on the fly
> — that only subdivided the sparse kept patches and did **not** recover density
> (the dense patches were never written by the 256 pipeline). The native pipeline
> below fixes that. The on-the-fly path is retained only as a sparse-data control.

**Box assignment (the key decision).** Each box is assigned to the **single quarter
that contains its center**, then clipped. A crown straddling a seam is owned by one
quarter; the adjacent quarter has no target there — so the model fires only when a
crown's center is in-tile, which prevents systematic double-counting at stitch time.

**Density filter — mirrors Evan.** Quarters with > `max_trees` (=150) crowns are
**dropped at write time** (same "drop tiles over threshold" rule Evan uses, now at
the quarter granularity). So the clean head-to-head is on the ≤150-crown parents
(Evan's domain); on those, all four quarters survive and both models fully cover them.

**Overlap.** None. Center-assignment handles seams; overlap's only payoff is
recovering crowns whose clipped half is too weak to detect, at the cost of more
windows + dedup. A future knob if seam error analysis shows it matters.

**Metrics — virtual stitching (head-to-head).** The model trains/predicts on 128
tiles, but evaluation is on the reassembled 256 patch: each quarter's predicted
boxes are mapped into the parent frame, NMS de-dups (IoU 0.6) any seam double-fire,
and the result is scored against the full 256 GT (`labels_parent.csv`) — the same GT
the un-quartered baseline uses. `stitch_eval` runs unchanged on a non-quartered
dataset, so Evan's baseline is scored the same way.

**Per-epoch parent metrics.** The per-quarter train/val loss isn't comparable to a
256 run's loss (different objects/tile + Q). So each epoch also logs **parent-level
count MAE, F1, and the stitched matching loss** (`stitch_eval` on val + a fixed
1000-parent train subset) — the cross-run-comparable signal. See `run_stage_b`
(`parent_eval*` args).

**Where it lives.**
- `dataset_dev/modal_pipeline.py` — `--quarter`: 2×2-aligned 128-px tifs to
  `/data/patches_quarter/`, plus `labels_quarter.csv` (per-quarter, with
  `parent`/`q_row`/`q_col`) and `labels_parent.csv` (full 256 GT).
- `clay/src/data/neon_dataset.py` — **pre-quartered mode** (`parent_labels_csv` set):
  reads 128 tifs directly, `parent_gt` from `labels_parent.csv`, separate read-window
  vs. parent stitch-offset; `group_keys` for the grouped split.
- `clay/configs/stage_b.yaml` → `data:` block:
  ```yaml
  prequartered: true
  tile_size: 128      # quarter tif size on disk
  parent_size: 256    # parent frame for stitching + GT
  max_trees: 0        # quarters already capped by the pipeline at write time
  ```
- `clay/src/eval/stitch.py` — `stitch_eval(...)`; `clay/modal_train.py` — `stage_b`
  + `eval_stage_b` (both `--out-tag patches_quarter`).

**Run.**
```bash
# 1. regenerate quarters (drops quarters >150 at write time)
modal run dataset_dev/modal_pipeline.py --quarter --max-trees 150
# 2. train on quarters; parent metrics logged each epoch
modal run clay/modal_train.py::stage_b --out-tag patches_quarter
# 3. parent-level (256) metrics on held-out val
modal run clay/modal_train.py::eval_stage_b --out-tag patches_quarter
# baseline: pipeline without --quarter + config prequartered:false, then same eval
```

**Caveats / open items.**
- `num_queries` (model block) **must match** the pipeline's per-quarter `--max-trees`
  (both 150 here). For `n_split: 4` / 64-px quarters you'd drop both to ~40.
- If the paper's Q degradation is *absolute* (not object-count-relative), Q=150 is
  still too high → regenerate with a finer split (64 px) and Q≈40.
- A parent with a dropped (>150) quarter is scored against its **full** 256 GT, so
  that quarter's trees count as misses — a deliberately conservative metric. Moot on
  the ≤150-parent head-to-head set (no quarter dropped there).
- Train/val split is **parent-grouped** (`_split_train_val`) — no quarter leakage.
- For a rigorous held-out number, build a separate test set and run both models with
  `eval_stage_b --split all` on identical parents/GT.

---

## Experiment B — Focal classification loss  (Planned)

**Problem.** The head's objectness branch uses plain BCE. After Hungarian matching
only M queries are positive (M = boxes in the tile) and the other (Q − M) are
pushed to 0; with Q≈150 vs ~57 objects, the loss is dominated by easy negatives.

**Change.** Replace BCE with focal loss in `matching_loss`
(`clay/src/train/losses.py`):
```
focal = -α (1 - p_t)^γ · log(p_t)        # α=0.25, γ=2.0 to start
```
i.e. `torchvision.ops.sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0)`.
The `(1 - p_t)^γ` factor down-weights confident-correct easy examples so gradient
flows to the hard ones. Standard in modern DETRs (Deformable DETR, DINO). The
`exhaustive` masking logic is unchanged (focal only replaces what's computed on the
contributing queries).

**Cost / risk.** ~5 lines. Adds two hyperparameters (defaults robust). Changes loss
scale, so `λ_cls` may want a small retune later.

---

## Experiment C — Deep supervision via auxiliary losses  (Planned)

**Problem.** DETR converges slowly; the 3-layer decoder is supervised only at its
final layer (`clay/src/model/head.py`), so layers 1–2 learn only indirectly. At a
~50-epoch budget the baseline may be undertrained.

**Change.** Attach the *same* `cls_head` + `box_head` (shared weights) to the output
of **every** decoder layer, run the Hungarian matching loss at each, and sum:
```
total = Σ over layers ℓ  matching_loss(cls_ℓ, box_ℓ, gt)
```
Each layer gets a direct "produce valid detections now" gradient → faster, more
stable convergence (this is what the DETR papers credit for making it train).
Requires returning intermediate decoder outputs (`nn.TransformerDecoder` doesn't by
default — loop the layers and collect each output, ~20 lines). **Inference is
unchanged** (use the last layer only), so zero deployment cost; training does ~3×
matching, which is negligible next to the Clay forward.

**Where it'll live.** `clay/src/model/head.py` (return per-layer outputs),
`clay/src/train/stage_b.py` / `losses.py` (sum the per-layer loss), with a config
flag in `stage_b.yaml` to toggle.

---

## Suggested run matrix

| Run | quarter | n_split / Q | focal | aux loss | Purpose |
|---|---|---|---|---|---|
| Evan baseline | false | — / 150 | no | no | reference |
| A | true | 2 / 150 | no | no | quartering effect |
| B | false | — / 150 | yes | no | focal effect |
| C | false | — / 150 | no | yes | deep-supervision effect |
| A+B+C | true | 2 / 150 | yes | yes | combined |

Each scored at the parent (256) level via `stitch_eval` for direct comparison.
