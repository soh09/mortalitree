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
`max_trees` filter throws away. Each 256-px source patch is split into an
`n_split × n_split` grid (default 2×2 → 128-px tiles, Q=150). A non-empty 256 patch
of ~581 crowns becomes four ~145-crown tiles, right at budget.

**Box assignment (the key decision).** Each box is assigned to the **single tile
that contains its center**, then clipped to that tile. A crown straddling an
internal seam is owned by exactly one tile; the adjacent tile sees the poke-in
pixels but has no target there — so the model learns to fire only when a crown's
center is in-tile. This is what prevents systematic double-counting when we stitch
predictions back together. (Mirrors what `modal_pipeline.py` already does at 256.)

**Overlap.** None for now. Center-assignment handles seams cleanly, so overlap's
only payoff is recovering crowns whose clipped half is too weak to detect — at the
cost of more windows + heavier dedup. Left as a future knob, to add only if seam
error analysis shows it matters.

**Metrics — virtual stitching (head-to-head).** The model trains/predicts on 128
tiles, but evaluation is on the reassembled 256 patch: each quarter's predicted
boxes are mapped into the parent frame, NMS de-dups (IoU 0.6) any seam double-fire,
and the result is scored against the full 256 GT — the same GT the un-quartered
baseline uses. `stitch_eval` also runs unchanged on a non-quartered dataset, so
Evan's baseline can be scored the same way.

**Where it lives.**
- `clay/src/data/neon_dataset.py` — `quarter` / `n_split` / `max_trees` args;
  per-tile center-assignment + clip; `parent_gt` (full 256 GT), `crop`, `group_keys`.
- `clay/configs/stage_b.yaml` → `data:` block:
  ```yaml
  quarter: true
  n_split: 2        # 2 -> 128px (Q=150); 4 -> 64px (set num_queries ~40)
  max_trees: 150    # drop tiles over the query budget at train time (0 = keep all)
  ```
- `clay/src/eval/stitch.py` — `stitch_eval(...)`.
- `clay/modal_train.py` — `eval_stage_b` entrypoint.

**Run.**
```bash
modal run modal_train.py::stage_b          # trains on quarters (config-driven)
modal run modal_train.py::eval_stage_b      # parent-level (256) metrics on held-out val
# baseline comparison: set quarter:false, retrain, eval the same way
```

**Caveats / open items.**
- `num_queries` (in the `model:` block) is **not** auto-coupled to `n_split`. If you
  switch to `n_split: 4`, lower `num_queries` to ~40 yourself.
- If the paper's Q degradation is absolute (not object-count-relative), Q=150 is
  still too high and we should run `n_split: 4` / Q≈40. Tile size is a one-line
  config sweep, no regen needed.
- Train/val split is now **parent-grouped** (`_split_train_val` in
  `clay/src/train/stage_b.py`) so a parent's quarters don't leak across the split.
- For a *rigorous* head-to-head, build a separate test `labels.csv` and run both
  models with `eval_stage_b --split all` so they're scored on identical parents/GT.

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
