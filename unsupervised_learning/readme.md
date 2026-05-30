# Implementation Spec — Swin-Based NIR-Aware Tree Detector

A developer-facing summary of the architecture and training pipeline, with tensor shapes, key code patterns, and hyperparameter defaults. Companion to the project architecture summary; this one is for actually writing code.

## Pipeline at a glance

```
4-band NAIP tile (R,G,B,NIR)
        │
        ▼
inflated patch-embed stem  (Conv2d 4→C, kernel=4, stride=4)
        │
        ▼
Swin backbone  (ImageNet-pretrained, then NAIP-MAE-pretrained)
        │
        ▼
FPN neck  (fuse multi-scale features → P2..P5 at 256 channels)
        │
        ▼
DETR-style box detection head on P2
   (Q learnable queries → transformer decoder → cls + box per query)
        │
        ▼
tree boxes  (cx, cy, w, h, score),   count = number of confident boxes
```

## Input format

- Tensor: `(B, 4, H, W)`, default H=W=224. Channel order R, G, B, NIR.
- Normalize each band independently using NAIP-corpus statistics. **Do not use ImageNet stats for the NIR channel.** Compute mean/std per band from a representative slice of your unlabeled NAIP pool.
- Labels: bounding boxes kept as boxes throughout. Internal format `(cx, cy, w, h)`, all normalized to `[0, 1]` relative to tile dimensions.
- Annotation flag: per tile, a boolean `exhaustive` indicates whether every tree in the tile is labeled. Sparsely-annotated tiles handle the classification loss differently (see §Loss).

## Component 1 — Inflated patch-embed stem

Take an ImageNet-pretrained Swin and replace its 3-channel first conv with a 4-channel version. Copy RGB weights; initialize NIR as the mean of RGB weights.

```python
import torch
import torch.nn as nn

def inflate_patch_embed_to_4ch(backbone):
    old = backbone.patch_embed.proj                       # Conv2d(3, C, 4, 4)
    new = nn.Conv2d(4, old.out_channels,
                    kernel_size=old.kernel_size,
                    stride=old.stride,
                    padding=old.padding)
    with torch.no_grad():
        new.weight[:, :3] = old.weight                    # copy RGB
        new.weight[:, 3:4] = old.weight.mean(             # NIR ← mean(RGB)
            dim=1, keepdim=True
        )
        if old.bias is not None:
            new.bias.copy_(old.bias)
    backbone.patch_embed.proj = new
    return backbone
```

Why mean-of-RGB init: gives the NIR channel a reasonable "vegetation-ish channel" starting behavior so MAE pretraining has somewhere sensible to start from. Random init wastes pretraining; identical-to-red init biases the model to ignore the new channel.

## Component 2 — Swin backbone (multi-scale features)

Use `timm` with `features_only=True` to retrieve all four stage outputs.

```python
import timm
backbone = timm.create_model(
    'swin_tiny_patch4_window7_224',
    pretrained=True,
    features_only=True,
    out_indices=(0, 1, 2, 3),
)
backbone = inflate_patch_embed_to_4ch(backbone)
```

Output shapes for Swin-Tiny at 224×224 input:

| Feature | Shape | Stride | Ground/cell at 60 cm |
|---|---|---|---|
| C2 | (B, 96, 56, 56) | 4 | 2.4 m |
| C3 | (B, 192, 28, 28) | 8 | 4.8 m |
| C4 | (B, 384, 14, 14) | 16 | 9.6 m |
| C5 | (B, 768, 7, 7) | 32 | 19.2 m |

The detection head runs on the finest level (P2, stride 4, 2.4 m/cell) since tree crowns are ~4 m. Coarser stages contribute semantic context via the FPN but are not where boxes come from.

## Component 3 — FPN neck

Standard top-down FPN. `torchvision` provides one out of the box.

```python
from torchvision.ops import FeaturePyramidNetwork

fpn = FeaturePyramidNetwork(
    in_channels_list=[96, 192, 384, 768],
    out_channels=256,
)

# usage
feats = {'C2': c2, 'C3': c3, 'C4': c4, 'C5': c5}
pyramid = fpn(feats)
p2 = pyramid['C2']   # (B, 256, 56, 56) — primary detection scale
```

Start with the head on P2 only. Adding P3 (multi-scale detection) is a later optimization if you see misses on larger crowns.

## Component 4 — Detection head (DETR-style box head)

A small DETR-style decoder with a fixed set of object queries. Each query attends to the P2 feature map and outputs a box `(cx, cy, w, h)` plus an objectness score. No anchors, no NMS — Hungarian matching during training enforces one-prediction-per-object behavior at inference.

```python
class BoxDetectionHead(nn.Module):
    """DETR-style box head with Q object queries, operating on a single neck level."""
    def __init__(self, feat_channels=256, num_queries=32, hidden=256,
                 n_heads=8, n_decoder_layers=3):
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Embedding(num_queries, hidden)
        self.pos_enc = SinCos2DPositionalEncoding(hidden)
        self.input_proj = nn.Conv2d(feat_channels, hidden, kernel_size=1)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden * 4,
            batch_first=True, norm_first=True, dropout=0.1,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_decoder_layers)
        self.cls_head = nn.Linear(hidden, 1)
        self.box_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 4),  # (cx, cy, w, h)
        )

    def forward(self, feat):
        # feat: (B, feat_channels, H_p2, W_p2)  e.g. (B, 256, 56, 56)
        B, _, H, W = feat.shape
        x = self.input_proj(feat)                            # (B, hidden, H, W)
        pos = self.pos_enc(H, W, x.device)                   # (H*W, hidden)
        memory = x.flatten(2).transpose(1, 2) + pos.unsqueeze(0)  # (B, H*W, hidden)
        q = self.queries.weight.unsqueeze(0).expand(B, -1, -1)    # (B, Q, hidden)
        decoded = self.decoder(q, memory)                    # (B, Q, hidden)
        cls_logits = self.cls_head(decoded).squeeze(-1)      # (B, Q)
        boxes = self.box_head(decoded).sigmoid()             # (B, Q, 4) in [0, 1]
        return cls_logits, boxes


class SinCos2DPositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert dim % 4 == 0
        self.dim = dim

    def forward(self, H, W, device):
        d = self.dim // 4
        y = torch.arange(H, device=device).float().unsqueeze(1).expand(H, W)
        x = torch.arange(W, device=device).float().unsqueeze(0).expand(H, W)
        freqs = torch.exp(
            torch.arange(0, d, device=device).float()
            * (-torch.log(torch.tensor(10000.0)) / d)
        )
        pe_y = torch.cat([torch.sin(y[..., None] * freqs),
                          torch.cos(y[..., None] * freqs)], dim=-1)
        pe_x = torch.cat([torch.sin(x[..., None] * freqs),
                          torch.cos(x[..., None] * freqs)], dim=-1)
        return torch.cat([pe_y, pe_x], dim=-1).reshape(H * W, self.dim)
```

**Why Q = 32 (small)?** Standard DETR uses 100 queries for COCO (many classes, many objects per image). Your tiles have 20–100 trees and only one class. A small `Q` is enough, and each query is parameters and compute. Start with Q = 32 if tiles routinely have >20 trees; drop to 16 if sparser, raise to 64 if denser. Rule of thumb: `Q ≈ 1.5 × max_trees_per_tile`.

**Why only 3 decoder layers?** Standard DETR uses 6. At ~200 tiles, fewer layers means fewer from-scratch parameters and more stable training. 3 is a reasonable compromise.

**Why sigmoid box outputs?** Normalized box coordinates in `[0, 1]` are scale-invariant and well-behaved for L1 loss. Convert to pixel coords only at evaluation / visualization.

## Loss — Hungarian matching with L1 + GIoU box losses

For each image, match the `Q` predicted boxes to ground-truth boxes with Hungarian assignment, then apply classification + L1 + generalized IoU losses on matched pairs.

```python
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops import generalized_box_iou, box_convert

def matching_loss(cls_logits, pred_boxes, gt_boxes, tile_exhaustive,
                  lam_cls=1.0, lam_l1=5.0, lam_giou=2.0):
    """
    cls_logits:      (Q,) raw scores
    pred_boxes:      (Q, 4) (cx, cy, w, h) in [0, 1]
    gt_boxes:        (M, 4) (cx, cy, w, h) in [0, 1]
    tile_exhaustive: bool — if True, unmatched queries are pushed to "no tree";
                     if False (sparsely annotated), only matched queries contribute
                     to the classification loss (avoids false-negative gradients
                     from real but unlabeled trees).
    """
    Q, M = cls_logits.shape[0], gt_boxes.shape[0]
    cls_prob = cls_logits.sigmoid()

    pred_xyxy = box_convert(pred_boxes, in_fmt="cxcywh", out_fmt="xyxy")
    gt_xyxy   = box_convert(gt_boxes,   in_fmt="cxcywh", out_fmt="xyxy")

    # --- Cost matrix ---
    cls_cost  = -cls_prob.unsqueeze(1).expand(Q, M)
    l1_cost   = torch.cdist(pred_boxes, gt_boxes, p=1)
    giou_cost = -generalized_box_iou(pred_xyxy, gt_xyxy)
    cost = lam_cls * cls_cost + lam_l1 * l1_cost + lam_giou * giou_cost

    q_idx, gt_idx = linear_sum_assignment(cost.detach().cpu().numpy())

    # --- Classification loss ---
    target = torch.zeros(Q, device=cls_logits.device)
    target[q_idx] = 1.0
    if tile_exhaustive:
        cls_loss = F.binary_cross_entropy_with_logits(cls_logits, target)
    else:
        mask = torch.zeros(Q, dtype=torch.bool, device=cls_logits.device)
        mask[q_idx] = True
        cls_loss = F.binary_cross_entropy_with_logits(
            cls_logits[mask], target[mask]
        )

    # --- Box regression losses on matched pairs ---
    matched_pred = pred_boxes[q_idx]
    matched_gt   = gt_boxes[gt_idx]
    l1_loss   = F.l1_loss(matched_pred, matched_gt)

    matched_pred_xyxy = box_convert(matched_pred, "cxcywh", "xyxy")
    matched_gt_xyxy   = box_convert(matched_gt,   "cxcywh", "xyxy")
    giou_loss = (1.0 - generalized_box_iou(
        matched_pred_xyxy, matched_gt_xyxy
    ).diag()).mean()

    return lam_cls * cls_loss + lam_l1 * l1_loss + lam_giou * giou_loss
```

The combined L1 + GIoU box loss is the DETR standard. L1 alone is poorly scale-aware (a 1-pixel error on a 7-pixel crown is much worse than on a 70-pixel building); GIoU adds the scale-aware component.

**Sparse-annotation handling.** The `tile_exhaustive` flag controls whether unmatched queries are pushed toward "no tree." On exhaustively-labeled tiles, all `Q` queries contribute to the classification loss (matched → 1, unmatched → 0). On sparsely-labeled tiles, only matched queries contribute — pushing unmatched queries toward 0 would create false-negative gradients on real but unlabeled trees. Track this flag per tile in your dataloader.

## Training Stage A — Domain-adaptive MAE pretraining

**Goal:** teach the 4-channel encoder NAIP's spectral structure, especially NIR.

- Use an MAE codebase (`facebookresearch/mae` or `timm`'s MAE) and swap in the inflated 4-channel patch embed. The MAE decoder must reconstruct all 4 channels.
- Data: 5,000–20,000 unlabeled NAIP tiles from your study region. More is better. Below ~5k the benefit shrinks.
- Mask ratio: 0.75.
- Loss: per-patch normalized MSE across all 4 channels (the original MAE trick — without it, the model "saves effort" by predicting patch means).
- Optimizer: AdamW, lr 1.5e-4 (cosine schedule, ~10 warmup epochs), weight decay 0.05, batch 64+.
- Epochs: 100–200. Many ablations show diminishing returns past 100 in domain-adaptive settings.
- After pretraining: discard the decoder. Keep encoder weights only.

## Training Stage B (optional) — Head pretrain on RGB tree datasets

**Goal:** teach the detection head what an overhead tree looks like using abundant labels before exposure to your tiny NAIP set.

- Data: DeepForest (NEON) or TreeFormer datasets — both RGB.
- Input: 3-channel RGB; set the NIR channel to zero, or use a 3-channel forward path that skips the NIR slot. (Zero-pad is simpler and works fine.)
- Train: detection head + last 1–2 Swin blocks. Freeze the rest.
- Epochs: ~50–100, validated against a holdout split.

Optional. Skip if pressed for time, but it is genuinely the lowest-risk way to compensate for your label scarcity.

## Training Stage C — Supervised finetuning on labeled NAIP

**Goal:** adapt the full model to your specific forests with real NIR.

- Data: your ~200 annotated NAIP tiles. Bounding boxes used directly (no center conversion). Track per-tile `exhaustive` flag.
- Input: full 4-channel NAIP, real NIR.
- Train: detection head + Swin upper blocks. Use differential learning rates (backbone 1e-5, head 1e-3).
- Epochs: 50–100 with early stopping on a held-out validation set (use mAP@0.5 or count F1).
- Loss: the Hungarian matching loss with L1 + GIoU box terms above.

## Data pipeline

**Tiling.** Extract 224×224 windows (134 m × 134 m at 60 cm). For very dense crowns try 160×160 (96 m) so each Swin patch covers fewer crowns. Use stride < tile_size for training (overlap = more samples); non-overlapping for evaluation.

**Augmentations.** Apply identical geometric transforms to all 4 channels:
- Random 90 / 180 / 270° rotation
- Random horizontal and vertical flip
- Mild uniform brightness/contrast jitter (applied identically across bands — do not jitter NIR independently of RGB; this breaks spectral relationships)
- Optional: small per-channel gain jitter to simulate inter-year NAIP radiometric drift (modest amplitude, e.g. ±5%)

**Normalization.** Compute per-band mean and std from the NAIP pool. Apply at dataload time. Do not skip the NIR-specific stats.

**Splits.** Always split train/val/test by **geographic region**, never by random crop within a region. Spatial autocorrelation will leak performance otherwise.

## Hyperparameter defaults to start from

| Hyperparameter | Default | Notes |
|---|---|---|
| Tile size | 224×224 | Try 160×160 in dense forest |
| Swin variant | Swin-Tiny | Only step up to Small if Tiny underfits |
| FPN channels | 256 | Standard |
| Object queries Q | 32 | Rule of thumb: ~1.5× max trees per tile |
| Decoder layers | 3 | Standard DETR uses 6; fewer is safer at 200 tiles |
| Decoder heads | 8 | |
| MAE mask ratio | 0.75 | Standard |
| MAE LR | 1.5e-4 cosine, 10 epoch warmup | AdamW, wd 0.05 |
| MAE batch | 64+ | Whatever your GPU allows |
| MAE epochs | 100–200 | Diminishing returns past 100 |
| Detection LR | head 1e-3, backbone 1e-5 | AdamW, wd 0.05 |
| Detection batch | 8–16 | Memory-bound |
| Detection epochs | 50–100 | Early stop on val |
| `λ_cls` | 1.0 | |
| `λ_l1` | 5.0 | DETR default |
| `λ_giou` | 2.0 | DETR default |
| Confidence threshold | 0.5 | Sweep on val for best F1 / AP |

## Things to verify before training

- **NAIP normalization stats are real, not ImageNet stats** — especially for NIR.
- **Per-tile `exhaustive` flag is correctly handled** in the classification loss (full BCE on exhaustive tiles; matched-only on sparse tiles).
- **MAE per-patch normalization** is on in Stage A.
- **Geographic train/val/test split**, not random.
- **Inflated stem weights actually loaded** — check that `new.weight[:, :3]` matches the original after instantiation.
- **MAE encoder weights actually transferred** to the detection model — easy to load the wrong checkpoint or have key-name mismatches.
- **Box outputs are in normalized [0, 1] coords**, not pixels. Sigmoid is on the box head output; targets must also be normalized.
- **GIoU is computed in (x1, y1, x2, y2) format** while L1 and matching use (cx, cy, w, h). Use `torchvision.ops.box_convert`; getting this wrong silently corrupts the loss.
- **`generalized_box_iou` returns a pairwise (Q, M) matrix.** After Hungarian assignment, you need the diagonal of the matched subset (the `.diag()` in the GIoU loss snippet), not the full matrix.
- **`Q ≥ max trees per tile`** with headroom, or Hungarian matching silently drops GT objects. Audit your dataset's max tree count before fixing Q.

## Eventual extension to pre/post-fire mortality

The architecture does not need to change. Apply the trained model independently to a pre-fire and a post-fire tile of the same area:

1. Run inference on both → two box sets per tile.
2. Per-hectare mortality estimate: aggregate to 100 m × 100 m grid cells; `Δcount = count_pre − count_post` per cell.
3. Per-tree mortality (harder): co-register the two tiles (phase correlation), match nearest boxes across time steps (e.g. by IoU or center distance), then classify each match as survived/died using a small post-process model that reads each tree's post-fire NIR signature (or ΔNDVI) at its location.
4. Crown-area-weighted analyses (uniquely possible with box output): use predicted box areas as proxy crown areas to weight mortality estimates by tree size or biomass.

The detector stays the same; mortality logic lives downstream of inference.

## File / module layout suggestion

```
src/
  data/
    naip_dataset.py        # tiling, normalization, box loading, augmentation, exhaustive flag
    deepforest_dataset.py  # optional Stage B
  model/
    stem.py                # inflate_patch_embed_to_4ch
    backbone.py            # Swin via timm, features_only
    neck.py                # FPN wrapper
    head.py                # BoxDetectionHead (DETR-style) + SinCos2DPositionalEncoding
    detector.py            # full pipeline module
  train/
    mae_pretrain.py        # Stage A
    head_pretrain.py       # Stage B (optional)
    finetune.py            # Stage C
    losses.py              # Hungarian matching loss (cls + L1 + GIoU)
  eval/
    metrics.py             # MAE/RMSE/R² per-tile count; mAP@0.5 and 0.5:0.95; crown-area R²/rRMSE
    inference.py           # tile → boxes
configs/
  mae.yaml
  head_pretrain.yaml
  finetune.yaml
```

Build and test components in isolation before stacking: a forward pass through stem → backbone → neck → head with random input should produce the expected shapes before any training loop is wired up.