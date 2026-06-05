# Project Spec — NAIP Tree Detection with Clay + ViTDet Neck

A complete specification for a vision-transformer-based tree detector that finetunes the Clay geospatial foundation model with a lightweight ViTDet-style neck and a point-detection head. Targeted at NAIP 4-band aerial imagery (RGB + NIR) at 60 cm resolution, with a small labeled training set on the order of 200 tiles. Eventual purpose: counting trees in matched pre-fire and post-fire imagery to estimate fire-induced mortality.

---

## 1. Why this design

### Task
Detect individual trees in NAIP 4-band aerial imagery (R, G, B, NIR) at 60 cm GSD and output a bounding box per tree. Counting falls out as `count = number_of_detected_boxes`. Available labels are bounding boxes (~200 annotated tiles, ~20–100 trees each); the head is trained to predict boxes directly. Crown extent is preserved in the output, which supports downstream analyses (e.g. crown-area-weighted mortality estimates).

### How this differs from prior work

The literature on tree detection from aerial imagery splits into three camps, each with a gap this design targets.

| Prior work | Backbone | Pretraining | NIR | Resolution | Key limitation for this task |
|---|---|---|---|---|---|
| Tong & Zhang (StarDist, 2025) | U-Net (CNN) | None (from scratch) | None | 15 cm | Crown polygons collapse to ~58% accuracy at 60 cm |
| TreeFormer (Amirkolaee et al., 2023) | Pyramid ViT | ImageNet | None | 12–80 cm | Density regression loses per-tree identity needed for pre/post matching |
| DeepForest (Weinstein et al.) | RetinaNet (CNN) | NEON tree dataset | None | sub-meter | RGB only, no NIR signal for live/dead discrimination |
| Cheng et al. (Nature Comms, 2024) | U-Net (CNN) | None (from scratch) | Real NIR (passive 4th channel) | NAIP 60 cm | CNN, no self-supervised pretraining, ~15–25% underestimation on fire-mortality |
| Zhang, Lei & Fan (RS, 2025) | Pretrained transformer + PEFT | ImageNet (generic ViT) | None | UAV (sub-meter) | Generic pretraining, not sensor-specific; UAV not NAIP; RGB only |

Two observations frame the contribution. First, **fire-related tree mortality is much harder than non-fire mortality detection** (Cheng et al. report fire MAE 4.56 vs non-fire 1.91 dead trees/ha), and **NAIP's NIR band is the most direct spectral signal for live/dead vegetation** but is underused — none of the transformer-based work above leverages it. Second, while Zhang et al. (2025) demonstrate that pretrained transformers with parameter-efficient fine-tuning work for tree crown detection, they use generic ImageNet-pretrained ViTs without sensor- or spectral-specific pretraining.

The defining choice of this design is the encoder: **Clay v1.5**, an MAE-pretrained ViT whose pretraining corpus is roughly 30% NAIP (~21 million NAIP chips, more than any other sensor in its training set). Clay has already done extensive self-supervised pretraining on NAIP imagery including the NIR band — at a scale individual researchers cannot replicate. Using Clay therefore gives the project a state-of-the-art NAIP-aware encoder for free.

The novelty contributions, stated precisely:

1. **First use of a geospatial foundation model (Clay) for individual-tree detection.** Geospatial foundation models have been evaluated on NAIP downstream tasks (e.g. SatMAE in the PhilEO benchmark, ~72% land-cover accuracy), but not for tree detection. Clay specifically has documented finetuning recipes for segmentation, classification, and regression; tree detection at sub-meter resolution is not among them.
2. **Contrast with Zhang et al. (2025):** that paper uses an ImageNet-pretrained generic transformer; this design uses an MAE-pretrained sensor-aware foundation model whose pretraining matches the deployment sensor (NAIP) and includes the spectral band most relevant to the eventual fire-mortality task (NIR).
3. **A ViTDet-style neck adapted for the small-object regime of 60 cm NAIP.** Plain ViTs output single-scale features; Clay v1.5 (patch size 8) emits them at stride 8, and the neck upsamples once more to stride 4 to better resolve small crowns. Building the multi-scale pyramid a detection head needs is non-trivial when training data is limited.
4. **Bounding box detection from a foundation-model backbone, sized for the data-limited regime.** A DETR-style box head replaces the original point-detection plan to preserve crown-size information for the eventual mortality-by-biomass analysis, while a modest number of object queries (124) and a stripped-down neck keep the from-scratch parameter count compatible with ~200 labeled tiles.
5. **A training schedule explicitly designed for the data-limited regime** — frozen encoder, mandatory Stage B pretraining on abundant RGB tree datasets, late unfreezing of upper encoder blocks, with a stripped-down neck to keep from-scratch parameters small.

### What this design is *not* claiming
- Not a new architecture (Clay, ViTDet neck, P2PNet head are all published).
- Not a new pretraining method (Clay's MAE is reused, not redone).
- Not a fire-mortality model (yet) — that's the downstream extension. The deliverable is a tree detector that supports the eventual mortality work.

The contribution is the synthesis: a recipe that combines existing components into a tree detector well-suited to 60 cm NAIP with NIR, trainable on ~200 labeled tiles.

---

## 2. Architecture overview

```
NAIP tile + metadata
        │
        ▼
Clay v1.5 dynamic embedding block
   (consumes pixels + wavelengths + GSD + lat/lon + time)
        │
        ▼
Clay ViT encoder    (~311M params, large variant, frozen → late-unfrozen)
   output: (B, 1024, 1024)    32×32 patches × 1024-dim
        │
        ▼  reshape tokens → spatial map
   (B, 1024, 32, 32)    stride 8
        │
        ▼
Stripped ViTDet neck    (~1M params, from scratch)
   single 2× deconv → P3: (B, 128, 64, 64)    stride 4
        │
        ▼
DETR-style box detection head    (~2M params, from scratch)
   Q learnable object queries (Q=124) → transformer decoder over neck features
       cls branch:  (B, Q, 1)   tree objectness per query
       box branch:  (B, Q, 4)   (cx, cy, w, h) per query, normalized to [0, 1]
        │
        ▼
At training: Hungarian matching between Q predicted boxes and box-center labels;
             BCE classification (masked) + L1 box loss + generalized IoU loss
At inference: keep boxes with cls_score > τ; count = |kept boxes|
```

**Total trainable params at Stage C with frozen encoder: ~3M (neck + head).** Larger than the point-head variant but still small enough to train on 200 tiles with the mitigations below.

---

## 3. Components

### 3.1 Input format

A batch is a dict with image and metadata:

```python
batch = {
    "pixels":      Tensor (B, 4, H, W),    # R, G, B, NIR  — note Clay's channel order
    "wavelengths": Tensor (4,),            # [0.665, 0.560, 0.493, 0.842] in μm for NAIP
    "gsd":         Tensor (B,),            # 1.0 (Clay's NAIP pretraining GSD)
    "time":        Tensor (B, 4),          # [sin(week), cos(week), sin(hour), cos(hour)]
    "latlon":      Tensor (B, 4),          # [sin(lat), cos(lat), sin(lon), cos(lon)]
}
```

**`time` and `latlon` are 4-dim sin/cos encodings, not raw values.** Clay's
`add_encodings` builds its positional encoding at width `dim - 8` and reserves
the trailing 8 dims for `hstack((time, latlon))`, so each must be a 4-element
cyclic encoding (total 8). Passing raw 2-dim `(week, hour)` / `(lat, lon)` makes
the metadata 4 wide, the encoding comes out at `dim - 4`, and it fails to add to
the `dim`-wide patch tokens. The exact transforms (from Clay's datamodule /
wall-to-wall tutorial), with `week` in 1..52, `hour` in 0..24, `lat`/`lon` in degrees:

```python
week_a = week * 2*pi/52;  hour_a = hour * 2*pi/24
time   = [sin(week_a), cos(week_a), sin(hour_a), cos(hour_a)]
lat_r  = lat * pi/180;    lon_r  = lon * pi/180
latlon = [sin(lat_r), cos(lat_r), sin(lon_r), cos(lon_r)]
```

Default tile size: `H = W = 256`. This matches NAIP's native 256×256 tile size, so no resampling is needed at inference. Clay v1.5 uses patch size 8 with dynamic (GSD-aware) positional encoding, so any multiple of 8 works; 256 → 32×32 token grid (1024 tokens). Stage B (RGB pretraining) and Stage C (NAIP finetuning) should both use 256×256 to avoid a train/test grid mismatch and to keep the normalized box-size prior consistent across stages. The encoder dim is also 1024, so the token tensor is `(B, 1024_tokens, 1024_dim)` — the two 1024s are coincidental, not the same axis.

**Channel order matters.** Clay's NAIP config uses **R, G, B, NIR** order (`configs/metadata.yaml` → `naip.band_order = [red, green, blue, nir]`; do not assume B,G,R or generic RGB). Match the dataloader's wavelength order and normalization stats to this band order. Native NAIP GeoTIFFs are already R,G,B,NIR, so reading bands in file order is correct.

**GSD.** Pass `gsd = 1.0` — the value Clay's NAIP dynamic embedding was pretrained against (`metadata.yaml` lists NAIP at gsd 1.0). This drives the GSD-aware positional encoding, so matching the pretraining value is the safe choice even though the imagery's true ground sampling is ~60 cm. Keep the true 60 cm resolution only for converting box sizes to physical crown-area / per-hectare metrics (§7) — that is a separate, real-world quantity from the metadata GSD fed to the encoder.

**Normalization.** Use Clay's NAIP per-band normalization stats from `configs/metadata.yaml` (`naip.bands.mean`/`naip.bands.std`, keyed by band name, in R,G,B,NIR order: mean `[110.16, 115.41, 98.15, 139.04]`, std `[47.23, 39.82, 35.43, 49.86]` on the 0–255 DN scale), not ImageNet stats and not ones you compute yourself. This is a free correctness win — Clay's encoder was trained against those exact stats.

**Labels.** Bounding boxes kept as boxes throughout. Internal format:
```python
box = (cx, cy, w, h)   # all normalized to [0, 1] relative to tile dims
```
Per tile, an `annotated_mask` (or equivalent flag) indicates whether the tile is exhaustively labeled or only sparsely annotated. The classification loss for unmatched queries is computed only on tiles flagged as exhaustively annotated — for sparsely annotated tiles, only matched-pair losses contribute (since "no tree" predictions in unlabeled regions can't be confirmed).

### 3.2 Clay encoder

Load Clay v1.5 from HuggingFace or the official repo. The released v1.5.0
checkpoint is the **large** encoder (dim 1024, depth 24, patch size 8):

```python
from claymodel.module import ClayMAEModule  # or equivalent loader

clay = ClayMAEModule.load_from_checkpoint(
    "path/to/clay-v1.5.ckpt",
    strict=False,
)
encoder = clay.model.encoder
encoder.mask_ratio = 0.0     # return all patch tokens
encoder.shuffle = False      # keep tokens in spatial (row-major) order
encoder.eval()
for p in encoder.parameters():
    p.requires_grad = False  # frozen for Stage B and most of Stage C
```

**Do not call the base `Encoder.forward`.** It runs `mask_out()`, which — when the
checkpoint was trained with `shuffle: True` (v1.5.0 was) — returns the patch tokens
in *shuffled* order even at `mask_ratio=0`, silently scrambling the spatial map.
Instead replicate Clay's own finetuning encoder (`claymodel/finetune/segment/factory.py`,
`SegmentEncoder.forward`): patch-embed → add positional/metadata encoding → prepend
cls token → transformer → drop cls token. The metadata dict uses the key `waves`
(not `wavelengths`), and `gsd` must be a **scalar** (Clay builds one positional
encoding per batch; pass `gsd[0]` if your collate produced a `(B,)` tensor):

```python
patches, _ = encoder.to_patch_embed(batch["pixels"], batch["wavelengths"])
patches = encoder.add_encodings(patches, batch["time"], batch["latlon"], gsd_scalar)
cls = encoder.cls_token.expand(B, -1, -1)
patches = encoder.transformer(torch.cat([cls, patches], dim=1))[:, 1:, :]  # drop cls
# patches: (B, 1024, 1024)   # N_tokens = 1024 (32×32), D = 1024
```

Reshape the patch tokens to spatial form (read the dims off the encoder rather
than hardcoding them — never hardcode 1024 here, since both the token count and
the encoder dim happen to be 1024 at tile size 256 and they are not the same axis):

```python
B, N, D = patches.shape                 # N = 1024, D = encoder.dim = 1024
H_p = W_p = H // encoder.patch_size      # 256 // 8 = 32
spatial = patches.transpose(1, 2).reshape(B, D, H_p, W_p)   # (B, 1024, 32, 32)
```

### 3.3 Stripped ViTDet neck

The neck manufactures one higher-resolution feature map from Clay's single-scale stride-8 output. A single 2× upsample takes it to stride 4 (P3 only), with 128 channels.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class StrippedViTDetNeck(nn.Module):
    """Single-scale neck: stride 8 → stride 4, 1024 → 128 channels."""
    def __init__(self, in_channels=1024, out_channels=128):
        super().__init__()
        # Project channels down before upsampling (cheaper, more stable)
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # 2× upsample. Use deconv with bilinear-init for stable from-scratch training.
        self.up = nn.ConvTranspose2d(
            out_channels, out_channels,
            kernel_size=4, stride=2, padding=1,
        )
        self.norm = nn.GroupNorm(8, out_channels)
        self.refine = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self._init_deconv_as_bilinear(self.up)

    @staticmethod
    def _init_deconv_as_bilinear(deconv):
        """Initialize a stride-2 transposed conv as a bilinear upsampler."""
        kh, kw = deconv.kernel_size
        assert kh == kw == 4
        bil = torch.tensor([
            [0.0625, 0.1875, 0.1875, 0.0625],
            [0.1875, 0.5625, 0.5625, 0.1875],
            [0.1875, 0.5625, 0.5625, 0.1875],
            [0.0625, 0.1875, 0.1875, 0.0625],
        ])
        weight = torch.zeros_like(deconv.weight)
        C_in, C_out = deconv.weight.shape[:2]
        assert C_in == C_out, "neck deconv should be channel-preserving"
        for c in range(C_in):
            weight[c, c] = bil
        with torch.no_grad():
            deconv.weight.copy_(weight)
            if deconv.bias is not None:
                deconv.bias.zero_()

    def forward(self, x):
        # x: (B, 1024, 32, 32)    stride 8
        x = self.proj(x)          # (B, 128, 32, 32)
        x = self.up(x)            # (B, 128, 64, 64)    stride 4
        x = F.relu(self.norm(x))
        x = self.refine(x)        # (B, 128, 64, 64)
        return x
```

**Why the bilinear init matters.** A from-scratch 2× deconv has to learn how to upsample on a tiny dataset. Initializing it as a literal bilinear upsampler means it *starts* as a reasonable upsampler, and training only has to teach it the delta from bilinear. This is a one-time init change with disproportionate impact on small-data trainability.

**Why a single 2× upsample, not the full P2/P3/P4/P5 pyramid.** The full ViTDet pyramid has ~3–8M params (mostly in the deeper deconv branches). At 200 tiles, that's risky. Clay v1.5 already outputs stride 8 (~4.8 m/cell at 60 cm); the single 2× deconv here takes it to stride 4 (~2.4 m/cell) — fine enough to resolve individual crowns while keeping the from-scratch parameter count small. Add a further branch (e.g. another 2× to stride 2) only if this is demonstrably missing small detections in validation.

### 3.4 Detection head — DETR-style box head

A small DETR-style decoder with a fixed set of object queries. Each query attends to the neck's feature map and outputs a box + objectness score. No anchors, no NMS — the Hungarian matching during training enforces one-prediction-per-object behavior at inference.

```python
import torch
import torch.nn as nn

class BoxDetectionHead(nn.Module):
    """DETR-style box head with Q object queries."""
    def __init__(self, feat_channels=128, num_queries=124, hidden=128,
                 n_heads=4, n_decoder_layers=3):
        super().__init__()
        self.num_queries = num_queries
        # Learnable object queries
        self.queries = nn.Embedding(num_queries, hidden)
        # 2D positional encoding for the neck features
        self.pos_enc = SinCos2DPositionalEncoding(hidden)
        # Project neck features to decoder dim if needed
        self.input_proj = nn.Conv2d(feat_channels, hidden, kernel_size=1)
        # Transformer decoder
        layer = nn.TransformerDecoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden * 4,
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_decoder_layers)
        # Output heads on each decoded query
        self.cls_head = nn.Linear(hidden, 1)
        self.box_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 4),  # (cx, cy, w, h)
        )

    def forward(self, feat):
        # feat: (B, feat_channels, H_p3, W_p3)
        B, _, H, W = feat.shape
        x = self.input_proj(feat)                       # (B, hidden, H, W)
        pos = self.pos_enc(H, W, x.device)              # (H*W, hidden)
        memory = x.flatten(2).transpose(1, 2)           # (B, H*W, hidden)
        memory = memory + pos.unsqueeze(0)
        # Queries: (Q, hidden) → (B, Q, hidden)
        q = self.queries.weight.unsqueeze(0).expand(B, -1, -1)
        decoded = self.decoder(q, memory)               # (B, Q, hidden)
        cls_logits = self.cls_head(decoded).squeeze(-1) # (B, Q)
        boxes = self.box_head(decoded).sigmoid()        # (B, Q, 4)  in [0, 1]
        return cls_logits, boxes
```

**Why only 3 decoder layers?** Standard DETR uses 6. At 200 tiles, fewer layers means fewer from-scratch parameters and more stable training. 3 layers is a reasonable compromise — enough capacity for the queries to specialize, not so many that overfitting dominates.

**Why box outputs in [0, 1] via sigmoid?** Normalized coordinates are scale-invariant and well-behaved for L1 loss. Convert to pixel coordinates only at inference / evaluation time.

A simple 2D sinusoidal positional encoding for the neck features:

```python
class SinCos2DPositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert dim % 4 == 0, "dim must be divisible by 4 for 2D sin/cos encoding"
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
        pe = torch.cat([pe_y, pe_x], dim=-1)             # (H, W, dim)
        return pe.reshape(H * W, self.dim)
```

### 3.5 Loss

Hungarian matching between Q predicted boxes and ground-truth boxes, with classification, L1 box, and generalized IoU (GIoU) losses on matched pairs.

```python
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops import generalized_box_iou, box_convert

def matching_loss(cls_logits, pred_boxes, gt_boxes, tile_exhaustive,
                  lam_cls=1.0, lam_l1=5.0, lam_giou=2.0):
    """
    cls_logits:       (Q,) raw scores
    pred_boxes:       (Q, 4) (cx, cy, w, h) in [0, 1]
    gt_boxes:         (M, 4) (cx, cy, w, h) in [0, 1]
    tile_exhaustive:  bool — whether unmatched queries should be pushed to "no tree"
    """
    Q, M = cls_logits.shape[0], gt_boxes.shape[0]
    cls_prob = cls_logits.sigmoid()

    # Convert to (x1, y1, x2, y2) for GIoU computation
    pred_xyxy = box_convert(pred_boxes, in_fmt="cxcywh", out_fmt="xyxy")
    gt_xyxy   = box_convert(gt_boxes,   in_fmt="cxcywh", out_fmt="xyxy")

    # --- Build matching cost ---
    cls_cost  = -cls_prob.unsqueeze(1).expand(Q, M)            # (Q, M)
    l1_cost   = torch.cdist(pred_boxes, gt_boxes, p=1)         # (Q, M)
    giou_cost = -generalized_box_iou(pred_xyxy, gt_xyxy)       # (Q, M)
    cost = lam_cls * cls_cost + lam_l1 * l1_cost + lam_giou * giou_cost

    q_idx, gt_idx = linear_sum_assignment(cost.detach().cpu().numpy())

    # --- Classification loss ---
    target = torch.zeros(Q, device=cls_logits.device)
    target[q_idx] = 1.0
    if tile_exhaustive:
        # All Q queries contribute: matched → 1, unmatched → 0
        cls_loss = F.binary_cross_entropy_with_logits(cls_logits, target)
    else:
        # Sparsely annotated tile: only push matched queries to 1.
        # Unmatched queries are not penalized (we don't know if their "no tree"
        # prediction is correct, since unannotated trees may be present).
        matched_mask = torch.zeros(Q, dtype=torch.bool, device=cls_logits.device)
        matched_mask[q_idx] = True
        cls_loss = F.binary_cross_entropy_with_logits(
            cls_logits[matched_mask], target[matched_mask]
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

The combined L1 + GIoU box loss is the DETR standard. L1 alone is poorly scale-aware (a 1-pixel error on a 7-pixel crown is much worse than on a 70-pixel building); GIoU adds the scale-aware component. Both together work best.

The exhaustive-tile flag controls whether unmatched queries are pushed toward "no tree." This is the box-detection analog of the point-detection masked classification loss: in sparsely annotated tiles, real unlabeled trees might be near unmatched queries, so pushing them all toward score 0 would create false-negative gradients.

---

## 4. Training pipeline

Two stages. Stage A (your own MAE pretraining) is *not* part of this design — Clay has already done it on a corpus you can't match.

### 4.1 Stage B — RGB tree-dataset pretraining (mandatory)

**Goal:** train the neck and head on abundant RGB tree-detection data before exposure to the small NAIP set.

- **Data:** DeepForest's NEON crown dataset (tens of thousands of annotated crowns) and/or the TreeFormer datasets. All RGB.
- **Input adaptation:** NAIP wavelengths are `[0.665, 0.560, 0.493, 0.842]` (R,G,B,NIR); the RGB datasets contribute only the first three, `[0.665, 0.560, 0.493]`. Pass just those 3 bands and 3 wavelengths to Clay's dynamic embedding — no zero NIR channel is needed, since the dynamic embedding handles variable band counts natively.
- **Encoder:** frozen.
- **Trainable:** neck + head (~1.5M params).
- **Optimizer:** AdamW, lr 1e-3, weight decay 0.05, cosine schedule with 5 epoch warmup.
- **Epochs:** ~50, with early stopping on a held-out validation split.
- **Why this is mandatory:** the neck is from scratch. Asking it to learn to upsample Clay features *and* detect trees on 200 tiles is the most likely failure mode. Pretraining on tens of thousands of RGB crowns means the neck arrives at Stage C already producing tree-shaped features, and Stage C only has to adapt them to NAIP's spectral domain.

### 4.2 Stage C — NAIP finetuning

**Goal:** adapt to your specific forests with real NIR.

- **Data:** your ~200 annotated NAIP tiles. Boxes converted to centers. Annotated masks per tile.
- **Input:** full 4-channel NAIP, real NIR, all 4 wavelengths to Clay.
- **Schedule:**
  - Epochs 1–30: encoder frozen. Train neck + head only. lr 5e-4.
  - Epochs 31+: unfreeze Clay's **last 2 transformer blocks**. lr 1e-5 for unfrozen Clay params, 5e-4 for neck + head. Cosine decay.
- **Epochs:** 50–100 total with early stopping on validation F1 at point level.
- **Augmentation:** rotations (90/180/270°), horizontal+vertical flips, mild brightness/contrast jitter applied identically across all 4 bands, optional small per-band gain jitter (±5%) to simulate NAIP inter-year radiometric drift. **Do not** jitter NIR independently of RGB — this breaks spectral relationships.
- **Splits:** train/val/test by geographic region, never by random crop within a region.

---

## 5. Data handling

### 5.1 Tiling
Extract 256×256 windows (154 m × 154 m at 60 cm). For NAIP this matches the native tile size — no resampling needed. For NEON (10 cm native, 1000×1000), downsample to ~60 cm GSD first so physical crown scale matches NAIP, then crop 256×256. Use overlapping windows at training (stride ≈ 128) for ~4× sample multiplication. Use non-overlapping for evaluation.

### 5.2 Metadata generation
For each tile, compute:
- `gsd = 1.0` (Clay's NAIP pretraining GSD; see §3.1 — distinct from the 60 cm physical resolution used for crown-area/per-hectare metrics)
- `latlon` from tile bounding box center (transform from native CRS to WGS84), then sin/cos-encode to 4 dims (see §3.1)
- `time` from NAIP acquisition date, sin/cos-encoded to 4 dims: `[sin(week·2π/52), cos(week·2π/52), sin(hour·2π/24), cos(hour·2π/24)]` (see §3.1)
- `wavelengths` is fixed for NAIP — store as a constant

### 5.3 Annotation mask
For each tile, build an `annotated_mask` of shape `(H, W)` marking regions where labeling was exhaustive. If the entire tile is exhaustively labeled, this is all ones. If only a sub-region was annotated, this is ones in that sub-region and zeros elsewhere. The mask determines where the classification loss applies.

### 5.4 Augmentation specifics
| Augmentation | Apply to | Notes |
|---|---|---|
| 90/180/270° rotation | all 4 bands identically | Free — NAIP is overhead, no preferred orientation |
| Horizontal/vertical flip | all 4 bands identically | Free |
| Brightness/contrast jitter | uniform across bands | Don't desync RGB from NIR |
| Per-band gain (±5%) | independently per band, small | Simulates inter-year radiometric drift |
| Random crop within tile | all 4 bands identically | Useful if tiles are larger than 256 |

Skip: anything that desyncs bands more than mildly. Skip Gaussian blur (NAIP is already at sensor resolution). Skip cutout/erasing initially — try only if overfitting.

---

## 6. Hyperparameter defaults

| Hyperparameter | Default | Notes |
|---|---|---|
| Tile size | 256×256 | Matches NAIP native tile size; multiple of Clay's patch size (8) → 32×32 tokens. Use same dim across Stage B, Stage C, and inference. |
| Clay variant | v1.5 (large, dim 1024, depth 24, patch 8, ~311M params) | The released v1.5.0 checkpoint |
| Neck input channels | 1024 | = Clay v1.5 encoder dim |
| Neck output channels | 128 | Halved from typical FPN 256 to stay trainable |
| Neck scales | P3 only (stride 8 → stride 4 via one 2× deconv) | Add another branch only if small detections are missed |
| Object queries Q | 124 | ≥ max trees per tile; raise if denser |
| Decoder layers | 3 | Standard DETR uses 6; fewer is safer at 200 tiles |
| Decoder heads | 4 | |
| λ_cls | 1.0 | |
| λ_l1 | 5.0 | DETR default |
| λ_giou | 2.0 | DETR default |
| Confidence threshold | 0.5 | Sweep on val for best F1 / AP |
| Stage B optimizer | AdamW, lr 1e-3, wd 0.05 | Cosine with 5-epoch warmup |
| Stage B epochs | ~50 | Early stop on val |
| Stage C frozen-encoder epochs | 30 | lr 5e-4 for neck+head |
| Stage C unfrozen-encoder epochs | 20–70 | lr 1e-5 Clay, 5e-4 neck+head |
| Stage C batch size | 8–16 | Memory-bound on Clay |
| Stage B batch size | 16–32 | Frozen encoder, lighter memory |
| Weight decay | 0.05 | |
| Dropout in neck/head | 0.1 | |

---

## 7. Evaluation

Per-tile metrics:
- **Count error:** `|predicted_count − true_count|`. Report MAE and RMSE per tile, plus R² of predicted vs. true count across the test set.
- **Box-level precision, recall, F1 at IoU = 0.5:** standard object-detection matching. A predicted box matches a ground-truth box if their IoU ≥ 0.5. Hungarian assignment for ambiguous cases.
- **mAP at IoU thresholds 0.5 and 0.5:0.95 (step 0.05):** the standard COCO-style metric. Report both.
- **Confidence threshold sweep:** generate a precision-recall curve over confidence thresholds; report AP and the operating-point F1.

Per-hectare metrics (for comparison with Cheng et al.):
- Aggregate predictions and labels to 100 m × 100 m grid cells. Report count MAE and RMSE in trees/ha. This is the comparable scale to NAIP-based mortality studies.

Crown-area metrics (uniquely possible with box output, not available from points):
- **Predicted-vs-true crown area R² and rRMSE**, computed on matched pairs at IoU ≥ 0.5 (the StarDist paper's standard reporting). This lets you directly compare crown-size accuracy with Tong & Zhang (R² 0.85–0.89, rRMSE 21–24%).

Qualitative sanity checks:
- Visualize predictions on ~10 held-out tiles. Look for systematic position errors (boxes consistently offset suggests undertrained decoder), clustered duplicate boxes on a single crown (suggests insufficient Hungarian matching pressure or too many queries), or whole-tile failures (suggests overfitting).

---

## 8. Implementation gotchas

Things that have bitten projects like this before. Check each before assuming a bug is somewhere else.

1. **Clay's channel order is R, G, B, NIR** for NAIP (`metadata.yaml` → `naip.band_order = [red, green, blue, nir]`). Native NAIP GeoTIFFs are already in this order, so read bands in file order — but make the `wavelengths` array and the normalization stats follow R,G,B,NIR too (`[0.665, 0.560, 0.493, 0.842]`). A common bug is mixing a B,G,R wavelength order with R,G,B-ordered stats, which silently swaps red/blue.
2. **Use Clay's NAIP normalization stats**, not ImageNet stats and not your own. They live under `naip.bands.mean`/`.std` as band-keyed dicts on the 0–255 DN scale (not reflectance ×10000). They are calibrated to match the encoder's pretraining.
2a. **Do not let the encoder shuffle tokens.** Set `mask_ratio=0` *and* `shuffle=False`, and bypass the base `Encoder.forward`/`mask_out` (see §3.2). With the v1.5.0 checkpoint's `shuffle: True`, calling the base forward returns spatially scrambled tokens even at `mask_ratio=0`, and the reshape to a feature map produces garbage with no error raised.
2b. **`gsd` must be a scalar.** Clay builds one positional encoding for the whole batch; passing a `(B,)` gsd raises a broadcast error in `posemb_sincos_2d_with_gsd`. Pass `gsd[0]` (all tiles share the same NAIP GSD of 1.0).
3. **Handle exhaustive vs. sparsely-annotated tiles correctly.** Track an `exhaustive` flag per tile and use it to gate whether unmatched queries are pushed toward "no tree." Skipping this makes the model under-predict.
4. **Bilinear init on the deconv** is a one-line change with large impact. Don't skip it.
5. **Geographic train/val/test split**, never random within a region. Spatial autocorrelation will leak performance dramatically.
6. **Box coordinates are normalized to [0, 1]** throughout the model. Convert to pixel coords only at inference / visualization / evaluation. Sigmoid the box head output before computing the L1 loss against the normalized targets.
7. **GIoU loss is computed in (x1, y1, x2, y2) format**, while L1 loss and Hungarian matching use (cx, cy, w, h). Use `torchvision.ops.box_convert` to switch; getting this wrong silently breaks the loss.
8. **Hungarian matching diagonal extraction.** `generalized_box_iou` returns a (Q, M) pairwise matrix; after matching, you need the diagonal of the (matched, matched) subset, not the full matrix. The `.diag()` call in the GIoU loss above is required.
9. **Late unfreezing schedule, not early.** Unfreezing Clay's last blocks too early — before the neck and head have stabilized — risks clobbering Clay's pretrained features with noisy gradients.
10. **Stage B is mandatory.** It's the single biggest risk mitigation. Don't skip it for time pressure; without it, the from-scratch neck and decoder have too little signal to converge on 200 tiles.
11. **Verify shapes after each component** with random input before wiring up training. Most failures-to-train in pipelines like this come from a silent shape error that produces garbage features.
12. **Number of queries `Q` matters for training stability.** Too few queries (Q < trees per tile) means Hungarian matching can't assign all GT objects — some are silently dropped. Too many means most queries are negative examples and the matching is noisy. Start with Q ≈ max trees per tile in your dataset, with some headroom.

---

## 9. File layout

```
src/
  data/
    naip_dataset.py        # NAIP tile loading, metadata extraction, augmentation
    deepforest_dataset.py  # Stage B RGB dataset wrapper (NIR=0, 3 wavelengths)
    augmentation.py        # rotations, flips, band-aware radiometric jitter
  model/
    clay_loader.py         # load Clay v1.5, wrap forward with batch dict
    neck.py                # StrippedViTDetNeck with bilinear-init deconv
    head.py                # PointHead (cls + reg branches)
    detector.py            # full pipeline module
  train/
    losses.py              # Hungarian matching, masked BCE, L1
    stage_b.py             # RGB pretraining loop
    stage_c.py             # NAIP finetune loop with unfreezing schedule
    schedulers.py          # cosine, warmup, layer-wise LR
  eval/
    metrics.py             # per-tile MAE/F1, per-hectare aggregation, PR curves
    inference.py           # tile → points, confidence threshold sweep
    visualize.py           # qualitative diagnostic plots
configs/
  stage_b.yaml
  stage_c.yaml
  naip_normalization.yaml  # band means/stds from Clay's metadata.yaml
```

Build and shape-test each component in isolation with random tensors before wiring training loops.

---

## 10. Eventual extension to pre/post-fire mortality

The architecture supports this with no fundamental changes:

1. Apply the trained detector independently to a pre-fire and a post-fire NAIP tile of the same region. Output: two sets of tree points.
2. **Per-hectare mortality estimate** (robust, simple): aggregate to 100 m × 100 m grid cells. `Δcount = count_pre − count_post`. Report mortality rate per hectare. Comparable to Cheng et al.'s framing.
3. **Per-tree mortality estimate** (harder, requires co-registration): phase-correlate pre/post tiles to estimate residual geo-shift, match nearest points across time, classify each matched tree as survived/died using a small post-process model that reads the post-fire NIR signature (or ΔNDVI) at each point.

The detector itself remains unchanged; all mortality logic lives downstream of inference. For the class project deliverable, per-hectare differencing is the recommended scope — it's defensible, comparable to published baselines, and doesn't require solving the co-registration problem.

---

## 11. Expected outcomes and honest limits

What this design should achieve, based on the surveyed literature:

- **Pre-fire tree counts:** ~10–15% relative error per tile, in the regime of TreeFormer's published numbers on RGB.
- **Dead-tree detection (eventual):** comparable to Cheng et al.'s ~15–25% underestimation, with the goal of improving on it via NIR-aware features rather than absolute scale of training data.
- **Per-hectare mortality estimates:** target MAE ~4–6 dead trees/ha, comparable to or modestly better than Cheng et al.'s 4.56 fire-mortality MAE.

What this design will *not* achieve:
- **Per-individual-tree mortality with high precision.** At 60 cm, two crowns within ~1.5 m are physically unresolvable. NAIP's inter-year geometric and radiometric drift add to this. Per-hectare aggregation is the honest reporting scale.
- **Beating Cheng et al.'s absolute accuracy on dead-tree mapping.** They had ~24,000 hand-digitized dead crowns. With 200 tiles, this design is in a different data regime and the contribution is methodological (transformer + NIR + foundation-model pretraining + limited labels), not raw accuracy.

Bounding box output (vs. the earlier point-detection plan) **does** enable:
- **Crown-area-weighted analyses.** Predicted box areas can be used directly as proxy crown areas, supporting biomass-weighted mortality estimates downstream.
- **Direct comparison with StarDist and Cheng et al.** Both report crown-area accuracy (R² and rRMSE on crown size). With box output, these metrics are comparable on the same axis.
