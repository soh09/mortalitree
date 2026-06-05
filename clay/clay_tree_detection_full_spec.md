# Project Spec — NAIP Tree Detection with Clay + ViTDet Neck + DQ-DETR

A complete specification for a vision-transformer-based tree detector that finetunes the Clay geospatial foundation model with a lightweight ViTDet-style neck and a **DQ-DETR-style dynamic-query detection head**. Targeted at NAIP 4-band aerial imagery (RGB + NIR) at 60 cm resolution, with a small labeled training set on the order of 200 tiles. Eventual purpose: counting trees in matched pre-fire and post-fire imagery to estimate fire-induced mortality.

The detection head follows **DQ-DETR** (Huang et al., ECCV 2024, *DETR with Dynamic Query for Tiny Object Detection*), which is purpose-built for the regime here: many tiny objects (trees are 4–10 px wide) at a count that varies wildly per image (20–500 per tile). DQ-DETR adds three pieces over a vanilla DETR head — a **Categorical Counting Module (CCM)**, **Category-Guided Feature Enhancement (CGFE)**, and **Dynamic Query Selection (DQS)** — so the number of object queries scales with the predicted object count instead of being a fixed constant. See §3.4–§3.5.

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
4. **Bounding box detection from a foundation-model backbone, with a DQ-DETR dynamic-query head sized for the data-limited regime.** A DETR-style box head replaces the original point-detection plan to preserve crown-size information for the eventual mortality-by-biomass analysis. On top of that, the **DQ-DETR** counting/dynamic-query design (CCM + CGFE + DQS) lets the number of object queries scale with the predicted tree count (200–600) rather than a single fixed value — important here because tree counts span 20–500 per tile, where a fixed query budget is either too few (drops GT on dense tiles) or mostly-negative noise (on sparse tiles). A stripped-down single-scale neck keeps the from-scratch parameter count (~3.2M) compatible with ~200 labeled tiles.
5. **A training schedule explicitly designed for the data-limited regime** — frozen encoder, mandatory Stage B pretraining on abundant RGB tree datasets, late unfreezing of upper encoder blocks, with a stripped-down neck to keep from-scratch parameters small.

### What this design is *not* claiming
- Not a new architecture (Clay, ViTDet neck, and the DQ-DETR head are all published).
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
   single 2× deconv → EMSV map: (B, 128, 64, 64)    stride 4
        │
        ├──────────────► CCM  (Categorical Counting Module)
        │                  ├─ count-category logits (B, 3)  ──► DQS
        │                  └─ density feature map  (B, 128, 64, 64) ──┐
        │                                                             │
        ▼                                                             ▼
   CGFE (Category-Guided Feature Enhancement)  ◄── density-gated  (Stage 2 only;
        │  spatial gate (from density) → channel gate              bypassed in Stage 1)
        ▼
   enhanced EMSV map  (B, 128, 64, 64)
        │
        ▼
Two-stage DQS detection head    (~2M params, from scratch)
   stage 1: every memory token → objectness + anchor-seeded box proposal
   DQS:     keep top-k proposals, k = dynamic_query_list[predicted count category]
            (k ∈ {200, 400, 600} — more queries for denser tiles)
   stage 2: DAB-style decoder over the k selected queries, iterative box refine
       cls branch:  (B, k, 1)   tree objectness per query
       box branch:  (B, k, 4)   (cx, cy, w, h) per query, normalized to [0, 1]
        │
        ▼
At training: Hungarian matching on decoder outputs (BCE cls + L1 + GIoU),
             same matching on the two-stage proposals (focal cls + L1 + GIoU),
             and a cross-entropy CCM loss on the count category.
At inference: keep boxes with cls_score > τ; count = |kept boxes|
```

**Total trainable params at Stage C with frozen encoder: ~3.2M (neck + CCM + CGFE + DQS head).** Larger than the point-head variant but still small enough to train on 200 tiles with the mitigations below.

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

### 3.4 Detection head — DQ-DETR dynamic-query head

The head follows DQ-DETR. Instead of a fixed set of learnable queries, the number of queries is chosen *per image* from a coarse count prediction, and the queries are seeded by a two-stage proposal step on the feature map. There are three sub-modules — **CCM**, **CGFE**, **DQS** — feeding a small DAB-style decoder. No anchors-as-NMS, no NMS at inference; Hungarian matching during training enforces one-prediction-per-object behavior.

Files: `src/model/ccm.py`, `src/model/cgfe.py`, `src/model/head.py`. The single-scale plumbing (one EMSV map, plain multi-head attention) is the adaptation of DQ-DETR's multi-scale deformable design to this Clay pipeline — it keeps the contribution (count-conditioned dynamic query number + two-stage anchor-seeded queries + density enhancement) without the CUDA deformable-attention ops.

#### CCM — Categorical Counting Module

Consumes the EMSV map (the neck output) and emits two things: a **count-category** logit vector (which coarse count bin the tile falls in) and a **density feature map** (a density-map-like tensor at the EMSV resolution). The count category drives DQS; the density map drives CGFE. A dilated-conv stack keeps the density map at full EMSV resolution — important for the small, densely packed crowns here.

```python
class CategoricalCountingModule(nn.Module):
    """EMSV map -> (count-category logits, density feature map)."""
    def __init__(self, in_channels=128, cls_num=3, density_channels=128):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, 256, kernel_size=1)
        self.ccm  = _make_dilated_layers([256, 256, density_channels, density_channels],
                                          in_channels=256, d_rate=2)  # dilated 3x3 + ReLU
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(density_channels, cls_num)

    def forward(self, feat):                  # feat: (B, in_channels, H, W)
        x = self.proj(feat)
        density = self.ccm(x)                 # (B, density_channels, H, W)
        count_logits = self.linear(self.pool(density).flatten(1))  # (B, cls_num)
        return count_logits, density
```

The count categories come from box-count thresholds `ccm_params = [100, 300]` → 3 bins (`<100`, `100–299`, `>=300`), matching the 20–500 trees/tile range. The CCM is supervised with a cross-entropy **CCM loss** against the bin a tile's GT count falls in.

#### CGFE — Category-Guided Feature Enhancement

A CBAM-style attention block that uses the CCM density map to enhance the EMSV features before the head sees them: a **spatial gate** computed from the density map (where, and how densely, objects are) followed by a **channel gate**. Single-scale, so it runs once on the `(B, 128, 64, 64)` map.

```python
class CGFE(nn.Module):
    def __init__(self, gate_channels=128, reduction_ratio=16):
        super().__init__()
        self.spatial_gate = SpatialGate()                 # 7x7 conv on (max,mean) channel-pool
        self.channel_gate = ChannelGate(gate_channels, reduction_ratio)  # SE-style avg+max MLP
    def forward(self, emsv, density):                     # both (B, C, H, W)
        feat = emsv * self.spatial_gate(density)
        feat = feat * self.channel_gate(feat)
        return feat
```

CGFE is **bypassed in Stage 1** (EMSV passes straight through) and **enabled in Stage 2**, once the density map is reliable — see §4.3.

#### DQS — Dynamic Query Selection + two-stage head

A lightweight two-stage proposal step scores every memory token (objectness) and regresses a box from a per-token grid anchor. DQS keeps the top-`k` proposals, where `k = dynamic_query_list[predicted count category]` (`{200, 400, 600}` here). To keep the batch tensor rectangular, `k` is taken from the **densest** predicted category in the batch (DQ-DETR's trick). The selected proposals seed a DAB-style decoder that derives its query positional encodings from the reference boxes and refines them iteratively.

```python
class TwoStageDQSHead(nn.Module):
    def __init__(self, feat_channels=128, hidden=128, max_queries=600,
                 n_heads=4, n_decoder_layers=3, dropout=0.1,
                 dynamic_query_list=(200, 400, 600), anchor_size=0.05):
        ...
        self.input_proj   = nn.Conv2d(feat_channels, hidden, 1)
        self.enc_cls_head = nn.Linear(hidden, 1)          # stage-1 objectness
        self.enc_box_head = MLP(hidden, hidden, 4, 3)      # stage-1 box (delta on grid anchor)
        self.tgt_embed    = nn.Embedding(max_queries, hidden)  # learnable content queries
        self.query_pos_mlp = MLP(hidden, hidden, hidden, 2)    # query pos from box sine-embed
        self.layers       = nn.ModuleList(DecoderLayer(...) for _ in range(n_decoder_layers))
        self.cls_head     = nn.Linear(hidden, 1)
        self.box_head     = MLP(hidden, hidden, 4, 3)

    def forward(self, feat, count_logits):
        memory = self.input_proj(feat).flatten(2).transpose(1, 2)   # (B, HW, hidden)
        # stage 1: proposals on every token (grid anchors -> sigmoid boxes)
        enc_cls_logits = self.enc_cls_head(memory + pos).squeeze(-1)        # (B, HW)
        enc_boxes      = (self.enc_box_head(memory + pos) + anchors).sigmoid()
        # DQS: k from the densest predicted category; keep top-k proposals
        k = dynamic_query_list[count_logits.argmax(1).max()]
        idx = enc_cls_logits.topk(k, dim=1).indices
        reference = gather(enc_boxes, idx).detach()                         # (B, k, 4)
        # stage 2: decode k queries, iterative box refinement
        tgt = self.tgt_embed.weight[:k].expand(B, -1, -1)
        for layer in self.layers:
            query_pos = self.query_pos_mlp(gen_sineembed_for_boxes(reference, hidden))
            tgt = layer(tgt, query_pos, memory, pos)
            box = (self.box_head(tgt) + inverse_sigmoid(reference)).sigmoid()
            reference = box.detach()
        return dict(cls_logits=self.cls_head(tgt).squeeze(-1), pred_boxes=box,
                    enc_cls_logits=enc_cls_logits, enc_boxes=enc_boxes, num_select=k)
```

**Why two-stage / dynamic queries instead of fixed Q?** Tree counts span 20–500 per tile. A fixed query budget is either too small (Hungarian matching silently drops GT on dense tiles — gotcha #12) or mostly-negative noise on sparse tiles. DQS sizes the query set to the predicted count, and seeding queries from high-objectness proposals (rather than free-floating learnable queries) gives the tiny-object decoder a much better starting point.

**Why only 3 decoder layers?** Standard DETR uses 6. At 200 tiles, fewer layers means fewer from-scratch parameters and more stable training.

**Why box outputs in [0, 1] via sigmoid?** Normalized coordinates are scale-invariant and well-behaved for L1 loss. Convert to pixel coordinates only at inference / evaluation time.

The model `forward` returns a **dict** (`cls_logits`, `pred_boxes`, `count_logits`, `enc_cls_logits`, `enc_boxes`, `num_select`), not a `(cls, boxes)` tuple — all call sites read from the dict.

### 3.5 Loss

Three terms, matching DQ-DETR: (1) the main **detection loss** — Hungarian matching between the `k` decoder predictions and GT boxes with classification + L1 + GIoU; (2) the **two-stage proposal loss** — the same matching on the stage-1 proposals (the "interm" supervision that teaches the proposal scorer what to select, since top-k selection is non-differentiable), using focal classification; (3) the **CCM loss** — cross-entropy on the count category. CGFE has no loss of its own; it is trained through the detection loss once enabled.

The main detection loss is Hungarian matching between the predicted boxes and ground-truth boxes, with classification, L1 box, and generalized IoU (GIoU) losses on matched pairs.

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

**Two-stage proposal loss (`encoder_proposal_loss`).** The same Hungarian matching is applied to the stage-1 proposals so the proposal scorer learns what to select (top-k selection is non-differentiable, so without this the scorer gets no gradient). To keep matching cheap, GT is matched only against the top-T = 1200 proposals by objectness — the candidates DQS actually draws from, and also the hardest negatives. Classification uses **sigmoid focal loss** (α=0.25, γ=2) rather than plain BCE, because there are thousands of mostly-negative proposals per tile and focal loss handles that imbalance. The same exhaustive-vs-sparse gating applies.

**CCM loss (`ccm_loss`).** Cross-entropy between the CCM count-category logits and the bin a tile's GT box count falls in (`ccm_params = [100, 300]` → 3 bins). This is what makes DQS's per-image query count meaningful.

The total training loss is `detection + λ_enc · proposal + λ_ccm · ccm`, with `λ_enc = λ_ccm = 1.0` (DQ-DETR defaults).

---

## 4. Training pipeline

Two stages (B, C). Stage A (your own MAE pretraining) is *not* part of this design — Clay has already done it on a corpus you can't match. Orthogonal to B/C, the DQ-DETR head runs a **two-phase CGFE schedule** within each stage — see §4.3.

### 4.1 Stage B — RGB tree-dataset pretraining (mandatory)

**Goal:** train the neck and head on abundant RGB tree-detection data before exposure to the small NAIP set.

- **Data:** DeepForest's NEON crown dataset (tens of thousands of annotated crowns) and/or the TreeFormer datasets. All RGB.
- **Input adaptation:** NAIP wavelengths are `[0.665, 0.560, 0.493, 0.842]` (R,G,B,NIR); the RGB datasets contribute only the first three, `[0.665, 0.560, 0.493]`. Pass just those 3 bands and 3 wavelengths to Clay's dynamic embedding — no zero NIR channel is needed, since the dynamic embedding handles variable band counts natively.
- **Encoder:** frozen.
- **Trainable:** neck + CCM + CGFE + DQS head (~3.2M params).
- **Optimizer:** AdamW, lr 1e-3, weight decay 0.05, cosine schedule with 5 epoch warmup.
- **Epochs:** ~50, with early stopping on a held-out validation split.
- **Why this is mandatory:** the neck is from scratch. Asking it to learn to upsample Clay features *and* detect trees on 200 tiles is the most likely failure mode. Pretraining on tens of thousands of RGB crowns means the neck arrives at Stage C already producing tree-shaped features, and Stage C only has to adapt them to NAIP's spectral domain.

### 4.2 Stage C — NAIP finetuning

**Goal:** adapt to your specific forests with real NIR.

- **Data:** your ~200 annotated NAIP tiles. Boxes converted to centers. Annotated masks per tile.
- **Input:** full 4-channel NAIP, real NIR, all 4 wavelengths to Clay.
- **Schedule:**
  - Epochs 1–30: encoder frozen. Train the from-scratch modules (neck + CCM + CGFE + DQS head) only. lr 5e-4.
  - Epochs 31+: unfreeze Clay's **last 2 transformer blocks**. lr 1e-5 for unfrozen Clay params, 5e-4 for neck + head. Cosine decay.
- **Epochs:** 50–100 total with early stopping on validation F1 at point level.
- **Augmentation:** rotations (90/180/270°), horizontal+vertical flips, mild brightness/contrast jitter applied identically across all 4 bands, optional small per-band gain jitter (±5%) to simulate NAIP inter-year radiometric drift. **Do not** jitter NIR independently of RGB — this breaks spectral relationships.
- **Splits:** train/val/test by geographic region, never by random crop within a region.

### 4.3 Two-phase CGFE schedule (within each stage)

DQ-DETR enables CGFE only once the counting head is producing a reasonable density map — feeding CGFE a noisy density map early on just injects noise. So both Stage B and Stage C run a two-phase schedule, controlled by `cgfe_start_epoch` (config; defaults to half the stage's epoch budget):

- **Phase 1 (stabilize counting).** CGFE disabled (`model.enable_cgfe = False`): EMSV features pass straight through to DQS. The CCM loss + detection loss + two-stage proposal loss are all active, so the counting head and the detector train together. Run until the CCM count-category accuracy plateaus — roughly the first half of the budget.
- **Phase 2 (add feature enhancement).** CGFE enabled: it now receives a reasonably accurate density map and gates the EMSV features. Resume with all losses active. CGFE's parameters simply start receiving gradient (through the detection loss) at this point.

The toggle is `model.set_cgfe_enabled(epoch >= cgfe_start_epoch)`, flipped at the top of each epoch in `stage_b.py` / `stage_c.py`, and the phase is logged (`*/cgfe_enabled`). The CCM and two-stage losses are on in **both** phases — only CGFE is gated.

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
| Neck output channels | 128 | Halved from typical FPN 256 to stay trainable; also the EMSV / density / CGFE width |
| Neck scales | P3 only (stride 8 → stride 4 via one 2× deconv) | Single-scale; CCM/CGFE/DQS all run on this one EMSV map |
| `dynamic_query_list` (DQS) | [200, 400, 600] | #queries per count category; `k` chosen from densest predicted category in the batch. Also sizes the learnable query bank (`max_queries = max(dynamic_query_list)`), so it must match across Stage B/C — it's in the checkpoint. There is no separate fixed-`Q` knob. |
| `ccm_params` (count bins) | [100, 300] | Thresholds → 3 categories: `<100`, `100–299`, `>=300` trees |
| `ccm_cls_num` | 3 | Number of count categories (= len(ccm_params) + 1) |
| `anchor_size` | 0.05 | Default normalized w/h of stage-1 grid-anchor proposals |
| `cgfe_start_epoch` | total_epochs // 2 | DQ-DETR two-phase: CGFE off (Phase 1) then on (Phase 2) |
| Decoder layers | 3 | Standard DETR uses 6; fewer is safer at 200 tiles |
| Decoder heads | 4 | |
| λ_cls | 1.0 | |
| λ_l1 | 5.0 | DETR default |
| λ_giou | 2.0 | DETR default |
| λ_ccm | 1.0 | CCM count-category cross-entropy weight (DQ-DETR default) |
| λ_enc | 1.0 | Two-stage proposal (interm) loss weight (DQ-DETR default) |
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
12. **Query count must cover the densest tile.** With DQS the per-image query count is `dynamic_query_list[count category]`, but the **learnable query pool `max_queries`** (and the largest `dynamic_query_list` entry) still has to be ≥ the densest tile, or Hungarian matching can't assign all GT objects and some are silently dropped. `max_queries` is part of the checkpoint (the `tgt_embed` table), so it must be **identical across Stage B and Stage C**. The modal trainer warns if the densest patch exceeds it.
13. **DQS needs the two-stage proposal loss.** Top-k selection is non-differentiable, so the stage-1 objectness scorer only learns from `encoder_proposal_loss`. If you drop that loss, query selection never improves past random and the detector stalls. Keep `λ_enc > 0`.
14. **Don't enable CGFE before the density map is meaningful.** CGFE gates features by the CCM density map; feeding it a noisy early density map injects noise. Keep `cgfe_start_epoch` at roughly half the budget (see §4.3). CCM and the two-stage loss, by contrast, are on from epoch 0.
15. **`ccm_params` (count bins) must match your data.** The defaults `[100, 300]` are tuned for 20–500 trees/tile. If your tiles are much sparser or denser, re-bin so the categories are reasonably balanced — a degenerate CCM (all tiles in one bin) makes DQS pick a constant query count, defeating the point.

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
    ccm.py                 # Categorical Counting Module (count category + density map)
    cgfe.py                # Category-Guided Feature Enhancement (spatial + channel gate)
    head.py                # TwoStageDQSHead (two-stage proposals + Dynamic Query Selection)
    detector.py            # full pipeline module (encoder→neck→CCM→CGFE→head); returns dict
  train/
    losses.py              # Hungarian matching (det + two-stage proposal), focal, CCM cross-entropy
    stage_b.py             # RGB pretraining loop (+ CGFE two-phase schedule)
    stage_c.py             # NAIP finetune loop with unfreezing + CGFE two-phase schedule
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
