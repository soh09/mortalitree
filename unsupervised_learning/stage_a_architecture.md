# Stage A — Architecture Summary (SwinSimMIM, 4-channel)

This document describes the model used for Stage A (domain-adaptive
pretraining) of the MortaliTREE pipeline. The companion file
`readme.md` is the broader implementation spec for the full detector
(stem → backbone → FPN → DETR head). This document is scoped to
**Stage A only** — what gets trained on unlabeled NAIP, what gets kept
for Stages B/C, and why.

## Pipeline at a glance

```
┌─────────────────────────────────────────────────────────────────┐
│  Input: (B, 4, 224, 224)  R, G, B, NIR — per-band normalized    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Inflated patch-embed stem                                      │
│    Conv2d(4 → 96, kernel=4, stride=4)                           │
│    Init: RGB ← ImageNet Swin-T patch_embed.proj                 │
│          NIR ← mean(R, G, B) along channel axis                 │
│    Output: (B, 56, 56, 96)                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Mask injection (SimMIM)                                        │
│    Random per-sample mask: (B, 7, 7) bool, mask_ratio=0.5       │
│    Upsample to patch-embed grid: (B, 56, 56)                    │
│    At masked positions, replace token with mask_token (96-d,    │
│      learnable, N(0, 0.02) init)                                │
│    Output: (B, 56, 56, 96)                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Swin-Tiny backbone  (ImageNet-pretrained except stem)          │
│    Stage 1: 2 Swin blocks, window=7, dim=96    → (B,28,28,192)  │
│    Stage 2: 2 Swin blocks, window=7, dim=192   → (B,14,14,384)  │
│    Stage 3: 6 Swin blocks, window=7, dim=384   → (B,7,7,768)    │
│    Stage 4: 2 Swin blocks, window=7, dim=768   → (B,7,7,768)    │
│    LayerNorm                                                    │
│    Output: (B, 7, 7, 768)  — 49 final tokens, 768-d each        │
│                                                                 │
│    Params: ~27.5 M  (Swin-T: 28.3M minus the original 3-ch stem │
│            8.5K, plus the new 4-ch stem 6.1K)                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Linear decoder (SimMIM-style)                                  │
│    Flatten: (B, 49, 768)                                        │
│    Linear: 768 → 4 * 32 * 32 = 4096                             │
│    Reshape: each token → (4, 32, 32) pixel block                │
│    Tile blocks back to image: (B, 4, 224, 224)                  │
│                                                                 │
│    Params: ~3.1 M                                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Output: (B, 4, 224, 224) — reconstructed normalized RGBN       │
└─────────────────────────────────────────────────────────────────┘
```

## Loss

**Per-mask-patch normalized MSE on masked patches only** — the MAE
`norm_pix_loss` trick, applied jointly across all 4 channels:

```python
target_p   = split(target, 7x7 grid of 32x32 patches)   # (B, 49, 4, 32, 32)
pred_p     = split(pred,   7x7 grid of 32x32 patches)
mean, var  = target_p.mean/var over (C, H, W) per patch  # joint over channels
target_p   = (target_p - mean) / sqrt(var + 1e-6)
per_patch  = mean_(C,H,W) (pred_p - target_p) ** 2        # (B, 49)
loss       = (per_patch * mask_flat).sum() / mask_flat.sum()
```

Normalizing the target per-patch jointly across all 4 channels prevents
the model from trivially learning to predict patch means, which has
zero loss under un-normalized MSE for any flat region.

Loss only fires on masked patches. Visible patches contribute through
attention but never directly through the reconstruction objective.

## Optimization

| Hyperparameter | Value | Notes |
|---|---|---|
| Optimizer | AdamW | betas = (0.9, 0.95) per MAE recipe |
| Base LR | 1.5 × 10⁻⁴ | scales with batch; below ImageNet's 2.4e-4 since dataset is smaller |
| Min LR | 1 × 10⁻⁶ | cosine floor |
| Schedule | linear warmup → cosine | 10 warmup epochs, 90 decay |
| Weight decay | 0.05 | not applied to biases, LayerNorm, mask_token |
| Batch size | 128 | A100-40GB, ~24 GB GPU memory |
| Grad clip | 1.0 | global L2 norm |
| Precision | bf16 autocast | A100 native; no loss-scaler overhead |
| Mask ratio | 0.5 | see "Why these choices" below |
| Epochs | 100 | diminishing returns past 50 in practice |

## Data

| | |
|---|---|
| Source | NAIP 2020 + 2022 over 15 California AOIs (10 forest + 5 fire) |
| Pretraining tiles | ~430 k train + ~65 k val (scene-level split) |
| Tile shape | (4, 224, 224) uint8 at 0.6 m GSD → 134 m × 134 m ground footprint |
| Normalization | per-band mean/std from NAIP corpus (not ImageNet) |
| Augmentation | D4 group only (rot 90°/180°/270° + h/v flip). No spectral jitter — spectral identity must be preserved across bands |

The train/val split operates on **whole .tif scenes**, never on
individual tiles. Tiles cropped from the same source orthomosaic share
crowns, shadows, and radiometry, so a tile-level random split would
leak nearly identical content across the val boundary. Within each
AOI, the val partition is the 10% of scenes on the AOI's geographic
edge, picked along the AOI's longer axis. Scenes overlapping the
supervised Stage C test set can be added to a "reserved" pool that's
excluded from pretraining entirely.

## What gets kept after pretraining

After 100 epochs:

- ✅ **Encoder** (`model.encoder.state_dict()`, ~27.5 M params) → goes
  into Stages B and C. This is the only thing Stage A is producing.
- ❌ Linear decoder, mask token, decoder bias → discarded.

Two checkpoint files are saved on the Modal volume:

```
/data/checkpoints/stage_a/encoder_best.pt    # selected on val loss
/data/checkpoints/stage_a/encoder_latest.pt  # most recent epoch
```

Loading the encoder into a fresh detector:

```python
import timm
from mortalitree.unsupervised_learning.model.stem import inflate_patch_embed_to_4ch

backbone = timm.create_model(
    "swin_tiny_patch4_window7_224",
    pretrained=False,
    features_only=True,
    out_indices=(0, 1, 2, 3),
)
backbone = inflate_patch_embed_to_4ch(backbone)

state = torch.load("encoder_best.pt", map_location="cpu")
backbone.load_state_dict(state["encoder"], strict=False)
```

`strict=False` is intentional — the SimMIM model used
`global_pool=""`, while the detector uses `features_only=True`, which
exposes slightly different layer names. The numeric weights for
patch_embed and all four Swin stages match exactly.

---

# Why SimMIM (and not MAE)

The implementation spec calls for "MAE pretraining." In practice we
chose SimMIM, a closely-related method, for one structural reason and
several practical ones. This section explains both.

## The structural reason

**MAE drops tokens. Swin can't.**

Facebook's original MAE recipe ([He et al. 2021](https://arxiv.org/abs/2111.06377))
works on Vision Transformers (ViTs). At each step it:

1. Splits the image into a regular grid of non-overlapping patches.
2. Computes patch embeddings for the whole grid.
3. **Discards 75% of patches entirely**, processing only the remaining
   25% through the encoder.
4. Inserts mask-token placeholders only at the decoder, where positional
   embeddings tell the decoder where the missing tokens belonged.

This 4× reduction in encoder tokens is half the reason MAE is
efficient — the encoder does a quarter the work per step.

But token-dropping requires that the encoder be permutation-invariant
to token positions, which ViTs are (their attention is global, every
token attends to every other token regardless of grid position).
**Swin is not.** Swin's attention is *windowed*: each token attends
only to other tokens inside a 7×7 spatial window, with windows
shifted between layers to enable cross-window communication. Dropping
75% of tokens breaks this structure:

- Windows become sparse, with holes where dropped tokens lived
- Window-shift operations no longer align cleanly
- The hierarchical patch-merging stages assume a regular grid

There are research forks that hack around this (custom sparse
attention, learnable padding in windows), but none are mature, and
all break compatibility with `timm`'s Swin implementation.

## How SimMIM solves this

**SimMIM** ([Xie et al. 2021](https://arxiv.org/abs/2111.09886)) was
developed specifically for Swin. It keeps every token at every stage:

1. Splits the image into patches (same as MAE).
2. Computes patch embeddings for the whole grid.
3. **Replaces masked patch embeddings with a learnable mask token**
   *before* the encoder, rather than dropping them.
4. The encoder sees a complete grid with mask tokens in masked
   positions, exactly as it sees its normal input.
5. Uses a **single linear layer** as the decoder, projecting each
   final-stage token back to its corresponding pixel patch.

Concretely, in our model:

```python
def forward(self, x, mask):
    x = self.encoder.patch_embed(x)        # (B, 56, 56, 96)
    x = self._apply_mask(x, mask)          # replace masked with mask_token
    x = self.encoder.layers(x)             # Swin runs on a full grid
    x = self.encoder.norm(x)
    x = x.flatten(1, 2)                    # (B, 49, 768)
    return self.decoder(x).reshape(...)    # linear → pixel patches
```

The structural problem disappears. Swin sees a complete grid; the
windowed attention works; the patch merging stages get aligned inputs.

## The trade-offs

SimMIM is not literally MAE — pretending otherwise leads to fixable
but irritating bugs. The actual differences:

| | MAE (ViT) | SimMIM (Swin) |
|---|---|---|
| **Encoder input** | Visible tokens only (25%) | Full grid with mask tokens |
| **Encoder FLOPs per step** | ~25% of full forward | ~100% of full forward |
| **Decoder** | Stack of transformer blocks (8 layers, dim 512) | Single linear layer |
| **Decoder params** | ~32 M | ~3 M |
| **Compatible backbones** | ViT only | Swin, ConvNeXt, ViT |
| **Position embedding** | Sinusoidal added at encoder input | Whatever the backbone uses (Swin: relative bias inside attention) |
| **Mask granularity** | Patch-level (encoder patches) | Mask-patch-level (typically 32×32, several encoder patches per mask patch) |
| **Reconstruction target** | Normalized pixels in masked positions | Normalized pixels in masked positions |
| **Reported ImageNet linear probe (ViT-B / Swin-B)** | 68% / — | 56% / 73% |
| **Reported ImageNet fine-tune (ViT-B / Swin-B)** | 83.6% / — | 83.8% / 84.0% |

The trade is roughly:

- MAE wins on **encoder efficiency** (4× faster forward).
- SimMIM wins on **simplicity** (linear decoder, no decoder-side
  position embeds, no token-dropping bookkeeping).
- For **downstream fine-tuning**, they're statistically indistinguishable
  on ImageNet — both produce strong encoders. The linear-probe gap is
  bigger because MAE's deeper decoder pushes the encoder to produce
  more linearly-separable features.

**Linear probing is irrelevant for our use case** — we fine-tune the
encoder for detection in Stage C, so fine-tune accuracy is the
relevant metric, and the two methods are equivalent there.

## Other practical reasons SimMIM fits this project

1. **No decoder to throw away** — MAE wastes ~32 M parameters of
   transformer decoder that exist only to make the encoder useful, and
   then get deleted. SimMIM's 3 M-param linear decoder is closer to
   "barely there" — less compute spent on a function we don't keep.

2. **Mask-patch granularity matches the problem**. SimMIM masks at a
   coarser scale than the encoder's patches: 32×32 pixel blocks rather
   than 4×4. For overhead forestry imagery, 32×32 (19 m at 0.6 m GSD)
   is approximately one crown width. The model is asked to reconstruct
   missing crowns from neighboring crowns — exactly the inductive bias
   that helps a tree detector.

3. **Compatibility with `timm`**. We use `timm.create_model(...)` to
   load ImageNet-pretrained Swin weights. SimMIM works with the
   stock `timm` Swin layout (`patch_embed → layers → norm`). MAE on
   Swin would require either a custom backbone or significant `timm`
   surgery.

4. **Stable on small datasets**. MAE was tuned for ImageNet-scale data
   (1.3 M images, 800 epochs). At our scale (~430k tiles, 100 epochs),
   SimMIM's simpler decoder reaches a better train/val loss in fewer
   steps because there are fewer free parameters and less risk of the
   decoder overfitting on a single token's local context.

## What "SimMIM-style" means in our code

Our `SwinSimMIM` follows SimMIM exactly with one minor parameter
choice:

- **Mask ratio 0.5 (vs SimMIM default 0.6, MAE default 0.75)**.
  We dropped to 0.5 because NAIP canopy is more uniformly textured
  than ImageNet objects — 0.75 masking starved the model of local
  context (large masked regions surrounded by limited unmasked
  vegetation looked too similar to other large masked regions). 0.5
  produced steady learning and stable val loss; 0.6 plateaued earlier.

- **Mask-patch size 32**. Standard SimMIM. With 224×224 input this
  gives a 7×7 mask grid (49 mask patches), aligned with the final
  Swin output grid so each output token decodes exactly one mask
  patch.

- **Per-patch normalized MSE on masked patches only**. The original
  MAE `norm_pix_loss` trick, jointly across all 4 channels. Without
  it, the model can achieve trivially-low loss by predicting per-patch
  means.

- **Mask token at patch-embed output, not at raw pixels**. SimMIM
  paper applies the mask token after the patch embedding (replacing
  embedded tokens), which is what we do. Some open-source forks
  incorrectly mask raw pixels (zeroing the input), which is a
  weaker signal because the patch_embed convolution can partially
  reconstruct masked regions from neighboring un-masked ones.

## When you might prefer MAE instead

The two reasons you'd switch to MAE proper:

1. **You switch from Swin to ViT** as the backbone. Then MAE's token-
   dropping efficiency wins, and the architecture supports it natively.

2. **You have orders of magnitude more data and compute**. MAE at
   scale (ImageNet, JFT) produces slightly stronger encoders than
   SimMIM at scale because its deeper decoder gives the encoder more
   pressure to produce semantic features rather than pixel-statistical
   ones. At ~430k tiles this advantage is invisible.

For this project — Swin backbone, ~430k tiles, A100-40GB, 100
epochs — SimMIM is strictly the right choice. The MAE wording in the
implementation spec describes the *approach* (masked image modeling,
domain-adaptive pretraining on unlabeled NAIP); the *specific recipe*
adapted to Swin is SimMIM.

---

# Summary in one paragraph

For Stage A we pretrain a 4-channel Swin-Tiny encoder using SimMIM:
50% of 32×32 pixel patches are masked, masked positions are filled
with a learnable mask token after the patch embedding, the encoder
runs over the full token grid, and a single linear layer reconstructs
the masked pixels from the encoder's final-stage outputs. Loss is
per-patch normalized MSE on masked patches only, computed jointly
across all 4 channels. ImageNet-pretrained Swin weights are loaded as
the starting point; the inflated 4-channel stem copies RGB weights
and initializes the NIR channel as the mean of the RGB channels.
Training runs for 100 epochs at batch 128 on an A100-40GB with
AdamW + cosine LR, scene-level train/val split, deterministic per-
sample masks on val so the validation loss is comparable across
epochs. After training the encoder weights (~27.5 M params) are saved
and loaded into Stages B and C; everything else is discarded.
