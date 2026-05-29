"""
SimMIM head for Swin-Tiny (4-channel). Conceptually equivalent to MAE for
the purpose of domain-adaptive pretraining, but uses a learnable mask token
applied after patch_embed instead of dropping tokens — Swin's windowed
attention is incompatible with the token-dropping MAE recipe.

Pipeline:
  input (B, 4, 224, 224)
    └── patch_embed         → (B, 56, 56, 96)
    └── inject mask token   → mask positions get learned mask vector
    └── Swin layers         → (B, 7, 7, 768)
    └── norm                → (B, 7, 7, 768)
    └── linear decoder      → (B, 49, 32*32*4)
    └── reshape             → (B, 4, 224, 224)

Loss: per-mask-patch normalized MSE on masked patches only. Per-patch
normalization (subtract patch mean, divide by patch std) follows the MAE
"norm_pix" trick — without it the model trivially learns to predict patch
means.
"""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .stem import inflate_patch_embed_to_4ch


class SwinSimMIM(nn.Module):
    def __init__(
        self,
        backbone_name: str = "swin_tiny_patch4_window7_224",
        in_chans: int = 4,
        img_size: int = 224,
        mask_patch_size: int = 32,
        model_patch_size: int = 4,
        pretrained: bool = True,
    ):
        super().__init__()
        # Build with 3 channels so timm loads ImageNet weights, then inflate.
        # global_pool='' + num_classes=0 keeps the (B, H, W, C) feature map.
        self.encoder = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=3,
            num_classes=0,
            global_pool="",
            img_size=img_size,
        )
        self.encoder = inflate_patch_embed_to_4ch(self.encoder)

        embed_dim = self.encoder.patch_embed.proj.out_channels  # 96 for Swin-T
        self.encoder_dim = self.encoder.num_features            # 768 for Swin-T
        self.in_chans = in_chans
        self.img_size = img_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        # how many model patches per mask patch on a side
        self.scale = mask_patch_size // model_patch_size

        # Learnable mask token, replaces patch embeddings at masked positions.
        self.mask_token = nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Decoder: project each final-feature token to one mask-patch worth of
        # pixels. With Swin-T at 224 input the final grid is 7x7 and each mask
        # patch is 32x32, so the linear maps 768 -> 4*32*32 = 4096.
        self.decoder = nn.Linear(
            self.encoder_dim, in_chans * mask_patch_size * mask_patch_size
        )

    def _apply_mask(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: (B, H_pe, W_pe, C); mask: (B, n_mp, n_mp) bool. Returns x with
        masked positions replaced by self.mask_token."""
        B, H_pe, W_pe, C = x.shape
        # Expand the coarse mask to the patch-embed grid (each mask patch
        # covers `scale` x `scale` patch-embed tokens).
        m = mask.repeat_interleave(self.scale, 1).repeat_interleave(self.scale, 2)
        if m.shape[1:] != (H_pe, W_pe):
            raise ValueError(
                f"mask grid {m.shape[1:]} does not match patch-embed grid {(H_pe, W_pe)}"
            )
        m = m.unsqueeze(-1).to(x.dtype)  # (B, H_pe, W_pe, 1)
        mt = self.mask_token.view(1, 1, 1, C).to(x.dtype)
        return x * (1.0 - m) + mt * m

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: (B, 4, H, W); mask: (B, n_mp, n_mp) bool. Returns reconstructed
        image (B, 4, H, W)."""
        B = x.shape[0]
        x = self.encoder.patch_embed(x)           # (B, H_pe, W_pe, 96)
        x = self._apply_mask(x, mask)             # mask injection
        x = self.encoder.layers(x)                # (B, 7, 7, 768)
        x = self.encoder.norm(x)                  # (B, 7, 7, 768)

        # Decode each final token to its mask-patch worth of pixels.
        x = x.flatten(1, 2)                       # (B, 49, 768)
        pred = self.decoder(x)                    # (B, 49, in_chans*mps*mps)

        n_mp = self.img_size // self.mask_patch_size
        mps = self.mask_patch_size
        pred = pred.view(B, n_mp, n_mp, self.in_chans, mps, mps)
        # (B, n_mp, n_mp, C, mps, mps) -> (B, C, n_mp, mps, n_mp, mps)
        pred = pred.permute(0, 3, 1, 4, 2, 5).contiguous()
        pred = pred.view(B, self.in_chans, n_mp * mps, n_mp * mps)
        return pred


def simmim_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    mask_patch_size: int = 32,
    norm_pix: bool = True,
) -> torch.Tensor:
    """Per-mask-patch normalized MSE on masked patches only.

    pred, target: (B, C, H, W) in the *normalized* input space.
    mask:         (B, n_mp, n_mp) bool, True = masked.
    """
    B, C, H, W = target.shape
    n_mp = H // mask_patch_size
    mps = mask_patch_size

    def split(t):
        # (B, C, n_mp, mps, n_mp, mps) -> (B, n_mp*n_mp, C, mps, mps)
        t = t.view(B, C, n_mp, mps, n_mp, mps)
        t = t.permute(0, 2, 4, 1, 3, 5).contiguous()
        return t.view(B, n_mp * n_mp, C, mps, mps)

    target_p = split(target)
    pred_p = split(pred)

    if norm_pix:
        # Normalize each (C, mps, mps) patch independently — joint across
        # channels and pixels, matching the MAE "norm_pix_loss" trick.
        mean = target_p.mean(dim=(-3, -2, -1), keepdim=True)
        var = target_p.var(dim=(-3, -2, -1), keepdim=True)
        target_p = (target_p - mean) / (var + 1e-6).sqrt()

    # per-patch MSE
    per_patch = (pred_p - target_p).pow(2).mean(dim=(-3, -2, -1))  # (B, n_mp*n_mp)
    mask_flat = mask.view(B, -1).to(per_patch.dtype)               # (B, n_mp*n_mp)
    loss = (per_patch * mask_flat).sum() / (mask_flat.sum() + 1e-6)
    return loss
