"""Categorical Counting Module (CCM), adapted from DQ-DETR (models/dqdetr/ccm.py).

Given the EMSV feature map (the ViTDet-neck output), the CCM produces:
  1. ``count_logits`` (B, cls_num): which count *category* the tile falls in
     (e.g. <100 / 100-300 / >=300 trees). This drives Dynamic Query Selection.
  2. ``density`` (B, density_channels, H, W): a density-map-like feature used by
     the CGFE to spatially gate the EMSV features.

DQ-DETR runs the CCM on multi-scale deformable-encoder memory; here the Clay
pipeline is single-scale (one P3 map, stride 4), so the module operates directly
on the (B, C, H, W) neck output -- no spatial_shapes bookkeeping needed.
"""
import torch
import torch.nn as nn


def _make_dilated_layers(cfg, in_channels, d_rate=2):
    """Stack of dilated 3x3 conv + ReLU, as in DQ-DETR's `make_layers`.

    Dilation widens the receptive field without downsampling, so the density
    feature stays at the input resolution -- important for resolving the small,
    densely packed crowns (4-10 px) this detector targets.
    """
    layers = []
    for v in cfg:
        layers += [
            nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate),
            nn.ReLU(inplace=True),
        ]
        in_channels = v
    return nn.Sequential(*layers)


class CategoricalCountingModule(nn.Module):
    """Single-scale CCM: EMSV feature map -> (count category logits, density map)."""

    def __init__(self, in_channels: int = 128, cls_num: int = 3, density_channels: int = 128):
        super().__init__()
        # Project to a wider working width, then a dilated-conv stack ending at
        # `density_channels` so the density map matches the EMSV width the CGFE
        # gates (single-scale, so density and EMSV share H x W and channels).
        self.proj = nn.Conv2d(in_channels, 256, kernel_size=1)
        self.ccm = _make_dilated_layers(
            [256, 256, density_channels, density_channels], in_channels=256, d_rate=2
        )
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear = nn.Linear(density_channels, cls_num)

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # feat: (B, in_channels, H, W)
        x = self.proj(feat)
        density = self.ccm(x)                       # (B, density_channels, H, W)
        pooled = self.pool(density).flatten(1)      # (B, density_channels)
        count_logits = self.linear(pooled)          # (B, cls_num)
        return count_logits, density
