"""Category-Guided Feature Enhancement (CGFE), adapted from DQ-DETR (cgfe.py).

A CBAM-style attention block that uses the CCM density feature to enhance the
EMSV feature map before it is handed to the detection head:

    spatial gate (from the density map)  ->  channel gate  ->  enhanced EMSV

The density map encodes *where* (and how densely) objects are, so using it to
drive the spatial attention focuses the detector's features on crowded crown
regions. DQ-DETR applies this per pyramid level; the Clay pipeline is
single-scale, so the gates run once on the (B, C, H, W) neck output.

Per the two-phase schedule, CGFE is bypassed in Stage 1 (EMSV passes straight
through to the head) and enabled in Stage 2, once the density map is reliable.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelGate(nn.Module):
    """Squeeze-excite style channel attention (avg + max pooled MLP)."""

    def __init__(self, gate_channels: int, reduction_ratio: int = 16):
        super().__init__()
        hidden = max(1, gate_channels // reduction_ratio)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, gate_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.mlp(F.adaptive_avg_pool2d(x, 1))
        mx = self.mlp(F.adaptive_max_pool2d(x, 1))
        scale = torch.sigmoid(avg + mx)             # (B, C)
        return scale.unsqueeze(-1).unsqueeze(-1)    # (B, C, 1, 1)


class SpatialGate(nn.Module):
    """Spatial attention from a (max, mean)-channel-pooled 7x7 conv."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(
            2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compress = torch.cat(
            [x.max(dim=1, keepdim=True)[0], x.mean(dim=1, keepdim=True)], dim=1
        )
        return torch.sigmoid(self.conv(compress))   # (B, 1, H, W)


class CGFE(nn.Module):
    """Single-scale Category-Guided Feature Enhancement."""

    def __init__(self, gate_channels: int = 128, reduction_ratio: int = 16):
        super().__init__()
        self.spatial_gate = SpatialGate()
        self.channel_gate = ChannelGate(gate_channels, reduction_ratio)

    def forward(self, emsv: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
        # emsv, density: (B, C, H, W) -- same spatial size (single-scale).
        c_spatial = self.spatial_gate(density)      # where objects are
        feat = emsv * c_spatial
        c_channel = self.channel_gate(feat)
        feat = feat * c_channel
        return feat
