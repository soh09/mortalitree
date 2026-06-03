"""Stripped ViTDet neck: single-scale 2x upsampler (Clay stride-8 -> stride-4)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class StrippedViTDetNeck(nn.Module):
    """Single-scale neck: 2x upsample, 1024 -> 128 channels.

    Clay v1.5 already emits stride-8 tokens (28x28 for a 224 input), so the
    single 2x deconv here takes that to stride 4 (56x56) -- finer than the
    original spec's stride-8 target, to better resolve small crowns at 60 cm.
    """

    def __init__(self, in_channels: int = 1024, out_channels: int = 128):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.up = nn.ConvTranspose2d(
            out_channels, out_channels, kernel_size=4, stride=2, padding=1
        )
        self.norm = nn.GroupNorm(8, out_channels)
        self.refine = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self._init_deconv_as_bilinear(self.up)

    @staticmethod
    def _init_deconv_as_bilinear(deconv: nn.ConvTranspose2d) -> None:
        kh, kw = deconv.kernel_size
        assert kh == kw == 4
        bil = torch.tensor([
            [0.0625, 0.1875, 0.1875, 0.0625],
            [0.1875, 0.5625, 0.5625, 0.1875],
            [0.1875, 0.5625, 0.5625, 0.1875],
            [0.0625, 0.1875, 0.1875, 0.0625],
        ])
        weight = torch.zeros_like(deconv.weight)
        C_in = deconv.weight.shape[0]
        for c in range(C_in):
            weight[c, c] = bil
        with torch.no_grad():
            deconv.weight.copy_(weight)
            if deconv.bias is not None:
                deconv.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1024, 28, 28)  stride 8
        x = self.proj(x)          # (B, 128, 28, 28)
        x = self.up(x)            # (B, 128, 56, 56)  stride 4
        x = F.relu(self.norm(x))
        x = self.refine(x)        # (B, 128, 56, 56)
        return x
