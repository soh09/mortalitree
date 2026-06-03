"""Full detection pipeline: Clay encoder → ViTDet neck → box detection head."""
import torch
import torch.nn as nn

from .clay_loader import ClayEncoderWrapper
from .neck import StrippedViTDetNeck
from .head import BoxDetectionHead


class TreeDetector(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        neck_in_channels: int = None,
        neck_out_channels: int = 128,
        num_queries: int = 124,
        hidden: int = 128,
        n_heads: int = 4,
        n_decoder_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = ClayEncoderWrapper(encoder)
        # Default the neck's input width to the encoder's actual embedding dim
        # (1024 for Clay v1.5 large) rather than hardcoding it.
        if neck_in_channels is None:
            neck_in_channels = self.encoder.dim
        self.neck = StrippedViTDetNeck(neck_in_channels, neck_out_channels)
        self.head = BoxDetectionHead(
            feat_channels=neck_out_channels,
            num_queries=num_queries,
            hidden=hidden,
            n_heads=n_heads,
            n_decoder_layers=n_decoder_layers,
            dropout=dropout,
        )

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            cls_logits: (B, Q) raw objectness scores
            boxes:      (B, Q, 4) (cx,cy,w,h) in [0,1]
        """
        spatial = self.encoder(batch)           # (B, 1024, 28, 28)  stride 8
        p3 = self.neck(spatial)                 # (B, 128, 56, 56)   stride 4
        cls_logits, boxes = self.head(p3)
        return cls_logits, boxes

    def unfreeze_last_n_encoder_blocks(self, n: int) -> None:
        self.encoder.unfreeze_last_n_blocks(n)

    def trainable_parameters(self) -> list:
        return [p for p in self.parameters() if p.requires_grad]

    def neck_head_parameters(self) -> list:
        return list(self.neck.parameters()) + list(self.head.parameters())

    def encoder_parameters(self) -> list:
        return [p for p in self.encoder.parameters() if p.requires_grad]
