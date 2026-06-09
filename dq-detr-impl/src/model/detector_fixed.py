"""Fixed-query tree detector (the original q150 / q500 Clay model).

Clay encoder -> ViTDet neck -> fixed-query DETR box head. Architecture and
parameter names match the original clay/src/model/detector.py so the q{N}_final.pt
Stage-B checkpoints load directly.

forward() returns a dict with the SAME keys the DQ-DETR detector exposes for
detection (``cls_logits`` / ``pred_boxes``) so the unified Stage-C loop, the
prediction-panel viz, and the metrics code consume both models unchanged. It has
no CCM / two-stage-proposal / density outputs, and ``enable_cgfe`` is a no-op
stub kept only so the shared loop can call it without branching.
"""
import torch
import torch.nn as nn

from .clay_loader import ClayEncoderWrapper
from .head_fixed import BoxDetectionHead
from .neck import StrippedViTDetNeck


class FixedQueryTreeDetector(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        neck_in_channels: int = None,
        neck_out_channels: int = 128,
        num_queries: int = 150,
        hidden: int = 128,
        n_heads: int = 4,
        n_decoder_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = ClayEncoderWrapper(encoder)
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
        # No CGFE in the fixed-query model; the attribute/stub keep the shared
        # Stage-C loop branch-free.
        self.enable_cgfe = False

    def forward(self, batch: dict) -> dict:
        spatial = self.encoder(batch)            # (B, 1024, 32, 32) stride 8
        p3 = self.neck(spatial)                  # (B, 128, 64, 64)  stride 4
        cls_logits, boxes = self.head(p3)
        return {"cls_logits": cls_logits, "pred_boxes": boxes}

    # --- compatibility stubs (no-ops; the DQ-DETR detector implements these) ---
    def set_cgfe_enabled(self, enabled: bool) -> None:
        self.enable_cgfe = False

    def unfreeze_last_n_encoder_blocks(self, n: int) -> None:
        self.encoder.unfreeze_last_n_blocks(n)

    def trainable_parameters(self) -> list:
        return [p for p in self.parameters() if p.requires_grad]

    def neck_head_parameters(self) -> list:
        return list(self.neck.parameters()) + list(self.head.parameters())

    def encoder_parameters(self) -> list:
        return [p for p in self.encoder.parameters() if p.requires_grad]
