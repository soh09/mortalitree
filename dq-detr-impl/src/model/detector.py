"""Full detection pipeline: Clay encoder -> ViTDet neck -> DQ-DETR head.

DQ-DETR is layered on top of the original Clay + ViTDet-neck design:

    Clay encoder -> neck -> EMSV feature map
        |-> CCM        : count-category logits  + density map
        |-> CGFE        : density-gated enhancement of EMSV  (Stage 2 only)
        '-> TwoStageDQSHead : two-stage proposals + Dynamic Query Selection
                              (#queries = dynamic_query_list[count category])

The two-phase schedule is controlled by ``enable_cgfe``: Stage 1 bypasses CGFE
(EMSV passes straight through) while the counting head stabilises; Stage 2 flips
it on so the head sees density-enhanced features.
"""
import torch
import torch.nn as nn

from .ccm import CategoricalCountingModule
from .cgfe import CGFE
from .clay_loader import ClayEncoderWrapper
from .head import TwoStageDQSHead
from .neck import StrippedViTDetNeck


class TreeDetector(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        neck_in_channels: int = None,
        neck_out_channels: int = 128,
        hidden: int = 128,
        n_heads: int = 4,
        n_decoder_layers: int = 3,
        dropout: float = 0.1,
        dynamic_query_list=(200, 400, 600),
        ccm_cls_num: int = 3,
        ccm_params=(100, 300),
        enable_cgfe: bool = False,
        anchor_size: float = 0.05,
    ):
        super().__init__()
        self.encoder = ClayEncoderWrapper(encoder)
        # Default the neck's input width to the encoder's actual embedding dim
        # (1024 for Clay v1.5 large) rather than hardcoding it.
        if neck_in_channels is None:
            neck_in_channels = self.encoder.dim
        self.neck = StrippedViTDetNeck(neck_in_channels, neck_out_channels)

        # DQ-DETR pieces operate on the single-scale EMSV (neck output) map.
        self.ccm = CategoricalCountingModule(
            in_channels=neck_out_channels,
            cls_num=ccm_cls_num,
            density_channels=neck_out_channels,
        )
        self.cgfe = CGFE(gate_channels=neck_out_channels)

        # The learnable query pool only needs to cover the densest DQS bin -- the
        # per-image query count is chosen dynamically from dynamic_query_list.
        max_queries = max(dynamic_query_list)
        self.head = TwoStageDQSHead(
            feat_channels=neck_out_channels,
            hidden=hidden,
            max_queries=max_queries,
            n_heads=n_heads,
            n_decoder_layers=n_decoder_layers,
            dropout=dropout,
            dynamic_query_list=dynamic_query_list,
            anchor_size=anchor_size,
        )

        # CCM count-category thresholds (used by the training loop to build
        # targets); kept on the model so callers don't re-specify them.
        self.ccm_params = list(ccm_params)
        self.ccm_cls_num = ccm_cls_num
        self.enable_cgfe = enable_cgfe

    def forward(self, batch: dict) -> dict:
        """
        Returns a dict with:
            cls_logits:     (B, Q) decoder objectness  (Q = dynamic num_select)
            pred_boxes:     (B, Q, 4) (cx,cy,w,h) in [0,1]
            count_logits:   (B, ccm_cls_num) CCM count-category logits
            enc_cls_logits: (B, HW) two-stage proposal objectness
            enc_boxes:      (B, HW, 4) two-stage proposal boxes
            num_select:     int, queries selected this forward pass
        """
        spatial = self.encoder(batch)            # (B, 1024, 32, 32)  stride 8
        emsv = self.neck(spatial)                # (B, 128, 64, 64)   stride 4
        count_logits, density = self.ccm(emsv)   # (B, cls_num), (B, 128, 64, 64)
        feat = self.cgfe(emsv, density) if self.enable_cgfe else emsv
        out = self.head(feat, count_logits)
        out["count_logits"] = count_logits
        out["density"] = density          # (B, C, H, W) CCM density map (for viz)
        return out

    def set_cgfe_enabled(self, enabled: bool) -> None:
        self.enable_cgfe = enabled

    def unfreeze_last_n_encoder_blocks(self, n: int) -> None:
        self.encoder.unfreeze_last_n_blocks(n)

    def trainable_parameters(self) -> list:
        return [p for p in self.parameters() if p.requires_grad]

    def neck_head_parameters(self) -> list:
        """All from-scratch params (neck + CCM + CGFE + DQS head)."""
        return (
            list(self.neck.parameters())
            + list(self.ccm.parameters())
            + list(self.cgfe.parameters())
            + list(self.head.parameters())
        )

    def encoder_parameters(self) -> list:
        return [p for p in self.encoder.parameters() if p.requires_grad]
