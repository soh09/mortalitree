"""Fixed-query DETR box head (the original q150/q500 Clay detector head).

Kept here so this repo can build and Stage-C-finetune the fixed-query models
(q150 / q500) alongside the DQ-DETR model. Architecture and parameter names match
the original clay/src/model/head.py exactly, so the q{N}_final.pt Stage-B
checkpoints load without renaming.
"""
import torch
import torch.nn as nn


class SinCos2DPositionalEncoding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 4 == 0, "dim must be divisible by 4 for 2D sin/cos encoding"
        self.dim = dim

    def forward(self, H: int, W: int, device: torch.device) -> torch.Tensor:
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
        pe = torch.cat([pe_y, pe_x], dim=-1)   # (H, W, dim)
        return pe.reshape(H * W, self.dim)


class BoxDetectionHead(nn.Module):
    """DETR-style box head with Q learnable object queries (fixed Q)."""

    def __init__(
        self,
        feat_channels: int = 128,
        num_queries: int = 124,
        hidden: int = 128,
        n_heads: int = 4,
        n_decoder_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Embedding(num_queries, hidden)
        self.pos_enc = SinCos2DPositionalEncoding(hidden)
        self.input_proj = nn.Conv2d(feat_channels, hidden, kernel_size=1)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_decoder_layers)
        self.cls_head = nn.Linear(hidden, 1)
        self.box_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 4),
        )

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, _, H, W = feat.shape
        x = self.input_proj(feat)                       # (B, hidden, H, W)
        pos = self.pos_enc(H, W, feat.device)           # (H*W, hidden)
        memory = x.flatten(2).transpose(1, 2)           # (B, H*W, hidden)
        memory = memory + pos.unsqueeze(0)
        q = self.queries.weight.unsqueeze(0).expand(B, -1, -1)  # (B, Q, hidden)
        decoded = self.decoder(q, memory)               # (B, Q, hidden)
        cls_logits = self.cls_head(decoded).squeeze(-1)  # (B, Q)
        boxes = self.box_head(decoded).sigmoid()        # (B, Q, 4) in [0, 1]
        return cls_logits, boxes
