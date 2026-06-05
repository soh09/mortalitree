"""Two-stage DETR head with Dynamic Query Selection (DQS), from DQ-DETR.

Replaces the original fixed-query DETR head. The EMSV feature map is flattened
into "encoder memory" tokens; a lightweight two-stage proposal head scores every
token (objectness) and regresses a box from a per-token grid anchor. Dynamic
Query Selection then keeps the top-k tokens, where k is chosen from the CCM's
predicted count category (`dynamic_query_list[category]`) -- so crowded tiles get
more queries than sparse ones. The selected proposals seed a DAB-style decoder
that iteratively refines the boxes.

This is the single-scale, plain-attention adaptation of DQ-DETR's deformable
two-stage decoder: the contribution (count-conditioned dynamic query number +
anchor-seeded two-stage queries) is preserved without the CUDA deformable ops.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(min=0, max=1)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))


class MLP(nn.Module):
    """Plain multi-layer perceptron (ReLU between layers), DETR's helper."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return x


class SinCos2DPositionalEncoding(nn.Module):
    """2D sin/cos positional encoding for the (flattened) memory tokens."""

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
        pe = torch.cat([pe_y, pe_x], dim=-1)        # (H, W, dim)
        return pe.reshape(H * W, self.dim)


def gen_sineembed_for_boxes(boxes: torch.Tensor, dim: int, temperature: float = 10000.0) -> torch.Tensor:
    """Sine positional embedding of (cx, cy, w, h) boxes -> (..., dim).

    Each of the 4 coordinates contributes ``dim // 4`` features (sin/cos
    interleaved over ``dim // 8`` frequencies), so the decoder query position is
    derived from the *box* itself (DAB-DETR style) rather than a free embedding.
    """
    assert dim % 8 == 0, "dim must be divisible by 8 for 4-coord box sine embedding"
    d = dim // 8
    scale = 2 * math.pi
    dim_t = torch.arange(d, device=boxes.device).float()
    dim_t = temperature ** (dim_t / d)
    embeds = []
    for i in range(4):
        val = boxes[..., i, None] * scale / dim_t           # (..., d)
        embeds.append(torch.cat([val.sin(), val.cos()], dim=-1))  # (..., 2d)
    return torch.cat(embeds, dim=-1)                         # (..., 8d == dim)


class DecoderLayer(nn.Module):
    """DETR-style decoder layer that injects query/memory positions into attention."""

    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.linear2 = nn.Linear(dim_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, tgt, query_pos, memory, memory_pos):
        # Self-attention among queries (positions added to Q and K only).
        q = k = tgt + query_pos
        sa = self.self_attn(q, k, tgt)[0]
        tgt = self.norm1(tgt + self.dropout(sa))
        # Cross-attention to memory (query pos on Q, memory pos on K).
        ca = self.cross_attn(tgt + query_pos, memory + memory_pos, memory)[0]
        tgt = self.norm2(tgt + self.dropout(ca))
        # Feed-forward.
        ff = self.linear2(self.dropout(self.act(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout(ff))
        return tgt


class TwoStageDQSHead(nn.Module):
    """Two-stage query selection + dynamic query number + iterative box refinement."""

    def __init__(
        self,
        feat_channels: int = 128,
        hidden: int = 128,
        max_queries: int = 600,
        n_heads: int = 4,
        n_decoder_layers: int = 3,
        dropout: float = 0.1,
        dynamic_query_list=(200, 400, 600),
        anchor_size: float = 0.05,
    ):
        super().__init__()
        self.hidden = hidden
        self.max_queries = max_queries
        self.anchor_size = anchor_size
        # DQS lookup: count category -> number of queries. Clamp the pool to the
        # largest entry so the learnable query bank is big enough for any tile.
        self.dynamic_query_list = list(dynamic_query_list)
        assert max_queries >= max(self.dynamic_query_list)

        self.input_proj = nn.Conv2d(feat_channels, hidden, kernel_size=1)
        self.pos_enc = SinCos2DPositionalEncoding(hidden)

        # Stage 1 (two-stage proposals over every memory token).
        self.enc_cls_head = nn.Linear(hidden, 1)
        self.enc_box_head = MLP(hidden, hidden, 4, 3)

        # Stage 2 (decoder over selected queries).
        self.tgt_embed = nn.Embedding(max_queries, hidden)
        self.query_pos_mlp = MLP(hidden, hidden, hidden, 2)
        self.layers = nn.ModuleList(
            DecoderLayer(hidden, n_heads, hidden * 4, dropout)
            for _ in range(n_decoder_layers)
        )
        self.cls_head = nn.Linear(hidden, 1)
        self.box_head = MLP(hidden, hidden, 4, 3)

        self._reset_parameters()

    def _reset_parameters(self):
        # Box heads start as identity-on-anchor (predict zero delta).
        for box_head in (self.enc_box_head, self.box_head):
            nn.init.constant_(box_head.layers[-1].weight, 0.0)
            nn.init.constant_(box_head.layers[-1].bias, 0.0)
        # Focal-loss style prior on the classification biases (rare positives).
        prior = 0.01
        bias = -math.log((1 - prior) / prior)
        nn.init.constant_(self.enc_cls_head.bias, bias)
        nn.init.constant_(self.cls_head.bias, bias)

    def _grid_anchors(self, H: int, W: int, device) -> torch.Tensor:
        """Per-token anchor boxes (cx, cy at pixel centers, fixed default w/h)."""
        ys = (torch.arange(H, device=device).float() + 0.5) / H
        xs = (torch.arange(W, device=device).float() + 0.5) / W
        cy, cx = torch.meshgrid(ys, xs, indexing="ij")
        wh = torch.full_like(cx, self.anchor_size)
        return torch.stack([cx.flatten(), cy.flatten(), wh.flatten(), wh.flatten()], dim=-1)

    def _num_select(self, count_logits: torch.Tensor) -> int:
        """DQS: choose k from the densest predicted category in the batch.

        Taking the max category across the batch keeps every tile's query tensor
        the same width (rectangular batch), matching DQ-DETR's behaviour.
        """
        category = count_logits.argmax(dim=1)                 # (B,)
        idx = min(int(category.max().item()), len(self.dynamic_query_list) - 1)
        return min(self.dynamic_query_list[idx], self.max_queries)

    def forward(self, feat: torch.Tensor, count_logits: torch.Tensor) -> dict:
        B, _, H, W = feat.shape
        device = feat.device

        x = self.input_proj(feat)                             # (B, hidden, H, W)
        memory = x.flatten(2).transpose(1, 2)                 # (B, HW, hidden)
        pos = self.pos_enc(H, W, device).unsqueeze(0)         # (1, HW, hidden)

        anchors = self._grid_anchors(H, W, device)            # (HW, 4)
        anchors_unsig = inverse_sigmoid(anchors).unsqueeze(0) # (1, HW, 4)

        # --- Stage 1: proposals over all memory tokens (position-aware) ---
        mem_pos = memory + pos
        enc_cls_logits = self.enc_cls_head(mem_pos).squeeze(-1)            # (B, HW)
        enc_boxes = (self.enc_box_head(mem_pos) + anchors_unsig).sigmoid()  # (B, HW, 4)

        # --- DQS: keep the top-k highest-objectness proposals ---
        k = self._num_select(count_logits)
        topk_idx = enc_cls_logits.topk(k, dim=1).indices                   # (B, k)
        reference = torch.gather(
            enc_boxes, 1, topk_idx.unsqueeze(-1).expand(-1, -1, 4)
        ).detach()                                                         # (B, k, 4)

        # --- Stage 2: decode selected queries with iterative box refinement ---
        tgt = self.tgt_embed.weight[:k].unsqueeze(0).expand(B, -1, -1)     # (B, k, hidden)
        mem_pos_full = pos.expand(B, -1, -1)
        for layer in self.layers:
            query_pos = self.query_pos_mlp(gen_sineembed_for_boxes(reference, self.hidden))
            tgt = layer(tgt, query_pos, memory, mem_pos_full)
            box = (self.box_head(tgt) + inverse_sigmoid(reference)).sigmoid()
            reference = box.detach()

        cls_logits = self.cls_head(tgt).squeeze(-1)                        # (B, k)
        pred_boxes = box                                                   # (B, k, 4)

        return {
            "cls_logits": cls_logits,
            "pred_boxes": pred_boxes,
            "enc_cls_logits": enc_cls_logits,
            "enc_boxes": enc_boxes,
            "num_select": k,
        }
