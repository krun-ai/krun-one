"""Lightweight fusion: project each modality into the Krun hidden space and hand a FEW tokens to the decision head.

    tower features [B, N, D_src]
      -> Projector: LayerNorm -> Linear(D_src, d) -> GELU -> Linear(d, d)                   (d = 1536 for K1)
      -> LearnedQueryPooler: Q learned queries cross-attend to the N projected tokens -> [B, Q, d]  (Q = 32 default)
      -> + modality embedding (image / document / audio / transcript)
      -> concatenated after the text tokens, attended to by the existing 2-layer decision head

Why this shape: the text encoder is untouched (text-only = the validated path), the extra cost is bounded by Q tokens
per modality in a 2-layer head, and there is no cross-modal transformer stack, no MoE, no generative decoder.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from .features import MODALITY_IDS, ModalityFeatures


class Projector(nn.Module):
    def __init__(self, src_dim: int, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(src_dim), nn.Linear(src_dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LearnedQueryPooler(nn.Module):
    """Perceiver-style resampler with ONE cross-attention: N tokens -> Q tokens (masked)."""

    def __init__(self, dim: int, num_queries: int, heads: int):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(self.queries)[None].expand(x.size(0), -1, -1)
        kv = self.norm_kv(x)
        empty = ~mask.any(1)
        key_pad = ~mask.clone()
        key_pad[empty, 0] = False                      # avoid all-masked rows (NaN); output zeroed below
        out, _ = self.attn(q, kv, kv, key_padding_mask=key_pad, need_weights=False)
        return out.masked_fill(empty[:, None, None], 0.0)


class ModalityAdapter(nn.Module):
    """Projector + pooler for one modality (one per tower; documents may share the vision adapter's tower)."""

    def __init__(self, src_dim: int, dim: int, num_queries: int, heads: int):
        super().__init__()
        self.projector = Projector(src_dim, dim)
        self.pooler = LearnedQueryPooler(dim, num_queries, heads)

    def forward(self, f: ModalityFeatures) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.projector(f.embeddings.float())
        pooled = self.pooler(x, f.mask.bool())
        m = f.mask.any(1)[:, None].expand(-1, pooled.size(1))
        return pooled, m


class LightweightFusion(nn.Module):
    """Holds one adapter per enabled modality (built lazily with the tower's width) + modality embeddings."""

    def __init__(self, dim: int, num_queries: int = 32, heads: int = 8, share_document_adapter: bool = True):
        super().__init__()
        self.dim, self.num_queries, self.heads = dim, num_queries, heads
        self.share_document_adapter = share_document_adapter
        self.adapters = nn.ModuleDict()
        self.modality_emb = nn.Embedding(len(MODALITY_IDS), dim)

    def adapter_key(self, modality: str) -> str:
        return "image" if (modality == "document" and self.share_document_adapter) else modality

    def ensure_adapter(self, modality: str, src_dim: int, num_queries: int | None = None) -> ModalityAdapter:
        key = self.adapter_key(modality)
        if key not in self.adapters:
            self.adapters[key] = ModalityAdapter(src_dim, self.dim, num_queries or self.num_queries, self.heads)
        return self.adapters[key]

    def forward(self, features: Sequence[ModalityFeatures]) -> tuple[torch.Tensor, torch.Tensor]:
        """features for ONE batch of requests (each [R, N_i, D_i]) -> tokens [R, M, d], mask [R, M]."""
        toks, masks = [], []
        for f in features:
            a = self.adapters[self.adapter_key(f.modality)]
            t, m = a(f)
            t = (t + self.modality_emb.weight[MODALITY_IDS[f.modality]]) * m[..., None].to(t.dtype)   # masked rows stay 0
            toks.append(t)
            masks.append(m)
        return torch.cat(toks, 1), torch.cat(masks, 1)


def expand_to_sequences(tokens: torch.Tensor, mask: torch.Tensor, seq_request: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-request tokens [R, M, d] -> per-sequence [B, M, d]: every question/chunk of a request sees its request's media."""
    idx = torch.as_tensor(list(seq_request), device=tokens.device)
    return tokens.index_select(0, idx), mask.index_select(0, idx)
