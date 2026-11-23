"""Function-preserving width growth for ModernBERT/mmBERT encoders (HyperCloning-style).

Idea (Samragh et al., 2024, "Scaling Smart: Accelerating LLM Pre-training with Small Model Initialization"):
a hidden state x of width h is represented in the wide model as [x, x] (width 2h). Every linear map
W (out x in) becomes the block matrix [[W, W], [W, W]] / 2 on the doubled input, so W' [x, x] = [Wx, Wx].
LayerNorm over [x, x] has the same mean/variance as over x, so it is preserved with duplicated weights.
Attention heads are duplicated (2x heads, same head_dim), so every attention pattern is unchanged, RoPE included.

ModernBERT specifics handled here:
* ``Wqkv`` output is laid out [q | k | v], each [head_0 .. head_{H-1}] -> each part is tiled separately so the
  wide model's q is [q_heads, q_heads] (heads 0..H-1 then their clones);
* ``Wi`` output is [input | gate] of the GLU -> each half is tiled separately;
* the token embedding is tiled as [e, e] (exact through the embedding LayerNorm); the tied MLM decoder would then
  double the logits, so the MLM head's final LayerNorm weight is tiled as [g, g] / 2 ([e,e] . [y,y]/2 = e . y).

``noise_std`` adds symmetric-breaking noise (without it the two halves receive identical gradients forever).
"""

from __future__ import annotations

import torch
from torch import nn


def _tile_in_out(w: torch.Tensor, parts_out: int = 1) -> torch.Tensor:
    """W (out, in) -> W' (2*out, 2*in) = [[W, W], [W, W]] / 2, tiling each of ``parts_out`` row blocks separately."""
    out, _ = w.shape
    blocks = w.chunk(parts_out, dim=0) if parts_out > 1 else (w,)
    tiled = []
    for b in blocks:
        row = torch.cat([b, b], dim=1) / 2          # (out_b, 2in)
        tiled.append(torch.cat([row, row], dim=0))  # (2out_b, 2in)
    return torch.cat(tiled, dim=0)


def _tile_vec(v: torch.Tensor) -> torch.Tensor:
    return torch.cat([v, v], dim=0)


@torch.no_grad()
def widen_modernbert(src, factor: int = 2, noise_std: float = 0.0, seed: int = 0):
    """Return a new ModernBertModel (or ForMaskedLM) whose width, heads and FFN are ``factor`` x the source.

    Only factor 2 is implemented (the one the Krun One v1 candidate needs).
    """
    if factor != 2:
        raise NotImplementedError("only factor=2")
    from transformers import ModernBertForMaskedLM, ModernBertModel

    is_mlm = isinstance(src, ModernBertForMaskedLM)
    enc = src.model if is_mlm else src
    cfg = enc.config.__class__(**enc.config.to_dict())
    cfg.hidden_size *= 2
    cfg.intermediate_size *= 2
    cfg.num_attention_heads *= 2
    cfg._attn_implementation = enc.config._attn_implementation
    dst = (ModernBertForMaskedLM(cfg) if is_mlm else ModernBertModel(cfg)).to(dtype=next(src.parameters()).dtype)
    denc = dst.model if is_mlm else dst

    denc.embeddings.tok_embeddings.weight.copy_(torch.cat([enc.embeddings.tok_embeddings.weight] * 2, dim=1))
    denc.embeddings.norm.weight.copy_(_tile_vec(enc.embeddings.norm.weight))
    for sl, dl in zip(enc.layers, denc.layers):
        if not isinstance(sl.attn_norm, nn.Identity):
            dl.attn_norm.weight.copy_(_tile_vec(sl.attn_norm.weight))
        dl.mlp_norm.weight.copy_(_tile_vec(sl.mlp_norm.weight))
        dl.attn.Wqkv.weight.copy_(_tile_in_out(sl.attn.Wqkv.weight, parts_out=3))
        dl.attn.Wo.weight.copy_(_tile_in_out(sl.attn.Wo.weight))
        dl.mlp.Wi.weight.copy_(_tile_in_out(sl.mlp.Wi.weight, parts_out=2))
        dl.mlp.Wo.weight.copy_(_tile_in_out(sl.mlp.Wo.weight))
    denc.final_norm.weight.copy_(_tile_vec(enc.final_norm.weight))
    if is_mlm:
        dst.head.dense.weight.copy_(_tile_in_out(src.head.dense.weight))
        dst.head.norm.weight.copy_(_tile_vec(src.head.norm.weight) / 2)
        if src.decoder.bias is not None:
            dst.decoder.bias.copy_(src.decoder.bias)
        dst.tie_weights()
    if noise_std > 0:
        g = torch.Generator().manual_seed(seed)
        for n, p in dst.named_parameters():
            if p.ndim == 2 and "tok_embeddings" not in n:
                p.add_(torch.randn(p.shape, generator=g, dtype=torch.float32).to(p.dtype) * noise_std * p.std())
    return dst
