"""M1 multimodal alignment model: FROZEN Krun One v1 text core + frozen towers + small trainable fusion.

    image / document page -> SigLIP2-base NaFlex (frozen) --\
    audio                 -> Whisper-small encoder (frozen) -+-> ModalityAdapter (projector + 32-query pooler) [trainable]
                                                               + modality embedding                           [trainable]
    question + candidates -> K1 text encoder (frozen, no grad) -> + type_emb (frozen)
                          -> [text tokens ; media tokens] -> 2-layer decision head (frozen) + LoRA adapters [trainable]
                          -> scorer (frozen) -> rank / apply / null / ans

The head LoRA is used ONLY on the multimodal forward (``forward_mm``); a text-only request goes through
``core(**batch)`` — the frozen artifact's own forward, bitwise unchanged. The core's parameters never receive
gradients (requires_grad False, encoder under no_grad) and their hash is checked before/after training.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from ..model import NEG, KrunOneV1Model
from .features import MODALITY_IDS, ModalityFeatures
from .fusion import ModalityAdapter


class LoRA(nn.Module):
    def __init__(self, d_in: int, d_out: int, r: int, alpha: float):
        super().__init__()
        self.a = nn.Parameter(torch.randn(r, d_in) / math.sqrt(d_in))
        self.b = nn.Parameter(torch.zeros(d_out, r))          # zero-init: step 0 == frozen head
        self.scale = alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(F.linear(x, self.a.to(x.dtype)), self.b.to(x.dtype)) * self.scale


class HeadLoRA(nn.Module):
    """LoRA on in_proj / out_proj / linear1 / linear2 of every decision-head layer."""

    def __init__(self, core: KrunOneV1Model, r: int = 16, alpha: float = 32.0):
        super().__init__()
        self.layers = nn.ModuleList()
        for layer in core.head.layers:
            d = layer.linear1.in_features
            ff = layer.linear1.out_features
            self.layers.append(nn.ModuleDict({"in_proj": LoRA(d, 3 * d, r, alpha), "out_proj": LoRA(d, d, r, alpha),
                                              "linear1": LoRA(d, ff, r, alpha), "linear2": LoRA(ff, d, r, alpha)}))


def head_layer_forward(layer: nn.TransformerEncoderLayer, lora: nn.ModuleDict | None, x: torch.Tensor,
                       pad: torch.Tensor) -> torch.Tensor:
    """Functional pre-norm TransformerEncoderLayer (eval: dropout off) with optional LoRA deltas."""
    sa = layer.self_attn
    B, L, d = x.shape
    H = sa.num_heads
    h = layer.norm1(x)
    qkv = F.linear(h, sa.in_proj_weight, sa.in_proj_bias)
    if lora is not None:
        qkv = qkv + lora["in_proj"](h)
    q, k, v = qkv.view(B, L, 3, H, d // H).permute(2, 0, 3, 1, 4)
    attn_mask = (~pad)[:, None, None, :]                       # True = attend
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)
    o = o.transpose(1, 2).reshape(B, L, d)
    out = F.linear(o, sa.out_proj.weight, sa.out_proj.bias)
    if lora is not None:
        out = out + lora["out_proj"](o)
    x = x + out
    h = layer.norm2(x)
    f = F.linear(h, layer.linear1.weight, layer.linear1.bias)
    if lora is not None:
        f = f + lora["linear1"](h)
    f = layer.activation(f)
    g = F.linear(f, layer.linear2.weight, layer.linear2.bias)
    if lora is not None:
        g = g + lora["linear2"](f)
    return x + g


class M1Fusion(nn.Module):
    """All trainable M1 parameters: one adapter per tower (image adapter shared by documents), modality embeddings,
    head LoRA."""

    def __init__(self, core: KrunOneV1Model, vision_dim: int, audio_dim: int, tokens: int = 32, heads: int = 8,
                 lora_r: int = 16, lora_alpha: float = 32.0):
        super().__init__()
        d = core.encoder.config.hidden_size
        self.adapters = nn.ModuleDict({"image": ModalityAdapter(vision_dim, d, tokens, heads),
                                       "audio": ModalityAdapter(audio_dim, d, tokens, heads)})
        self.modality_emb = nn.Embedding(len(MODALITY_IDS), d)
        nn.init.normal_(self.modality_emb.weight, std=0.02)
        self.lora = HeadLoRA(core, lora_r, lora_alpha) if lora_r > 0 else None
        self.config = {"vision_dim": vision_dim, "audio_dim": audio_dim, "tokens": tokens, "heads": heads,
                       "lora_r": lora_r, "lora_alpha": lora_alpha}

    def media_tokens(self, f: ModalityFeatures) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.adapters["audio" if f.modality == "audio" else "image"]
        t, m = a(f)
        t = (t + self.modality_emb.weight[MODALITY_IDS[f.modality]]) * m[..., None].to(t.dtype)
        return t, m


class M1Model(nn.Module):
    def __init__(self, core: KrunOneV1Model, fusion: M1Fusion, freeze_core: bool = True):
        """freeze_core=True for TRAINING (no gradient can reach the core). SERVING must pass freeze_core=False: flipping
        requires_grad on the core's parameters changes PyTorch's numeric path under bf16 autocast (measured on RTX 4090:
        text-only signals differ at bf16 resolution, max |dlogit| 0.06, although the weights are identical), so a
        serving stack leaves the frozen artifact's parameters exactly as loaded; gradients are excluded there by
        inference_mode / no optimizer."""
        super().__init__()
        self.core, self.fusion = core, fusion
        if freeze_core:
            for p in self.core.parameters():
                p.requires_grad_(False)
        self.core.eval()

    def train(self, mode: bool = True):          # the core stays in eval mode (dropout off) forever
        super().train(mode)
        self.core.eval()
        return self

    def forward_text(self, batch: dict) -> dict[str, torch.Tensor]:
        return self.core(**batch)

    def forward_mm(self, batch: dict, media: tuple[torch.Tensor, torch.Tensor] | None, seq_request: Sequence[int] | None,
                   use_lora: bool = True) -> dict[str, torch.Tensor]:
        """batch: collate_chunks output (one row per chunk); media: per-request tokens [R, M, d] + mask; seq_request:
        request index of each chunk row. media=None -> text-only head (the frozen text decision, "blind" baseline
        when the context is empty)."""
        c = self.core
        with torch.no_grad():
            h = c.encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
            h = h + c.type_emb(batch["qtype"])[:, None, :].to(h.dtype)
        att = batch["attention_mask"]
        if media is not None:
            idx = torch.as_tensor(list(seq_request), device=h.device)
            t, m = media[0].index_select(0, idx), media[1].index_select(0, idx)
            h = torch.cat([h, t.to(h.dtype)], 1)
            att = torch.cat([att, m.to(att.dtype)], 1)
        pad = ~att.bool()
        for i, layer in enumerate(c.head.layers):
            lora = self.fusion.lora.layers[i] if (use_lora and self.fusion.lora is not None and media is not None) else None
            h = head_layer_forward(layer, lora, h, pad)
        d = h.size(-1)
        mk = torch.gather(h, 1, batch["marker_pos"].clamp(min=0)[:, :, None].expand(-1, -1, d))
        s = c.scorer(mk).float()
        out = {"rank": s[..., 0].masked_fill(~batch["marker_mask"], NEG)}
        if s.size(-1) > 1:
            out["apply"] = s[..., 1].masked_fill(~batch["marker_mask"], NEG)
        null_pos = batch.get("null_pos")
        if null_pos is not None:
            has_null = null_pos >= 0
            hn = torch.gather(h, 1, null_pos.clamp(min=0)[:, None, None].expand(-1, 1, d))[:, 0]
            out["null_rank"] = c.scorer(hn).float()[:, 0].masked_fill(~has_null, NEG)
            if c.answer_head is not None:
                out["ans"] = c.answer_head(torch.cat([hn, h[:, 0]], -1)).float()[:, 0]
            out["has_null"] = has_null
        return out


def trainable_parameter_count(fusion: M1Fusion) -> dict[str, int]:
    n = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    out = {"image_adapter": n(fusion.adapters["image"]), "audio_adapter": n(fusion.adapters["audio"]),
           "modality_emb": fusion.modality_emb.weight.numel(), "head_lora": n(fusion.lora) if fusion.lora else 0}
    out["total"] = sum(out.values())
    return out


def core_digest(core: nn.Module) -> str:
    """sha256 over the core state dict (names + raw bytes): proves the text core never changed."""
    import hashlib

    h = hashlib.sha256()
    for k, v in sorted(core.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes() if v.dtype != torch.bool
                 else v.cpu().numpy().tobytes())
    return h.hexdigest()
