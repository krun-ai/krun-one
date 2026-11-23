"""Krun One v1.1 model variants (experimental): the v1 model + an optional candidate-interaction block.

    h  = head(encoder(ids) + type_emb)                      # exactly v1
    m  = h[markers]                                         # one vector per candidate (+ [CLS] as a state token)
    m  = m + CandidateInteraction(m)                        # NEW, optional: candidates attend to each other only
    s  = scorer(m)                                          # exactly v1

Why a separate block: in v1 a candidate can only be compared with the others through full-sequence attention
(8 global encoder layers + 2 head layers over up to 1024 tokens). The interaction block runs on K + 1 vectors, so
its cost is independent of the sequence length (K <= 64 per chunk), and it is a pure "A vs B vs C" computation.

Function preservation: every interaction layer is initialized as the identity (attention out_proj and the second
FFN linear are zero), so a v1.1 model built from v1 weights returns v1's logits bit for bit until it is trained.
The public outputs (rank / apply / null_rank / ans) and the request/answer contract are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..v1.arch import ModelSpec, count_parameters
from ..v1.model import NEG, KrunOneV1Model


@dataclass(frozen=True)
class V11Spec:
    base: ModelSpec
    cand_layers: int = 0          # candidate-interaction layers (0 = plain v1 architecture)
    cand_ffn_mult: int = 2
    adapter_dim: int = 0          # R1/R2 "technical adapter": residual bottleneck on the encoder output (0 = none)
    aux_q: int = 0                # training-only Q(state, action) outputs per candidate (0 = no head)
    aux_v: int = 0                # training-only V(state) outputs (0 = no head)

    @property
    def name(self) -> str:
        return self.base.name + (f"+ci{self.cand_layers}" if self.cand_layers else "")

    def to_dict(self) -> dict:
        return {"base": self.base.to_dict(), "cand_layers": self.cand_layers, "cand_ffn_mult": self.cand_ffn_mult,
                "adapter_dim": self.adapter_dim, "aux_q": self.aux_q, "aux_v": self.aux_v}

    @classmethod
    def from_dict(cls, d: dict) -> "V11Spec":
        return cls(ModelSpec.from_dict(d["base"]), d.get("cand_layers", 0), d.get("cand_ffn_mult", 2), d.get("adapter_dim", 0),
                   d.get("aux_q", 0), d.get("aux_v", 0))

    @property
    def v1_compatible(self) -> bool:
        """True when the inference modules are exactly v1's (aux heads are training-only and dropped at export)."""
        return self.cand_layers == 0 and self.adapter_dim == 0


def cand_counts(spec: V11Spec) -> int:
    h, ff = spec.base.encoder.hidden_size, spec.cand_ffn_mult * spec.base.encoder.hidden_size
    layer = (3 * h * h + 3 * h) + (h * h + h) + (ff * h + ff) + (h * ff + h) + 4 * h
    return spec.cand_layers * layer


def count_parameters_v11(spec: V11Spec) -> dict[str, int]:
    c = dict(count_parameters(spec.base))
    extra = cand_counts(spec)
    c["candidate_interaction"] = extra
    c["decision_total"] += extra
    c["total"] += extra
    c["non_embedding"] += extra
    return c


class KrunOneV11Model(KrunOneV1Model):
    def __init__(self, encoder: nn.Module, spec: V11Spec, dropout: float = 0.1):
        super().__init__(encoder, spec.base, dropout)
        self.v11 = spec
        d = encoder.config.hidden_size
        if spec.cand_layers > 0:
            layer = nn.TransformerEncoderLayer(d, max(1, d // 64), spec.cand_ffn_mult * d, dropout, batch_first=True,
                                               norm_first=True)
            self.cand = nn.TransformerEncoder(layer, spec.cand_layers, enable_nested_tensor=False)
            with torch.no_grad():
                for lyr in self.cand.layers:      # identity at init: x + 0 (attention) + 0 (FFN)
                    lyr.self_attn.out_proj.weight.zero_()
                    lyr.self_attn.out_proj.bias.zero_()
                    lyr.linear2.weight.zero_()
                    lyr.linear2.bias.zero_()
        else:
            self.cand = None
        if spec.adapter_dim > 0:                 # zero-initialized residual bottleneck: h + up(gelu(down(norm(h))))
            self.adapter = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, spec.adapter_dim), nn.GELU(), nn.Linear(spec.adapter_dim, d))
            with torch.no_grad():
                self.adapter[-1].weight.zero_()
                self.adapter[-1].bias.zero_()
        else:
            self.adapter = None
        self.q_head = nn.Linear(d, spec.aux_q) if spec.aux_q > 0 else None
        self.v_head = nn.Linear(d, spec.aux_v) if spec.aux_v > 0 else None

    @classmethod
    def from_spec(cls, spec: V11Spec, attn_implementation: str = "sdpa") -> "KrunOneV11Model":
        from transformers import ModernBertModel

        cfg = spec.base.encoder.hf_config()
        cfg._attn_implementation = attn_implementation
        return cls(ModernBertModel(cfg), spec)

    @classmethod
    def from_v1(cls, v1: KrunOneV1Model, cand_layers: int = 0, cand_ffn_mult: int = 2, adapter_dim: int = 0, aux_q: int = 0,
                aux_v: int = 0) -> "KrunOneV11Model":
        """v1 weights + identity-initialized new blocks (function-preserving on rank / apply / null / ans).
        ``v1`` is not modified."""
        m = cls.from_spec(V11Spec(v1.spec, cand_layers, cand_ffn_mult, adapter_dim, aux_q, aux_v))
        missing, unexpected = m.load_state_dict(v1.state_dict(), strict=False)
        new = ("cand.", "adapter.", "q_head.", "v_head.")
        if unexpected or any(not k.startswith(new) for k in missing):
            raise ValueError(f"v1 -> v1.1 mismatch: {missing[:4]} {unexpected[:4]}")
        return m

    def v1_state_dict(self) -> dict:
        """The weights a plain KrunOneV1Model needs (only valid when spec.v1_compatible)."""
        if not self.v11.v1_compatible:
            raise ValueError("this model has inference modules that v1 does not have")
        return {k: v for k, v in self.state_dict().items() if not k.startswith(("q_head.", "v_head."))}

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype, null_pos=None,
                extra_tokens=None, extra_mask=None) -> dict[str, torch.Tensor]:
        if self.cand is None and self.adapter is None and self.q_head is None and self.v_head is None:
            return super().forward(input_ids, attention_mask, marker_pos, marker_mask, qtype, null_pos, extra_tokens,
                                   extra_mask)
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if self.adapter is not None:
            h = h + self.adapter(h)
        h = h + self.type_emb(qtype)[:, None, :].to(h.dtype)
        if extra_tokens is not None:
            h = torch.cat([h, extra_tokens.to(h.dtype)], 1)
            attention_mask = torch.cat([attention_mask, extra_mask.to(attention_mask.dtype)], 1)
        if self.head is not None:
            pad = ~attention_mask.bool()
            for layer in self.head.layers:
                h = layer(h, src_key_padding_mask=pad)
        d = h.size(-1)
        m = torch.gather(h, 1, marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, d))
        if self.cand is not None:       # candidates (+ [CLS] as the state token) attend to each other; padded markers masked
            x = torch.cat([h[:, :1], m], 1)
            kpm = torch.cat([torch.zeros_like(marker_mask[:, :1]), ~marker_mask], 1)
            for layer in self.cand.layers:
                x = layer(x, src_key_padding_mask=kpm)
            m = x[:, 1:]
        s = self.scorer(m).float()
        out = {"rank": s[..., 0].masked_fill(~marker_mask, NEG)}
        if s.size(-1) > 1:
            out["apply"] = s[..., 1].masked_fill(~marker_mask, NEG)
        if null_pos is not None:
            has_null = null_pos >= 0
            hn = torch.gather(h, 1, null_pos.clamp(min=0)[:, None, None].expand(-1, 1, d))[:, 0]
            out["null_rank"] = self.scorer(hn).float()[:, 0].masked_fill(~has_null, NEG)
            if self.answer_head is not None:
                out["ans"] = self.answer_head(torch.cat([hn, h[:, 0]], -1)).float()[:, 0]
            out["has_null"] = has_null
        if self.q_head is not None:
            out["q"] = self.q_head(m).float()                 # [chunks, K, aux_q] — training only
        if self.v_head is not None:
            out["v"] = self.v_head(h[:, 0]).float()            # [chunks, aux_v]    — training only
        return out
