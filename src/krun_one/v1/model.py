"""Krun One v1 decision model.

Laya's DecisionModel (Apache-2.0, Convai Innovations; see NOTICE) generalized:

    h      = encoder(ids)                                  # HF ModernBertModel (mmBERT architecture)
    h      = h + type_emb[qtype]                            # choice / score / noul / multi
    h      = head(h)                                        # Laya's pre-norm transformer layers
    s      = scorer(h[markers])  -> [rank, apply]           # per candidate marker (Laya: rank only)
    null   = scorer(h[NULL])[rank]                          # the "none of them" slot, same scorer
    ans    = answer_head([h[NULL]; h[CLS]])                  # answerability logit

Outputs are raw logits; which ones are trained and read is decided by the loss / decoding config, so the
answerability variants (margin baseline, null-in-softmax, answerability head, max applicability) are all
computed from the same forward and can be compared on the same checkpoint.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.utils.checkpoint

from .arch import ModelSpec

NEG = -1e4


class KrunOneV1Model(nn.Module):
    def __init__(self, encoder: nn.Module, spec: ModelSpec, dropout: float = 0.1):
        super().__init__()
        self.spec = spec
        self.encoder = encoder
        d = encoder.config.hidden_size
        ds = spec.decision
        self.type_emb = nn.Embedding(ds.n_qtypes, d)
        if ds.head_layers > 0:
            layer = nn.TransformerEncoderLayer(d, max(1, d // 64), ds.head_ffn_mult * d, dropout, batch_first=True,
                                               norm_first=True)
            self.head = nn.TransformerEncoder(layer, ds.head_layers, enable_nested_tensor=False)
        else:
            self.head = None
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, ds.scorer_outputs))
        self.answer_head = (nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
                            if ds.answerability_head else None)
        self.head_checkpointing = False

    @classmethod
    def from_spec(cls, spec: ModelSpec, attn_implementation: str = "sdpa") -> "KrunOneV1Model":
        from transformers import ModernBertModel

        cfg = spec.encoder.hf_config()
        cfg._attn_implementation = attn_implementation
        return cls(ModernBertModel(cfg), spec)

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype, null_pos=None,
                extra_tokens=None, extra_mask=None) -> dict[str, torch.Tensor]:
        """``extra_tokens`` [B, M, d] / ``extra_mask`` [B, M]: optional non-text tokens (image / document / audio),
        already projected to the hidden size (krun_one.v1.multimodal). They are appended AFTER the text encoder and
        attended to only by the decision head, so a text-only call (extra_tokens=None) runs exactly the text path."""
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = h + self.type_emb(qtype)[:, None, :].to(h.dtype)
        if extra_tokens is not None:
            if self.head is None:
                raise ValueError("multimodal fusion needs head_layers >= 1 (the head is where modalities meet)")
            h = torch.cat([h, extra_tokens.to(h.dtype)], 1)
            attention_mask = torch.cat([attention_mask, extra_mask.to(attention_mask.dtype)], 1)
        if self.head is not None:
            pad = ~attention_mask.bool()
            for layer in self.head.layers:
                if self.head_checkpointing and self.training and torch.is_grad_enabled():
                    h = torch.utils.checkpoint.checkpoint(layer, h, None, pad, use_reentrant=False)
                else:
                    h = layer(h, src_key_padding_mask=pad)
        d = h.size(-1)
        m = torch.gather(h, 1, marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, d))
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
        return out


# ------------------------------------------------------------------------------------------------ batching
def collate_chunks(chunks: Sequence[Any], pad_id: int) -> dict[str, torch.Tensor]:
    """chunks: EncodedChunk-like objects (ids, markers, null_pos, qtype) -> right-padded tensors."""
    n, L = len(chunks), max(len(c.ids) for c in chunks)
    kmax = max(len(c.markers) for c in chunks)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    for i, c in enumerate(chunks):
        ids[i, :len(c.ids)] = torch.tensor(c.ids)
        att[i, :len(c.ids)] = 1
        mpos[i, :len(c.markers)] = torch.tensor(c.markers)
        mmask[i, :len(c.markers)] = True
    return {"input_ids": ids, "attention_mask": att, "marker_pos": mpos, "marker_mask": mmask,
            "qtype": torch.tensor([c.qtype for c in chunks], dtype=torch.long),
            "null_pos": torch.tensor([c.null_pos for c in chunks], dtype=torch.long)}


def chunk_index(questions: Sequence[Any]) -> tuple[list[Any], list[tuple[int, int, list[int]]]]:
    """Flatten EncodedQuestions into chunks + (question, chunk-row, candidate indices) map."""
    flat, where = [], []
    for qi, q in enumerate(questions):
        for c in q.chunks:
            where.append((qi, len(flat), c.candidate_index))
            flat.append(c)
    return flat, where


def merge_chunks(out: Mapping[str, torch.Tensor], where: Sequence[tuple[int, int, list[int]]], n_questions: int,
                 n_candidates: Sequence[int]) -> dict[str, torch.Tensor]:
    """Per-chunk outputs -> per-question outputs.

    rank / apply: candidates are placed at their question-level index (chunks are disjoint);
    null_rank:    mean over the question's chunks (every chunk has its own [NULL] slot);
    ans:          max over chunks — a question is answerable if *some* chunk contains an applicable candidate.
    Single-chunk questions are an identity map (asserted in tests).
    """
    dev = out["rank"].device
    kmax = max(n_candidates)
    q_idx, c_idx, r_idx, j_idx = [], [], [], []
    for qi, row, cidx in where:
        q_idx += [qi] * len(cidx)
        c_idx += list(cidx)
        r_idx += [row] * len(cidx)
        j_idx += list(range(len(cidx)))
    qi_t, ci_t = torch.tensor(q_idx, device=dev), torch.tensor(c_idx, device=dev)
    ri_t, ji_t = torch.tensor(r_idx, device=dev), torch.tensor(j_idx, device=dev)
    res = {}
    for k in ("rank", "apply"):
        if k in out:  # out-of-place index_put keeps autograd happy
            res[k] = torch.full((n_questions, kmax), NEG, device=dev, dtype=out[k].dtype).index_put(
                (qi_t, ci_t), out[k][ri_t, ji_t])
    res["mask"] = torch.zeros((n_questions, kmax), dtype=torch.bool, device=dev).index_put(
        (qi_t, ci_t), torch.ones(len(q_idx), dtype=torch.bool, device=dev))
    row_q = torch.tensor([qi for qi, _, _ in where], device=dev)
    rows = torch.tensor([row for _, row, _ in where], device=dev)
    if "null_rank" in out:
        has = out["has_null"][rows].float()
        s_ = torch.zeros(n_questions, device=dev).index_add(0, row_q, out["null_rank"][rows] * has)
        n_ = torch.zeros(n_questions, device=dev).index_add(0, row_q, has)
        res["null_rank"] = torch.where(n_ > 0, s_ / n_.clamp(min=1), torch.full_like(s_, NEG))
    if "ans" in out:
        res["ans"] = torch.full((n_questions,), NEG, device=dev).scatter_reduce(
            0, row_q, out["ans"][rows], reduce="amax", include_self=True)
    return res
