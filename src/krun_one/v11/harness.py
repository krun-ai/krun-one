"""Shared model plumbing for the v1.1 experiments: encode probe problems, dump rank logits, build model variants.

The v1 text path is used as is (krun_one.v1 Serializer / KrunOneV1Model / merge_chunks, fp32 weights under bf16
autocast = the serving numerics of TextArtifact.forward). Nothing here writes to a v1 artifact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ..v1.arch import DecisionSpec, EncoderSpec, ModelSpec
from ..v1.model import KrunOneV1Model, chunk_index, collate_chunks, merge_chunks
from ..v1.serialization import Budget, Serializer
from . import probe

MMBERT = {
    "mmbert-small": ("jhu-clsp/mmBERT-small", EncoderSpec(hidden_size=384, num_layers=22, num_heads=6, intermediate_size=1152)),
    "mmbert-base": ("jhu-clsp/mmBERT-base", EncoderSpec(hidden_size=768, num_layers=22, num_heads=12, intermediate_size=1152)),
}
V1_DECISION = DecisionSpec(head_layers=2, head_ffn_mult=2)


def load_problems(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8").read().split("\n") if line]


def serializer(artifact: str | Path) -> Serializer:
    from transformers import AutoTokenizer

    artifact = Path(artifact)
    sq = json.loads((artifact / "decision_config.json").read_text())["sequence"]
    return Serializer(AutoTokenizer.from_pretrained(str(artifact / "tokenizer")),
                      Budget(sq["max_len"], sq["head_max_len"], max_candidates_per_chunk=sq["max_candidates_per_chunk"]))


def encode(ser: Serializer, problem: Mapping[str, Any], cond: str):
    return ser.encode("choice", problem["context"], probe.options(problem, cond), problem["instructions"])


def length(q) -> int:
    return sum(len(c.ids) for c in q.chunks)


def batches_by_length(lengths: Sequence[int], max_tokens: int, max_items: int = 128) -> list[list[int]]:
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    out, cur, mx = [], [], 0
    for i in order:
        new = max(mx, lengths[i])
        if cur and (new * (len(cur) + 1) > max_tokens or len(cur) >= max_items):
            out.append(cur)
            cur, new = [], lengths[i]
        cur.append(i)
        mx = new
    if cur:
        out.append(cur)
    return out


def forward_questions(model, qs: Sequence[Any], pad_id: int, dev: torch.device, amp: bool = True) -> dict[str, torch.Tensor]:
    flat, where = chunk_index(qs)
    b = {k: v.to(dev, non_blocking=True) for k, v in collate_chunks(flat, pad_id).items()}
    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp and dev.type == "cuda"):
        out = model(**b)
    return merge_chunks(out, where, len(qs), [q.n_candidates for q in qs])


@torch.inference_mode()
def rank_logits(model, qs: Sequence[Any], pad_id: int, dev: torch.device, max_tokens: int = 24_000) -> list[np.ndarray]:
    """Rank logits per encoded question (fp32 weights, bf16 autocast on CUDA; plain fp32 on CPU)."""
    model.eval()
    res: list[np.ndarray | None] = [None] * len(qs)
    for idx in batches_by_length([length(q) for q in qs], max_tokens):
        m = forward_questions(model, [qs[i] for i in idx], pad_id, dev)
        z = m["rank"].float().cpu().numpy()
        for n, i in enumerate(idx):
            res[i] = z[n, : qs[i].n_candidates].copy()
    return res  # type: ignore[return-value]


# ------------------------------------------------------------------------------------------------ models
def load_v1(artifact: str | Path) -> KrunOneV1Model:
    """The frozen v1 text model (read-only use: weights are loaded into a fresh module)."""
    from safetensors.torch import load_model

    artifact = Path(artifact)
    spec = ModelSpec.from_dict(json.loads((artifact / "spec.json").read_text()))
    model = KrunOneV1Model.from_spec(spec, attn_implementation="sdpa")
    load_model(model, str(artifact / "model.safetensors"))
    return model


def load_mmbert(name: str, widen: bool = False, seed: int = 0) -> KrunOneV1Model:
    """mmBERT encoder (MLM-pretrained, no decision training) + a randomly initialized v1 decision stack.

    widen=True applies v1's function-preserving HyperCloning (x2 width): mmbert-base -> the K1-W1536-L22 encoder that
    computes the SAME function as mmBERT-base at step 0 ("k1-0b", the init Krun One v1 was trained from)."""
    from transformers import ModernBertModel

    from ..v1.growth import widen_modernbert

    hf_id, enc = MMBERT[name]
    src = ModernBertModel.from_pretrained(hf_id)
    if widen:
        src = widen_modernbert(src, 2)
        enc = EncoderSpec(hidden_size=enc.hidden_size * 2, num_layers=enc.num_layers, num_heads=enc.num_heads * 2,
                          intermediate_size=enc.intermediate_size * 2)
    spec = ModelSpec(name + ("-x2" if widen else ""), enc, V1_DECISION)
    torch.manual_seed(seed)
    model = KrunOneV1Model.from_spec(spec, attn_implementation="sdpa")
    missing, unexpected = model.encoder.load_state_dict(src.state_dict(), strict=False)
    if missing or unexpected:
        raise ValueError(f"encoder init mismatch: {missing[:4]} {unexpected[:4]}")
    return model
