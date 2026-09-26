"""Krun One model: the Laya decision architecture, self-contained (no dependency on ./laya).

Ported from upstream Laya ``rl_common.py`` (Apache-2.0, Convai Innovations): ``DecisionModel``,
option rendering and ``build_sequence`` are kept *numerically identical* to upstream
(tests/test_laya_equivalence.py checks this against the ./laya checkout when it is present).
Nothing architectural is changed: ModernBERT-large encoder + type embedding + 2-layer
transformer decision head + option-marker scorer (+ the upstream act head, which is kept only
so upstream checkpoints load strictly; Krun One never uses it — it saturates at ~1.0).

Input layout (upstream):
    [CLS] "choice question: {instructions}" [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] context [SEP]
Each option is scored at its own [MASK] marker; softmax over the options of the decision.

Public inference API::

    from krun_one import KrunOne
    m = KrunOne.from_pretrained("artifacts/krun-one-v0")
    m.decide("Find my meetings tomorrow",
             [{"name": "calendar_search", "description": "Search calendar events"},
              {"name": "send_email", "description": "Send an email"}])
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint

QTYPES = {"choice": 0, "score": 1, "noul": 2}
OPTION_TOKEN_CAP = 48  # upstream: every option is cut to 48 tokens (+ its marker)

DEFAULT_INSTRUCTIONS = {
    "intent": "Choose the option that best describes the intent of the user message.",
    "tool": "Choose the function that should be called to fulfil the user request.",
}

_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def humanize_label(label: str) -> str:
    """Same rule as krunbench.options.humanize_label (duplicated so serving has no krunbench dependency)."""
    t = _CAMEL_RE.sub(" ", label)
    t = re.sub(r"[_\-./:]+", " ", t)
    return " ".join(t.lower().split())


def option_texts(names: Sequence[str], descriptions: Mapping[str, str | None] | None = None) -> list[str]:
    """Text shown to the model for each option.

    * with a description (tools): ``"{name}: {description}"`` — the upstream Laya rendering of a
      choice criterion ``{name: description}``; the name is kept verbatim (it is the tool id).
    * without one (intent labels): the humanized label (``card_arrival`` -> ``card arrival``), as in
      KRUN-004; falls back to the raw names if humanizing creates duplicates.
    """
    descriptions = descriptions or {}
    if any(descriptions.get(n) for n in names):
        return [f"{n}: {descriptions[n]}" if descriptions.get(n) else n for n in names]
    shown = [humanize_label(n) for n in names]
    return shown if len(set(shown)) == len(shown) else list(names)


def build_sequence(tok, context: str, options: Sequence[str], instructions: str, max_len: int, head_max_len: int,
                   qtype: str = "choice") -> tuple[list[int], list[int]]:
    """Upstream ``build_sequence`` for one choice question whose rendered option texts are ``options``.

    Returns input ids and the positions of the per-option [MASK] markers (same order as ``options``).
    """
    mask_tok = tok.mask_token
    ins = str(instructions).replace(mask_tok, " ")
    head_ids = tok("%s question: %s" % (qtype, ins), add_special_tokens=False)["input_ids"]
    opt_ids = []
    for o in options:
        opt_ids.append([tok.mask_token_id]
                       + tok(" " + o.replace(mask_tok, " "), add_special_tokens=False)["input_ids"][:OPTION_TOKEN_CAP])
    opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    if opt_budget < 16:  # too many / too long options: shrink every option text evenly
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]
        opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    head_ids = head_ids[:max(8, opt_budget)]
    ids = [tok.cls_token_id] + head_ids + [tok.sep_token_id]
    markers = []
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(tok.sep_token_id)
    room = max(0, max_len - len(ids) - 1)
    st = tok(str(context).replace(mask_tok, " "), add_special_tokens=False)["input_ids"]
    st = st[:room]
    ids = ids + st + [tok.sep_token_id]
    return ids[:max_len], [m for m in markers if m < max_len]


class DecisionModel(nn.Module):
    """Upstream Laya DecisionModel (same parameter names, so upstream state dicts load with strict=True)."""

    def __init__(self, encoder: nn.Module, head_layers: int = 2, n_act: int = 2, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        d = encoder.config.hidden_size
        nhead = max(1, d // 64)
        layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout, batch_first=True, norm_first=True)
        self.head = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False) if head_layers > 0 else None
        self.type_emb = nn.Embedding(3, d)
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.act_head = nn.Sequential(nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, n_act))  # unused by Krun One
        self.register_buffer("temperature", torch.ones(3))
        self.head_checkpointing = False

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype, with_act: bool = False):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = h + self.type_emb(qtype)[:, None, :]
        if self.head is not None:
            pad = ~attention_mask.bool()
            for layer in self.head.layers:
                if self.head_checkpointing and self.training and torch.is_grad_enabled():
                    h = torch.utils.checkpoint.checkpoint(layer, h, None, pad, use_reentrant=False)
                else:
                    h = layer(h, src_key_padding_mask=pad)
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)
        logits = self.scorer(m).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, -1e4)
        if not with_act:
            return logits
        p = torch.softmax(logits.detach(), -1)
        k = marker_mask.sum(-1).clamp(min=2).float()
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        top2 = p.topk(2, -1).values
        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
        act_logits = self.act_head(torch.cat([h[:, 0].float(), feats], -1))
        return logits, act_logits


def build_encoder(encoder_config_dir: str | os.PathLike):
    """Architecture only (weights come from the Krun/Laya state dict)."""
    from transformers import AutoConfig, AutoModel

    cfg = AutoConfig.from_pretrained(str(encoder_config_dir))
    cfg.reference_compile = False  # torch.compile is a loss for small inference batches
    return AutoModel.from_config(cfg, attn_implementation="sdpa")


def collate(items: Sequence[Mapping[str, Any]], pad_id: int) -> dict[str, torch.Tensor]:
    """items: {"ids": [...], "markers": [...], optional "target": [...]} -> padded batch tensors."""
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, :len(it["ids"])] = torch.tensor(it["ids"])
        att[i, :len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        if it.get("target") is not None:
            target[i, :k] = torch.tensor(it["target"], dtype=torch.float32)
    return {"input_ids": ids, "attention_mask": att, "marker_pos": mpos, "marker_mask": mmask, "target": target,
            "qtype": torch.tensor([int(it.get("qtype", 0)) for it in items], dtype=torch.long)}  # default: choice


def softmax_np(z: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64) / temperature
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def confidence_score(p: np.ndarray, name: str) -> float:
    """Abstention scores (higher = more confident). Same definitions as krunbench.metrics."""
    s = np.sort(p)[::-1]
    if name == "top1_probability":
        return float(s[0])
    if name == "top1_top2_margin":
        return float(s[0] - s[1]) if len(s) > 1 else float(s[0])
    if name == "neg_normalized_entropy":
        nz = p[p > 0]
        return float(1.0 - (-(nz * np.log(nz)).sum()) / math.log(max(len(p), 2)))
    raise ValueError(f"unknown confidence score {name!r}")


class KrunOne:
    """Self-contained inference wrapper: tokenizer + DecisionModel + calibration + abstention.

    Artifact layout (see export.py): model.safetensors, config.json (encoder architecture),
    tokenizer/, krun_config.json (sequence budgets, temperatures, abstention rule).
    """

    def __init__(self, model: DecisionModel, tokenizer, krun_config: Mapping[str, Any], device: torch.device):
        self.model, self.tok, self.cfg, self.device = model, tokenizer, dict(krun_config), device
        self.dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8) \
            else torch.float16

    @classmethod
    def from_pretrained(cls, path: str | os.PathLike, device: str | None = None, dtype: str | None = None) -> "KrunOne":
        from safetensors.torch import load_file
        from transformers import AutoTokenizer

        path = Path(path)
        kcfg = json.loads((path / "krun_config.json").read_text())
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        tok = AutoTokenizer.from_pretrained(str(path / "tokenizer"))
        model = DecisionModel(build_encoder(path), kcfg["head_layers"], kcfg.get("n_act", 2))
        model.load_state_dict(load_file(str(path / "model.safetensors")), strict=True)
        model.to(dev).eval()
        obj = cls(model, tok, kcfg, dev)
        if dtype:
            obj.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
        return obj

    # -- encoding ----------------------------------------------------------------------------
    @staticmethod
    def _normalize_options(options) -> tuple[list[str], dict[str, str | None]]:
        names, desc = [], {}
        for o in options:
            if isinstance(o, str):
                names.append(o)
            else:
                names.append(o["name"])
                desc[o["name"]] = o.get("description")
        if len(set(names)) != len(names):
            raise ValueError("option names must be unique")
        if len(names) < 2:
            raise ValueError("a decision needs at least 2 options")
        return names, desc

    def encode(self, context: str, names: Sequence[str], desc: Mapping[str, str | None], instructions: str | None):
        task = "tool" if any(desc.get(n) for n in names) else "intent"
        ins = instructions or DEFAULT_INSTRUCTIONS[task]
        ids, markers = build_sequence(self.tok, context, option_texts(names, desc), ins, self.cfg["max_len"],
                                      self.cfg["head_max_len"])
        if len(markers) != len(names):
            raise ValueError(f"{len(names)} options do not fit in head_max_len={self.cfg['head_max_len']}")
        return {"ids": ids, "markers": markers, "task": task, "qtype": QTYPES["choice"]}

    # -- inference ---------------------------------------------------------------------------
    @torch.no_grad()
    def logits(self, items: Sequence[Mapping[str, Any]]) -> list[np.ndarray]:
        b = collate(items, self.tok.pad_token_id)
        use_amp = self.device.type == "cuda" and self.dtype != torch.float32
        with torch.autocast(self.device.type, dtype=self.dtype, enabled=use_amp):
            z = self.model(b["input_ids"].to(self.device), b["attention_mask"].to(self.device),
                           b["marker_pos"].to(self.device), b["marker_mask"].to(self.device), b["qtype"].to(self.device))
        z = z.float().cpu().numpy()
        return [z[i, :len(it["markers"])] for i, it in enumerate(items)]

    def decide_batch(self, requests: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """requests: [{"context": str, "options": [str | {"name", "description"}], "instructions": optional}]"""
        enc, names_all = [], []
        for r in requests:
            names, desc = self._normalize_options(r["options"])
            enc.append(self.encode(r["context"], names, desc, r.get("instructions")))
            names_all.append(names)
        out = []
        ab = self.cfg["abstention"]
        for it, names, z in zip(enc, names_all, self.logits(enc)):
            T = self.cfg["temperature"].get(it["task"], self.cfg["temperature"].get("default", 1.0))
            p = softmax_np(z, T)
            conf = confidence_score(p, ab["score"])
            thr = ab["threshold"].get(it["task"], ab["threshold"].get("default")) if isinstance(ab["threshold"], dict) \
                else ab["threshold"]
            best = int(np.argmax(p))
            abstain = bool(conf < thr)
            out.append({
                "choice": None if abstain else names[best],
                "top_option": names[best],
                "abstain": abstain,
                "confidence": round(conf, 6),
                "confidence_score": ab["score"],
                "threshold": thr,
                "probabilities": {n: round(float(v), 6) for n, v in zip(names, p)},
                "task": it["task"],
            })
        return out

    def decide(self, context: str, options, instructions: str | None = None) -> dict[str, Any]:
        return self.decide_batch([{"context": context, "options": options, "instructions": instructions}])[0]

    # -- KRUN-011: typed multi-primitive requests ----------------------------------------------------------------
    @property
    def primitives(self) -> list[str]:
        return list(self.cfg.get("primitives", ["choice"]))

    def decide_questions(self, context: str, questions: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
        """One request, N heterogeneous questions, ONE forward pass (upstream Laya ``system_one`` semantics).

        questions: {id: {"type": "choice", "options": [str | {"name","description"}], "instructions"?}
                        | {"type": "noul", "instructions": str, "criteria"?: {"true","false"}}
                        | {"type": "score", "instructions": str, "levels": [str, ...]}}
        Choice answers keep the v0 shape (+ "type"); noul -> {"type","noul"}; score -> {"type","score","confidence",
        "legend","probabilities"}.
        """
        from .primitives import encode_question, noul_answer, score_answer, validate_question

        items, meta = [], []
        for qid, q in questions.items():
            t = q.get("type", "choice")
            if t not in self.primitives:
                raise ValueError(f"question {qid!r}: type {t!r} is not supported by this model ({self.primitives})")
            if t == "choice":
                names, desc = self._normalize_options(q["options"])
                it = self.encode(context, names, desc, q.get("instructions"))
                meta.append((qid, t, names, it["task"]))
            else:
                validate_question(q)
                it = encode_question(self.tok, context, q, self.cfg["max_len"], self.cfg["head_max_len"])
                meta.append((qid, t, q.get("levels"), None))
            items.append(it)
        out = {}
        ab = self.cfg["abstention"]
        for (qid, t, extra, task), z in zip(meta, self.logits(items)):
            if t == "choice":
                T = self.cfg["temperature"].get(task, self.cfg["temperature"].get("default", 1.0))
                p = softmax_np(z, T)
                conf = confidence_score(p, ab["score"])
                thr = ab["threshold"].get(task, ab["threshold"].get("default")) if isinstance(ab["threshold"], dict) \
                    else ab["threshold"]
                best = int(np.argmax(p))
                abstain = bool(conf < thr)
                out[qid] = {"type": "choice", "choice": None if abstain else extra[best], "top_option": extra[best],
                            "abstain": abstain, "confidence": round(conf, 6), "confidence_score": ab["score"],
                            "threshold": thr, "probabilities": {n: round(float(v), 6) for n, v in zip(extra, p)},
                            "task": task}
            elif t == "noul":
                out[qid] = noul_answer(softmax_np(z, self.cfg["temperature"]["noul"]))
            else:
                out[qid] = score_answer(softmax_np(z, self.cfg["temperature"]["score"]), extra,
                                        self.cfg.get("score_confidence", "normalized_entropy"))
        return out
