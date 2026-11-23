"""Krun One v1 text runtime: load a frozen artifact directory and answer decisions (the production answer contract).

Artifact layout (artifacts/krun-one-v1-text-final)::

    model.safetensors  spec.json  tokenizer/  calibration.json  decision_config.json  provenance.json  SHA256SUMS

Request (same shape as the krun-worker decision payload; ``multi`` is internal only, not in the public contract)::

    {"context": "...", "questions": {"q1": {"type": "choice", "options": [{"id": .., "description"?: ..}],
                                            "task_type"?: "intent"|"tool", "instructions"?: ..},
                                     "q2": {"type": "noul", "instructions": .., "criteria"?: {"true"?, "false"?}},
                                     "q3": {"type": "score", "instructions": .., "levels": [..]},
                                     "q4": {"type": "multi", "options": [..], "instructions"?: ..}}}

Answers follow the krun-worker (v0.3) answer schema field for field for choice / noul / score:
    choice: probabilities = softmax(z / T[family]); confidence = top1 - top2 margin; abstain if < threshold[family]
    noul:   noul = softmax(z / T_noul)[true]
    score:  score = sum(i * p_i), confidence = ordinal dispersion
    multi:  selected = sigmoid(apply) > multi_threshold (internal)
family = task_type if given, else "tool" when any option has a description, else "intent".
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .arch import ModelSpec
from .model import KrunOneV1Model, chunk_index, collate_chunks, merge_chunks
from .serialization import DEFAULT_INSTRUCTIONS, Budget, Serializer

REQUIRED = ("model.safetensors", "spec.json", "calibration.json", "decision_config.json", "tokenizer/tokenizer.json")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def softmax(z, T: float = 1.0) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64) / T
    e = np.exp(z - z.max())
    return e / e.sum()


def ordinal_dispersion(p: np.ndarray) -> float:
    k = len(p)
    idx = np.arange(k)
    mu = float((idx * p).sum())
    return float(1.0 - float(((idx - mu) ** 2 * p).sum()) / (((k - 1) / 2.0) ** 2))


class TextArtifact:
    def __init__(self, root: Path, model: KrunOneV1Model, tok, ser: Serializer, calib: dict, dcfg: dict,
                 device: torch.device, dtype: torch.dtype | None):
        self.root, self.model, self.tok, self.ser = root, model, tok, ser
        self.calib, self.dcfg, self.device, self.dtype = calib, dcfg, device, dtype
        self.model_id = dcfg["model_id"]

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, root: str | Path, device: str = "cpu", dtype: str = "fp32", verify: bool = True) -> "TextArtifact":
        from safetensors.torch import load_model
        from transformers import AutoTokenizer

        root = Path(root)
        for f in REQUIRED:
            if not (root / f).exists():
                raise FileNotFoundError(f"artifact incomplete: {f}")
        if verify:
            bad = [line for line in (root / "SHA256SUMS").read_text().splitlines()
                   if line and sha256_file(root / line.split(None, 1)[1].strip()) != line.split()[0]]
            if bad:
                raise ValueError(f"checksum mismatch: {bad[:3]}")
        spec = ModelSpec.from_dict(json.loads((root / "spec.json").read_text()))
        model = KrunOneV1Model.from_spec(spec, attn_implementation="sdpa")
        load_model(model, str(root / "model.safetensors"))
        dev = torch.device(device)
        model.to(dev).eval()
        dt = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[dtype]
        tok = AutoTokenizer.from_pretrained(str(root / "tokenizer"))
        dcfg = json.loads((root / "decision_config.json").read_text())
        sq = dcfg["sequence"]
        ser = Serializer(tok, Budget(sq["max_len"], sq["head_max_len"], max_candidates_per_chunk=sq["max_candidates_per_chunk"]))
        return cls(root, model, tok, ser, json.loads((root / "calibration.json").read_text()), dcfg, dev, dt)

    # ------------------------------------------------------------------ encode / forward
    def encode(self, context: str, q: Mapping[str, Any]):
        t = q.get("type", "choice")
        if t in ("choice", "multi"):
            opts = [{"id": o["id"], **({"description": o["description"]} if o.get("description") else {})} for o in q["options"]]
            fam = q.get("task_type") or ("tool" if any(o.get("description") for o in opts) else "intent")
            ins = q.get("instructions") or DEFAULT_INSTRUCTIONS["multi" if t == "multi" else fam]
            return self.ser.encode(t, context, opts, ins), fam
        if t == "noul":
            return self.ser.encode("noul", context, [{"id": "false"}, {"id": "true"}], q["instructions"], q.get("criteria")), "noul"
        if t == "score":
            return self.ser.encode("score", context, [{"id": str(i), "label": lv} for i, lv in enumerate(q["levels"])],
                                   q["instructions"]), "score"
        raise ValueError(f"unsupported question type {t!r}")

    @torch.inference_mode()
    def forward(self, encoded: list) -> dict[str, np.ndarray]:
        flat, where = chunk_index(encoded)
        b = {k: v.to(self.device) for k, v in collate_chunks(flat, self.tok.pad_token_id).items()}
        with torch.autocast(self.device.type, dtype=self.dtype or torch.float32, enabled=self.dtype is not None):
            out = self.model(**b)
        q = merge_chunks(out, where, len(encoded), [e.n_candidates for e in encoded])
        return {k: v.float().cpu().numpy() for k, v in q.items() if v.is_floating_point()}

    # ------------------------------------------------------------------ answers
    def answer(self, q: Mapping[str, Any], fam: str, rank: np.ndarray, apply: np.ndarray | None) -> dict[str, Any]:
        t = q.get("type", "choice")
        T = self.calib["temperature"][fam if t in ("choice", "multi") else t] if t != "multi" else 1.0
        p = softmax(rank, T)
        if t == "noul":
            return {"model": self.model_id, "type": "noul", "noul": round(float(p[1]), 6)}
        if t == "score":
            return {"model": self.model_id, "type": "score", "score": round(float((np.arange(len(p)) * p).sum()), 6),
                    "confidence": round(ordinal_dispersion(p), 6),
                    "legend": {str(i): lv for i, lv in enumerate(q["levels"])},
                    "probabilities": {str(i): round(float(v), 6) for i, v in enumerate(p)}}
        ids = [o["id"] for o in q["options"]]
        if t == "multi":
            s = 1 / (1 + np.exp(-np.asarray(apply, dtype=np.float64)))
            th = self.calib["multi_threshold"]
            return {"model": self.model_id, "type": "multi", "selected": [i for i, v in zip(ids, s) if v > th],
                    "scores": {i: round(float(v), 6) for i, v in zip(ids, s)}, "threshold": th, "internal": True}
        srt = np.sort(p)[::-1]
        margin = float(srt[0] - (srt[1] if len(srt) > 1 else 0.0))
        ent = float(-(p * np.log(np.clip(p, 1e-12, 1))).sum() / np.log(len(p))) if len(p) > 1 else 0.0
        thr = self.calib["margin_threshold"][fam]
        best = int(np.argmax(p))
        abstain = bool(margin < thr)
        return {"model": self.model_id, "type": "choice", "decision": None if abstain else ids[best],
                "top_option": ids[best], "probabilities": {i: round(float(v), 6) for i, v in zip(ids, p)},
                "confidence": round(margin, 6), "margin": round(margin, 6), "top1_probability": round(float(srt[0]), 6),
                "normalized_entropy": round(ent, 6), "abstain": abstain,
                "abstention_status": self.dcfg["abstention_status"][fam], "task_type": q.get("task_type") or fam,
                "option_count": len(ids)}

    def decide(self, request: Mapping[str, Any], return_raw: bool = False) -> dict[str, Any]:
        qids = list(request["questions"])
        enc = [self.encode(request["context"], request["questions"][k]) for k in qids]
        raw = self.forward([e for e, _ in enc])
        answers = {}
        for i, k in enumerate(qids):
            n = enc[i][0].n_candidates
            answers[k] = self.answer(request["questions"][k], enc[i][1], raw["rank"][i, :n],
                                     raw["apply"][i, :n] if "apply" in raw else None)
            if return_raw:
                answers[k]["_raw_rank"] = [float(x) for x in raw["rank"][i, :n]]
        return {"model": self.model_id, "answers": answers,
                "usage": {"input_tokens": int(sum(len(c.ids) for e, _ in enc for c in e.chunks))}}
