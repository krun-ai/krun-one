"""Decision primitives (KRUN-011): choice, noul, score — upstream-Laya serialization and typed answers.

Serialization is upstream ``rl_common.render_options`` / ``build_sequence`` (see
results/primitives_architecture_audit.md):

* choice  options in request order; described options render as ``"name: description"``, bare labels humanized
          (Krun One v0 rendering, unchanged)
* noul    always two options, **false first**: ``"false: {criteria.false | default}"``, ``"true: {criteria.true | default}"``;
          ``noul = p[1]``
* score   ``"level {i}: {level_i}"`` in the given order (order is semantic: never permuted)

The header is ``"{type} question: {instructions}"`` and ``qtype`` selects the type-embedding row
(``choice`` 0, ``score`` 1, ``noul`` 2), exactly as upstream.

Answers (temperature-scaled probabilities ``p`` over the rendered options):

* noul   ``{"type": "noul", "noul": p_true}``
* score  ``{"type": "score", "score": Σ i·p_i, "confidence": 1 − H(p)/ln K, "legend", "probabilities"}``
         (confidence definition chosen in results/primitives_baseline/score_confidence.md)
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .model import QTYPES, build_sequence, option_texts

PRIMITIVES = ("choice", "noul", "score")
NOUL_DEFAULT_FALSE = "no, the statement does not hold"  # upstream defaults (rl_common.render_options)
NOUL_DEFAULT_TRUE = "yes, the statement holds"
MIN_LEVELS, MAX_LEVELS = 2, 16


def render_options(q: Mapping[str, Any]) -> list[str]:
    t = q["type"]
    if t == "choice":
        return option_texts(q["options"], q.get("option_descriptions"))
    if t == "score":
        return ["level %d: %s" % (i, c) for i, c in enumerate(q["levels"])]
    if t == "noul":
        crit = q.get("criteria") or {}
        return ["false: " + (crit.get("false") or NOUL_DEFAULT_FALSE), "true: " + (crit.get("true") or NOUL_DEFAULT_TRUE)]
    raise ValueError(f"unknown question type {t!r}")


def validate_question(q: Mapping[str, Any]) -> None:
    t = q.get("type")
    if t not in PRIMITIVES:
        raise ValueError(f"unknown question type {t!r}")
    if t == "choice":
        if len(q["options"]) < 2 or len(set(q["options"])) != len(q["options"]):
            raise ValueError("choice needs >= 2 unique options")
        return
    if not str(q.get("instructions") or "").strip():
        raise ValueError(f"{t} question needs non-empty instructions")
    if t == "noul":
        crit = q.get("criteria")
        if crit is not None:
            if set(crit) - {"true", "false"}:
                raise ValueError("noul criteria keys must be 'true' / 'false'")
            if any(v is not None and not str(v).strip() for v in crit.values()):
                raise ValueError("noul criteria must be non-empty when present")
    if t == "score":
        lv = q.get("levels") or []
        if not MIN_LEVELS <= len(lv) <= MAX_LEVELS:
            raise ValueError(f"score needs {MIN_LEVELS}..{MAX_LEVELS} levels")
        if any(not str(x).strip() for x in lv):
            raise ValueError("score levels must be non-empty")


def encode_question(tok, context: str, q: Mapping[str, Any], max_len: int, head_max_len: int) -> dict[str, Any]:
    """-> {"ids", "markers", "qtype"}; raises if the options do not fit (never silently truncates the answer space)."""
    opts = render_options(q)
    ids, markers = build_sequence(tok, context, opts, q.get("instructions") or "", max_len, head_max_len, qtype=q["type"])
    if len(markers) != len(opts):
        raise ValueError(f"{len(opts)} options do not fit in head_max_len={head_max_len}")
    return {"ids": ids, "markers": markers, "qtype": QTYPES[q["type"]]}


# ------------------------------------------------------------------------------------------------ answers
def normalized_entropy_confidence(p: np.ndarray) -> float:
    k = len(p)
    if k < 2:
        return 1.0
    nz = p[p > 0]
    return float(1.0 - (-(nz * np.log(nz)).sum()) / math.log(k))


def ordinal_dispersion_confidence(p: np.ndarray) -> float:
    k = len(p)
    idx = np.arange(k)
    mu = float((idx * p).sum())
    var = float(((idx - mu) ** 2 * p).sum())
    return float(1.0 - var / (((k - 1) / 2.0) ** 2))


SCORE_CONFIDENCE = {
    "normalized_entropy": normalized_entropy_confidence,
    "ordinal_dispersion": ordinal_dispersion_confidence,
    "top1_probability": lambda p: float(np.max(p)),
}


def expected_score(p: np.ndarray) -> float:
    return float((np.arange(len(p)) * p).sum())


def noul_answer(p: np.ndarray) -> dict[str, Any]:
    return {"type": "noul", "noul": round(float(p[1]), 6)}


def score_answer(p: np.ndarray, levels: Sequence[str], confidence: str = "normalized_entropy") -> dict[str, Any]:
    return {"type": "score", "score": round(expected_score(p), 6),
            "confidence": round(SCORE_CONFIDENCE[confidence](p), 6),
            "legend": {str(i): lv for i, lv in enumerate(levels)},
            "probabilities": {str(i): round(float(v), 6) for i, v in enumerate(p)}}
