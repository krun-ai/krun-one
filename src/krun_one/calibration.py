"""Calibration: temperature scaling per task family (intent / tool), fitted on val_fit only.

The scalar-T fit is KRUN-004's (``bakeoff.calibration.fit_temperature``: LBFGS on log T, NLL of
answerable decisions). Krun One fits one T per family because tool decisions (long described options)
and intent decisions (short labels) have different logit scales; the family is decided exactly like
the serving API does it: options carry descriptions -> ``tool``.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from bakeoff.calibration import fit_temperature, softmax  # noqa: F401
from krunbench.schema import DecisionExample

FAMILIES = ("intent", "tool")


def family(e: DecisionExample) -> str:
    return "tool" if e.option_descriptions and any(e.option_descriptions.values()) else "intent"


def fit_family_temperatures(fit: Sequence[DecisionExample], logits: Mapping[str, Sequence[float]]) -> dict[str, float]:
    out = {}
    for fam in FAMILIES:
        ans = [e for e in fit if e.answer is not None and family(e) == fam]
        out[fam] = fit_temperature([np.asarray(logits[e.id]) for e in ans], [e.answer_index for e in ans]) if ans else 1.0
    return out


def probs(exs: Sequence[DecisionExample], logits: Mapping[str, Sequence[float]], temps: Mapping[str, float] | None) -> list[np.ndarray]:
    return [softmax(np.asarray(logits[e.id]), (temps or {}).get(family(e), 1.0)) for e in exs]


def reliability(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.ceil(conf * bins).astype(int) - 1, 0, bins - 1)
    return [{"bin": f"{edges[b]:.3f}-{edges[b + 1]:.3f}", "n": int((idx == b).sum()),
             "confidence": float(conf[idx == b].mean()) if (idx == b).any() else None,
             "accuracy": float(correct[idx == b].mean()) if (idx == b).any() else None} for b in range(bins)]
