"""Abstention from the answer distribution (no act head): top-1 probability, top1-top2 margin,
1 - normalized entropy. Thresholds are chosen on val_fit (in-distribution answerable decisions,
per task family) for a target selective accuracy and applied unchanged to every reported set.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np
from bakeoff.calibration import confidence_scores  # top1_probability, top1_top2_margin, neg_normalized_entropy
from bakeoff.evaluation import choose_threshold, curve
from krunbench import metrics as M
from krunbench.schema import DecisionExample
from sklearn.metrics import roc_auc_score

SCORES = ("top1_probability", "top1_top2_margin", "neg_normalized_entropy")


def correct_flags(exs: Sequence[DecisionExample], ps: Sequence[np.ndarray]) -> np.ndarray:
    return np.array([e.answer is not None and int(np.argmax(p)) == e.answer_index for e, p in zip(exs, ps)], dtype=bool)


def thresholds(fit_id: Sequence[DecisionExample], fit_probs: Sequence[np.ndarray], targets: Sequence[float]) -> dict:
    """{score: {target: threshold}} from val_fit in-distribution answerable decisions."""
    s = confidence_scores(fit_probs)
    c = correct_flags(fit_id, fit_probs)
    return {name: {f"{t:.2f}": choose_threshold(s[name], c, t) for t in targets} for name in SCORES}


def evaluate_abstention(id_exs, id_probs, ood_exs, ood_probs, thr: Mapping[str, Mapping[str, float]],
                        grid: Sequence[float]) -> dict[str, Any]:
    s_id, s_ood = confidence_scores(id_probs), confidence_scores(ood_probs) if ood_exs else {}
    c = correct_flags(id_exs, id_probs)
    out: dict[str, Any] = {"n_id": len(id_exs), "n_ood": len(ood_exs)}
    for name in SCORES:
        e: dict[str, Any] = {"aurc": M.aurc(s_id[name], c) if len(c) else float("nan"),
                             "curve": curve(s_id[name], c, grid) if len(c) else []}
        if ood_exs and len(id_exs):
            y = np.r_[np.ones(len(id_exs)), np.zeros(len(ood_exs))]
            e["ood_auroc"] = float(roc_auc_score(y, np.r_[s_id[name], s_ood[name]]))
            kinds = defaultdict(list)
            for ex, v in zip(ood_exs, s_ood[name]):
                kinds[ex.metadata.get("ood_kind") or ex.suite].append(v)
            e["ood_auroc_by_kind"] = {k: float(roc_auc_score(np.r_[np.ones(len(id_exs)), np.zeros(len(v))], np.r_[s_id[name], v]))
                                      for k, v in sorted(kinds.items())}
        ops = {}
        for tgt, t in thr.get(name, {}).items():
            acc_id = s_id[name] >= t
            op = {"threshold": t, "coverage": M.coverage(acc_id) if len(c) else float("nan"),
                  "selective_accuracy": M.selective_accuracy(acc_id, c) if len(c) else float("nan")}
            if ood_exs:
                acc_ood = s_ood[name] >= t
                op["ood_false_acceptance"] = float(acc_ood.mean())
                by = defaultdict(list)
                for ex, a in zip(ood_exs, acc_ood):
                    by[ex.metadata.get("ood_kind") or ex.suite].append(a)
                op["ood_false_acceptance_by_kind"] = {k: float(np.mean(v)) for k, v in sorted(by.items())}
                dec = np.r_[acc_id & c, ~acc_ood]
                op["decision_accuracy"] = float(dec.mean())
            ops[tgt] = op
        e["operating_points"] = ops
        out[name] = e
    return out
