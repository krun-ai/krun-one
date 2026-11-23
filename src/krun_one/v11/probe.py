"""Technical Capacity Probe (v1.1 R0): conditions and metrics over envs.py problems.

Conditions (the same problem, the same candidate order):
    raw    state + candidate actions                      -> decision
    feat   state + candidate actions + explicit outcomes  -> decision   ("feature-assisted")
Pairwise items are 2-candidate versions of the k=8 problems (gold vs its hardest negative, gold vs a random one).

Rendering goes through the public request shape: raw = options with ids only, feat = options with id + description,
``task_type`` fixed to "tool" in both so the calibration family (temperature, abstention threshold) does not move.

Metrics are computed offline from dumped rank logits (``signals``), so they can be re-derived without a GPU.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

CONDITIONS = ("raw", "feat")


def options(problem: Mapping[str, Any], cond: str) -> list[dict]:
    if cond == "raw":
        return [{"id": c["name"]} for c in problem["candidates"]]
    if cond == "feat":
        return [{"id": c["name"], "description": c["feat"]} for c in problem["candidates"]]
    raise ValueError(cond)


def pairwise(problem: Mapping[str, Any], rng: random.Random) -> list[dict]:
    """Two 2-candidate problems from an answerable problem: gold vs hardest negative, gold vs a random negative."""
    g = problem["gold"]
    neg = [i for i in range(problem["k"]) if i != g]
    val = lambda i: -1.0 if problem["candidates"][i]["value"] is None else problem["candidates"][i]["value"]  # noqa: E731
    hardest = max(neg, key=val)
    out = []
    for kind, j in (("hardest", hardest), ("random", rng.choice(neg))):
        order = [g, j] if rng.random() < 0.5 else [j, g]
        out.append({**problem, "id": f"{problem['id']}/pair-{kind}", "k": 2, "candidates": [problem["candidates"][i] for i in order],
                    "gold": order.index(g), "pair_kind": kind, "parent": problem["id"], "flip": None})
    return out


# ------------------------------------------------------------------------------------------------ per-item signals
def softmax(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64) / T
    e = np.exp(z - z.max())
    return e / e.sum()


def item_signals(problem: Mapping[str, Any], z: Sequence[float], T: float, margin_threshold: float) -> dict[str, Any]:
    """Everything the probe reports for one (problem, condition), from its K rank logits."""
    z = np.asarray(z, dtype=np.float64)
    k = len(z)
    p1, pT = softmax(z), softmax(z, T)
    srt = np.sort(pT)[::-1]
    margin = float(srt[0] - srt[1])
    pred = int(np.argmax(z))
    row = {"id": problem["id"], "k": k, "pred": pred, "top1_p": float(p1.max()), "top1_pT": float(srt[0]),
           "entropy": float(-(p1 * np.log(np.clip(p1, 1e-12, 1))).sum() / math.log(k)), "margin_T": margin,
           "abstain": bool(margin < margin_threshold), "answerable": bool(problem["answerable"])}
    if not problem["answerable"]:
        return row
    g = problem["gold"]
    vals = np.array([-1.0 if c["value"] is None else c["value"] for c in problem["candidates"]])
    neg = [i for i in range(k) if i != g]
    hardest = max(neg, key=lambda i: vals[i])
    rank = int((z > z[g]).sum()) + 1                       # 1 = gold on top (ties broken against the model)
    rel = (vals + 1.0) / (vals.max() + 1.0)                # infeasible = 0, best = 1
    order = np.argsort(-z, kind="stable")
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    ndcg = float((rel[order] * disc).sum() / (np.sort(rel)[::-1] * disc).sum())
    a, b = np.triu_indices(k, 1)
    diff = vals[a] != vals[b]
    conc = ((z[a] - z[b]) * (vals[a] - vals[b]) > 0)[diff]
    row.update({"correct": pred == g, "rank": rank, "mrr": 1.0 / rank, "ndcg": ndcg, "gold_p": float(p1[g]),
                "gold_pT": float(pT[g]), "pair_all": float(conc.mean()) if diff.any() else None,
                "gold_vs_hardest": bool(z[g] > z[hardest]), "margin_logit": float(z[g] - z[hardest]),
                "margin_p": float(p1[g] - p1[hardest]), "pred_name": problem["candidates"][pred]["name"],
                "gold_name": problem["candidates"][g]["name"]})
    return row


# ------------------------------------------------------------------------------------------------ aggregation
def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)


def auroc(pos: np.ndarray, neg: np.ndarray) -> float | None:
    if len(pos) == 0 or len(neg) == 0:
        return None
    s = np.concatenate([pos, neg])
    order = s.argsort(kind="stable")
    r = np.empty(len(s))
    r[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):                                   # average ranks on ties
        m = s == v
        r[m] = r[m].mean()
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def boot_ci(x: np.ndarray, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, len(x))).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def aggregate(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Metrics of one cell (answerable items of one slice x condition x k)."""
    rows = [r for r in rows if r["answerable"]]
    if not rows:
        return {"n": 0}
    k = float(np.mean([r["k"] for r in rows]))
    acc = np.array([r["correct"] for r in rows], dtype=float)
    f = lambda key: float(np.mean([r[key] for r in rows if r.get(key) is not None]))  # noqa: E731
    chance = float(np.mean([1.0 / r["k"] for r in rows]))
    lo, hi = boot_ci(acc)
    kept = np.array([not r["abstain"] for r in rows])
    return {"n": len(rows), "k": k, "top1": float(acc.mean()), "top1_ci95": [round(lo, 4), round(hi, 4)], "chance": chance,
            "skill": float((acc.mean() - chance) / (1 - chance)), "mrr": f("mrr"), "ndcg": f("ndcg"),
            "pair_all": f("pair_all"), "gold_vs_hardest": f("gold_vs_hardest"), "gold_p": f("gold_p"),
            "gold_p_over_uniform": f("gold_p") / chance, "top1_p": f("top1_p"), "entropy": f("entropy"),
            "margin_logit": f("margin_logit"), "margin_p": f("margin_p"),
            "ece_T": ece(np.array([r["top1_pT"] for r in rows]), acc), "abstain_rate": float(1 - kept.mean()),
            "selective_acc": float(acc[kept].mean()) if kept.any() else None}


def group(rows: Sequence[Mapping[str, Any]], problems: Mapping[str, Mapping[str, Any]], keys: Sequence[str]) -> dict:
    cells = defaultdict(list)
    for r in rows:
        p = problems[r["id"]]
        cells[tuple(str(p.get(k_, r.get(k_))) for k_ in keys)].append(r)
    return {"/".join(k_): aggregate(v) for k_, v in sorted(cells.items())}


def flip_metrics(rows: Mapping[str, Mapping[str, Any]], problems: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Constraint-flip pairs: base -> X, base + one constraint -> Y (same candidates, same order)."""
    by = defaultdict(lambda: defaultdict(dict))
    for pid, p in problems.items():
        if p.get("flip") and pid in rows:
            by[p["slice"]][p["flip"]["pair"]][p["flip"]["role"]] = (p, rows[pid])
    out = {}
    for sl, pairs in sorted(by.items()):
        b_ok = f_ok = both = stuck = n = 0
        for pr in pairs.values():
            if "base" not in pr or "flip" not in pr:
                continue
            (pb, rb), (pf, rf) = pr["base"], pr["flip"]
            n += 1
            b_ok += rb["correct"]
            f_ok += rf["correct"]
            both += rb["correct"] and rf["correct"]
            stuck += rf["pred_name"] == pb["candidates"][pb["gold"]]["name"]   # still answers the base winner
        if n:
            out[sl] = {"pairs": n, "base_acc": b_ok / n, "flip_acc": f_ok / n, "both_correct": both / n,
                       "stuck_on_base_answer": stuck / n, "flip_sensitivity": both / max(1, b_ok)}
    return out


def ood_metrics(rows: Sequence[Mapping[str, Any]], problems: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """None-of-the-above: false acceptance at the production threshold, margin AUROC vs answerable k=8 items."""
    out = {}
    for sl in sorted({problems[r["id"]]["slice"] for r in rows}):
        una = [r for r in rows if problems[r["id"]]["slice"] == sl and not r["answerable"]]
        ans = [r for r in rows if problems[r["id"]]["slice"] == sl and r["answerable"] and r["k"] == 8
               and not problems[r["id"]].get("flip") and "parent" not in problems[r["id"]]]
        if una and ans:
            out[sl] = {"n_unanswerable": len(una), "false_acceptance": float(np.mean([not r["abstain"] for r in una])),
                       "answerable_coverage": float(np.mean([not r["abstain"] for r in ans])),
                       "auroc_margin": auroc(np.array([r["margin_T"] for r in ans]), np.array([r["margin_T"] for r in una]))}
    return out
