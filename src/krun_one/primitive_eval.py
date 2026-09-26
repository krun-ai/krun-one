"""KRUN-011 evaluation for noul / score / mixed requests — identical code path for Laya and every Krun One checkpoint.

* Encoding: ``primitives.encode_question`` (upstream serialization, real ``qtype``); a question whose options do not
  fit is counted in ``unfit`` and scored as uniform (never silently dropped).
* Calibration: one temperature per primitive (``noul``, ``score``) fitted by NLL (LBFGS) on **val_fit**; every
  number reported on val is on **val_eval** (group split, ``bakeoff.data.split_validation`` seed). Test sets use the
  val_fit temperature of the same model. Raw (T = 1) numbers are reported too.
* Noul metrics on ``p_true``: ROC-AUC, PR-AUC (average precision), accuracy @ 0.5, NLL, Brier, binary ECE (15 equal
  bins of p_true vs outcome) + reliability curve.
* Score metrics: MAE / RMSE of the expected score, argmax accuracy, off-by-≤1 accuracy, NLL, multi-class Brier,
  top-label ECE, RPS (ranked probability score, normalized by K−1), Spearman / Kendall-τb of expected score vs label,
  distance-weighted error = E_p|i − y| (expected absolute level distance under the predicted distribution), and
  the three score-confidence candidates' Spearman with −|score − y|.
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from bakeoff.calibration import fit_temperature
from krunbench.splits import stable_fraction

from .model import QTYPES, collate
from .primitives import SCORE_CONFIDENCE, encode_question, render_options

VAL_FIT_SEED = 20260923  # = KRUN-004/005 val_fit seed
N_BINS = 15


def load_jsonl(p: Path) -> list[dict]:
    with open(p) as f:
        return [json.loads(l) for l in f]


def is_val_fit(r: Mapping[str, Any]) -> bool:
    return stable_fraction(VAL_FIT_SEED, "val-fit", r["group_id"]) < 0.5


def question_of(r: Mapping[str, Any]) -> dict[str, Any]:
    q = {"type": r["qtype"], "instructions": r.get("instructions") or ""}
    if r["qtype"] == "noul":
        q["criteria"] = r.get("criteria")
    elif r["qtype"] == "score":
        q["levels"] = r["levels"]
    else:
        q["options"], q["option_descriptions"] = r["options"], r.get("option_descriptions")
    return q


def encode_records(tok, records: Sequence[Mapping[str, Any]], max_len: int, head_max_len: int):
    items, unfit = [], []
    for r in records:
        try:
            it = encode_question(tok, r["context"], question_of(r), max_len, head_max_len)
        except ValueError:
            unfit.append(r["id"])
            continue
        it["id"] = r["id"]
        items.append(it)
    return items, unfit


@torch.no_grad()
def predict(model, tok, items: Sequence[Mapping[str, Any]], device: torch.device, dtype: torch.dtype,
            max_tokens: int = 32768, max_seqs: int = 256) -> dict[str, list[float]]:
    model.eval()
    order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
    out, i = {}, 0
    while i < len(order):
        j, L = i, 0
        while j < len(order) and j - i < max_seqs and max(L, len(items[order[j]]["ids"])) * (j - i + 1) <= max_tokens:
            L = max(L, len(items[order[j]]["ids"]))
            j += 1
        j = max(j, i + 1)
        sel = [items[order[t]] for t in range(i, j)]
        b = collate(sel, tok.pad_token_id)
        with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda" and dtype != torch.float32):
            z = model(b["input_ids"].to(device), b["attention_mask"].to(device), b["marker_pos"].to(device),
                      b["marker_mask"].to(device), b["qtype"].to(device))
        z = z.float().cpu().numpy()
        for r, it in enumerate(sel):
            out[it["id"]] = z[r, :len(it["markers"])].tolist()
        i = j
    return out


def softmax(z, T=1.0):
    z = np.asarray(z, dtype=np.float64) / T
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def probs_for(records, logits, T):
    out = []
    for r in records:
        k = 2 if r["qtype"] == "noul" else len(r["levels"]) if r["qtype"] == "score" else len(r["options"])
        z = logits.get(r["id"])
        out.append(np.full(k, 1.0 / k) if z is None else softmax(z, T))
    return out


def fit_T(records, logits) -> float:
    rs = [r for r in records if r["id"] in logits]
    return fit_temperature([np.asarray(logits[r["id"]]) for r in rs], [r["target"] for r in rs]) if rs else 1.0


# ------------------------------------------------------------------------------------------------- metrics
def reliability(conf, correct, bins=N_BINS):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    idx = np.clip(np.ceil(conf * bins).astype(int) - 1, 0, bins - 1)
    return [{"bin": f"{b / bins:.3f}-{(b + 1) / bins:.3f}", "n": int((idx == b).sum()),
             "mean_p": float(conf[idx == b].mean()) if (idx == b).any() else None,
             "freq": float(correct[idx == b].mean()) if (idx == b).any() else None} for b in range(bins)]


def ece(conf, correct, bins=N_BINS):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    if not len(conf):
        return float("nan")
    idx = np.clip(np.ceil(conf * bins).astype(int) - 1, 0, bins - 1)
    return float(sum((idx == b).mean() * abs(conf[idx == b].mean() - correct[idx == b].mean())
                     for b in range(bins) if (idx == b).any()))


def noul_metrics(p_true: np.ndarray, y: np.ndarray, with_curve: bool = False) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    p = np.clip(np.asarray(p_true, float), 1e-7, 1 - 1e-7)
    y = np.asarray(y, int)
    both = 0 < y.sum() < len(y)
    m = {"n": int(len(y)), "positive_rate": float(y.mean()) if len(y) else float("nan"),
         "roc_auc": float(roc_auc_score(y, p)) if both else float("nan"),
         "pr_auc": float(average_precision_score(y, p)) if both else float("nan"),
         "accuracy@0.5": float(((p >= 0.5) == y).mean()),
         "nll": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
         "brier": float(((p - y) ** 2).mean()),
         "ece": ece(p, y),
         "mean_p": float(p.mean())}
    if with_curve:
        m["reliability"] = reliability(p, y)
    return m


def score_metrics(P: Sequence[np.ndarray], y: Sequence[int], with_curve: bool = False) -> dict[str, Any]:
    from scipy.stats import kendalltau, spearmanr

    y = np.asarray(y, int)
    if not len(y):
        return {"n": 0}
    s = np.array([(np.arange(len(p)) * p).sum() for p in P])
    am = np.array([int(np.argmax(p)) for p in P])
    ptrue = np.array([max(p[t], 1e-12) for p, t in zip(P, y)])
    brier = np.array([((p - np.eye(len(p))[t]) ** 2).sum() for p, t in zip(P, y)])
    rps = np.array([((np.cumsum(p) - np.cumsum(np.eye(len(p))[t])) ** 2).sum() / (len(p) - 1) for p, t in zip(P, y)])
    dwe = np.array([(np.abs(np.arange(len(p)) - t) * p).sum() for p, t in zip(P, y)])
    top = np.array([p.max() for p in P])
    err = np.abs(s - y)
    const = np.std(s) == 0 or np.std(y) == 0
    m = {"n": int(len(y)), "mae": float(err.mean()), "rmse": float(np.sqrt(((s - y) ** 2).mean())),
         "accuracy": float((am == y).mean()), "within1_accuracy": float((np.abs(am - y) <= 1).mean()),
         "nll": float(-np.log(ptrue).mean()), "brier": float(brier.mean()), "ece": ece(top, am == y),
         "rps": float(rps.mean()), "distance_weighted_error": float(dwe.mean()),
         "spearman": float("nan") if const else float(spearmanr(s, y).statistic),
         "kendall_tau_b": float("nan") if const else float(kendalltau(s, y).statistic),
         "mean_score": float(s.mean()), "mean_label": float(y.mean()),
         "confidence_vs_error_spearman": {}}
    for name, fn in SCORE_CONFIDENCE.items():
        c = np.array([fn(p) for p in P])
        m["confidence_vs_error_spearman"][name] = float("nan") if np.std(c) == 0 else float(spearmanr(c, -err).statistic)
    if with_curve:
        m["reliability"] = reliability(top, am == y)
    return m


def slices(records, key_fn):
    d = defaultdict(list)
    for i, r in enumerate(records):
        for k in key_fn(r):
            d[k].append(i)
    return d


def noul_keys(r):
    md = r.get("metadata", {})
    ks = ["all", f"source:{r['source']}", f"suite:{r['suite']}"]
    if md.get("unseen_proposition"):  # held-out propositions never count as in-domain (KRUN-011 v0.3 data)
        ks += ["unseen_proposition", f"unseen_proposition:{r['source']}"]
    elif r["source"] != "boolq":
        ks.append("in_domain")
    ks.append("criteria:" + ("yes" if r.get("criteria") else "no"))
    if md.get("unseen_tool"):
        ks.append("unseen_tool")
    if md.get("language"):
        ks.append(f"lang:{md['language']}")
    if r["suite"].startswith("hard_negative") or r["suite"] == "positive":
        ks.append("positive_vs_hard_negative")
    if r["suite"] in ("cross_domain", "positive"):
        ks.append("positive_vs_cross_domain")
    return ks


def score_keys(r):
    ks = ["all", f"source:{r['source']}", f"attr:{r['source']}:{r.get('metadata', {}).get('attribute')}"]
    if r["source"] != "sst5":
        ks.append("in_domain")
    return ks


def report_split(records, logits, T, qtype):
    P = probs_for(records, logits, T)
    out = {}
    for k, idx in sorted(slices(records, noul_keys if qtype == "noul" else score_keys).items()):
        if qtype == "noul":
            out[k] = noul_metrics(np.array([P[i][1] for i in idx]), np.array([records[i]["target"] for i in idx]),
                                  with_curve=k == "all")
        else:
            out[k] = score_metrics([P[i] for i in idx], [records[i]["target"] for i in idx], with_curve=k == "all")
    return out


def evaluate_primitive(qtype: str, val: list[dict], test: list[dict] | None, logits: Mapping[str, list[float]]):
    fit = [r for r in val if is_val_fit(r)]
    ev = [r for r in val if not is_val_fit(r)]
    T = fit_T(fit, logits)
    rep = {"temperature": T, "n_val_fit": len(fit), "n_val_eval": len(ev),
           "val_eval": {"raw": report_split(ev, logits, 1.0, qtype), "calibrated": report_split(ev, logits, T, qtype)}}
    if test is not None:
        rep["test"] = {"raw": report_split(test, logits, 1.0, qtype), "calibrated": report_split(test, logits, T, qtype)}
    return rep


# ------------------------------------------------------------------------------------------------- mixed
def mixed_questions(req):
    out = []
    for qid, q in req["questions"].items():
        r = {"id": f"{req['id']}/{qid}", "qtype": q["type"], "context": req["context"],
             "instructions": q.get("instructions"), "target": q["target"], "group_id": req["id"]}
        if q["type"] == "score":
            r["levels"] = q["levels"]
        elif q["type"] == "choice":
            r["options"], r["option_descriptions"] = q["options"], q.get("option_descriptions")
        out.append(r)
    return out


@torch.no_grad()
def evaluate_mixed(model, tok, requests, max_len, head_max_len, device, dtype, temps: Mapping[str, float],
                   max_tokens: int = 32768):
    """Per request: all questions in ONE forward (as the worker does) vs each question alone (batch of 1).
    Reports per-type quality on the mixed forward and the invariance |Δp| / decision flips alone vs mixed."""
    alone, mixed, recs = {}, {}, []
    t_mixed = t_alone = 0.0
    for req in requests:
        qs = mixed_questions(req)
        items, unfit = encode_records(tok, qs, max_len, head_max_len)
        if unfit:
            continue
        recs += qs
        t0 = time.time()
        mixed.update(predict(model, tok, items, device, dtype, max_tokens=max_tokens))
        t_mixed += time.time() - t0
        t0 = time.time()
        for it in items:
            alone.update(predict(model, tok, [it], device, dtype))
        t_alone += time.time() - t0
    by_type = defaultdict(list)
    for r in recs:
        by_type[r["qtype"]].append(r)
    quality, inv = {}, {}
    for t, rs in by_type.items():
        T = temps.get(t, 1.0)
        Pm, Pa = probs_for(rs, mixed, T), probs_for(rs, alone, T)
        dp = np.array([np.abs(a - m).max() for a, m in zip(Pa, Pm)])
        flips = np.array([int(np.argmax(a)) != int(np.argmax(m)) for a, m in zip(Pa, Pm)])
        inv[t] = {"n": len(rs), "max_abs_dp": float(dp.max()), "mean_abs_dp": float(dp.mean()),
                  "p99_abs_dp": float(np.quantile(dp, 0.99)), "decision_flip_rate": float(flips.mean())}
        y = [r["target"] for r in rs]
        if t == "noul":
            quality[t] = noul_metrics(np.array([p[1] for p in Pm]), np.array(y))
        elif t == "score":
            quality[t] = score_metrics(Pm, y)
            for attr in sorted({r["id"].rsplit("/", 1)[1] for r in rs}):
                sel = [i for i, r in enumerate(rs) if r["id"].endswith("/" + attr)]
                quality[f"score:{attr}"] = score_metrics([Pm[i] for i in sel], [y[i] for i in sel])
        else:
            am = np.array([int(np.argmax(p)) for p in Pm])
            quality[t] = {"n": len(rs), "accuracy": float((am == np.array(y)).mean()),
                          "ece": ece([p.max() for p in Pm], am == np.array(y))}
    all_ok = [all(int(np.argmax(probs_for([r], mixed, temps.get(r["qtype"], 1.0))[0])) == r["target"]
                  if r["qtype"] != "noul" else (probs_for([r], mixed, temps.get("noul", 1.0))[0][1] >= 0.5) == bool(r["target"])
                  for r in qs_) for qs_ in _group(recs)]
    return {"n_requests": len(_group(recs)), "quality": quality, "invariance_alone_vs_mixed": inv,
            "all_questions_correct_rate": float(np.mean(all_ok)) if all_ok else float("nan"),
            "seconds_mixed_forward": round(t_mixed, 2), "seconds_one_forward_per_question": round(t_alone, 2)}


def _group(recs):
    g = defaultdict(list)
    for r in recs:
        g[r["group_id"]].append(r)
    return list(g.values())


__all__ = ["QTYPES", "render_options", "load_jsonl", "encode_records", "predict", "evaluate_primitive", "evaluate_mixed",
           "noul_metrics", "score_metrics", "is_val_fit"]
