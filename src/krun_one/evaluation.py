"""Evaluation protocol shared by the Laya baseline and Krun One (identical examples, options, serialization,
metrics).

Sets
    kb_val         KrunBench v0 validation, verbatim (intents; HWU64 = secondary slice)      [validation]
    val_hard       KrunBench validation items with hard_v1 option sets, k = 4..32 (+OOD)       [validation]
    val_tools      tool routing validation (seen + unseen tools, OOD, permutations)           [validation]
    holdout_tools  same construction, disjoint queries/tools                                   [final, once]
    kb_test        KrunBench v0 test, verbatim, incl. BFCL (external tool benchmark)           [final, once]
Validation sets are split by group into val_fit (temperatures, thresholds) and val_eval (reported numbers),
with KRUN-004's seed, so KrunBench validation's split is identical to KRUN-004's.

Every model is run through the same ``build_sequence`` with the same (max_len, head_max_len); a decision
whose options do not fit is scored as uniform (counted in ``unfit``), never silently dropped.
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from bakeoff.data import split_validation
from krunbench import metrics as M
from krunbench.schema import DecisionExample, read_jsonl

from .abstention import SCORES, correct_flags, evaluate_abstention, thresholds
from .calibration import family, fit_family_temperatures, probs, reliability
from .model import build_sequence, collate, humanize_label, option_texts

VALIDATION_SETS = ("kb_val", "val_hard", "val_tools")
FINAL_SETS = ("holdout_tools", "kb_test")
OOD_SUITES = ("ood_label_removed", "ood_cross_dataset", "ood_native")
KB_DATASETS = ("banking77", "massive", "hwu64", "bfcl")


def load_set(name: str, project_root: Path, data_dir: Path) -> list[DecisionExample]:
    kb = project_root / "benchmarks" / "krunbench" / "data" / "v0"
    if name in ("kb_val", "kb_test"):
        split = "validation" if name == "kb_val" else "test"
        out = []
        for ds in KB_DATASETS:
            p = kb / ds / f"{split}.jsonl"
            if p.exists():
                out += read_jsonl(p)
        return out
    return read_jsonl(data_dir / f"{name}.jsonl")


# ------------------------------------------------------------------------------------------ inference
@torch.no_grad()
def predict_logits(model, tok, exs: Sequence[DecisionExample], max_len: int, head_max_len: int, device: torch.device,
                   dtype: torch.dtype = torch.bfloat16, max_tokens: int = 32768, max_seqs: int = 256,
                   progress: str = "") -> tuple[dict[str, list[float]], list[str]]:
    model.eval()
    enc, unfit = [], []
    for e in exs:
        ids, markers = build_sequence(tok, e.context, option_texts(e.options, e.option_descriptions), e.instructions or "",
                                      max_len, head_max_len)
        if len(markers) != len(e.options):
            unfit.append(e.id)
            continue
        enc.append({"id": e.id, "ids": ids, "markers": markers})
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]["ids"]))
    out: dict[str, list[float]] = {}
    t0, i = time.time(), 0
    while i < len(order):
        j, L = i, 0
        while j < len(order) and j - i < max_seqs and max(L, len(enc[order[j]]["ids"])) * (j - i + 1) <= max_tokens:
            L = max(L, len(enc[order[j]]["ids"]))
            j += 1
        j = max(j, i + 1)
        sel = [enc[order[t]] for t in range(i, j)]
        b = collate(sel, tok.pad_token_id)
        with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
            z = model(b["input_ids"].to(device), b["attention_mask"].to(device), b["marker_pos"].to(device),
                      b["marker_mask"].to(device), b["qtype"].to(device))
        z = z.float().cpu().numpy()
        for r, it in enumerate(sel):
            out[it["id"]] = z[r, :len(it["markers"])].tolist()
        i = j
    for e in exs:  # unfit decisions: uniform (never dropped)
        if e.id in unfit:
            out[e.id] = [0.0] * len(e.options)
    if progress:
        print(f"  [{progress}] {len(exs)} decisions in {time.time() - t0:.0f}s (unfit {len(unfit)})", flush=True)
    return out, unfit


# ------------------------------------------------------------------------------------------ slices
def suite_key(e: DecisionExample) -> str:
    d = e.provenance.dataset
    return f"massive/{e.language}" if d == "massive" else d


def _words(s: str) -> set[str]:
    return {w for w in humanize_label(s).split() if len(w) > 2}


def tool_slices(e: DecisionExample) -> list[str]:
    """BFCL-style difficulty slices for an answerable tool decision."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    out = []
    k = len(e.options)
    out.append("toolset_large(k>=16)" if k >= 16 else "toolset_medium(5-15)" if k >= 5 else "toolset_small(<=4)")
    docs = [f"{o} {(e.option_descriptions or {}).get(o, '')}" for o in e.options]
    gi = e.answer_index
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit(docs)
    m = v.transform(docs)
    sims = (m @ m[gi].T).toarray().ravel()
    sims[gi] = 0
    out.append("near_duplicate_distractor" if sims.max() >= 0.5 else "no_near_duplicate")
    name_words = _words(e.answer)
    out.append("description_dependent" if not (name_words & _words(e.context)) else "name_matches_query")
    if "tool_seen" in e.metadata:
        out.append("seen_tool" if e.metadata["tool_seen"] else "unseen_tool")
    return out


# ------------------------------------------------------------------------------------------ metrics
def quality(exs: Sequence[DecisionExample], ps: Sequence[np.ndarray]) -> dict[str, Any]:
    ans = [(e, p) for e, p in zip(exs, ps) if e.answer is not None]
    if not ans:
        return {"n": 0}
    ex, pr = zip(*ans)
    gold = [e.answer_index for e in ex]
    pred = [int(np.argmax(p)) for p in pr]
    top1 = M.top1_probability(pr)
    corr = np.array([a == b for a, b in zip(pred, gold)])
    return {"n": len(ex), "accuracy": float(corr.mean()),
            "macro_f1": M.macro_f1([e.options[i] for e, i in zip(ex, pred)], [e.answer for e in ex]),
            "nll": M.nll(pr, gold), "brier": M.brier(pr, gold), "ece": M.ece(top1, corr),
            "mean_top1_probability": float(top1.mean())}


def permutation_consistency(exs: Sequence[DecisionExample], logits: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    perm = [e for e in exs if e.suite == "permutation" and e.generation.consistency_group]
    if not perm:
        return {"groups": 0}
    groups = [e.generation.consistency_group for e in perm]
    pred = [e.options[int(np.argmax(logits[e.id]))] for e in perm]
    acc = [p == e.answer for p, e in zip(pred, perm) if e.answer is not None]
    return {"groups": len(set(groups)), "consistency": M.permutation_consistency(groups, pred),
            "accuracy": float(np.mean(acc)) if acc else float("nan")}


def by(exs: Sequence[DecisionExample], key: Callable[[DecisionExample], Any]) -> dict[str, list[int]]:
    g = defaultdict(list)
    for i, e in enumerate(exs):
        for k in (key(e) if isinstance(key(e), list) else [key(e)]):
            g[str(k)].append(i)
    return dict(sorted(g.items()))


def set_report(name: str, exs: Sequence[DecisionExample], logits: Mapping[str, Sequence[float]],
               temps: Mapping[str, float], thr_by_family: Mapping[str, Mapping], grid: Sequence[float]) -> dict[str, Any]:
    ps_raw = probs(exs, logits, None)
    ps = probs(exs, logits, temps)
    core = [i for i, e in enumerate(exs) if e.suite == "core" and e.answer is not None]
    rep: dict[str, Any] = {"n": len(exs)}
    sub = lambda idx, P: ([exs[i] for i in idx], [P[i] for i in idx])  # noqa: E731
    rep["quality_raw"] = {"core": quality(*sub(core, ps_raw))}
    q = {"core": quality(*sub(core, ps))}
    cex = [exs[i] for i in core]
    for label, fn in (("family", family), ("dataset", suite_key), ("option_count", lambda e: e.generation.option_count),
                      ("negatives", lambda e: e.metadata.get("negatives", e.generation.negative_strategy)),
                      ("language", lambda e: e.language)):
        q[f"by_{label}"] = {k: quality(*sub([core[i] for i in idx], ps)) for k, idx in by(cex, fn).items()}
    tool_core = [i for i in core if family(exs[i]) == "tool"]
    if tool_core:
        tex = [exs[i] for i in tool_core]
        q["tool_slices"] = {k: quality(*sub([tool_core[i] for i in idx], ps)) for k, idx in by(tex, tool_slices).items()}
    dist = [i for i, e in enumerate(exs) if e.suite == "distractors" and e.answer is not None]
    if dist:
        dex = [exs[i] for i in dist]
        q["distractors_nested"] = {k: quality(*sub([dist[i] for i in idx], ps))
                                   for k, idx in by(dex, lambda e: f"k{e.generation.option_count}-{e.generation.negative_strategy}").items()}
    rep["quality"] = q
    ece_core = [exs[i] for i in core]
    if ece_core:
        pc = [ps[i] for i in core]
        rep["reliability"] = reliability(M.top1_probability(pc), correct_flags(ece_core, pc))
    rep["permutation"] = permutation_consistency(exs, logits)
    # abstention per family: ID = answerable core, OOD = unanswerable suites
    rep["abstention"] = {}
    for fam in ("intent", "tool", "all"):
        id_idx = [i for i in core if fam == "all" or family(exs[i]) == fam]
        # OOD family = family of the *options* shown (cross-task OOD with tool options counts as tool)
        ood_idx = [i for i, e in enumerate(exs) if e.suite in OOD_SUITES and (fam == "all" or family(e) == fam)]
        if not id_idx and not ood_idx:
            continue
        if fam == "all":
            # thresholds are per family: evaluate with each decision's own family threshold
            thr = {s: {} for s in SCORES}
            rep["abstention"]["all"] = _mixed_family(exs, ps, id_idx, ood_idx, thr_by_family, grid)
            continue
        rep["abstention"][fam] = evaluate_abstention([exs[i] for i in id_idx], [ps[i] for i in id_idx],
                                                     [exs[i] for i in ood_idx], [ps[i] for i in ood_idx],
                                                     thr_by_family.get(fam, {}), grid)
    return rep


def _mixed_family(exs, ps, id_idx, ood_idx, thr_by_family, grid) -> dict[str, Any]:
    """AUROC/AURC over both families at once, and operating points with per-family thresholds."""
    from bakeoff.calibration import confidence_scores
    from sklearn.metrics import roc_auc_score

    out: dict[str, Any] = {"n_id": len(id_idx), "n_ood": len(ood_idx)}
    all_idx = id_idx + ood_idx
    s = confidence_scores([ps[i] for i in all_idx])
    c = correct_flags([exs[i] for i in id_idx], [ps[i] for i in id_idx])
    fams = [family(exs[i]) for i in all_idx]
    for name in SCORES:
        sid, sood = s[name][: len(id_idx)], s[name][len(id_idx):]
        e = {"aurc": M.aurc(sid, c) if len(c) else float("nan")}
        if len(ood_idx) and len(id_idx):
            e["ood_auroc"] = float(roc_auc_score(np.r_[np.ones(len(id_idx)), np.zeros(len(ood_idx))], s[name]))
        ops = {}
        targets = sorted({t for f in thr_by_family.values() for t in f.get(name, {})})
        for tgt in targets:
            th = np.array([thr_by_family.get(f, {}).get(name, {}).get(tgt, np.inf) for f in fams])
            acc = s[name] >= th
            aid, aood = acc[: len(id_idx)], acc[len(id_idx):]
            op = {"coverage": M.coverage(aid), "selective_accuracy": M.selective_accuracy(aid, c)}
            if len(ood_idx):
                op["ood_false_acceptance"] = float(aood.mean())
                op["decision_accuracy"] = float(np.r_[aid & c, ~aood].mean())
            ops[tgt] = op
        e["operating_points"] = ops
        out[name] = e
    return out


def full_report(sets: Mapping[str, Sequence[DecisionExample]], logits: Mapping[str, Mapping[str, Sequence[float]]],
                eval_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Fit temperatures/thresholds on the val_fit halves of the validation sets; report val_eval halves + final sets."""
    fit, reports = [], {}
    split_sets: dict[str, list[DecisionExample]] = {}
    all_logits: dict[str, Sequence[float]] = {}
    for name, exs in sets.items():
        all_logits.update(logits[name])
        if name in VALIDATION_SETS:
            f, ev = split_validation(exs, eval_cfg["val_fit_seed"], eval_cfg["val_fit_fraction"])
            fit += f
            split_sets[name] = ev
        else:
            split_sets[name] = list(exs)
    temps = fit_family_temperatures(fit, all_logits)
    targets = eval_cfg["target_selective_accuracy"]
    thr = {}
    for fam in ("intent", "tool"):
        fid = [e for e in fit if e.suite == "core" and e.answer is not None and family(e) == fam]
        thr[fam] = thresholds(fid, probs(fid, all_logits, temps), targets) if fid else {}
    for name, exs in split_sets.items():
        reports[name] = set_report(name, exs, all_logits, temps, thr, eval_cfg["coverage_grid"])
    return {"temperatures": temps, "thresholds": thr, "n_val_fit": len(fit),
            "report_sets": {k: len(v) for k, v in split_sets.items()}, "sets": reports}
