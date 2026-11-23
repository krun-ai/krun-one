"""Answerability Bakeoff analysis (CPU, offline): variants A-D, J and the fusion calibrator E.

Reads runs/v1/<variant>_s<seed>/val_signals.jsonl.gz (validation only), splits val_fit / val_eval by
split_group_id, fits per-family temperatures, the E calibrator (logistic regression) and thresholds on val_fit,
and reports everything on val_eval. Rule and gates: configs/v1/proxy.yaml (registered before training).

    uv run python scripts/v1/answerability_bakeoff.py            # -> results/v1/answerability_bakeoff.{json,md}
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load((ROOT / "configs/v1/proxy.yaml").read_text())
BK = CFG["bakeoff"]
TARGET = 0.98
OOD_BUCKETS = ("tool_irrelevance", "tool_gold_removed", "intent_near_ood")
FEATURES_FULL = ("top1", "margin", "negent", "p_null", "ans_logit", "apply_logit", "log_k", "is_tool")
FEATURES_SOFTMAX = ("top1", "margin", "negent", "log_k", "is_tool")


# ------------------------------------------------------------------------------------------------ loading
def load(run: Path) -> list[dict]:
    with gzip.open(run / "val_signals.jsonl.gz", "rt") as f:
        return [json.loads(l) for l in f]


def is_fit(group: str) -> bool:
    h = hashlib.sha256((BK["val_split"]["salt"] + "|" + group).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64 < BK["val_split"]["fit_fraction"]


def bucket(r: dict) -> str | None:
    s = set(r["slices"])
    if r["answerable"]:
        return None
    if "native_irrelevance" in s:
        return "tool_irrelevance"
    if "gold_removed" in s and r["family"] == "tool":
        return "tool_gold_removed"
    if ("ood_near" in s or "gold_removed" in s) and r["family"] == "intent":
        return "intent_near_ood"
    if "cross_task" in s:
        return "cross_task"
    if "ood_far" in s:
        return "far_ood"
    return "other_ood"


# ------------------------------------------------------------------------------------------------ features
def softmax(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def fit_temperature(rows: list[dict]) -> float:
    grid = np.exp(np.linspace(np.log(0.3), np.log(5.0), 120))
    best, bt = math.inf, 1.0
    for T in grid:
        nll = 0.0
        for r in rows:
            p = softmax(np.asarray(r["rank"]) / T)
            nll -= math.log(max(p[r["gold"][0]], 1e-12))
        if nll < best:
            best, bt = nll, float(T)
    return bt


def featurize(r: dict, T: float) -> dict:
    z = np.asarray(r["rank"]) / T
    p = softmax(z)
    s = np.sort(p)[::-1]
    k = len(p)
    ent = -(p[p > 0] * np.log(p[p > 0])).sum() / math.log(max(k, 2))
    f = {"top1": float(s[0]), "margin": float(s[0] - (s[1] if k > 1 else 0)), "negent": float(1 - ent),
         "log_k": math.log(k), "is_tool": float(r["family"] == "tool"), "argmax": int(np.argmax(p)), "p": p}
    if "null" in r:
        f["p_null"] = float(softmax(np.append(np.asarray(r["rank"]), r["null"]) / T)[-1])
    if "ans" in r:
        f["ans_logit"] = float(r["ans"])
    if "apply" in r:
        f["apply_logit"] = float(max(r["apply"]))
    f["correct"] = bool(r["answerable"] and f["argmax"] in r["gold"])
    return f


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def native_scores(fs: list[dict], name: str) -> np.ndarray:
    if name == "margin":
        return np.array([f["margin"] for f in fs])
    if name == "null_softmax":
        return np.array([1 - f["p_null"] for f in fs])
    if name == "ans_head":
        return np.array([sigmoid(f["ans_logit"]) for f in fs])
    if name == "max_apply":
        return np.array([sigmoid(f["apply_logit"]) for f in fs])
    raise ValueError(name)


def fit_calibrator(fs: list[dict], feats: tuple[str, ...]):
    X = np.array([[f.get(k, 0.0) for k in feats] for f in fs])
    y = np.array([f["correct"] for f in fs], dtype=int)
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(X), y)
    return lambda fs2: lr.predict_proba(sc.transform(np.array([[f.get(k, 0.0) for k in feats] for f in fs2])))[:, 1], \
        {"features": list(feats), "coef": [round(float(c), 4) for c in lr.coef_[0]], "intercept": float(lr.intercept_[0])}


# ------------------------------------------------------------------------------------------------ metrics
def threshold(scores: np.ndarray, correct: np.ndarray, target: float = TARGET) -> float:
    o = np.argsort(-scores, kind="stable")
    acc = np.cumsum(correct[o]) / np.arange(1, len(o) + 1)
    ok = np.flatnonzero(acc >= target)
    return float(scores[o][ok[-1]]) if len(ok) else math.inf


def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return None
    s = np.concatenate([pos, neg])
    r = s.argsort(kind="stable").argsort().astype(float) + 1
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def aurc(scores, correct):
    o = np.argsort(-scores, kind="stable")
    risk = np.cumsum(1 - correct[o]) / np.arange(1, len(o) + 1)
    return float(risk.mean())


def ece(conf, correct, bins=15):
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)


def evaluate_signal(rows_e, fs_e, scores_e, tau: dict) -> dict:
    acc = np.array([scores_e[i] >= tau[r["family"]] for i, r in enumerate(rows_e)])
    correct = np.array([f["correct"] for f in fs_e])
    ans = np.array([r["answerable"] for r in rows_e])
    out = {"coverage": float(acc[ans].mean()), "selective_accuracy": float(correct[acc].mean()) if acc.any() else None,
           "auroc": auroc(scores_e[ans], scores_e[~ans]), "aurc": aurc(scores_e, correct.astype(float)),
           "false_acceptance": {}, "answerable_buckets": {}, "by_lang": {}}
    b = [bucket(r) for r in rows_e]
    for name in sorted({x for x in b if x}):
        m = np.array([x == name for x in b])
        out["false_acceptance"][name] = {"fa": float(acc[m].mean()), "n": int(m.sum())}
    out["macro_fa"] = float(np.mean([out["false_acceptance"][k]["fa"] for k in OOD_BUCKETS
                                     if k in out["false_acceptance"]]))
    for name, pred in (("unseen_tool", lambda r: "unseen_tool" in r["slices"]),
                       ("hard", lambda r: r.get("difficulty") == "hard"),
                       ("cross_language", lambda r: "cross_language" in r["slices"])):
        m = np.array([r["answerable"] and pred(r) for r in rows_e])
        if m.any():
            a = acc & m
            out["answerable_buckets"][name] = {"coverage": float(a.sum() / m.sum()),
                                               "selective_risk": float(1 - correct[a].mean()) if a.any() else None,
                                               "n": int(m.sum())}
    for lang in sorted({r["lang"] for r in rows_e}):
        m = np.array([r["lang"] == lang for r in rows_e])
        un = m & ~ans
        out["by_lang"][lang] = {"coverage": float(acc[m & ans].mean()) if (m & ans).any() else None,
                                "fa": float(acc[un].mean()) if un.any() else None}
    return out


def model_metrics(rows_e, fs_e) -> dict:
    res = {}
    for fam in ("intent", "tool"):
        m = [f for r, f in zip(rows_e, fs_e) if r["answerable"] and r["family"] == fam]
        res[f"accuracy_{fam}"] = float(np.mean([f["correct"] for f in m]))
    m = [(r, f) for r, f in zip(rows_e, fs_e) if r["answerable"]]
    conf = np.array([f["top1"] for _, f in m])
    cor = np.array([f["correct"] for _, f in m], dtype=float)
    res["ece"] = ece(conf, cor)
    res["nll"] = float(np.mean([-math.log(max(f["p"][r["gold"][0]], 1e-12)) for r, f in m]))
    groups = defaultdict(list)
    for r, f in zip(rows_e, fs_e):
        if r.get("consistency_group"):
            groups[r["consistency_group"]].append(r["candidate_ids"][f["argmax"]])
    g = [len(set(v)) == 1 for v in groups.values() if len(v) > 1]
    res["permutation_consistency"] = float(np.mean(g)) if g else None
    res["permutation_groups"] = len(g)
    return res


def bootstrap_diff(rows_by_seed_A, sc_A, rows_by_seed_X, sc_X, reps=1000, seed=0) -> tuple[float, float]:
    """Paired group bootstrap of macro FA(A) - macro FA(X), pooled over seeds (same eval rows per seed)."""
    rng = np.random.default_rng(seed)
    per = []  # (group, bucket, accA, accX)
    for s in rows_by_seed_A:
        for r, a, x in zip(rows_by_seed_A[s], sc_A[s], sc_X[s]):
            b = bucket(r)
            if b in OOD_BUCKETS:
                per.append((r["split_group_id"], b, a, x))
    groups = sorted({p[0] for p in per})
    gi = {g: i for i, g in enumerate(groups)}
    by_g = defaultdict(list)
    for p in per:
        by_g[gi[p[0]]].append(p)
    diffs = []
    for _ in range(reps):
        pick = rng.integers(0, len(groups), len(groups))
        tot = defaultdict(lambda: [0, 0, 0])
        for g in pick:
            for _, b, a, x in by_g[g]:
                t = tot[b]
                t[0] += a
                t[1] += x
                t[2] += 1
        diffs.append(np.mean([t[0] / t[2] - t[1] / t[2] for t in tot.values() if t[2]]))
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


# ------------------------------------------------------------------------------------------------ main
def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "runs/v1"))
    ap.add_argument("--out", default=str(ROOT / "results/v1"))
    args = ap.parse_args()
    runs = Path(args.runs)
    variants = ["A_margin", "B_null", "C_ans_head", "D_apply", "J_joint"]
    results: dict = {"target_selective_accuracy": TARGET, "variants": {}}
    accept_cache: dict = defaultdict(dict)   # (mechanism) -> seed -> (rows_e, accepted array)
    score_cache: dict = defaultdict(dict)    # (mechanism) -> seed -> (rows_e, scores)
    for v in variants:
        for seed in CFG["seeds"]:
            run = runs / f"{v}_s{seed}"
            if not (run / "val_signals.jsonl.gz").exists():
                continue
            rows = [r for r in load(run) if r["type"] == "choice"]
            fit = [r for r in rows if is_fit(r["split_group_id"])]
            ev = [r for r in rows if not is_fit(r["split_group_id"])]
            T = {fam: fit_temperature([r for r in fit if r["answerable"] and r["family"] == fam]) for fam in ("intent", "tool")}
            ff = [featurize(r, T[r["family"]]) for r in fit]
            fe = [featurize(r, T[r["family"]]) for r in ev]
            mech = {}
            natives = {"A_margin": ["margin"], "B_null": ["null_softmax"], "C_ans_head": ["ans_head"],
                       "D_apply": ["max_apply"], "J_joint": ["margin", "null_softmax", "ans_head", "max_apply"]}[v]
            for sig in natives:
                mech[sig] = (native_scores(ff, sig), native_scores(fe, sig), None)
            feats = FEATURES_FULL if v == "J_joint" else FEATURES_SOFTMAX
            cal, info = fit_calibrator(ff, feats)
            mech["E" if v == "J_joint" else "E_softmax"] = (cal(ff), cal(fe), info)
            entry = {"temperature": T, "model": model_metrics(ev, fe), "signals": {}}
            ext = runs / f"{v}_s{seed}" / "eval_validation.json"
            if ext.exists():
                e = json.loads(ext.read_text())
                entry["other_primitives"] = {k: e.get(k) for k in ("noul_auroc", "score_mae", "score_spearman",
                                                                    "multi_micro_f1")}
            for name, (sf, se, info) in mech.items():
                tau = {}
                for fam in ("intent", "tool"):
                    idx = [i for i, r in enumerate(fit) if r["family"] == fam]
                    tau[fam] = threshold(sf[idx], np.array([ff[i]["correct"] for i in idx], dtype=float))
                if v == "A_margin" and name == "margin":
                    for fam in tau:
                        _A_TAUS[(fam, seed)] = tau[fam]
                m = evaluate_signal(ev, fe, se, tau)
                m["tau"] = tau
                if info:
                    m["calibrator"] = info
                entry["signals"][name] = m
                accept_cache[f"{v}:{name}"][seed] = (ev, [bool(se[i] >= tau[r["family"]]) for i, r in enumerate(ev)])
                score_cache[f"{v}:{name}"][seed] = (ev, se)
            results["variants"].setdefault(v, {})[str(seed)] = entry
            print(f"{v} seed {seed}: " + ", ".join(f"{n} macroFA={m['macro_fa']:.3f} cov={m['coverage']:.3f} "
                                                   f"auroc={m['auroc']:.3f}" for n, m in entry["signals"].items()))
    results["summary"] = summarize(results, accept_cache)
    results["summary"]["matched_coverage"] = matched_coverage(score_cache)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "answerability_bakeoff.json").write_text(json.dumps(results, indent=2, default=str))
    (out / "answerability_bakeoff.md").write_text(render(results))
    print(render(results))


MECHANISMS = [("A_margin", "margin", "A margin (baseline)"), ("A_margin", "E_softmax", "E[A] softmax-only fusion"),
              ("B_null", "null_softmax", "B [NULL] in softmax"), ("C_ans_head", "ans_head", "C answerability head"),
              ("D_apply", "max_apply", "D max applicability"), ("J_joint", "margin", "J margin"),
              ("J_joint", "null_softmax", "J p_null"), ("J_joint", "ans_head", "J ans head"),
              ("J_joint", "max_apply", "J max apply"), ("J_joint", "E", "E fusion (on J)")]


def summarize(results, cache) -> dict:
    g = BK["gates_vs_A"]
    rows = {}
    for v, sig, label in MECHANISMS:
        seeds = results["variants"].get(v, {})
        if not seeds:
            continue
        get = lambda f: [f(e) for e in seeds.values()]  # noqa: E731
        sm = lambda e: e["signals"].get(sig)  # noqa: E731
        if any(sm(e) is None for e in seeds.values()):
            continue
        rows[label] = {
            "variant": v, "signal": sig, "seeds": len(seeds),
            "macro_fa": get(lambda e: sm(e)["macro_fa"]),
            **{f"fa_{b}": get(lambda e, b=b: sm(e)["false_acceptance"].get(b, {}).get("fa")) for b in OOD_BUCKETS + ("cross_task",)},
            "coverage": get(lambda e: sm(e)["coverage"]), "auroc": get(lambda e: sm(e)["auroc"]),
            "aurc": get(lambda e: sm(e)["aurc"]),
            "unseen_tool_risk": get(lambda e: (sm(e)["answerable_buckets"].get("unseen_tool") or {}).get("selective_risk")),
            "hard_risk": get(lambda e: (sm(e)["answerable_buckets"].get("hard") or {}).get("selective_risk")),
            "acc_intent": get(lambda e: e["model"]["accuracy_intent"]), "acc_tool": get(lambda e: e["model"]["accuracy_tool"]),
            "ece": get(lambda e: e["model"]["ece"]), "nll": get(lambda e: e["model"]["nll"]),
            "perm": get(lambda e: e["model"]["permutation_consistency"]),
        }
    base = rows.get("A margin (baseline)")
    decision = {"winner": None, "candidates": {}}
    if base:
        mean = lambda x: float(np.mean([v for v in x if v is not None]))  # noqa: E731
        for label, r in rows.items():
            if label == "A margin (baseline)":
                continue
            gates = {
                "intent_acc": mean(base["acc_intent"]) - mean(r["acc_intent"]) <= g["choice_intent_accuracy_drop_pp"] / 100,
                "tool_acc": mean(base["acc_tool"]) - mean(r["acc_tool"]) <= g["choice_tool_accuracy_drop_pp"] / 100,
                "perm": True if not any(v is not None for v in base["perm"] + r["perm"]) else
                        (mean(base["perm"]) - mean(r["perm"])) <= g["permutation_consistency_drop_pp"] / 100,
                "ece": mean(r["ece"]) - mean(base["ece"]) <= g["ece_rise"],
                "nll": mean(r["nll"]) <= mean(base["nll"]) * (1 + g["nll_rise_relative"]),
                "coverage": mean(base["coverage"]) - mean(r["coverage"]) <= g["coverage_drop_pp"] / 100,
            }
            d_abs = mean(base["macro_fa"]) - mean(r["macro_fa"])
            d_rel = d_abs / max(mean(base["macro_fa"]), 1e-9)
            every = len(r["macro_fa"]) == len(base["macro_fa"]) and all(
                x < y for x, y in zip(r["macro_fa"], base["macro_fa"]))
            ci = None
            ka, kx = f"{base['variant']}:{base['signal']}", f"{r['variant']}:{r['signal']}"
            if ka in cache and kx in cache and set(cache[ka]) == set(cache[kx]) and all(
                    [x["id"] for x in cache[ka][sd][0]] == [x["id"] for x in cache[kx][sd][0]] for sd in cache[ka]):
                rows_by_seed = {sd: cache[ka][sd][0] for sd in cache[ka]}
                ci = bootstrap_diff(rows_by_seed, {sd: cache[ka][sd][1] for sd in cache[ka]},
                                    rows_by_seed, {sd: cache[kx][sd][1] for sd in cache[kx]})
            clear = d_abs >= 0.03 and d_rel >= 0.25 and every and ci is not None and ci[0] > 0
            decision["candidates"][label] = {"gates": gates, "all_gates": all(gates.values()), "delta_macro_fa": d_abs,
                                             "delta_rel": d_rel, "better_every_seed": every, "bootstrap_ci_A_minus_X": ci,
                                             "clear_win": bool(clear and all(gates.values()))}
        winners = [(mean(rows[l]["macro_fa"]), l) for l, c in decision["candidates"].items() if c["clear_win"]]
        if winners:
            winners.sort()
            decision["winner"] = winners[0][1]
    return {"rows": rows, "decision": decision}


def matched_coverage(score_cache) -> dict:
    """POST-HOC DIAGNOSTIC (not part of the registered rule): per seed and family, set each mechanism's threshold on
    val_eval so that its answerable coverage equals the baseline A-margin coverage at A's own operating point; report
    macro OOD FA and selective accuracy there. Isolates OOD discrimination from 'abstain more'."""
    base = score_cache.get("A_margin:margin", {})
    out = {}
    for key, per_seed in score_cache.items():
        fas = []
        for seed, (rows, sc) in per_seed.items():
            if seed not in base:
                continue
            brows, bsc = base[seed]
            if [r["id"] for r in brows] != [r["id"] for r in rows]:
                continue
            acc = np.zeros(len(rows), dtype=bool)
            for fam in ("intent", "tool"):
                idx = np.array([i for i, r in enumerate(rows) if r["family"] == fam])
                ans = np.array([rows[i]["answerable"] for i in idx])
                # A's answerable coverage at its own val_fit operating point, as a count on val_eval
                k = int(round(float(np.mean(bsc[idx][ans] >= _a_tau(fam, seed))) * ans.sum()))
                if k == 0:
                    continue
                t = np.sort(sc[idx][ans])[::-1][k - 1]
                acc[idx] = sc[idx] >= t
            b = [bucket(r) for r in rows]
            fas.append(float(np.mean([acc[[x == name for x in b]].mean() for name in OOD_BUCKETS
                                      if any(x == name for x in b)])))
        out[key] = {"macro_fa_at_A_coverage": fas}
    return out


_A_TAUS: dict = {}


def _a_tau(fam, seed):
    return _A_TAUS[(fam, seed)]


def render(results) -> str:
    s = results["summary"]
    rows = s["rows"]

    def fmt(x, pct=True):
        v = [y for y in x if y is not None]
        if not v:
            return "—"
        m, sd = float(np.mean(v)), float(np.std(v))
        return f"{100 * m:.1f} ±{100 * sd:.1f}" if pct else f"{m:.3f} ±{sd:.3f}"

    out = ["# Answerability Bakeoff — proxy (mmBERT-small + v1 stack), validation only\n",
           f"Operating point: selective accuracy {TARGET} on val_fit per family; numbers on val_eval; mean ± sd over "
           "seeds. FA = false acceptance (%) of unanswerable decisions; risk = accepted-and-wrong / accepted (%).\n",
           "| mechanism | macro FA | tool irrelevance FA | tool gold-removed FA | intent near-OOD FA | cross-task FA | "
           "coverage | AUROC | AURC | unseen-tool risk | hard risk |", "|---|" + "---|" * 10]
    for label, r in rows.items():
        out.append(f"| {label} | {fmt(r['macro_fa'])} | {fmt(r['fa_tool_irrelevance'])} | {fmt(r['fa_tool_gold_removed'])} | "
                   f"{fmt(r['fa_intent_near_ood'])} | {fmt(r['fa_cross_task'])} | {fmt(r['coverage'])} | "
                   f"{fmt(r['auroc'], False)} | {fmt(r['aurc'], False)} | {fmt(r['unseen_tool_risk'])} | {fmt(r['hard_risk'])} |")
    out += ["", "## Model-level gates (per trained variant)\n",
            "| variant | intent acc | tool acc | ECE | NLL | permutation consistency |", "|---|---|---|---|---|---|"]
    seen = set()
    for label, r in rows.items():
        if r["variant"] in seen:
            continue
        seen.add(r["variant"])
        out.append(f"| {r['variant']} | {fmt(r['acc_intent'])} | {fmt(r['acc_tool'])} | {fmt(r['ece'], False)} | "
                   f"{fmt(r['nll'], False)} | {fmt(r['perm'])} |")
    d = s["decision"]
    out += ["", "## Decision (rule registered in configs/v1/proxy.yaml before training)\n",
            "| mechanism | Δ macro FA vs A (pp) | relative | every seed | 95% CI (A−X) | gates | clear win |",
            "|---|---|---|---|---|---|---|"]
    for label, c in d["candidates"].items():
        failed = [k for k, ok in c["gates"].items() if not ok]
        ci = c["bootstrap_ci_A_minus_X"]
        out.append(f"| {label} | {100 * c['delta_macro_fa']:+.1f} | {100 * c['delta_rel']:+.0f}% | {c['better_every_seed']} | "
                   f"{'—' if ci is None else f'[{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]'} | "
                   f"{'pass' if not failed else 'FAIL: ' + ','.join(failed)} | {'**yes**' if c['clear_win'] else 'no'} |")
    out += ["", f"**Winner:** {d['winner'] or 'none -> keep top1-top2 margin as the public abstention mechanism'}"]
    mc = s.get("matched_coverage") or {}
    if mc:
        out += ["", "## Post-hoc diagnostic (NOT part of the registered rule): macro OOD FA at A's answerable coverage\n",
                "Each mechanism's threshold is moved (on val_eval) until it accepts the same fraction of answerable "
                "decisions as A-margin does at its operating point. Lower = better OOD discrimination at equal coverage.\n",
                "| mechanism | macro FA @ A coverage (%) |", "|---|---|"]
        for label_key, v in sorted(mc.items(), key=lambda kv: np.mean(kv[1]["macro_fa_at_A_coverage"] or [9])):
            f = v["macro_fa_at_A_coverage"]
            if f:
                out.append(f"| {label_key} | {100 * np.mean(f):.1f} ±{100 * np.std(f):.1f} |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    main()
