"""Krun One v1 — KrunBench v1 V metrics + calibration from signal dumps (v1 candidate or v0.3, same code path).

Protocol (docs/KRUNBENCH_V1_SPEC.md §6): temperatures (per family intent / tool / noul / score) and the public
abstention threshold (top1 - top2 margin, per family, lowest tau with selective accuracy >= 0.98 on fit) are fitted on
the FIT part only; every number below is reported on the EVAL part. Fit/eval: KrunBench v1 val_fit / val_eval files;
val_ml by its metadata.val_part (split_group hash). final_protected is never read.

    python scripts/v1/kd1_evaluate.py --run runs/v1/ft/<name> [--baseline runs/v1/ft/v03_baseline] --tag <tag>
    -> results/v1/ft/eval_<tag>.json (+ calibration.json in the run dir)
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TARGET = 0.98
LANGS = {"EN": lambda l: l.split("-")[0] == "en", "PT-BR": lambda l: l == "pt-BR", "PT-PT": lambda l: l == "pt-PT",
         "ES": lambda l: l.split("-")[0] == "es", "FR": lambda l: l.split("-")[0] == "fr"}
DATA = {"val_fit": ROOT / "data/krunbench_v1/val_fit.jsonl", "val_eval": ROOT / "data/krunbench_v1/val_eval.jsonl",
        "val_ml": ROOT / "data/kd1/val_ml.jsonl", "val_tools_v2": ROOT / "data/kd1_r2/val_tools_v2.jsonl"}


def softmax(z, T=1.0):
    z = np.asarray(z, dtype=np.float64) / T
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def load(run: Path) -> list[dict]:
    rows = []
    for stem, path in DATA.items():
        f = run / f"signals_{stem}.jsonl.gz"
        if not f.exists():
            continue
        part = {}
        if stem in ("val_ml", "val_tools_v2"):
            for l in open(path):
                r = json.loads(l)
                part[r["id"]] = r["metadata"].get("val_part", "eval")
        with gzip.open(f, "rt") as fh:
            for l in fh:
                r = json.loads(l)
                if "final_protected" in r["id"] or r["id"].startswith("kbv1/final"):
                    raise SystemExit("protected item in signals: refusing")
                r["part"] = {"val_fit": "fit", "val_eval": "eval"}.get(stem) or part.get(r["id"], "eval")
                r["file"] = stem
                rows.append(r)
    return rows


def fit_T(rows) -> float:
    rows = [r for r in rows if r["gold"]]
    if not rows:
        return 1.0
    grid = np.exp(np.linspace(np.log(0.3), np.log(5.0), 80))
    best, bt = math.inf, 1.0
    for T in grid:
        nll = -sum(math.log(max(softmax(r["rank"], T)[r["gold"][0]], 1e-12)) for r in rows)
        if nll < best:
            best, bt = nll, float(T)
    return bt


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return None
    s = np.concatenate([pos, neg])
    r = s.argsort(kind="stable").argsort().astype(float) + 1
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def ece(conf, correct, bins=15):
    conf, correct = np.asarray(conf), np.asarray(correct, float)
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)


def threshold(scores, correct, target=TARGET):
    o = np.argsort(-np.asarray(scores), kind="stable")
    acc = np.cumsum(np.asarray(correct, float)[o]) / np.arange(1, len(o) + 1)
    ok = np.flatnonzero(acc >= target)
    return float(np.asarray(scores)[o][ok[-1]]) if len(ok) else math.inf


def aurc(scores, correct):
    o = np.argsort(-np.asarray(scores), kind="stable")
    risk = np.cumsum(1 - np.asarray(correct, float)[o]) / np.arange(1, len(o) + 1)
    return float(risk.mean())


def mean(x):
    x = [v for v in x if v is not None]
    return float(np.mean(x)) if x else None


def evaluate(rows: list[dict], calib: dict | None = None) -> tuple[dict, dict]:
    perm = [r for r in rows if r["part"] == "eval" and (r.get("consistency_group"))]
    rows = [r for r in rows if "permutation" not in r["slices"]]        # rotated copies: consistency metric only
    fit = [r for r in rows if r["part"] == "fit"]
    ev = [r for r in rows if r["part"] == "eval"]
    if calib is None:
        calib = {"temperature": {}, "margin_threshold": {}, "target_selective_accuracy": TARGET, "multi_threshold": 0.5}
        for fam in ("intent", "tool"):
            calib["temperature"][fam] = fit_T([r for r in fit if r["type"] == "choice" and r["family"] == fam and r["answerable"]])
        for t in ("noul", "score"):
            calib["temperature"][t] = fit_T([r for r in fit if r["type"] == t])
        for fam in ("intent", "tool"):
            sub = [r for r in fit if r["type"] == "choice" and r["family"] == fam]
            sc, cor = [], []
            for r in sub:
                p = softmax(r["rank"], calib["temperature"][fam])
                s = np.sort(p)[::-1]
                sc.append(s[0] - s[1])
                cor.append(bool(r["answerable"] and int(np.argmax(p)) in r["gold"]))
            calib["margin_threshold"][fam] = threshold(np.array(sc), np.array(cor))
        # multi: threshold on sigmoid(apply) maximizing micro-F1 on fit
        mu = [r for r in fit if r["type"] == "multi" and "apply" in r]
        if mu:
            best = (0, 0.5)
            for th in np.linspace(0.2, 0.8, 25):
                tp = fp = fn = 0
                for r in mu:
                    pr = 1 / (1 + np.exp(-np.asarray(r["apply"]))) > th
                    g = np.zeros(len(pr), bool)
                    g[r["gold"]] = True
                    tp += int((pr & g).sum()); fp += int((pr & ~g).sum()); fn += int((~pr & g).sum())
                f1 = 2 * tp / max(1, 2 * tp + fp + fn)
                if f1 > best[0]:
                    best = (f1, float(th))
            calib["multi_threshold"] = best[1]
    res: dict = {"n_eval": len(ev), "n_fit": len(fit)}
    # ---------------- choice
    ch = []
    for r in ev:
        if r["type"] != "choice":
            continue
        T = calib["temperature"].get(r["family"], 1.0)
        p = softmax(r["rank"], T)
        s = np.sort(p)[::-1]
        am = int(np.argmax(p))
        ch.append({"r": r, "p": p, "top1": float(s[0]), "margin": float(s[0] - s[1]), "argmax": am,
                   "correct": bool(r["answerable"] and am in r["gold"]),
                   "accept": float(s[0] - s[1]) >= calib["margin_threshold"].get(r["family"], math.inf)})
    ans = [x for x in ch if x["r"]["answerable"]]

    def acc(pred):
        v = [x["correct"] for x in ans if pred(x["r"])]
        return (float(np.mean(v)), len(v)) if v else (None, 0)

    res["choice_accuracy_intent"], res["n_intent"] = acc(lambda r: r["family"] == "intent")
    res["tool_accuracy"], res["n_tool"] = acc(lambda r: r["family"] == "tool")
    res["unseen_tool_accuracy"], res["n_unseen_tool"] = acc(lambda r: "unseen_tool" in r["slices"])
    res["hard_negative_accuracy"], res["n_hard"] = acc(lambda r: r.get("difficulty") == "hard")
    res["near_duplicate_accuracy"], _ = acc(lambda r: "near_duplicate" in r["slices"])
    res["cross_language_accuracy"], res["n_cross_language"] = acc(lambda r: "cross_language" in r["slices"])
    res["by_language"] = {}
    for L, f in LANGS.items():
        a, n = acc(lambda r, f=f: f(r["lang"]))
        ai, ni = acc(lambda r, f=f: f(r["lang"]) and r["family"] == "intent")
        at, nt = acc(lambda r, f=f: f(r["lang"]) and r["family"] == "tool")
        un = [x for x in ch if not x["r"]["answerable"] and f(x["r"]["lang"])]
        res["by_language"][L] = {"choice_accuracy": a, "n": n, "intent_accuracy": ai, "n_intent": ni, "tool_accuracy": at,
                                 "n_tool": nt, "ood_false_acceptance": mean([x["accept"] for x in un]), "n_ood": len(un)}
    res["by_k"] = {}
    for k in (2, 4, 8, 16, 32, 64, 128):
        a, n = acc(lambda r, k=k: (k // 2 < r["k"] <= k) if k > 2 else r["k"] <= 2)
        res["by_k"][k] = {"accuracy": a, "n": n}
    conf = np.array([x["top1"] for x in ans])
    cor = np.array([x["correct"] for x in ans], float)
    res["ece"] = ece(conf, cor) if len(ans) else None
    res["nll_choice"] = mean([-math.log(max(x["p"][x["r"]["gold"][0]], 1e-12)) for x in ans])
    sc = np.array([x["margin"] for x in ch])
    corr_all = np.array([x["correct"] for x in ch], float)
    is_ans = np.array([x["r"]["answerable"] for x in ch])
    res["ood_auroc_margin"] = auroc(sc[is_ans], sc[~is_ans])
    res["aurc"] = aurc(sc, corr_all) if len(ch) else None
    acc_mask = np.array([x["accept"] for x in ch])
    res["coverage_at_target"] = float(acc_mask[is_ans].mean()) if is_ans.any() else None
    res["selective_accuracy_at_target"] = float(corr_all[acc_mask].mean()) if acc_mask.any() else None
    res["ood_false_acceptance"] = float(acc_mask[~is_ans].mean()) if (~is_ans).any() else None
    buckets = defaultdict(list)
    for x in ch:
        if x["r"]["answerable"]:
            continue
        s = set(x["r"]["slices"])
        b = ("native_irrelevance" if "native_irrelevance" in s else "gold_removed" if "gold_removed" in s else
             "cross_task" if "cross_task" in s else "ood_far" if "ood_far" in s else "other")
        buckets[f"{x['r']['family']}/{b}"].append(x["accept"])
    un_ut = [x["accept"] for x in ch if not x["r"]["answerable"] and "unseen_tool" in x["r"]["slices"]]
    res["unseen_tool_false_acceptance"], res["n_unseen_tool_ood"] = (mean(un_ut), len(un_ut))
    res["ood_false_acceptance_by_bucket"] = {k: {"fa": float(np.mean(v)), "n": len(v)} for k, v in sorted(buckets.items())}
    for tgt in (0.95, 0.99):
        cov = []
        for fam in ("intent", "tool"):
            f_sc = [(x["margin"], x["correct"]) for x in [
                {"margin": float(np.sort(softmax(r["rank"], calib["temperature"][fam]))[::-1][:2] @ [1, -1]),
                 "correct": bool(r["answerable"] and int(np.argmax(r["rank"])) in r["gold"])}
                for r in fit if r["type"] == "choice" and r["family"] == fam]]
            if not f_sc:
                continue
            tau = threshold(np.array([a for a, _ in f_sc]), np.array([b for _, b in f_sc]), tgt)
            m = [x for x in ans if x["r"]["family"] == fam]
            cov += [x["margin"] >= tau for x in m]
        res[f"coverage_at_{tgt}"] = mean(cov)
    groups = defaultdict(list)
    for r in perm:
        if r["type"] == "choice":
            groups[r["consistency_group"]].append(r["candidate_ids"][int(np.argmax(r["rank"]))])
    g = [len(set(v)) == 1 for v in groups.values() if len(v) > 1]
    res["permutation_consistency"] = mean(g)
    res["permutation_groups"] = len(g)
    # cross-language consistency: same semantic group answered in >= 2 languages (MASSIVE / BANKING77 parallel rows)
    sg = defaultdict(dict)
    for x in ans:
        gid = x["r"]["split_group_id"]
        sg[gid][x["r"]["lang"]] = x["correct"]
    pairs = [v for v in sg.values() if len(v) >= 2]
    res["cross_language_consistency"] = mean([len(set(v.values())) == 1 for v in pairs])
    res["cross_language_groups"] = len(pairs)
    # ---------------- noul / score / multi
    nl = [r for r in ev if r["type"] == "noul"]
    if nl:
        pt = np.array([softmax(r["rank"], calib["temperature"]["noul"])[1] for r in nl])
        y = np.array([1 in r["gold"] for r in nl])
        res["noul_auroc"] = auroc(pt[y], pt[~y])
        res["noul_accuracy"] = float(((pt > 0.5) == y).mean())
        res["noul_ece"] = ece(np.maximum(pt, 1 - pt), ((pt > 0.5) == y))
        res["n_noul"] = len(nl)
        res["noul_by_language"] = {L: mean([float((p > 0.5) == yy) for p, yy, r in zip(pt, y, nl) if f(r["lang"])])
                                   for L, f in LANGS.items()}
    scr = [r for r in ev if r["type"] == "score"]
    if scr:
        from scipy.stats import spearmanr

        P = [softmax(r["rank"], calib["temperature"]["score"]) for r in scr]
        e = np.array([float((np.arange(len(p)) * p).sum()) for p in P])
        y = np.array([r["gold"][0] for r in scr])
        res["score_mae"] = float(np.abs(e - y).mean())
        res["score_spearman"] = float(spearmanr(e, y).correlation)
        res["score_nll"] = float(np.mean([-math.log(max(p[g], 1e-12)) for p, g in zip(P, y)]))
        res["n_score"] = len(scr)
    mu = [r for r in ev if r["type"] == "multi" and "apply" in r]
    if mu:
        th = calib["multi_threshold"]
        tp = fp = fn = 0
        per_label = defaultdict(lambda: [0, 0, 0])
        empty_ok = []
        for r in mu:
            pr = 1 / (1 + np.exp(-np.asarray(r["apply"]))) > th
            g = np.zeros(len(pr), bool)
            g[r["gold"]] = True
            tp += int((pr & g).sum()); fp += int((pr & ~g).sum()); fn += int((~pr & g).sum())
            for i, cid in enumerate(r["candidate_ids"]):
                t = per_label[cid]
                t[0] += int(pr[i] and g[i]); t[1] += int(pr[i] and not g[i]); t[2] += int(g[i] and not pr[i])
            if not g.any():
                empty_ok.append(not pr.any())
        res["multi_micro_f1"] = 2 * tp / max(1, 2 * tp + fp + fn)
        tpu = fpu = fnu = 0
        for r in mu:
            if "unseen_tool" not in r["slices"]:
                continue
            pr = 1 / (1 + np.exp(-np.asarray(r["apply"]))) > th
            g = np.zeros(len(pr), bool)
            g[r["gold"]] = True
            tpu += int((pr & g).sum()); fpu += int((pr & ~g).sum()); fnu += int((~pr & g).sum())
        res["multi_unseen_tool_micro_f1"] = 2 * tpu / max(1, 2 * tpu + fpu + fnu) if tpu + fpu + fnu else None
        res["multi_macro_f1"] = mean([2 * a / max(1, 2 * a + b + c) for a, b, c in per_label.values() if a + b + c])
        res["multi_empty_set_accuracy"] = mean(empty_ok)
        res["n_multi"] = len(mu)
    return res, calib


def paired_delta(rows_a, rows_b, pred, reps=1000, seed=0):
    """Paired group bootstrap of accuracy(b) - accuracy(a) on shared answerable choice ids (b = candidate)."""
    def correct(rows):
        out = {}
        for r in rows:
            if r["part"] == "eval" and r["type"] == "choice" and r["answerable"] and pred(r) and "permutation" not in r["slices"]:
                out[r["id"]] = (r["split_group_id"], float(int(np.argmax(r["rank"])) in r["gold"]))
        return out
    a, b = correct(rows_a), correct(rows_b)
    ids = sorted(set(a) & set(b))
    if not ids:
        return None
    grp = defaultdict(list)
    for i in ids:
        grp[a[i][0]].append(b[i][1] - a[i][1])
    keys = list(grp)
    rng = np.random.default_rng(seed)
    d = float(np.mean([b[i][1] - a[i][1] for i in ids]))
    bs = []
    for _ in range(reps):
        pick = rng.integers(0, len(keys), len(keys))
        v = [x for k in pick for x in grp[keys[k]]]
        bs.append(np.mean(v))
    return {"delta": d, "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "n": len(ids)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--baseline")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    rows = load(Path(args.run))
    res, calib = evaluate(rows)
    (Path(args.run) / "calibration.json").write_text(json.dumps(calib, indent=2))
    out = {"run": args.run, "metrics": res, "calibration": calib}
    if args.baseline:
        base = load(Path(args.baseline))
        bres, bcal = evaluate(base)
        out["baseline"] = {"run": args.baseline, "metrics": bres, "calibration": bcal}
        out["paired_deltas"] = {
            name: paired_delta(base, rows, pred) for name, pred in (
                ("all_choice", lambda r: True), ("intent", lambda r: r["family"] == "intent"),
                ("tool", lambda r: r["family"] == "tool"), ("cross_language", lambda r: "cross_language" in r["slices"]),
                ("hard", lambda r: r.get("difficulty") == "hard"), ("unseen_tool", lambda r: "unseen_tool" in r["slices"]),
                ("non_en", lambda r: not r["lang"].startswith("en")), ("PT-BR", lambda r: r["lang"] == "pt-BR"),
                ("PT-PT", lambda r: r["lang"] == "pt-PT"), ("ES", lambda r: r["lang"].startswith("es")),
                ("FR", lambda r: r["lang"].startswith("fr")), ("EN", lambda r: r["lang"].startswith("en")))}
    p = ROOT / "results/v1/ft"
    p.mkdir(parents=True, exist_ok=True)
    (p / f"eval_{args.tag}.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}, indent=1, default=str))


if __name__ == "__main__":
    main()
