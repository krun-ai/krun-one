"""M1 evaluation from signal dumps (runs/v1/m1/<run>/signals_{val_fit,val_eval}.jsonl.gz) -> results/v1/m1/m1_eval.json.

Temperatures (per modality x primitive) and the multi threshold are fitted on val_fit; every number is on val_eval.
Baselines: trivial (train-split majority), blind (frozen text model, media removed), teacher (frozen text model on
the paired text), and no-LoRA ablation when dumped. Gate G1 (configs/v1/m1.yaml) is computed here.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]


def softmax(z, T=1.0):
    z = np.asarray(z, np.float64) / T
    e = np.exp(z - z.max())
    return e / e.sum()


def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return None
    allv = np.concatenate([pos, neg])
    r = allv.argsort().argsort() + 1.0
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def ece(conf, cor, bins=15):
    conf, cor = np.asarray(conf), np.asarray(cor, float)
    e = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - cor[m].mean())
    return float(e)


def fit_T(rows, key="rank"):
    rows = [r for r in rows if r["gold"] and (r["type"] != "choice" or r["answerable"])]
    if not rows:
        return 1.0
    best = (1e18, 1.0)
    for T in np.exp(np.linspace(math.log(0.25), math.log(8), 60)):
        nll = -sum(math.log(max(softmax(r[key], T)[r["gold"][0]], 1e-12)) for r in rows)
        best = min(best, (nll, float(T)))
    return best[1]


def load(run: Path, sp: str) -> list[dict]:
    return [json.loads(l) for l in gzip.open(run / f"signals_{sp}.jsonl.gz", "rt")]


def majority(train_path: Path) -> dict:
    maj = defaultdict(Counter)
    for l in open(train_path, encoding="utf-8"):
        r = json.loads(l)
        key = (r["source"], r["task"]["type"], r["task"].get("family"))
        for lab in r["target"]["labels"]:
            maj[key][lab] += 1
        if r["task"]["type"] == "multi":
            for c in r["candidates"]:
                maj[key + ("cand",)][c["id"]] += 1
    return maj


def cell_key(r):
    return f"{r['modality']}/{r['type']}"


def evaluate(fit: list[dict], ev: list[dict], train_path: Path | None) -> dict:
    cells_fit = defaultdict(list)
    for r in fit:
        cells_fit[cell_key(r)].append(r)
    calib = {}
    for k, rs in cells_fit.items():
        if k.endswith("multi"):
            best = (-1, 0.5)
            for th in np.linspace(0.2, 0.8, 25):
                tp = fp = fn = 0
                for r in rs:
                    pr = 1 / (1 + np.exp(-np.asarray(r["apply"]))) > th
                    g = np.zeros(len(pr), bool)
                    g[r["gold"]] = True
                    tp += int((pr & g).sum()); fp += int((pr & ~g).sum()); fn += int((~pr & g).sum())
                best = max(best, (2 * tp / max(1, 2 * tp + fp + fn), float(th)))
            calib[k] = {"multi_threshold": best[1]}
        else:
            calib[k] = {"T": fit_T(rs), "T_blind": fit_T(rs, "blind_rank")}
    maj = majority(train_path) if train_path and train_path.exists() else None
    ids_by = {}

    def trivial(r):
        if maj is None:
            return None
        src = r["source"]
        key = (src, r["type"], r.get("family"))
        if r["type"] == "multi":
            # per candidate: positive if it was positive in more than half of its train appearances
            src_cand = maj.get(key + ("cand",), Counter())
            return None, [maj[key][c] > 0.5 * src_cand[c] if src_cand[c] else False for c in r["_cands"]]
        for lab, _ in maj[key].most_common():
            if lab in r["_cands"]:
                return r["_cands"].index(lab), None
        return 0, None

    out = {"calibration_fit": calib, "cells": {}, "slices": {}, "sources": {}}
    agg = defaultdict(lambda: defaultdict(list))
    extra = defaultdict(lambda: defaultdict(list))
    for r in ev:
        k = cell_key(r)
        c = calib.get(k, {})
        g = np.zeros(r["k"], bool)
        g[r["gold"]] = True
        res = {}
        if r["type"] == "multi":
            th = c.get("multi_threshold", 0.5)
            for name, key in (("student", "apply"), ("blind", "blind_apply"), ("nolora", "nolora_apply"), ("teacher", "teacher_apply")):
                if r.get(key) is not None:
                    pr = 1 / (1 + np.exp(-np.asarray(r[key]))) > (th if name in ("student", "nolora") else 0.35)
                    res[name] = float((pr == g).all())
                    extra[(k, name)]["tp"].append(int((pr & g).sum()))
                    extra[(k, name)]["fp"].append(int((pr & ~g).sum()))
                    extra[(k, name)]["fn"].append(int((~pr & g).sum()))
            res["random"] = 0.5 ** r["k"]
        else:
            if r["type"] == "choice" and not r["answerable"]:
                p = softmax(r["rank"], c.get("T", 1))
                s = np.sort(p)[::-1]
                extra[(k, "ood")]["neg"].append(float(s[0] - s[1]))
                continue
            for name, key in (("student", "rank"), ("blind", "blind_rank"), ("nolora", "nolora_rank"), ("teacher", "teacher_rank")):
                if r.get(key) is not None:
                    res[name] = float(g[int(np.argmax(r[key]))])
            res["random"] = 1.0 / r["k"]
            p = softmax(r["rank"], c.get("T", 1))
            s = np.sort(p)[::-1]
            extra[(k, "calib")]["conf"].append(float(s[0]))
            extra[(k, "calib")]["cor"].append(res["student"])
            if r["type"] == "choice":
                extra[(k, "ood")]["pos"].append(float(s[0] - (s[1] if len(s) > 1 else 0)))
            if r["type"] == "noul":
                extra[(k, "noul")]["p"].append(float(p[1]))
                extra[(k, "noul")]["y"].append(bool(g[1]))
            if r["type"] == "score":
                extra[(k, "score")]["e"].append(float((np.arange(len(p)) * p).sum()))
                extra[(k, "score")]["y"].append(int(r["gold"][0]))
        if maj is not None:
            tv = trivial(r)
            if tv is not None:
                idx, mp = tv
                res["trivial"] = float((np.asarray(mp) == g).all()) if r["type"] == "multi" else float(g[idx])
        for name, v in res.items():
            agg[k][name].append(v)
            agg[f"source:{r['source']}/{r['type']}"][name].append(v)
            for sl in r["slices"]:
                if sl in ("hard_negative", "cross_language", "real_speech", "synthetic_speech", "screenshot", "real_document"):
                    agg[f"slice:{r['modality']}/{sl}"][name].append(v)
            agg[f"lang:{r['modality']}/{r['lang_media']}"][name].append(v)
    for k, d in agg.items():
        row = {name: float(np.mean(v)) for name, v in d.items()}
        row["n"] = len(d["student"])
        dest = out["cells"] if "/" in k and ":" not in k else (out["sources"] if k.startswith("source:") else out["slices"])
        dest[k] = row
    for (k, what), d in extra.items():
        cell = out["cells"].setdefault(k, {})
        if what == "calib":
            cell["ece"] = ece(d["conf"], d["cor"])
        elif what == "ood" and d.get("neg"):
            cell["ood_auroc_margin"] = auroc(d["pos"], d["neg"])
            cell["n_unanswerable"] = len(d["neg"])
        elif what == "noul":
            y = np.array(d["y"])
            p = np.array(d["p"])
            cell["noul_auroc"] = auroc(p[y], p[~y])
        elif what == "score":
            e, y = np.array(d["e"]), np.array(d["y"])
            cell["score_mae"] = float(np.abs(e - y).mean())
            cell["score_spearman"] = float(spearmanr(e, y).correlation)
        elif what in ("student", "blind", "nolora", "teacher"):
            tp, fp, fn = sum(d["tp"]), sum(d["fp"]), sum(d["fn"])
            cell[f"micro_f1_{what}"] = 2 * tp / max(1, 2 * tp + fp + fn)
    # per modality means + G1
    mods = defaultdict(lambda: defaultdict(list))
    for k, row in out["cells"].items():
        mod = k.split("/")[0]
        for name in ("student", "blind", "trivial", "random", "teacher", "nolora"):
            if name in row:
                mods[mod][name].append(row[name])
    out["modality_mean"] = {m: {n: float(np.mean(v)) for n, v in d.items()} for m, d in mods.items()}
    g1 = {}
    for m, d in out["modality_mean"].items():
        g1[m] = {"student": d["student"], "trivial": d.get("trivial"), "blind": d["blind"],
                 "pass": d["student"] >= (d.get("trivial") or 0) + 0.10 and d["student"] >= d["blind"] + 0.10}
    out["G1_above_trivial"] = {"per_modality": g1, "pass": all(v["pass"] for v in g1.values()) and
                               {"image", "document", "audio"} <= set(g1)}
    # cross-modal consistency (kd1 decisions present as text teacher, speech and screenshot)
    grp = defaultdict(dict)
    for r in ev:
        if r.get("cross_modal_group") and r["type"] != "multi":
            grp[r["cross_modal_group"]][r["modality"]] = r
    pairs = defaultdict(list)
    for gdict in grp.values():
        am = {m: int(np.argmax(x["rank"])) for m, x in gdict.items()}
        t = next(iter(gdict.values())).get("teacher_rank")
        if t is not None:
            am["text"] = int(np.argmax(t))
        gold = next(iter(gdict.values()))["gold"]
        for a, b in (("audio", "document"), ("audio", "text"), ("document", "text")):
            if a in am and b in am:
                pairs[f"{a}~{b}"].append(am[a] == am[b])
        if len(am) == 3:
            pairs["all_three_agree"].append(len(set(am.values())) == 1)
            pairs["all_three_correct"].append(all(v in gold for v in am.values()))
    out["cross_modal_consistency"] = {k: {"agreement": float(np.mean(v)), "n": len(v)} for k, v in pairs.items()}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--train", default=str(ROOT / "data/m1/train.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "results/v1/m1/m1_eval.json"))
    a = ap.parse_args()
    run = Path(a.run)
    cands = {}
    for sp in ("val_fit", "val_eval"):
        for l in open(ROOT / f"data/m1/{sp}.jsonl", encoding="utf-8"):
            r = json.loads(l)
            cands[r["id"]] = [c["id"] for c in r["candidates"]]
    fit, ev = load(run, "val_fit"), load(run, "val_eval")
    for r in fit + ev:
        r["_cands"] = cands[r["id"]]
    res = evaluate(fit, ev, Path(a.train))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({"modality_mean": res["modality_mean"], "G1": res["G1_above_trivial"],
                      "consistency": res["cross_modal_consistency"]}, indent=1))
