"""Krun One v1.1 R3 — technical report on KrunTechBench v1 (labels restored) + the R3 extension, any question type.

    uv run python scripts/v11/r3_report.py --split protected --models v1 r2l4na_s1 r3f_s1 r3f_s2 --out results/v11/r3/techbench/protected
    uv run python scripts/v11/r3_report.py --split val ...

Reads results/v11/eval/<model>/tech_<stem>.jsonl.gz for the two files of the split:
  v1 part   data/v11/techbench/<split>_v1_labeled.jsonl      (dumps of the unlabeled file are accepted: same ids, same inputs)
  r3 part   data/v11/techbench/<split>_r3.jsonl
Metrics (raw rendering is the headline; ``feat`` = with each candidate's explicit consequence):
  top-1 per track and macro; capability tracks by question type and by seen / unseen scenario; unseen requirement
  combinations (by set size, and on invented-product catalogs); constraint flips / counterfactual families:
      both_correct  = share of (base, variant) pairs with both members right
      stuck_rate    = P(variant answered with the BASE's correct answer | base answered correctly)   <- R3 headline
  hard-negative margin; OOD false acceptance; paired bootstrap vs a baseline.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from krun_one.v11 import probe, training as T  # noqa: E402

EVAL = ROOT / "results/v11/eval"
D = ROOT / "data/v11/techbench"
V1_TRACKS = ["api", "backend", "cloud", "database", "devops", "distributed", "mlinfra", "security", "software_repair", "state_action"]
CAP_TRACKS = ["numeric", "temporal", "multi_hop", "long_policy", "tradeoff", "mixed"]


def cond(e: dict) -> str:
    return "raw" if e["render"] == "raw" or e["track"] == "software_repair" else "feat"


def load(model: str, stems: list[str], ex: dict, Tm: float, thr: float) -> dict[str, dict]:
    rows = {}
    for stem in stems:
        p = EVAL / model / f"tech_{stem}.jsonl.gz"
        if not p.exists():
            continue
        with gzip.open(p, "rt") as f:
            for line in f:
                r = json.loads(line)
                e = ex.get(r["id"])
                if e is None or r["id"] in rows:
                    continue
                qt = e.get("qtype", "choice")
                if qt == "choice":
                    row = probe.item_signals(e, r["z"], Tm, thr)
                    if e["answerable"]:
                        row["pred_label"] = row["pred_name"]
                elif qt in ("noul", "score"):
                    if e["gold"] is None:
                        continue
                    pred = int(np.argmax(r["z"]))
                    row = {"id": r["id"], "correct": pred == e["gold"], "answerable": True, "pred_label": e["candidates"][pred]["name"],
                           "abs_err": abs(pred - e["gold"])}
                else:
                    pred = 1 / (1 + np.exp(-np.asarray(r["apply"]))) > 0.5
                    gold = np.zeros(len(pred), bool)
                    gold[e["gold_multi"]] = True
                    row = {"id": r["id"], "answerable": True, "multi": [int((pred & gold).sum()), int((pred & ~gold).sum()), int((~pred & gold).sum())],
                           "correct": bool((pred == gold).all())}
                rows[r["id"]] = row
    return rows


def acc(rows: list[dict]) -> dict:
    v = [r["correct"] for r in rows if r.get("answerable") and "correct" in r]
    return {"n": len(v), "top1": float(np.mean(v)) if v else None}


def pairs(ex: dict, rows: dict, ids: list[str]) -> dict:
    """(base, variant) pairs: R1/R2 constraint flips and R3 families; variant 0 is the base."""
    fam = defaultdict(dict)
    for i in ids:
        e = ex[i]
        if e.get("family") and e["family"].get("size", 1) > 1:
            fam[(e["family"]["id"], e["render"])][e["family"]["variant"]] = i
        elif e.get("flip"):
            fam[(e["flip"]["pair"], e["render"])][0 if e["flip"]["role"] == "base" else 1] = i
    n = both = base_ok = stuck = var_ok = 0
    for members in fam.values():
        if 0 not in members or members[0] not in rows:
            continue
        b = ex[members[0]]
        if not b["answerable"]:
            continue
        gold_b = b["candidates"][b["gold"]]["name"]
        rb = rows[members[0]]
        for v, i in members.items():
            if v == 0 or i not in rows or not ex[i]["answerable"]:
                continue
            if ex[i]["candidates"][ex[i]["gold"]]["name"] == gold_b:
                continue                                      # not a decision flip
            rv = rows[i]
            n += 1
            base_ok += rb["correct"]
            var_ok += rv["correct"]
            both += rb["correct"] and rv["correct"]
            stuck += rb["correct"] and rv.get("pred_label") == gold_b
    if not n:
        return {"pairs": 0}
    return {"pairs": n, "base_acc": base_ok / n, "variant_acc": var_ok / n, "both_correct": both / n,
            "stuck_rate": stuck / max(1, base_ok), "stuck_unconditional": stuck / n}


def metrics(ex: dict, rows: dict) -> dict:
    out = {}
    for cd in ("raw", "feat"):
        ids = [i for i, e in ex.items() if cond(e) == cd and i in rows]
        if not ids:
            continue
        R = [rows[i] for i in ids]
        by_track = defaultdict(list)
        for i in ids:
            if ex[i].get("qtype", "choice") != "multi":
                by_track[ex[i]["track"]].append(rows[i])
        c = {"by_track": {t: acc(v) for t, v in sorted(by_track.items())}}
        sel = lambda f: [rows[i] for i in ids if f(ex[i])]  # noqa: E731
        v1 = lambda e: e["id"].startswith(("ktb-", "ktv-"))  # noqa: E731
        ch = lambda e: e.get("qtype", "choice") == "choice"  # noqa: E731
        tr1 = {t: acc(sel(lambda e, t=t: v1(e) and ch(e) and e["track"] == t)) for t in V1_TRACKS}
        c["v1_macro_top1"] = float(np.mean([x["top1"] for x in tr1.values() if x["n"]])) if any(x["n"] for x in tr1.values()) else None
        c["v1_by_track"] = tr1
        cap = {t: acc(sel(lambda e, t=t: "capability" in e["tags"] and e["track"] == t)) for t in CAP_TRACKS}
        c["capability"] = cap
        c["capability_macro"] = float(np.mean([x["top1"] for x in cap.values() if x["n"]])) if any(x["n"] for x in cap.values()) else None
        c["capability_by_qtype"] = {t: {q: acc(sel(lambda e, t=t, q=q: "capability" in e["tags"] and e["track"] == t and e.get("qtype", "choice") == q))
                                        for q in ("choice", "noul", "score")} for t in CAP_TRACKS}
        c["capability_by_scenario"] = {s: acc(sel(lambda e, s=s: "capability" in e["tags"] and s in e["tags"])) for s in ("seen_scenario", "unseen_scenario")}
        c["capability_by_slice"] = {s: acc(sel(lambda e, s=s: "capability" in e["tags"] and e["slice"] == s))
                                    for s in sorted({e["slice"] for e in ex.values() if "capability" in e["tags"]})}
        c["hops"] = {h: acc(sel(lambda e, h=h: h in e["tags"])) for h in ("hops_2", "hops_3", "hops_4")}
        tag = lambda t: (lambda e: t in e["tags"])  # noqa: E731
        uc = lambda e: v1(e) and ch(e) and "unseen_combination" in e["tags"]  # noqa: E731
        c["v1_slices"] = {t: acc(sel(lambda e, t=t: v1(e) and ch(e) and t in e["tags"]))
                          for t in ("hard_negative", "constraint_flip", "unseen_combination", "seen_combination", "unseen_technology", "large_k", "unseen_family")}
        c["unseen_combination_by_size"] = {str(n): acc(sel(lambda e, n=n: uc(e) and len(e["meta"]["requirements"]) == n)) for n in (1, 2, 3)}
        c["unseen_combination_by_table"] = {s: acc(sel(lambda e, s=s: uc(e) and e["slice"] == s)) for s in sorted({e["slice"] for e in ex.values() if uc(e)})}
        c["r3_tables"] = {"catalog_unseen_combination": acc(sel(lambda e: "catalog" in e["tags"] and e["answerable"])),
                          "table_family_unseen_combination": acc(sel(tag("table_family")))}
        c["noul_swe_involved"] = acc(sel(lambda e: e["slice"] == "swe_involved"))
        c["flips_v1"] = pairs(ex, rows, [i for i in ids if v1(ex[i])])
        c["families_tables"] = pairs(ex, rows, [i for i in ids if not v1(ex[i]) and "capability" not in ex[i]["tags"]])
        c["families_capability"] = pairs(ex, rows, [i for i in ids if "capability" in ex[i]["tags"]])
        c["families_all"] = pairs(ex, rows, ids)
        hn = [r for r in R if "margin_logit" in r]
        v1c = [rows[i] for i in ids if v1(ex[i]) and "margin_logit" in rows[i]]
        bt = defaultdict(list)
        for i in ids:
            if v1(ex[i]) and "margin_logit" in rows[i]:
                bt[ex[i]["track"]].append(rows[i])
        c["v1_macro_margin_logit"] = float(np.mean([np.mean([r["margin_logit"] for r in v]) for v in bt.values()])) if bt else None
        c["v1_macro_gold_vs_hardest"] = float(np.mean([np.mean([r["gold_vs_hardest"] for r in v]) for v in bt.values()])) if bt else None
        c["all_margin_logit"] = float(np.mean([r["margin_logit"] for r in hn])) if hn else None
        una = [rows[i] for i in ids if not ex[i]["answerable"] and "abstain" in rows[i]]
        una1 = [rows[i] for i in ids if not ex[i]["answerable"] and "abstain" in rows[i] and v1(ex[i])]
        c["ood"] = {"n": len(una), "false_acceptance": float(np.mean([not r["abstain"] for r in una])) if una else None,
                    "v1_n": len(una1), "v1_false_acceptance": float(np.mean([not r["abstain"] for r in una1])) if una1 else None}
        mu = [r["multi"] for r in R if "multi" in r]
        if mu:
            tp, fp, fn = (sum(x[i] for x in mu) for i in range(3))
            c["multi_micro_f1"] = 2 * tp / max(1, 2 * tp + fp + fn)
        out[cd] = c
    return out


def paired(ex, a: dict, b: dict, f, B: int = 2000, seed: int = 0) -> dict | None:
    ids = [i for i in ex if cond(ex[i]) == "raw" and f(ex[i]) and i in a and i in b and "correct" in a[i] and "correct" in b[i]]
    if not ids:
        return None
    d = np.array([float(b[i]["correct"]) - float(a[i]["correct"]) for i in ids])
    rng = np.random.default_rng(seed)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]
    return {"n": len(ids), "delta": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="protected", choices=["protected", "val"])
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--baseline", default="v1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ex = {e["id"]: e for f in (f"{args.split}_v1_labeled.jsonl", f"{args.split}_r3.jsonl") for e in T.load_jsonl(D / f)}
    calib = json.loads((ROOT / "artifacts/krun-one-v1-text-final/calibration.json").read_text())
    Tm, thr = calib["temperature"]["tool"], calib["margin_threshold"]["tool"]
    stems = [args.split, f"{args.split}_v1_labeled", f"{args.split}_r3"]
    rows = {m: load(m, stems, ex, Tm, thr) for m in args.models}
    res = {"split": args.split, "models": {m: {"n_rows": len(rows[m]), **metrics(ex, rows[m])} for m in args.models}, "paired_vs_baseline": {}}
    if args.baseline in rows:
        for m in args.models:
            if m != args.baseline:
                res["paired_vs_baseline"][m] = {"v1_bench_choice": paired(ex, rows[args.baseline], rows[m], lambda e: e["id"].startswith(("ktb-", "ktv-")) and e.get("qtype", "choice") == "choice"),
                                                "capability": paired(ex, rows[args.baseline], rows[m], lambda e: "capability" in e["tags"])}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(res, indent=1))
    f3 = lambda v: "—" if v is None else f"{100 * v:.1f}"  # noqa: E731
    g = lambda d, *ks: (lambda x: x)(_get(d, ks))  # noqa: E731
    md = [f"# R3 technical report — {args.split} (KrunTechBench v1 with restored labels + R3 extension)", ""]
    for cd in ("raw", "feat"):
        ms = [m for m in args.models if cd in res["models"][m]]
        if not ms:
            continue
        md += [f"## {cd}: KrunTechBench v1 part (choice items; macro over the 10 tracks)", "",
               "| model | macro | " + " | ".join(V1_TRACKS) + " | hard neg. | flips | unseen comb. | seen comb. | unseen tech. | large k | unseen family | gold>hardest | margin | OOD FA | involved (noul) |",
               "|---|" + "---:|" * (len(V1_TRACKS) + 12)]
        for m in ms:
            c = res["models"][m][cd]
            md.append(f"| {m} | {f3(c['v1_macro_top1'])} | " + " | ".join(f3(c["v1_by_track"][t]["top1"]) for t in V1_TRACKS) + " | "
                      + " | ".join(f3(c["v1_slices"][t]["top1"]) for t in ("hard_negative", "constraint_flip", "unseen_combination", "seen_combination",
                                                                              "unseen_technology", "large_k", "unseen_family"))
                      + f" | {f3(c['v1_macro_gold_vs_hardest'])} | {c['v1_macro_margin_logit']:+.2f} | {f3(c['ood']['v1_false_acceptance'])} | {f3(c['noul_swe_involved']['top1'])} |")
        md += ["", f"## {cd}: unseen requirement combinations and counterfactual families", "",
               "| model | unseen 1 req | unseen 2 req | unseen 3 req | catalog (unseen sets) | table families (unseen sets) | v1 flips: both / stuck | table families: both / stuck | capability families: both / stuck |",
               "|---|---:|---:|---:|---:|---:|---|---|---|"]
        pf = lambda p: "—" if not p.get("pairs") else f"{100 * p['both_correct']:.1f} / {100 * p['stuck_rate']:.1f} (n={p['pairs']})"  # noqa: E731
        for m in ms:
            c = res["models"][m][cd]
            md.append(f"| {m} | " + " | ".join(f3(c["unseen_combination_by_size"][n]["top1"]) for n in ("1", "2", "3"))
                      + f" | {f3(c['r3_tables']['catalog_unseen_combination']['top1'])} | {f3(c['r3_tables']['table_family_unseen_combination']['top1'])} | "
                      + f"{pf(c['flips_v1'])} | {pf(c['families_tables'])} | {pf(c['families_capability'])} |")
        md += ["", f"## {cd}: capability tracks (R3 extension; all question types)", "",
               "| model | macro | " + " | ".join(CAP_TRACKS) + " | seen scenario | unseen scenario | choice | true/false | ordinal | 2 hops | 3 hops | 4 hops |",
               "|---|" + "---:|" * (len(CAP_TRACKS) + 9)]
        for m in ms:
            c = res["models"][m][cd]
            qt = {q: [c["capability_by_qtype"][t][q] for t in CAP_TRACKS] for q in ("choice", "noul", "score")}
            qa = {q: (sum(x["top1"] * x["n"] for x in v if x["n"]) / max(1, sum(x["n"] for x in v))) if any(x["n"] for x in v) else None for q, v in qt.items()}
            md.append(f"| {m} | {f3(c['capability_macro'])} | " + " | ".join(f3(c["capability"][t]["top1"]) for t in CAP_TRACKS)
                      + f" | {f3(c['capability_by_scenario']['seen_scenario']['top1'])} | {f3(c['capability_by_scenario']['unseen_scenario']['top1'])} | "
                      + " | ".join(f3(qa[q]) for q in ("choice", "noul", "score")) + " | " + " | ".join(f3(c["hops"][h]["top1"]) for h in ("hops_2", "hops_3", "hops_4")) + " |")
        md.append("")
    if res["paired_vs_baseline"]:
        md += [f"## Paired top-1 difference vs {args.baseline} (raw, bootstrap 95% CI)", "", "| model | KrunTechBench v1 choice items | capability items |", "|---|---|---|"]
        cell = lambda x: "—" if not x else f"{100 * x['delta']:+.1f} pp [{100 * x['ci95'][0]:+.1f}, {100 * x['ci95'][1]:+.1f}] (n={x['n']})"  # noqa: E731
        for m, p in res["paired_vs_baseline"].items():
            md.append(f"| {m} | {cell(p['v1_bench_choice'])} | {cell(p['capability'])} |")
    out.with_suffix(".md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def _get(d, ks):
    for k in ks:
        d = d[k]
    return d


if __name__ == "__main__":
    main()
