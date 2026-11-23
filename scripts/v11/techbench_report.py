"""Krun One v1.1 — technical metrics from eval dumps (offline): KrunTechBench / KrunTechVal / Tetris representation.

    uv run python scripts/v11/techbench_report.py --set data/v11/techbench/val.jsonl --models v1 r2_s1 r2_s2 --out results/v11/techbench/val

Each model name is a directory under results/v11/eval/. Conditions: ``raw`` = candidates only (the headline; SWE
decisions have a single rendering and count as raw), ``feat`` = candidates + explicit consequence (diagnostic).
Temperature / abstention threshold: the frozen v1 ``tool`` calibration, for every model (a promoted model would be
recalibrated; AUROC is threshold-free).
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
TAGS = ["easy", "hard", "hard_negative", "constraint_flip", "unseen_combination", "seen_combination", "unseen_technology",
        "large_k", "unseen_family"]


def cond(e: dict) -> str:
    return "raw" if e["render"] == "raw" or e["track"] == "software_repair" else "feat"


def load_rows(model: str, stem: str, ex: dict, Tm: float, thr: float) -> dict[str, dict]:
    rows = {}
    with gzip.open(EVAL / model / f"tech_{stem}.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            e = ex.get(r["id"])
            if e is None:
                continue
            qt = e.get("qtype", "choice")
            if qt == "choice":
                rows[r["id"]] = probe.item_signals(e, r["z"], Tm, thr)
            elif qt == "noul":
                if e["gold"] is not None:      # the R1 / R2 build stored no label for swe_involved: not scored
                    rows[r["id"]] = {"id": r["id"], "noul_correct": int(np.argmax(r["z"])) == e["gold"]}
            else:
                pred = 1 / (1 + np.exp(-np.asarray(r["apply"]))) > 0.5
                gold = np.zeros(len(pred), bool)
                gold[e["gold_multi"]] = True
                rows[r["id"]] = {"id": r["id"], "multi": [int((pred & gold).sum()), int((pred & ~gold).sum()), int((~pred & gold).sum())]}
    return rows


def macro(cells: dict, key: str = "top1") -> float | None:
    v = [c[key] for c in cells.values() if c.get("n")]
    return float(np.mean(v)) if v else None


def metrics(rows: dict, ex: dict) -> dict:
    out = {}
    for cd in ("raw", "feat"):
        ch = [r for i, r in rows.items() if cond(ex[i]) == cd and "pred" in r]
        ans = [r for r in ch if r["answerable"]]
        if not ans:
            continue
        by_track = probe.group(ans, ex, ("track",))
        c = {"overall": probe.aggregate(ans), "by_track": by_track, "macro_top1": macro(by_track),
             "macro_gold_vs_hardest": macro(by_track, "gold_vs_hardest"), "macro_margin_logit": macro(by_track, "margin_logit"),
             "by_slice": probe.group(ans, ex, ("slice",)), "by_tag": {}, "flip": {}, "ood": {}}
        for t in TAGS:
            sub = [r for r in ans if t in ex[r["id"]]["tags"]]
            if sub:
                c["by_tag"][t] = {**probe.aggregate(sub), "macro_top1": macro(probe.group(sub, ex, ("track",)))}
        by_id = {r["id"]: r for r in ans}
        fl = probe.flip_metrics(by_id, {i: {**ex[i], "slice": ex[i]["track"]} for i in by_id})
        tot = sum(v["pairs"] for v in fl.values())
        if tot:
            c["flip"] = {"by_track": fl, "pairs": tot, **{k: sum(v[k] * v["pairs"] for v in fl.values()) / tot
                                                          for k in ("base_acc", "flip_acc", "both_correct", "stuck_on_base_answer")}}
        una = [r for r in ch if not r["answerable"]]
        if una:
            c["ood"] = {"n": len(una), "false_acceptance": float(np.mean([not r["abstain"] for r in una])),
                        "answerable_coverage": float(np.mean([not r["abstain"] for r in ans])),
                        "auroc_margin": probe.auroc(np.array([r["margin_T"] for r in ans]), np.array([r["margin_T"] for r in una]))}
        out[cd] = c
    nl = [r["noul_correct"] for r in rows.values() if "noul_correct" in r]
    if nl:
        out["swe_noul_accuracy"] = {"n": len(nl), "acc": float(np.mean(nl))}
    mu = [r["multi"] for r in rows.values() if "multi" in r]
    if mu:
        tp, fp, fn = (sum(x[i] for x in mu) for i in range(3))
        out["swe_multi_micro_f1"] = {"n": len(mu), "f1": 2 * tp / max(1, 2 * tp + fp + fn)}
    return out


def paired(a: dict, b: dict, ex: dict, cd: str, reps: int = 1000) -> dict:
    """b - a on top-1 over the answerable choice items both models scored, bootstrap over problems."""
    ids = sorted(i for i in set(a) & set(b) if cond(ex[i]) == cd and a[i].get("answerable") and "correct" in a[i] and "correct" in b[i])
    if not ids:
        return {}
    d = np.array([float(b[i]["correct"]) - float(a[i]["correct"]) for i in ids])
    rng = np.random.default_rng(0)
    bs = rng.choice(d, (reps, len(d))).mean(1)
    return {"n": len(ids), "delta_top1": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline", default="v1")
    args = ap.parse_args()
    ex = {e["id"]: e for e in T.load_jsonl(ROOT / args.set)}
    stem = Path(args.set).stem
    calib = json.loads((ROOT / "artifacts/krun-one-v1-text-final/calibration.json").read_text())
    Tm, thr = calib["temperature"]["tool"], calib["margin_threshold"]["tool"]
    rows = {m: load_rows(m, stem, ex, Tm, thr) for m in args.models if (EVAL / m / f"tech_{stem}.jsonl.gz").exists()}
    res = {"set": args.set, "n_examples": len(ex), "models": {m: metrics(r, ex) for m, r in rows.items()}, "paired_vs_baseline": {}}
    if args.baseline in rows:
        for m, r in rows.items():
            if m != args.baseline:
                res["paired_vs_baseline"][m] = {cd: paired(rows[args.baseline], r, ex, cd) for cd in ("raw", "feat")}
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(res, indent=1))
    f3 = lambda v: "—" if v is None else f"{100 * v:.1f}"  # noqa: E731
    md = [f"# Technical metrics — `{args.set}` ({len(ex)} examples)", ""]
    for cd in ("raw", "feat"):
        ms = [m for m in rows if cd in res["models"][m]]
        if not ms:
            continue
        tracks = sorted({t for m in ms for t in res["models"][m][cd]["by_track"]})
        md += [f"## {cd}: top-1 by track (macro = mean over tracks)", "", "| model | macro | overall | " + " | ".join(tracks) + " |",
               "|---|---:|---:|" + "---:|" * len(tracks)]
        for m in ms:
            c = res["models"][m][cd]
            md.append(f"| {m} | {f3(c['macro_top1'])} | {f3(c['overall']['top1'])} | "
                      + " | ".join(f3(c["by_track"].get(t, {}).get("top1")) for t in tracks) + " |")
        md += ["", f"## {cd}: slices (top-1), separation and flips", "",
               "| model | " + " | ".join(TAGS) + " | gold > hardest | margin (logit) | flips both | flips stuck | OOD FA | OOD AUROC | entropy | ECE |",
               "|---|" + "---:|" * (len(TAGS) + 8)]
        for m in ms:
            c = res["models"][m][cd]
            md.append(f"| {m} | " + " | ".join(f3(c["by_tag"].get(t, {}).get("top1")) for t in TAGS)
                      + f" | {f3(c['macro_gold_vs_hardest'])} | {c['macro_margin_logit']:+.2f} | {f3(c['flip'].get('both_correct'))} | "
                        f"{f3(c['flip'].get('stuck_on_base_answer'))} | {f3(c['ood'].get('false_acceptance'))} | "
                        f"{'—' if not c['ood'] or c['ood']['auroc_margin'] is None else format(c['ood']['auroc_margin'], '.3f')} | "
                        f"{c['overall']['entropy']:.3f} | {c['overall']['ece_T']:.3f} |")
        md.append("")
    if res["paired_vs_baseline"]:
        md += [f"## Paired top-1 difference vs {args.baseline} (bootstrap 95% CI)", "", "| model | raw | feat |", "|---|---|---|"]
        for m, p in res["paired_vs_baseline"].items():
            cell = lambda x: "—" if not x else f"{100 * x['delta_top1']:+.1f} pp [{100 * x['ci95'][0]:+.1f}, {100 * x['ci95'][1]:+.1f}] (n={x['n']})"  # noqa: E731
            md.append(f"| {m} | {cell(p['raw'])} | {cell(p['feat'])} |")
    out.with_suffix(".md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
