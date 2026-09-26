#!/usr/bin/env python
"""Error analysis on val_eval (never on the final sets) -> results/error_analysis_<version>.md/.json

    .venv/bin/python scripts/error_analysis.py --tag krun_one_A --baseline laya --version v0

Categories (a decision can fall into several):
  semantic_confusion     answerable, wrong, predicted option is among the gold's top-5 kNN-confusion/similar labels
                         (intents) or near-duplicate/description-sharing tools (char TF-IDF >= 0.5)
  option_count           error rate by number of options
  multilingual           error rate by language (MASSIVE en-US / pt-PT / es-ES)
  tool_description       tool errors on description-dependent decisions (gold name shares no word with the query)
  ood_false_accept       unanswerable decisions accepted at the primary operating point, by OOD kind
  ambiguous              wrong with top1-top2 margin < 0.1, or the context occurs with a different gold elsewhere
  permutation_flip       consistency groups whose rotations disagree
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

import krun_one  # noqa: F401
from bakeoff.calibration import confidence_scores
from bakeoff.data import split_validation
from krunbench.splits import normalize_text
from krun_one.calibration import family, probs
from krun_one.evaluation import VALIDATION_SETS, load_set, tool_slices

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]


def analyse(tag: str, cfg: dict) -> dict:
    rep = json.loads((ROOT / "results" / f"eval_{tag}.json").read_text())
    temps, thr = rep["temperatures"], rep["thresholds"]
    dec = {}
    for f in ("krun_one_decisions.json", "krun_one_decisions_v0.json"):
        if (ROOT / "results" / f).exists():
            dec = json.loads((ROOT / "results" / f).read_text())
            break
    score = dec.get("abstention_score", "top1_probability")
    target = dec.get("operating_point_target", f"{cfg['evaluation']['target_selective_accuracy'][0]:.2f}")
    out: dict = {"tag": tag, "score": score, "target": target, "sets": {}}
    for s in VALIDATION_SETS:
        exs = load_set(s, PROJECT, PROJECT / cfg["data"]["dir"])
        _, ev = split_validation(exs, cfg["evaluation"]["val_fit_seed"], cfg["evaluation"]["val_fit_fraction"])
        logits = json.loads((ROOT / "runs" / "eval" / tag / f"{s}.json").read_text())["logits"]
        ps = probs(ev, logits, temps)
        sc = confidence_scores(ps)[score]
        gold_by_ctx = defaultdict(set)
        for e in exs:
            if e.answer is not None:
                gold_by_ctx[normalize_text(e.context)].add(e.answer)
        cats: dict[str, Counter] = defaultdict(Counter)
        tot: dict[str, Counter] = defaultdict(Counter)
        pairs, examples = Counter(), defaultdict(list)
        for e, p, c in zip(ev, ps, sc):
            fam = family(e)
            thr_e = thr[fam][score][target]
            pred = e.options[int(np.argmax(p))]
            if e.answer is None:
                kind = e.metadata.get("ood_kind") or e.suite
                tot["ood_false_accept"][kind] += 1
                if c >= thr_e:
                    cats["ood_false_accept"][kind] += 1
                    if len(examples[f"ood:{kind}"]) < 4:
                        examples[f"ood:{kind}"].append({"context": e.context[:160], "accepted": pred, "conf": round(float(c), 3),
                                                        "options": e.options[:6]})
                continue
            if e.suite not in ("core", "distractors"):
                continue
            wrong = pred != e.answer
            k = str(len(e.options))
            tot["option_count"][f"{fam}:k{k}"] += 1
            cats["option_count"][f"{fam}:k{k}"] += wrong
            if e.language:
                tot["multilingual"][e.language] += 1
                cats["multilingual"][e.language] += wrong
            if fam == "tool":
                for sl in tool_slices(e):
                    tot["tool_slices"][sl] += 1
                    cats["tool_slices"][sl] += wrong
            if not wrong:
                continue
            srt = np.sort(p)[::-1]
            if srt[0] - srt[1] < 0.1 or len(gold_by_ctx[normalize_text(e.context)]) > 1:
                cats["ambiguous"][fam] += 1
            if fam == "tool":
                sl = tool_slices(e)
                if "near_duplicate_distractor" in sl:
                    cats["semantic_confusion"]["tool:near_duplicate"] += 1
                if "description_dependent" in sl:
                    cats["tool_description"]["description_dependent"] += 1
            else:
                cats["semantic_confusion"][f"intent:{e.provenance.dataset}"] += 1
                pairs[f"{e.answer} -> {pred}"] += 1
            key = f"wrong:{fam}"
            if len(examples[key]) < 8:
                examples[key].append({"context": e.context[:160], "gold": e.answer, "pred": pred, "k": len(e.options),
                                      "p_pred": round(float(np.max(p)), 3), "language": e.language})
        perm = [e for e in ev if e.suite == "permutation" and e.generation.consistency_group]
        flips = defaultdict(set)
        for e in perm:
            flips[e.generation.consistency_group].add(e.options[int(np.argmax(logits[e.id]))])
        cats["permutation_flip"]["groups_disagreeing"] = sum(len(v) > 1 for v in flips.values())
        tot["permutation_flip"]["groups_disagreeing"] = len(flips)
        out["sets"][s] = {"errors": {k: dict(v) for k, v in cats.items()}, "totals": {k: dict(v) for k, v in tot.items()},
                          "top_confusions": pairs.most_common(15), "examples": examples}
    return out


def md(res: dict, base: dict | None) -> str:
    L = [f"# Error analysis — {res['tag']} (val_eval only)", "",
         f"Abstention score `{res['score']}`, operating point {res['target']}. Rates = errors / decisions of that slice."
         + (f" Baseline column: `{base['tag']}` on the same decisions." if base else ""), ""]
    for s, r in res["sets"].items():
        L += [f"## {s}", "", "| category | slice | errors / n | rate |" + (" baseline rate |" if base else ""),
              "|---|---|---:|---:|" + ("---:|" if base else "")]
        b = base["sets"][s] if base else None
        for cat in ("option_count", "multilingual", "tool_slices", "ood_false_accept", "permutation_flip"):
            for sl, n in sorted(r["totals"].get(cat, {}).items()):
                e = r["errors"].get(cat, {}).get(sl, 0)
                row = f"| {cat} | {sl} | {e} / {n} | {100 * e / n:.1f}% |" if n else ""
                if b and n:
                    bn = b["totals"].get(cat, {}).get(sl, 0)
                    be = b["errors"].get(cat, {}).get(sl, 0)
                    row += f" {100 * be / bn:.1f}% |" if bn else " N/A |"
                if row:
                    L.append(row)
        for cat in ("semantic_confusion", "tool_description", "ambiguous"):
            for sl, e in sorted(r["errors"].get(cat, {}).items()):
                L.append(f"| {cat} (count of errors) | {sl} | {e} | |" + (" |" if base else ""))
        if r["top_confusions"]:
            L += ["", "Top intent confusions (gold -> predicted): " + "; ".join(f"`{a}` ×{n}" for a, n in r["top_confusions"][:10])]
        for k, xs in r["examples"].items():
            L += ["", f"<details><summary>examples {k}</summary>", "", "```json", json.dumps(xs, ensure_ascii=False, indent=1), "```",
                  "</details>"]
        L.append("")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--baseline", default="laya")
    ap.add_argument("--version", default="v0")
    ap.add_argument("--config", default=str(ROOT / "configs" / "krun_one_v0.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    res = analyse(args.tag, cfg)
    base = analyse(args.baseline, cfg) if args.baseline else None
    (ROOT / "results" / f"error_analysis_{args.version}.json").write_text(json.dumps({"model": res, "baseline": base}, indent=1,
                                                                                    ensure_ascii=False) + "\n")
    (ROOT / "results" / f"error_analysis_{args.version}.md").write_text(md(res, base))
    print(f"wrote results/error_analysis_{args.version}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
