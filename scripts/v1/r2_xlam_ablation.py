"""xLAM ablation probe analysis: probe_with_xlam vs probe_without_xlam (identical short fine-tunes from v1-lr1e5).

    python scripts/v1/r2_xlam_ablation.py  -> results/v1/r2/XLAM_ABLATION.{json,md}
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("C", ROOT / "scripts/v1/r2_compare.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


def main():
    for v in ("with_xlam", "without_xlam"):
        C.RUNS[v] = ROOT / f"runs/v1/ft/probe_{v}"
        C.TOOL_REG[v] = ROOT / f"runs/v1/ft/probe_{v}/regression_toolace_v0/signals_probe_{v}.jsonl.gz"
    A, B = C.model_block("without_xlam"), C.model_block("with_xlam")
    recs = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "runs/v1/ft/regression_toolace_v0/records.jsonl")}
    hold = lambda i: recs[i]["metadata"]["v0_file"] == "holdout_tools"  # noqa: E731
    big = lambda i: recs[i]["metadata"]["v0_file"] == "holdout_tools" and len(recs[i]["candidates"]) >= 16  # noqa: E731
    nd = lambda i: recs[i]["metadata"]["v0_file"] == "holdout_tools" and "native" in str(recs[i]["candidate_set"]["strategy"])  # noqa: E731
    K = C.K
    rows = {
        "toolace_holdout": C.paired(A["toolace"]["per"], B["toolace"]["per"], hold),
        "toolace_holdout_k>=16": C.paired(A["toolace"]["per"], B["toolace"]["per"], big),
        "toolace_holdout_native+hard": C.paired(A["toolace"]["per"], B["toolace"]["per"], nd),
        "V_tool": K.paired_delta(A["rows"], B["rows"], lambda r: r["family"] == "tool"),
        "V_near_duplicate": K.paired_delta(A["rows"], B["rows"], lambda r: "near_duplicate" in r["slices"]),
        "V_k>=32": K.paired_delta(A["rows"], B["rows"], lambda r: r["k"] >= 32),
        "V_cross_language": K.paired_delta(A["rows"], B["rows"], lambda r: "cross_language" in r["slices"]),
    }
    ood = {cov: {"without": C.at_cov(A, "tool", cov, "ood_false_acceptance"), "with": C.at_cov(B, "tool", cov, "ood_false_acceptance"),
                 "unseen_without": C.at_cov(A, "tool", cov, "unseen_tool_false_acceptance"),
                 "unseen_with": C.at_cov(B, "tool", cov, "unseen_tool_false_acceptance")} for cov in C.COVS}
    out = {"paired_with_minus_without": rows, "tool_ood_curves": ood,
           "toolace_acc": {"without": A["toolace"]["acc"], "with": B["toolace"]["acc"]},
           "tool_aurc": {"without": A["curves"]["tool"]["aurc"], "with": B["curves"]["tool"]["aurc"]}}
    p = ROOT / "results/v1/r2"
    (p / "XLAM_ABLATION.json").write_text(json.dumps(out, indent=1, default=str))
    f = lambda x: "—" if x is None else f"{100 * x:.1f}"  # noqa: E731
    md = ["# xLAM ablation probe (R2)\n",
          "Two identical short fine-tunes (1 epoch over 210k decisions: 150k new tool decisions + the same 60k base replay)",
          "from the v1-lr1e5 checkpoint, encoder LR 1e-5 / head 5e-5, same seed and order. Only difference: the 150k tool",
          "decisions are sampled from ALL Tool Corpus v2 decisions (64k xLAM-sourced) vs from the non-xLAM ones (0 xLAM).",
          "A probe, not two full trainings: it answers *direction and size* of the xLAM effect under a fixed budget.\n",
          "| slice (with − without) | n | Δ accuracy (pp) | 95% CI |", "|---|---:|---:|---|"]
    for k, v in rows.items():
        if v:
            md.append(f"| {k} | {v['n']} | {100 * v['delta']:+.2f} | [{100 * v['ci95'][0]:+.2f}, {100 * v['ci95'][1]:+.2f}] |")
    md += ["", f"ToolACE accuracy without / with xLAM: holdout {f(A['toolace']['acc'].get('holdout_tools'))} / "
               f"{f(B['toolace']['acc'].get('holdout_tools'))}; val_tools {f(A['toolace']['acc'].get('val_tools'))} / "
               f"{f(B['toolace']['acc'].get('val_tools'))}", "",
           "| tool coverage | OOD FA without / with | unseen-tool FA without / with |", "|---:|---|---|"]
    for cov, v in ood.items():
        md.append(f"| {int(cov * 100)}% | {f(v['without'])} / {f(v['with'])} | {f(v['unseen_without'])} / {f(v['unseen_with'])} |")
    md.append(f"\nTool AURC without / with: {out['tool_aurc']['without']:.4f} / {out['tool_aurc']['with']:.4f}")
    (p / "XLAM_ABLATION.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
