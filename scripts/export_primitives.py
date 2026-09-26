#!/usr/bin/env python
"""KRUN-011: export the selected multi-primitive candidate as a self-contained artifact (never overwrites v0).

    .venv/bin/python scripts/export_primitives.py --variant M50_CE --out artifacts/krun-one-v0.2-candidate

Choice temperatures / abstention thresholds come from the candidate's choice evaluation (results/eval_v02_<V>.json,
fit on val_fit, same rule as v0: top1_top2_margin @ 0.98); noul / score temperatures from its primitives evaluation
(fit on val_fit); score confidence definition from results/primitives_decisions.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

import krun_one  # noqa: F401
from safetensors.torch import load_file

from krun_one.export import build_krun_config, export_artifact
from krun_one.laya_adapter import LAYA_DIR, laya_provenance, load_laya, sha256_file

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]

README = """# Krun One multi-primitive candidate (NOT promoted to production)

One decision model, three primitives, one forward pass per request:

```python
from krun_one import KrunOne                    # experiments/krun_one/src/krun_one/model.py (no ./laya needed)
m = KrunOne.from_pretrained("artifacts/<this artifact>")
m.decide_questions(
    "Customer says this is the third time exports failed and wants a human immediately.",
    {"department": {"type": "choice", "options": ["billing", "support", "sales"]},
     "needs_human": {"type": "noul", "instructions": "Is the customer asking for human assistance?"},
     "severity": {"type": "score", "instructions": "How severe is the reported issue?",
                  "levels": ["Minor issue", "Feature degraded", "Blocking issue"]}})
# department  -> {"type": "choice", "choice": ..., "confidence": ..., "probabilities": {...}, "abstain": ...}
# needs_human -> {"type": "noul", "noul": p_true}
# severity    -> {"type": "score", "score": Σ i·p_i, "confidence": ..., "legend": {...}, "probabilities": {...}}
```

* choice: unchanged from krun-one-v0 (`decide` / `decide_batch` still work).
* noul: probability that the proposition holds (options rendered `[false, true]`, temperature `T_noul`).
* score: levels are ORDERED (never permuted); `score` is the expected level, not the argmax.

Continued supervised fine-tuning of Krun One v0 on choice replay + noul + score (see provenance.json).
Base model: Laya (Convai Innovations, Apache-2.0). Training data licenses: BANKING77, MASSIVE, xLAM,
xlam-irrelevance (CC-BY-4.0), ToolACE (Apache-2.0), Civil Comments (CC0-1.0), HelpSteer2 (NVIDIA, CC-BY-4.0),
Measuring Hate Speech (UC Berkeley D-Lab, CC-BY-4.0). BoolQ / SST-5 / HWU64 / BFCL: evaluation only.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--round", default="v03", help="run/tag prefix; config = configs/krun_one_v0_<n>.yaml")
    ap.add_argument("--out", default=None, help="default artifacts/krun-one-<round>-candidate")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    ver = args.round.replace("v0", "v0.")
    args.name = args.name or f"krun-one-{ver}-candidate"
    args.out = args.out or str(ROOT / "artifacts" / args.name)
    if Path(args.out).resolve() == (ROOT / "artifacts" / "krun-one-v0").resolve():
        raise SystemExit("refusing to overwrite krun-one-v0")
    tag = f"{args.round}_{args.variant}"
    cfg_file = ROOT / "configs" / f"krun_one_{args.round.replace('v0', 'v0_')}.yaml"
    dec = json.loads((ROOT / "results" / "primitives_decisions.json").read_text())
    v0dec = json.loads((ROOT / "results" / "krun_one_decisions.json").read_text())
    if dec["selected_variant"] != args.variant or dec.get("round", args.round) != args.round:
        raise SystemExit(f"{args.variant} is not the selected variant ({dec['selected_variant']})")
    crep = json.loads((ROOT / "results" / f"eval_{tag}.json").read_text())
    prep = json.loads((ROOT / "results" / "primitives_candidates" / f"eval_{tag}.json").read_text())
    score, target = v0dec["abstention_score"], v0dec["operating_point_target"]
    thr = {fam: crep["thresholds"][fam][score][target] for fam in ("intent", "tool")}
    run = ROOT / "runs" / tag / "final"
    model, tok, laya_cfg = load_laya(device="cpu")
    model.load_state_dict(load_file(str(run / "model.safetensors")), strict=True)
    ptemps = {"noul": prep["noul"]["temperature"], "score": prep["score"]["temperature"]}
    kcfg = build_krun_config(max_len=crep["max_len"], head_max_len=crep["head_max_len"],
                             head_layers=laya_cfg["head_layers"], n_act=len(laya_cfg["act_costs"]) + 1,
                             temperatures=crep["temperatures"], score=score, thresholds=thr, target=target, name=args.name,
                             primitive_temperatures=ptemps, score_confidence=dec["score_confidence"])
    final = json.loads((ROOT / "results" / "primitives_final.json").read_text()) \
        if (ROOT / "results" / "primitives_final.json").exists() else None
    metrics = {"choice_report": f"results/eval_{tag}.json", "primitives_report": f"results/primitives_candidates/eval_{tag}.json",
               "temperatures": kcfg["temperature"], "thresholds": thr,
               "choice": {s: r["quality"]["core"] for s, r in crep["sets"].items()},
               "noul": {p: prep["noul"][p]["calibrated"]["in_domain"] for p in ("val_eval", "test") if p in prep["noul"]},
               "score": {p: prep["score"][p]["calibrated"]["in_domain"] for p in ("val_eval", "test") if p in prep["score"]},
               "mixed": prep.get("mixed"), "final_table": "results/primitives_final_table.md", "final": final}
    for q in ("noul", "score"):
        for p in list(metrics[q]):
            metrics[q][p] = {k: v for k, v in metrics[q][p].items() if k != "reliability"}
    cfg = yaml.safe_load(cfg_file.read_text())
    train = json.loads((ROOT / "results" / "primitives_candidates" / f"train_{tag}.json").read_text())
    readme = README + f"\nScore confidence: `{dec['score_confidence']}` (see krun_config.json).\n"
    train.pop("log", None)
    pman = json.loads((PROJECT / cfg["data"]["primitives_dir"] / "manifest.json").read_text())
    v0man = json.loads((ROOT / "data" / "v0" / "manifest.json").read_text())
    prov = {"exported_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "version": args.name, "status": "candidate — not promoted, not published",
            "base_checkpoint": laya_provenance(),
            "initialized_from": {"artifact": "artifacts/krun-one-v0",
                                 "sha256": sha256_file(ROOT / "artifacts" / "krun-one-v0" / "model.safetensors")},
            "fine_tuned_weights": {"path": str(run / "model.safetensors"), "sha256": sha256_file(run / "model.safetensors")},
            "decisions": dec, "training": train,
            "data": {"choice_replay": {"files": v0man["files"], "tool_sources": v0man["sources"]["tools"]},
                     "primitives": {"sources": pman["config"]["sources"], "files": pman["files"]},
                     "audits": ["results/noul_dataset_audit.md", "results/score_dataset_audit.md"]}}
    out = export_artifact(model, LAYA_DIR / "tokenizer", LAYA_DIR / "encoder" / "config.json", Path(args.out), kcfg,
                          metrics, prov, readme)
    print(f"exported {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
