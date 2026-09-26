#!/usr/bin/env python
"""Export the chosen Krun One run as a self-contained inference artifact.

    .venv/bin/python scripts/export_model.py --run runs/A/final --eval results/eval_krun_one_A.json \
        --out artifacts/krun-one-v0

Temperatures (per family) and the abstention threshold come from the evaluation report (fit on val_fit);
the abstention score / operating point come from results/krun_one_decisions.json.
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

README = """# Krun One v0 (release candidate, closed beta — not published)

Decision model: `context + runtime options -> choice | abstain`, calibrated probabilities, one forward pass.
Fine-tuned (supervised, cross-entropy) from the public **Laya** checkpoint (`convaiinnovations/laya`, ModernBERT-large,
421M parameters, Apache-2.0). Architecture unchanged.

```python
from krun_one import KrunOne            # experiments/krun_one/src/krun_one/model.py (no ./laya needed)
m = KrunOne.from_pretrained("artifacts/krun-one-v0")          # cuda if available (bf16 autocast), else cpu fp32
m.decide("Find my meetings tomorrow",
         [{"name": "calendar_search", "description": "Search calendar events"},
          {"name": "send_email", "description": "Send an email"}])
# -> {"choice": "calendar_search" | None, "abstain": bool, "confidence": float, "probabilities": {...}, "task": "tool", ...}
m.decide("my card was charged twice", ["transaction_charged_twice", "card_payment_not_recognised", "pending_card_payment"])
```

* Options: strings (labels; humanized for the model) or `{"name", "description"}` (tools; shown as `name: description`).
* `task` family = `tool` when options carry descriptions, else `intent`; each family has its own temperature and
  abstention threshold (`krun_config.json`).
* `choice` is `null` when the model abstains (confidence below the family threshold) — "none of the options applies".
* Budgets: 1024 tokens per decision, 896 for instructions + options, 48 tokens per option (Laya serialization).

Files: `model.safetensors` (fp32 state dict), `config.json` (encoder architecture), `tokenizer/`, `krun_config.json`,
`metrics.json`, `provenance.json`.

Attribution / licenses of the training data: MASSIVE (Amazon, CC-BY-4.0), BANKING77 (PolyAI, CC-BY-4.0), xLAM
function-calling-60k (Salesforce, CC-BY-4.0, via argilla/apigen-function-calling), xlam-irrelevance-7.5k (MadeAgents,
CC-BY-4.0), ToolACE (Team-ACE, Apache-2.0). Base model: Laya (Convai Innovations, Apache-2.0). HWU64 and BFCL were not
used for training. See provenance.json. Limitations: see experiments/krun_one/README.md.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="dir with the fine-tuned model.safetensors")
    ap.add_argument("--eval", required=True, help="evaluation report of that model (results/eval_<tag>.json)")
    ap.add_argument("--train-summary", default=None, help="results/train_<run>.json")
    ap.add_argument("--decisions", default=str(ROOT / "results" / "krun_one_decisions.json"))
    ap.add_argument("--config", default=str(ROOT / "configs" / "krun_one_v0.yaml"))
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "krun-one-v0"))
    ap.add_argument("--name", default="krun-one-v0")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    rep = json.loads(Path(args.eval).read_text())
    dec = json.loads(Path(args.decisions).read_text())
    score, target = dec["abstention_score"], dec["operating_point_target"]
    thr = {fam: rep["thresholds"][fam][score][target] for fam in ("intent", "tool")}
    run = Path(args.run) if Path(args.run).is_absolute() else ROOT / args.run
    model, tok, laya_cfg = load_laya(device="cpu")
    model.load_state_dict(load_file(str(run / "model.safetensors")), strict=True)
    kcfg = build_krun_config(max_len=rep["max_len"], head_max_len=rep["head_max_len"], head_layers=laya_cfg["head_layers"],
                             n_act=len(laya_cfg["act_costs"]) + 1, temperatures=rep["temperatures"], score=score,
                             thresholds=thr, target=target, name=args.name)
    metrics = {"source_report": str(Path(args.eval).resolve().relative_to(ROOT)), "temperatures": rep["temperatures"],
               "thresholds": thr, "abstention_score": score, "operating_point_target": target,
               "sets": {s: {"core": r["quality"]["core"], "permutation": r["permutation"],
                            "abstention_all": {k: v for k, v in r["abstention"].get("all", {}).get(score, {}).items()}}
                        for s, r in rep["sets"].items()}}
    manifest = json.loads((PROJECT / cfg["data"]["dir"] / "manifest.json").read_text())
    train_summary = json.loads(Path(args.train_summary).read_text()) if args.train_summary else None
    if train_summary:
        train_summary.pop("log", None)
    prov = {"exported_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "base_checkpoint": laya_provenance(),
            "fine_tuned_weights": {"path": str(run / "model.safetensors"), "sha256": sha256_file(run / "model.safetensors")},
            "decisions": dec, "training": train_summary,
            "data": {"files": manifest["files"], "tool_sources": manifest["sources"]["tools"],
                     "intent_sources": "KrunBench v0 train (banking77, massive en-US/pt-PT/es-ES); HWU64 excluded"},
            "eval_report": str(Path(args.eval).resolve().relative_to(ROOT))}
    out = export_artifact(model, LAYA_DIR / "tokenizer", LAYA_DIR / "encoder" / "config.json", Path(args.out), kcfg, metrics,
                          prov, README)
    print(f"exported {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
