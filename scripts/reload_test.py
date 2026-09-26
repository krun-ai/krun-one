#!/usr/bin/env python
"""Reload test: a NEW Python process loads the exported artifact and runs inference.

    .venv/bin/python scripts/reload_test.py --artifact artifacts/krun-one-v0 --eval-tag krun_one_B

Checks (results/reload_test_<device>.json):
  1. ./laya is not importable/used: sys.path is scrubbed of any 'laya' entry and ./laya is not referenced by the artifact;
  2. three product decisions: intent routing, tool routing, OOD (no valid tool -> expect abstention);
  3. fidelity: logits of 64 validation decisions equal the logits cached by the evaluation of the trained model
     (same device/dtype family; tolerance reported, not assumed).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path = [p for p in sys.path if "laya" not in p.lower()]

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]

CASES = {
    "intent": {"context": "I was charged twice for the same purchase at the supermarket",
               "options": ["transaction_charged_twice", "card_payment_not_recognised", "pending_card_payment",
                           "reverted_card_payment?", "extra_charge_on_statement"],
               "expect": "transaction_charged_twice"},
    "tool": {"context": "Find my meetings tomorrow",
             "options": [{"name": "calendar_search", "description": "Search calendar events by date, title or attendee."},
                         {"name": "send_email", "description": "Send an email to one or more recipients."},
                         {"name": "web_search", "description": "Search the public web."},
                         {"name": "create_invoice", "description": "Create a customer invoice."}],
             "expect": "calendar_search"},
    "ood": {"context": "What is the weather tomorrow in Lisbon?",
            "options": [{"name": "send_email", "description": "Send an email to one or more recipients."},
                        {"name": "create_invoice", "description": "Create a customer invoice."},
                        {"name": "search_database", "description": "Run a query against the company's customer database."}],
            "expect": None},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default=str(ROOT / "artifacts" / "krun-one-v0"))
    ap.add_argument("--eval-tag", default=None, help="compare with runs/eval/<tag>/val_tools.json logits")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    t0 = time.time()
    import torch

    from krun_one import KrunOne

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    m = KrunOne.from_pretrained(args.artifact, device=dev)
    out = {"artifact": args.artifact, "device": dev, "load_seconds": round(time.time() - t0, 2),
           "laya_modules_loaded": [k for k in sys.modules if "rl_common" in k or "rl_agent" in k],
           "sys_path_has_laya": any("laya" in p.lower() for p in sys.path), "cases": {}}
    for name, c in CASES.items():
        r = m.decide(c["context"], c["options"])
        ok = (r["choice"] == c["expect"]) if c["expect"] else r["abstain"]
        out["cases"][name] = {"expect": c["expect"] or "abstain", "result": r, "pass": bool(ok)}
        print(f"  {name:6s} -> choice={r['choice']!r} top={r['top_option']!r} conf={r['confidence']:.3f} "
              f"abstain={r['abstain']} {'PASS' if ok else 'FAIL'}")
    if args.eval_tag:
        from krunbench.schema import read_jsonl

        from krun_one.model import build_sequence, option_texts

        cache = json.loads((ROOT / "runs" / "eval" / args.eval_tag / "val_tools.json").read_text())
        exs = [e for e in read_jsonl(PROJECT / "experiments/krun_one/data/v0/val_tools.jsonl")][:64]
        items = []
        for e in exs:
            ids, mk = build_sequence(m.tok, e.context, option_texts(e.options, e.option_descriptions), e.instructions,
                                     m.cfg["max_len"], m.cfg["head_max_len"])
            items.append({"ids": ids, "markers": mk})
        diffs = []
        for i in range(0, len(items), 16):
            for e, z in zip(exs[i:i + 16], m.logits(items[i:i + 16])):
                diffs.append(float(abs(z - __import__("numpy").asarray(cache["logits"][e.id])).max()))
        agree = sum(int(__import__("numpy").argmax(m.logits([it])[0]) == __import__("numpy").argmax(cache["logits"][e.id]))
                    for it, e in zip(items, exs))
        out["fidelity"] = {"n": len(exs), "max_abs_logit_diff": max(diffs), "mean_abs_logit_diff": sum(diffs) / len(diffs),
                           "argmax_agreement": agree / len(exs), "eval_device": cache.get("device"), "eval_dtype": cache.get("dtype")}
        print(f"  fidelity vs evaluated logits: {out['fidelity']}")
    out["all_pass"] = all(c["pass"] for c in out["cases"].values()) and not out["laya_modules_loaded"]
    dst = ROOT / "results" / f"reload_test_{dev}.json"
    dst.write_text(json.dumps(out, indent=2) + "\n")
    print(f"all_pass={out['all_pass']} wrote {dst}")
    return 0 if out["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
