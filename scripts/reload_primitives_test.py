#!/usr/bin/env python
"""KRUN-011 reload test: a NEW process loads the multi-primitive artifact (no ./laya) and answers
choice, noul, score and a mixed request; checks fidelity against the evaluated logits.

    .venv/bin/python scripts/reload_primitives_test.py --artifact artifacts/krun-one-v0.2-candidate --eval-tag v02_M50_CE
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path = [p for p in sys.path if "laya" not in p.lower()]

ROOT = Path(__file__).resolve().parents[1]

MIXED_CTX = "Customer says this is the third time exports failed and wants a human immediately."
MIXED = {
    "department": {"type": "choice", "options": ["billing", "support", "sales"]},
    "needs_human": {"type": "noul", "instructions": "Is the customer asking for human assistance?"},
    "severity": {"type": "score", "instructions": "How severe is the reported issue?",
                 "levels": ["Minor issue", "Feature degraded", "Blocking issue"]},
}
CASES = [  # (name, context, questions, check)
    ("choice", "I was charged twice for the same purchase at the supermarket",
     {"q": {"type": "choice", "options": ["transaction_charged_twice", "card_payment_not_recognised", "pending_card_payment"]}},
     lambda a: a["q"]["choice"] == "transaction_charged_twice"),
    ("noul_true", "Hi, I have been waiting for 20 minutes. Please let me talk to a real person, not a bot.",
     {"q": {"type": "noul", "instructions": "Is the customer asking to speak with a human?"}},
     lambda a: a["q"]["noul"] > 0.5),
    ("noul_false", "How do I change the email address on my account?",
     {"q": {"type": "noul", "instructions": "Is the customer asking to speak with a human?"}},
     lambda a: a["q"]["noul"] < 0.5),
    ("noul_criteria", "My card hasn't arrived yet and it's been two weeks.",
     {"q": {"type": "noul", "instructions": "Is this about a duplicated transaction?",
            "criteria": {"true": "the customer was charged twice", "false": "any other issue"}}},
     lambda a: a["q"]["noul"] < 0.5),
    ("score_high", "Production is completely down. Nobody can log in and we have no workaround.",
     {"q": {"type": "score", "instructions": "How severe is the reported issue?",
            "levels": ["Cosmetic issue; no functionality affected", "Feature degraded; workaround available",
                       "Critical functionality blocked; no workaround"]}},
     lambda a: a["q"]["score"] > 1.0),
    ("score_low", "There is a small typo in the footer of the settings page.",
     {"q": {"type": "score", "instructions": "How severe is the reported issue?",
            "levels": ["Cosmetic issue; no functionality affected", "Feature degraded; workaround available",
                       "Critical functionality blocked; no workaround"]}},
     lambda a: a["q"]["score"] < 1.0),
    ("mixed", MIXED_CTX, MIXED,
     lambda a: a["department"]["type"] == "choice" and a["needs_human"]["noul"] > 0.5 and a["severity"]["type"] == "score"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default=str(ROOT / "artifacts" / "krun-one-v0.2-candidate"))
    ap.add_argument("--eval-tag", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    t0 = time.time()
    import numpy as np
    import torch

    from krun_one import KrunOne

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    m = KrunOne.from_pretrained(args.artifact, device=dev)
    out = {"artifact": args.artifact, "device": dev, "load_seconds": round(time.time() - t0, 2),
           "primitives": m.primitives, "laya_modules_loaded": [k for k in sys.modules if "rl_common" in k or "rl_agent" in k],
           "sys_path_has_laya": any("laya" in p.lower() for p in sys.path), "cases": {}}
    for name, ctx, qs, check in CASES:
        a = m.decide_questions(ctx, qs)
        ok = bool(check(a))
        out["cases"][name] = {"answers": a, "pass": ok}
        print(f"  {name:13s} {'PASS' if ok else 'FAIL'} {json.dumps(a)[:220]}")
    # legacy v0 API still works
    legacy = m.decide("Find my meetings tomorrow", [{"name": "calendar_search", "description": "Search calendar events"},
                                                    {"name": "send_email", "description": "Send an email"}])
    out["cases"]["legacy_decide"] = {"answers": legacy, "pass": legacy["top_option"] == "calendar_search"}
    if args.eval_tag:
        from krun_one.primitive_eval import encode_records, load_jsonl

        fid = {}
        for f in ("noul_val", "score_val"):
            cache = json.loads((ROOT / "runs" / "eval_primitives" / args.eval_tag / f"{f}.json").read_text())
            recs = load_jsonl(ROOT / "data" / "primitives_v2" / f"{f}.jsonl")[:64]
            items, _ = encode_records(m.tok, recs, m.cfg["max_len"], m.cfg["head_max_len"])
            zs = []
            for i in range(0, len(items), 16):
                zs += m.logits(items[i:i + 16])
            d = [float(np.abs(z - np.asarray(cache["logits"][it["id"]])).max()) for z, it in zip(zs, items)]
            agree = np.mean([int(np.argmax(z) == np.argmax(cache["logits"][it["id"]])) for z, it in zip(zs, items)])
            fid[f] = {"n": len(items), "max_abs_logit_diff": max(d), "mean_abs_logit_diff": float(np.mean(d)),
                      "argmax_agreement": float(agree), "eval_dtype": cache.get("dtype")}
        out["fidelity"] = fid
        print(f"  fidelity: {fid}")
    out["all_pass"] = all(c["pass"] for c in out["cases"].values()) and not out["laya_modules_loaded"] \
        and not out["sys_path_has_laya"]
    dst = ROOT / "results" / f"reload_primitives_test_{dev}.json"
    dst.write_text(json.dumps(out, indent=2) + "\n")
    print(f"all_pass={out['all_pass']} wrote {dst}")
    return 0 if out["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
