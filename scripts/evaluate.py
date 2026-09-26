#!/usr/bin/env python
"""Evaluate a model on the Krun One protocol (identical for Laya and Krun One).

    .venv/bin/python scripts/evaluate.py --model laya                    # Laya, Krun One budgets (1024/896)
    .venv/bin/python scripts/evaluate.py --model laya --tag laya_default --max-len 512 --head-max-len 192
    .venv/bin/python scripts/evaluate.py --model runs/A/final --tag krun_one_A
    .venv/bin/python scripts/evaluate.py --model artifacts/krun-one-v0 --tag krun_one_v0 --final

Logits are cached in runs/eval/<tag>/<set>.json; the report goes to results/eval_<tag>.json.
``--final`` adds holdout_tools + kb_test; it is refused until results/krun_one_decisions.json exists
(decision rules in configs/krun_one_v0.yaml: nothing is tuned on the final sets).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import yaml

import krun_one  # noqa: F401
import torch
from safetensors.torch import load_file

from krun_one.evaluation import FINAL_SETS, VALIDATION_SETS, full_report, load_set, predict_logits
from krun_one.laya_adapter import LAYA_DIR, load_laya

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]


def clean(o):
    if isinstance(o, float):
        return None if math.isnan(o) or math.isinf(o) else o
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    return o


def load_model(spec: str, device: torch.device):
    """'laya' -> upstream checkpoint; a run dir with model.safetensors (Laya architecture); or an exported artifact."""
    if spec == "laya":
        model, tok, _ = load_laya(device="cpu")
        return model.to(device), tok, "upstream ./laya checkpoint (no training)"
    p = Path(spec)
    if not p.is_absolute():
        p = ROOT / p
    if (p / "krun_config.json").exists():
        from krun_one import KrunOne

        k = KrunOne.from_pretrained(p, device=str(device))
        return k.model, k.tok, f"artifact {p}"
    model, tok, _ = load_laya(device="cpu")
    model.load_state_dict(load_file(str(p / "model.safetensors")), strict=True)
    return model.to(device), tok, f"fine-tuned weights {p / 'model.safetensors'}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--config", default=str(ROOT / "configs" / "krun_one_v0.yaml"))
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--head-max-len", type=int, default=None)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--sets", default=None, help="comma list (default: validation sets [+ final])")
    ap.add_argument("--limit", type=int, default=0, help="smoke: N decisions per set")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    tag = args.tag or Path(args.model).name
    max_len = args.max_len or cfg["sequence"]["max_len"]
    head = args.head_max_len or cfg["sequence"]["head_max_len"]
    data_dir = Path(args.data_dir) if args.data_dir else PROJECT / cfg["data"]["dir"]
    sets = args.sets.split(",") if args.sets else list(VALIDATION_SETS) + (list(FINAL_SETS) if args.final else [])
    if any(s in FINAL_SETS for s in sets) and not (ROOT / "results" / "krun_one_decisions.json").exists():
        raise SystemExit("final sets are locked until results/krun_one_decisions.json is written (see decision_rules)")
    if any(s in FINAL_SETS for s in sets) and tag.startswith("v02") and not (ROOT / "results" / "primitives_decisions.json").exists():
        raise SystemExit("KRUN-011 candidates: final sets are locked until results/primitives_decisions.json exists")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16

    cache_dir = ROOT / "runs" / "eval" / tag
    cache_dir.mkdir(parents=True, exist_ok=True)
    model, tok, desc = None, None, None
    examples, logits, unfit, timing = {}, {}, {}, {}
    for s in sets:
        exs = load_set(s, PROJECT, data_dir)
        if args.limit:
            exs = exs[:: max(1, len(exs) // args.limit)]
        examples[s] = exs
        f = cache_dir / f"{s}.json"
        if f.exists() and not args.limit:
            d = json.loads(f.read_text())
            if set(d["logits"]) >= {e.id for e in exs} and d["max_len"] == max_len and d["head_max_len"] == head:
                logits[s], unfit[s], timing[s] = d["logits"], d["unfit"], d.get("seconds")
                continue
        if model is None:
            model, tok, desc = load_model(args.model, device)
        t0 = time.time()
        logits[s], unfit[s] = predict_logits(model, tok, exs, max_len, head, device, dtype, progress=s)
        timing[s] = round(time.time() - t0, 1)
        if not args.limit:
            f.write_text(json.dumps({"logits": logits[s], "unfit": unfit[s], "max_len": max_len, "head_max_len": head,
                                     "seconds": timing[s], "dtype": str(dtype), "device": str(device)}))
    rep = full_report(examples, logits, cfg["evaluation"])
    out = {"tag": tag, "model": args.model, "model_description": desc, "max_len": max_len, "head_max_len": head,
           "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
           "dtype": str(dtype), "sets": sets, "n_per_set": {k: len(v) for k, v in examples.items()},
           "unfit_per_set": {k: len(v) for k, v in unfit.items()}, "inference_seconds": timing,
           "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **rep}
    dst = ROOT / "results" / (f"eval_{tag}.json" if not args.limit else f"_scratch/eval_{tag}_limit{args.limit}.json")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(clean(out), indent=1) + "\n")
    for s, r in rep["sets"].items():
        c = r["quality"]["core"]
        ab = r["abstention"].get("all", {}).get("top1_probability", {})
        print(f"  {s:14s} n={c.get('n', 0):6d} acc={c.get('accuracy', float('nan')):.4f} ece={c.get('ece', float('nan')):.4f} "
              f"ood_auroc={ab.get('ood_auroc', float('nan')):.4f}")
    print(f"T={rep['temperatures']}  wrote {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
