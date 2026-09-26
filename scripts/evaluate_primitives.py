#!/usr/bin/env python
"""KRUN-011: evaluate a checkpoint on noul / score / mixed (same code path for Laya and Krun One).

    .venv/bin/python scripts/evaluate_primitives.py --model laya --tag laya                       # 1024/896
    .venv/bin/python scripts/evaluate_primitives.py --model laya --tag laya_default --max-len 512 --head-max-len 192
    .venv/bin/python scripts/evaluate_primitives.py --model artifacts/krun-one-v0 --tag krun_one_v0
    .venv/bin/python scripts/evaluate_primitives.py --model runs/v02_A/final --tag v02_A --no-test   # candidates: val only

Logits are cached in runs/eval_primitives/<tag>/; the report goes to <out-dir>/eval_<tag>.json.
Test sets for a *candidate* are refused until results/primitives_decisions.json exists (selection frozen first).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import krun_one  # noqa: F401
import torch

from krun_one.primitive_eval import encode_records, evaluate_mixed, evaluate_primitive, load_jsonl, predict

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate import clean, load_model  # noqa: E402

BASELINES = {"laya", "laya_default", "krun_one_v0", "laya_p2", "laya_default_p2", "krun_one_v0_p2"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "primitives_v1"))
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--head-max-len", type=int, default=896)
    ap.add_argument("--no-test", action="store_true")
    ap.add_argument("--no-mixed", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="smoke: every n-th record so that ~N remain per file")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-tokens", type=int, default=32768, help="padded tokens per forward (4096 on a 4 GB GPU)")
    args = ap.parse_args()
    with_test = not args.no_test
    if with_test and args.tag not in BASELINES and not (ROOT / "results" / "primitives_decisions.json").exists():
        raise SystemExit("test sets are locked for candidates until results/primitives_decisions.json exists")
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "results" / ("primitives_baseline" if args.tag in BASELINES
                                                                          else "primitives_candidates")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8 else \
        (torch.float16 if device.type == "cuda" else torch.float32)
    data = Path(args.data_dir)
    files = {f"{q}_{s}": data / f"{q}_{s}.jsonl" for q in ("noul", "score") for s in ("val", "test")
             if s == "val" or with_test}
    recs = {k: load_jsonl(p) for k, p in files.items()}
    if args.limit:
        recs = {k: v[:: max(1, len(v) // args.limit)] for k, v in recs.items()}
    cache = ROOT / "runs" / "eval_primitives" / args.tag
    cache.mkdir(parents=True, exist_ok=True)
    model = tok = desc = None
    logits, unfit, timing = {}, {}, {}
    for k, rs in recs.items():
        f = cache / f"{k}.json"
        if f.exists() and not args.limit:
            d = json.loads(f.read_text())
            if d["max_len"] == args.max_len and d["head_max_len"] == args.head_max_len and \
                    set(d["logits"]) | set(d["unfit"]) >= {r["id"] for r in rs}:
                logits.update(d["logits"])
                unfit[k], timing[k] = d["unfit"], d["seconds"]
                continue
        if model is None:
            model, tok, desc = load_model(args.model, device)
        t0 = time.time()
        items, unfit[k] = encode_records(tok, rs, args.max_len, args.head_max_len)
        z = predict(model, tok, items, device, dtype, max_tokens=args.max_tokens)
        timing[k] = round(time.time() - t0, 1)
        logits.update(z)
        print(f"  {k}: {len(items)} questions in {timing[k]}s ({len(unfit[k])} unfit)", flush=True)
        if not args.limit:
            f.write_text(json.dumps({"logits": z, "unfit": unfit[k], "max_len": args.max_len,
                                     "head_max_len": args.head_max_len, "seconds": timing[k], "dtype": str(dtype)}))
    rep = {}
    for q in ("noul", "score"):
        rep[q] = evaluate_primitive(q, recs[f"{q}_val"], recs.get(f"{q}_test"), logits)
    if with_test and not args.no_mixed:
        if model is None:
            model, tok, desc = load_model(args.model, device)
        mixed = load_jsonl(data / "mixed_test.jsonl")
        if args.limit:
            mixed = mixed[:: max(1, len(mixed) // args.limit)]
        rep["mixed"] = evaluate_mixed(model, tok, mixed, args.max_len, args.head_max_len, device, dtype,
                                      {"noul": rep["noul"]["temperature"], "score": rep["score"]["temperature"],
                                       "choice": 1.0}, max_tokens=args.max_tokens)
    out = {"tag": args.tag, "model": args.model, "model_description": desc, "max_len": args.max_len,
           "head_max_len": args.head_max_len, "device": str(device),
           "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None, "dtype": str(dtype),
           "n": {k: len(v) for k, v in recs.items()}, "unfit": {k: len(v) for k, v in unfit.items()},
           "inference_seconds": timing, "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **rep}
    dst = out_dir / (f"eval_{args.tag}.json" if not args.limit else f"_smoke_eval_{args.tag}.json")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(clean(out), indent=1) + "\n")
    for q in ("noul", "score"):
        for part in ("val_eval", "test"):
            if part not in rep[q]:
                continue
            a = rep[q][part]["calibrated"]["all"]
            if q == "noul":
                print(f"  {q} {part}: auroc {a['roc_auc']:.4f} acc {a['accuracy@0.5']:.4f} brier {a['brier']:.4f} "
                      f"ece {a['ece']:.4f} nll {a['nll']:.4f} (T={rep[q]['temperature']:.3f})")
            else:
                print(f"  {q} {part}: mae {a['mae']:.4f} acc {a['accuracy']:.4f} spearman {a['spearman']:.4f} "
                      f"ece {a['ece']:.4f} nll {a['nll']:.4f} (T={rep[q]['temperature']:.3f})")
    if "mixed" in rep:
        print("  mixed:", json.dumps(rep["mixed"]["invariance_alone_vs_mixed"]))
    print("wrote", dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
