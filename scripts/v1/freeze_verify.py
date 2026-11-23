"""Reload verification + latency regression of the FROZEN text artifact (artifacts/krun-one-v1-text-final).

    python scripts/v1/freeze_verify.py reload  --device cpu  --dtype fp32  --out results/v1/freeze/reload_cpu_fp32.json
    python scripts/v1/freeze_verify.py reload  --device cuda --dtype bf16  --out results/v1/freeze/reload_cuda_bf16.json
    python scripts/v1/freeze_verify.py latency --out results/v1/freeze/latency_rtx4090.json          (GPU)

reload (fresh process, artifact loaded from scratch with SHA256SUMS verification):
  1. API reference cases -> answers compared to the reference outputs stored IN the artifact (CPU fp32):
     same decision / abstention / noul / score / multi set, max |dprob|;
  2. V eval sample (reference/v_eval_sample.jsonl) -> raw rank logits, artifact path vs the ORIGINAL training/eval
     path (runs/.../step_003299 weights + HF-cache tokenizer + training.encode_records/make_batch): token ids equal,
     max |dlogit|, argmax agreement;
  3. (cuda bf16) the same sample vs the Stage 2 signal dumps (RTX 4090 bf16, gate harness): max |dlogit|,
     argmax + abstention agreement with the frozen calibration.
latency: end-to-end artifact.decide (serialize + forward + answers) and forward-only on the 4 regression workloads.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/v1"))
ART = ROOT / "artifacts/krun-one-v1-text-final"
SRC = ROOT / "runs/v1/ft/krun-one-v1-stage2/milestones/step_003299"


def compare_answers(a: dict, b: dict) -> dict:
    same, dprob, diffs = True, 0.0, []
    for k in a:
        for q in a[k]["answers"]:
            x, y = a[k]["answers"][q], b[k]["answers"][q]
            for f in ("decision", "abstain", "top_option", "selected"):
                if x.get(f) != y.get(f):
                    same = False
                    diffs.append((k, q, f, x.get(f), y.get(f)))
            for f in ("noul", "score"):
                if f in x:
                    dprob = max(dprob, abs(x[f] - y[f]))
            for f in ("probabilities", "scores"):
                if f in x:
                    dprob = max(dprob, max(abs(x[f][i] - y[f][i]) for i in x[f]))
    return {"decisions_identical": same, "max_abs_prob_or_value_diff": dprob, "differences": diffs}


def original_logits(records, device, dtype):
    """The Stage 2 eval path: model from runs/ weights, tokenizer from the HF cache, training.encode_records."""
    from safetensors.torch import load_model

    from krun_one.v1.arch import ModelSpec
    from krun_one.v1.model import KrunOneV1Model, merge_chunks
    from krun_one.v1.serialization import Budget, Serializer
    from krun_one.v1.training import encode_records, make_batch
    from train_decisions import tokenizer

    tok = tokenizer()
    ser = Serializer(tok, Budget(1024, 896, max_candidates_per_chunk=64))
    items = encode_records(ser, records)
    m = KrunOneV1Model.from_spec(ModelSpec.from_dict(json.loads((ROOT / "runs/v1/ft/krun-one-v1-stage2/final/spec.json").read_text())))
    load_model(m, str(SRC / "model.safetensors"))
    m.to(device).eval()
    out, ids = {}, {}
    with torch.inference_mode():
        for i in range(0, len(items), 16):
            batch = items[i:i + 16]
            b, where, nc, _ = make_batch(batch, tok.pad_token_id, device)
            with torch.autocast(device.type, dtype=dtype or torch.float32, enabled=dtype is not None):
                o = m(**b)
            q = merge_chunks(o, where, len(batch), nc)
            for j, it in enumerate(batch):
                out[it["id"]] = q["rank"][j, :len(it["target"])].float().cpu().numpy()
                ids[it["id"]] = [list(map(int, c.ids)) for c in it["q"].chunks]
    del m
    return out, ids


def artifact_logits(a, records):
    out, ids = {}, {}
    for i in range(0, len(records), 16):
        rs = records[i:i + 16]
        enc = [a.ser.encode_record(r) for r in rs]
        raw = a.forward(enc)
        for j, (r, e) in enumerate(zip(rs, enc)):
            out[r["id"]] = raw["rank"][j, :e.n_candidates]
            ids[r["id"]] = [list(map(int, c.ids)) for c in e.chunks]
    return out, ids


def reload(args) -> dict:
    from krun_one.v1.runtime import TextArtifact

    t0 = time.perf_counter()
    a = TextArtifact.load(ART, device=args.device, dtype=args.dtype, verify=True)
    res = {"device": args.device, "dtype": args.dtype, "load_seconds_incl_sha_verification": time.perf_counter() - t0,
           "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None, "torch": torch.__version__}
    cases = json.loads((ART / "reference/api_cases.json").read_text())
    ref = json.loads((ART / "reference/api_outputs_cpu_fp32.json").read_text())
    got = {k: a.decide(v, return_raw=True) for k, v in cases.items()}
    res["api_cases"] = len(cases)
    res["api_vs_reference_cpu_fp32"] = compare_answers(ref, got)
    got2 = {k: a.decide(v) for k, v in cases.items()}
    res["api_repeat_identical"] = all(json.dumps(got2[k]["answers"], sort_keys=True) ==
                                      json.dumps({q: {f: v for f, v in x.items() if f != "_raw_rank"} for q, x in got[k]["answers"].items()}, sort_keys=True)
                                      for k in cases)
    recs = [json.loads(l) for l in open(ART / "reference/v_eval_sample.jsonl")]
    la, ia = artifact_logits(a, recs)
    dev = torch.device(args.device)
    del a
    if args.device == "cuda":
        torch.cuda.empty_cache()
    lo, io = original_logits(recs, dev, {"fp32": None, "bf16": torch.bfloat16}[args.dtype])
    d = [float(np.abs(la[k] - lo[k]).max()) for k in la]
    res["v_sample"] = {"n": len(recs), "token_ids_identical": all(ia[k] == io[k] for k in ia),
                       "artifact_vs_original_path_max_abs_logit_diff": max(d),
                       "argmax_agreement": float(np.mean([int(np.argmax(la[k])) == int(np.argmax(lo[k])) for k in la]))}
    if args.device == "cuda":
        calib = json.loads((ART / "calibration.json").read_text())
        stored = {}
        for f in SRC.glob("signals_*.jsonl.gz"):
            for l in gzip.open(f, "rt"):
                r = json.loads(l)
                if r["id"] in la:
                    stored[r["id"]] = r
        fam = {r["id"]: (r["task"].get("family") if r["task"]["type"] == "choice" else r["task"]["type"]) for r in recs}

        def acc(z, k):
            if fam[k] not in ("intent", "tool"):
                return None
            from krun_one.v1.runtime import softmax
            s = np.sort(softmax(z, calib["temperature"][fam[k]]))[::-1]
            return bool(s[0] - s[1] >= calib["margin_threshold"][fam[k]])

        ks = [k for k in la if k in stored]
        dd = [float(np.abs(la[k] - np.asarray(stored[k]["rank"])).max()) for k in ks]
        ab = [acc(la[k], k) == acc(np.asarray(stored[k]["rank"], dtype=np.float64), k) for k in ks]
        res["v_sample_vs_stage2_signals_4090_bf16"] = {
            "n": len(ks), "max_abs_logit_diff": max(dd), "p99_abs_logit_diff": float(np.percentile(dd, 99)),
            "argmax_agreement": float(np.mean([int(np.argmax(la[k])) == int(np.argmax(stored[k]["rank"])) for k in ks])),
            "abstention_agreement": float(np.mean(ab))}
    res["PASS"] = bool(res["api_vs_reference_cpu_fp32"]["decisions_identical"] and res["api_repeat_identical"]
                       and res["v_sample"]["token_ids_identical"]
                       and (res["v_sample"]["artifact_vs_original_path_max_abs_logit_diff"] == 0.0 or args.device == "cuda")
                       and res["v_sample"]["argmax_agreement"] == 1.0)
    return res


def latency(args) -> dict:
    from latency_proxy import build_context, question

    from krun_one.v1.runtime import TextArtifact

    a = TextArtifact.load(ART, device="cuda", dtype="bf16", verify=False)
    corpus = json.loads((ROOT / "results/v1/latency_corpus.json").read_text())
    W = {"typical_1q_k4_ctx48": (48, [("labels", 4)]),
         "typical_4q_mixed_ctx96": (96, [("tools", 8), ("noul", 2), ("score", 5), ("labels", 4)]),
         "k_32_tools_ctx64": (64, [("tools", 32)]), "k_128_tools_ctx64": (64, [("tools", 128)])}
    out = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "dtype": "bf16", "iters": args.iters,
           "results": {}}
    for name, (ctx_n, qs) in W.items():
        ctx = build_context(corpus, "pt-PT", ctx_n, a.tok, offset=7)
        req = {"context": ctx, "questions": {}}
        for j, (kind, k) in enumerate(qs):
            q = question(corpus, kind, k, offset=13 * j)
            if q["type"] == "choice":
                req["questions"][f"q{j}"] = {"type": "choice", "task_type": "tool" if kind == "tools" else "intent",
                                             "options": [{"id": c["id"], **({"description": c["description"]} if c.get("description") else {})} for c in q["candidates"]]}
            elif q["type"] == "noul":
                req["questions"][f"q{j}"] = {"type": "noul", "instructions": q["instructions"]}
            else:
                req["questions"][f"q{j}"] = {"type": "score", "instructions": q["instructions"], "levels": [c["label"] for c in q["candidates"]]}
        enc = [a.encode(ctx, q)[0] for q in req["questions"].values()]
        e2e, fwd = [], []
        for it in range(args.warmup + args.iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            a.decide(req)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            a.forward(enc)
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            if it >= args.warmup:
                e2e.append((t1 - t0) * 1000)
                fwd.append((t2 - t1) * 1000)
        out["results"][name] = {"e2e_p50_ms": float(np.percentile(e2e, 50)), "e2e_p95_ms": float(np.percentile(e2e, 95)),
                                "forward_p50_ms": float(np.percentile(fwd, 50)), "forward_p95_ms": float(np.percentile(fwd, 95))}
        print(name, out["results"][name], flush=True)
    r = out["results"]
    out["reference_stage2_p50_ms"] = {"typical_1q_k4_ctx48": 16.923, "typical_4q_mixed_ctx96": 18.584,
                                      "k_32_tools_ctx64": 18.771, "k_128_tools_ctx64": 43.887}
    out["PASS"] = bool(r["typical_1q_k4_ctx48"]["forward_p50_ms"] <= 20 and r["k_128_tools_ctx64"]["forward_p50_ms"] < 50)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["reload", "latency"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="fp32")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.device == "cpu":
        torch.set_num_threads(11)
    res = reload(args) if args.cmd == "reload" else latency(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=1, default=str)[:3000])
