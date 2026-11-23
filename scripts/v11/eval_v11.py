"""Krun One v1.1 — evaluation dumps of one model (the frozen v1 or an R1 / R2 checkpoint), same harness for all.

    python scripts/v11/eval_v11.py --ckpt v1 --out results/v11/eval/v1 --tech <files...> --regression --latency
    python scripts/v11/eval_v11.py --ckpt runs/v11/r2/r2_last2_s1 --out results/v11/eval/r2_last2_s1 --tech ... --regression

--tech        rendered example files (KrunTechBench, KrunTechVal, Tetris representation, mining pool): rank logits per id
--regression  the v1 regression inputs: KrunBench v1 val_fit / val_eval, val_ml, val_tools_v2 (signals in the v1 dump
              format, read by scripts/v1/kd1_evaluate.py) and the ToolACE v0 records. final_protected is never read.
--latency     text latency of THIS model object (p50 / p95, VRAM), v1 harness workloads
Metrics are computed offline from the dumps.
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from krun_one.v1.training import dump_signals, encode_records, load_records  # noqa: E402
from krun_one.v11 import harness, training as T  # noqa: E402
from krun_one.v11.model import KrunOneV11Model, V11Spec  # noqa: E402

ARTIFACT = ROOT / "artifacts/krun-one-v1-text-final"
V_FILES = {"val_fit": "data/krunbench_v1/val_fit.jsonl", "val_eval": "data/krunbench_v1/val_eval.jsonl",
           "val_ml": "data/kd1/val_ml.jsonl", "val_tools_v2": "data/kd1_r2/val_tools_v2.jsonl"}
TOOLACE = "runs/v1/ft/regression_toolace_v0/records.jsonl"
WORKLOADS = ["typical_1q_k4_ctx48", "typical_1q_tools8_ctx48", "typical_4q_mixed_ctx96", "k_8_tools_ctx64", "k_32_tools_ctx64",
             "k_128_tools_ctx64"]


def load(ckpt: str):
    if ckpt == "v1":
        return harness.load_v1(ARTIFACT), {"model": "krun-one-v1-text-final"}
    from safetensors.torch import load_model

    d = Path(ckpt)
    if not (d / "model.safetensors").exists() and (d / "model_v1.safetensors").exists():      # exported v1-architecture weights
        from krun_one.v1.arch import ModelSpec
        from krun_one.v1.model import KrunOneV1Model

        base = json.loads((d / "spec.json").read_text())
        m = KrunOneV1Model.from_spec(ModelSpec.from_dict(base.get("base", base)))
        load_model(m, str(d / "model_v1.safetensors"))
        return m, {"model": str(d), "weights": "model_v1.safetensors"}
    spec = V11Spec.from_dict(json.loads((d / "spec.json").read_text()))
    m = KrunOneV11Model.from_spec(spec)
    load_model(m, str(d / "model.safetensors"))
    m.q_head = m.v_head = None            # training-only heads: never part of inference
    return m, {"model": str(d), "spec": spec.to_dict()}


@torch.inference_mode()
def dump_tech(model, ser, path: Path, out: Path, dev) -> int:
    ex = T.load_jsonl(path)
    qs = [T.encode_tech(ser, e) for e in ex]
    model.eval()
    n = 0
    with gzip.open(out, "wt") as f:
        for idx in harness.batches_by_length([harness.length(q) for q in qs], 24_000):
            m = harness.forward_questions(model, [qs[i] for i in idx], ser.tok.pad_token_id, dev)
            z, ap = m["rank"].float().cpu().numpy(), m["apply"].float().cpu().numpy()
            for j, i in enumerate(idx):
                k = qs[i].n_candidates
                f.write(json.dumps({"id": ex[i]["id"], "z": [round(float(x), 5) for x in z[j, :k]],
                                    "apply": [round(float(x), 5) for x in ap[j, :k]], "chunks": len(qs[i].chunks)}) + "\n")
                n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tech", nargs="*", default=[])
    ap.add_argument("--regression", action="store_true")
    ap.add_argument("--v-files", default=",".join(V_FILES), help="subset of the V files (pilot: val_fit only); ToolACE only with all")
    ap.add_argument("--latency", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    model, meta = load(args.ckpt)
    model.to(dev).eval()
    ser = harness.serializer(ARTIFACT)
    meta.update(device=str(dev), gpu=torch.cuda.get_device_name(0) if dev.type == "cuda" else None, torch=torch.__version__, done={})
    for f in args.tech:
        t0 = time.time()
        n = dump_tech(model, ser, ROOT / f, out / f"tech_{Path(f).stem}.jsonl.gz", dev)
        meta["done"][f] = {"rows": n, "seconds": round(time.time() - t0, 1)}
        print(json.dumps({f: meta["done"][f]}), flush=True)
    if args.regression:
        want = args.v_files.split(",")
        for stem, f in V_FILES.items():
            if stem not in want:
                continue
            if "final_protected" in f:
                raise SystemExit("final_protected is never read")
            recs = load_records(ROOT / f, args.limit)
            items = encode_records(ser, recs)
            t0 = time.time()
            n = dump_signals(model, items, recs, ser.tok.pad_token_id, out / f"signals_{stem}.jsonl.gz", device=dev)
            meta["done"][stem] = {"rows": n, "seconds": round(time.time() - t0, 1)}
            print(json.dumps({stem: meta["done"][stem]}), flush=True)
        if len(want) == len(V_FILES):
            recs = load_records(ROOT / TOOLACE, args.limit)
            (out / "regression_toolace_v0").mkdir(exist_ok=True)
            n = dump_signals(model, encode_records(ser, recs), recs, ser.tok.pad_token_id,
                             out / "regression_toolace_v0/signals.jsonl.gz", device=dev, batch_tokens=8192)
            meta["done"]["toolace_v0"] = {"rows": n}
    if args.latency:
        spec_ = importlib.util.spec_from_file_location("latency_proxy", ROOT / "scripts/v1/latency_proxy.py")
        lp = importlib.util.module_from_spec(spec_)
        spec_.loader.exec_module(lp)
        corpus = json.loads((ROOT / "results/v1/latency_corpus.json").read_text())
        res = []
        for dtype in ("autocast", "bf16"):
            if dtype == "bf16":
                model = model.to(torch.bfloat16)
            torch.cuda.reset_peak_memory_stats()
            for wl in [w for w in lp.workloads() if w["name"] in WORKLOADS]:
                r = lp.measure(model, ser, ser.tok, corpus, wl, "en-US", dev, dtype, 40, 8)
                r.update(dtype=dtype, workload=wl["name"], peak_vram_mib=round(torch.cuda.max_memory_allocated() / 2**20))
                res.append(r)
                print(json.dumps({k: r[k] for k in ("dtype", "workload", "total_p50_ms", "total_p95_ms", "peak_vram_mib")}), flush=True)
        import resource

        (out / "latency.json").write_text(json.dumps({"gpu": meta["gpu"], "results": res,
                                                      "process_peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)}, indent=1))
    (out / "eval_meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
