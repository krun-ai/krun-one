"""M1 serving checks on the GPU pod: text-only regression (G2) and latency by modality (G4).

    python scripts/v1/m1_serving_checks.py textreg --run runs/v1/m1/<run>   -> results/v1/m1/text_regression.json
    python scripts/v1/m1_serving_checks.py latency --run runs/v1/m1/<run>   -> results/v1/m1/latency_rtx4090.json

textreg: A = the frozen artifact alone; B = a SECOND fresh load of the artifact wrapped in the M1 model with the
trained fusion + both towers loaded, after serving image / document / audio requests. Both dump the full V signal
sets (val_fit, val_eval, val_ml, val_tools_v2) through the text path; the files must be bitwise identical, the core
digests must equal each other. V metrics of A (kd1_evaluate protocol) are reported as the text-only numbers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/v1"))
import train_m1 as T  # noqa: E402
from krun_one.v1.multimodal.features import ModalityFeatures  # noqa: E402
from krun_one.v1.multimodal.m1 import M1Fusion, M1Model, core_digest  # noqa: E402
from krun_one.v1.runtime import TextArtifact  # noqa: E402

ART = ROOT / "artifacts/krun-one-v1-text-final"
VFILES = {"val_fit": ROOT / "data/krunbench_v1/val_fit.jsonl", "val_eval": ROOT / "data/krunbench_v1/val_eval.jsonl",
          "val_ml": ROOT / "data/kd1/val_ml.jsonl", "val_tools_v2": ROOT / "data/kd1_r2/val_tools_v2.jsonl"}


def m1_stack(run: Path):
    from safetensors.torch import load_file

    from krun_one.v1.multimodal.m1_towers import AudioFrontend, image_preprocessor

    a = TextArtifact.load(ART, device="cuda", dtype="bf16", verify=False)
    towers = T.Towers("cuda")
    fus = M1Fusion(a.model, towers.vision.output_dim, towers.audio.output_dim, **T.CFG["fusion"]).cuda()
    tr = json.loads((run / "train_result.json").read_text())
    fus.load_state_dict(load_file(str(run / tr["selected"] / "fusion.safetensors")))
    m = M1Model(a.model, fus, freeze_core=False).eval()   # serving: core untouched (see M1Model)
    return a, m, towers, image_preprocessor(T.CFG["towers"]["vision"]), AudioFrontend(T.CFG["towers"]["audio"])


def dump_all(model, ser, pad_id, out: Path) -> dict:
    from krun_one.v1.training import dump_signals, encode_records, load_records

    out.mkdir(parents=True, exist_ok=True)
    sums = {}
    for name, p in VFILES.items():
        recs = load_records(p)
        items = encode_records(ser, recs)
        f = out / f"signals_{name}.jsonl.gz"
        dump_signals(model, items, recs, pad_id, f, device="cuda")
        import gzip

        sums[name] = hashlib.sha256(gzip.open(f).read()).hexdigest()   # content hash (gzip header has a timestamp)
    return sums


def mm_request(m, towers, ip, fe, a, kind: str, seconds: float = 5.0):
    from PIL import Image

    from krun_one.v1.model import chunk_index, collate_chunks, merge_chunks
    from krun_one.v1.multimodal.m1_towers import PATCHES

    rec = {"task": {"type": "choice", "instructions": "Which option matches the input?"}, "context": {"text": ""},
           "candidates": [{"id": x} for x in ("cancel_subscription", "check_balance", "report_fraud", "book_flight")]}
    timing = {}
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    if kind == "audio":
        w = (np.random.default_rng(0).standard_normal(int(16000 * seconds)) * 0.05).astype(np.float32)
        feats, n = fe([w])
    else:
        im = Image.new("RGB", (850, 1100) if kind == "document" else (640, 480), (200, 180, 160))
        mv = dict(ip(images=[im], max_num_patches=PATCHES[kind], return_tensors="pt"))
    t1 = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        if kind == "audio":
            h, mk = towers.audio.encode(feats, n)
        else:
            h, mk = towers.vision.encode(mv)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        tok, tm = m.fusion.media_tokens(ModalityFeatures(kind, h, mk))
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        enc = [a.ser.encode_record(rec)]
        flat, where = chunk_index(enc)
        b = {k: v.cuda() for k, v in collate_chunks(flat, a.tok.pad_token_id).items()}
        o = m.forward_mm(b, (tok, tm), [0])
        q = merge_chunks(o, where, 1, [4])
        _ = q["rank"].float().cpu()
    torch.cuda.synchronize()
    t4 = time.perf_counter()
    timing = {"preprocess_cpu_ms": (t1 - t0) * 1e3, "tower_ms": (t2 - t1) * 1e3, "projection_fusion_ms": (t3 - t2) * 1e3,
              "krun_decision_ms": (t4 - t3) * 1e3, "total_ms": (t4 - t0) * 1e3}
    return timing


def cmd_textreg(args) -> None:
    run = Path(args.run)
    A = TextArtifact.load(ART, device="cuda", dtype="bf16", verify=True)
    dA = core_digest(A.model)
    sA = dump_all(A.model, A.ser, A.tok.pad_token_id, run / "textreg/A")
    del A
    torch.cuda.empty_cache()
    a, m, towers, ip, fe = m1_stack(run)
    for kind in ("image", "document", "audio"):            # serve multimodal traffic first
        mm_request(m, towers, ip, fe, a, kind)
    dB = core_digest(m.core)
    sB = dump_all(m.core, a.ser, a.tok.pad_token_id, run / "textreg/B")
    res = {"core_digest_A": dA, "core_digest_B": dB, "core_digest_equal": dA == dB, "signals_sha_A": sA,
           "signals_sha_B": sB, "signals_bitwise_identical": sA == sB,
           "towers_loaded_in_B": True}
    # V metrics of the text path (identical for A and B when bitwise equal)
    import importlib.util

    spec = importlib.util.spec_from_file_location("K", ROOT / "scripts/v1/kd1_evaluate.py")
    K = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(K)
    rows = [r for r in K.load(run / "textreg/A") if r["file"] != "val_tools_v2"]
    met, _ = K.evaluate(rows)
    keep = ("choice_accuracy_intent", "tool_accuracy", "cross_language_accuracy", "by_language", "by_k", "ece",
            "ood_auroc_margin", "aurc", "ood_false_acceptance", "permutation_consistency", "score_mae", "score_spearman",
            "noul_auroc", "noul_accuracy", "multi_micro_f1")
    res["text_v_metrics_this_pod"] = {k: met.get(k) for k in keep}
    res["PASS"] = bool(res["core_digest_equal"] and res["signals_bitwise_identical"])
    Path(args.out or ROOT / "results/v1/m1/text_regression.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k != "text_v_metrics_this_pod"}, indent=1))


def cmd_latency(args) -> None:
    from latency_proxy import build_context, question

    run = Path(args.run)
    base = TextArtifact.load(ART, device="cuda", dtype="bf16", verify=False)
    a, m, towers, ip, fe = m1_stack(run)
    corpus = json.loads((ROOT / "results/v1/latency_corpus.json").read_text())
    W = {"typical_1q_k4_ctx48": (48, [("labels", 4)]), "typical_4q_mixed_ctx96": (96, [("tools", 8), ("noul", 2), ("score", 5), ("labels", 4)]),
         "k_32_tools_ctx64": (64, [("tools", 32)]), "k_128_tools_ctx64": (64, [("tools", 128)])}
    out = {"gpu": torch.cuda.get_device_name(0), "dtype": "bf16", "iters": args.iters, "text_only": {}, "multimodal": {}}
    for name, (n_ctx, qs) in W.items():
        ctx = build_context(corpus, "pt-PT", n_ctx, a.tok, offset=7)
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
        tA, tB = [], []
        for it in range(args.warmup + args.iters):
            for who, obj, acc in ((("A", base, tA), ("B", a, tB)) if it % 2 == 0 else (("B", a, tB), ("A", base, tA))):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                r = obj.decide(req)
                torch.cuda.synchronize()
                if it >= args.warmup:
                    acc.append((time.perf_counter() - t0) * 1e3)
        same = json.dumps(base.decide(req)["answers"], sort_keys=True) == json.dumps(a.decide(req)["answers"], sort_keys=True)
        out["text_only"][name] = {"frozen_artifact_p50_ms": float(np.percentile(tA, 50)), "with_m1_loaded_p50_ms": float(np.percentile(tB, 50)),
                                  "delta_pct": 100 * (np.percentile(tB, 50) / np.percentile(tA, 50) - 1), "answers_identical": same}
        print(name, out["text_only"][name], flush=True)
    for kind, secs in (("image", 0), ("document", 0), ("audio", 5.0), ("audio", 15.0)):
        rows = []
        for it in range(args.warmup + args.iters):
            t = mm_request(m, towers, ip, fe, a, kind, secs or 5.0)
            if it >= args.warmup:
                rows.append(t)
        key = f"{kind}" + (f"_{int(secs)}s" if kind == "audio" else "")
        out["multimodal"][key] = {f"{k}_p50": float(np.percentile([r[k] for r in rows], 50)) for k in rows[0]}
        out["multimodal"][key].update({"total_ms_p95": float(np.percentile([r["total_ms"] for r in rows], 95))})
        print(key, out["multimodal"][key], flush=True)
    out["G4_pass"] = bool(all(v["total_ms_p50"] <= 150 for v in out["multimodal"].values())
                          and all(v["delta_pct"] <= 5 for v in out["text_only"].values()))
    Path(args.out or ROOT / "results/v1/m1/latency_rtx4090.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["textreg", "latency"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out")
    a = ap.parse_args()
    {"textreg": cmd_textreg, "latency": cmd_latency}[a.cmd](a)
