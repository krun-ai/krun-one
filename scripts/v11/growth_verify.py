"""Krun One v1.1 — does a grown model reproduce the frozen v1 BEFORE any training?

For every growth route (krun_one.v11.candidates): build the larger model from the v1 weights (v1 is only read),
run the same encoded probe questions through both, and report the numerical differences of every output.

    python scripts/v11/growth_verify.py --device cpu --n 48        # fp32 (the exactness check)
    python scripts/v11/growth_verify.py --device cuda --n 400      # + bf16 autocast (serving numerics)
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from krun_one.v1.arch import ModelSpec  # noqa: E402
from krun_one.v11 import candidates, growth, growth_width, harness  # noqa: E402
from krun_one.v11.model import KrunOneV11Model  # noqa: E402

ARTIFACT = ROOT / "artifacts/krun-one-v1-text-final"


def build(src, name: str):
    """Returns (model, route description, report)."""
    by = {s.name: s for s, _ in candidates.ENCODER_CANDIDATES + candidates.HEAD_CANDIDATES}
    spec = by[name]
    if spec.cand_layers:
        return KrunOneV11Model.from_v1(src, spec.cand_layers), "identity-initialized candidate-interaction layers", {}
    e, v1e = spec.base.encoder, src.spec.encoder
    if e.hidden_size == v1e.hidden_size:
        m, rep = growth.grow_exact(src, spec.base)
        return m, "exact: FFN columns zero / identity blocks", rep
    wide = ModelSpec(spec.base.name + "-L22", replace(e, num_layers=v1e.num_layers), spec.base.decision)
    m, rep = growth_width.widen_stream(src, wide)
    route = "width: centered stream + carriers"
    if e.num_layers > v1e.num_layers:
        eps = {k: mod.eps for k, mod in m.named_modules() if isinstance(mod, torch.nn.LayerNorm)}
        m2, rep2 = growth.grow_exact(m, spec.base)
        for k, mod in m2.named_modules():                # LayerNorm eps is not part of the spec: carry it over
            if isinstance(mod, torch.nn.LayerNorm):
                mod.eps = eps.get(k, m.encoder.final_norm.eps)
        # new encoder blocks are identities, but their LayerNorms must not see un-normalized pads: weights irrelevant (Wo = 0)
        m, rep = m2, {**rep, "depth": rep2["new_identity_blocks"]}
        route += " + appended identity layers"
    return m, route, rep


@torch.inference_mode()
def outputs(model, qs, pad, dev, amp: bool):
    model.eval()
    res = [None] * len(qs)
    for idx in harness.batches_by_length([harness.length(q) for q in qs], 12_000, 32):
        m = harness.forward_questions(model, [qs[i] for i in idx], pad, dev, amp=amp)
        for n_, i in enumerate(idx):
            k = qs[i].n_candidates
            res[i] = {"rank": m["rank"][n_, :k].float().cpu().numpy(), "apply": m["apply"][n_, :k].float().cpu().numpy(),
                      "null": float(m["null_rank"][n_]), "ans": float(m["ans"][n_])}
    return res


def compare(a, b) -> dict:
    d = lambda key: float(max(np.abs(np.asarray(x[key]) - np.asarray(y[key])).max() for x, y in zip(a, b)))  # noqa: E731
    sm = lambda z: np.exp(z - z.max()) / np.exp(z - z.max()).sum()  # noqa: E731
    return {"max_abs_dlogit_rank": d("rank"), "max_abs_dlogit_apply": d("apply"), "max_abs_d_null": d("null"),
            "max_abs_d_ans": d("ans"), "max_abs_dp": float(max(np.abs(sm(x["rank"]) - sm(y["rank"])).max() for x, y in zip(a, b))),
            "argmax_agreement": float(np.mean([int(x["rank"].argmax() == y["rank"].argmax()) for x, y in zip(a, b)])),
            "mean_abs_dlogit_rank": float(np.mean([np.abs(x["rank"] - y["rank"]).mean() for x, y in zip(a, b)]))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default=str(ROOT / "results/v11/architecture_search/growth_verification.json"))
    args = ap.parse_args()
    dev = torch.device(args.device)
    ser = harness.serializer(ARTIFACT)
    pad = ser.tok.pad_token_id
    probs = random.Random(7).sample(harness.load_problems(ROOT / "data/v11/probe/probe.jsonl"), args.n)
    qs = [harness.encode(ser, p, "feat" if i % 2 else "raw") for i, p in enumerate(probs)]
    src = harness.load_v1(ARTIFACT).eval()
    modes = [("fp32", False)] + ([("bf16_autocast", True)] if dev.type == "cuda" else [])
    src.to(dev)
    ref = {name: outputs(src, qs, pad, dev, amp) for name, amp in modes}
    src.cpu()
    names = [s.name for s, _ in candidates.ENCODER_CANDIDATES[1:] + candidates.HEAD_CANDIDATES[1:]]
    if args.only:
        names = [x for x in names if x in args.only.split(",")]
    out_path = Path(args.out).with_name(Path(args.out).stem + f"_{dev.type}.json")
    out = {"device": str(dev), "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None, "torch": torch.__version__,
           "n_questions": len(qs), "v1_sha256": (ARTIFACT / "SHA256SUMS").read_text().split()[0], "candidates": {}}
    if dev.type == "cuda":   # reference noise floor: v1 bf16 vs v1 fp32 on the same questions
        out["v1_bf16_vs_fp32"] = compare(ref["fp32"], ref["bf16_autocast"])
    for name in names:
        t0 = time.time()
        model, route, rep = build(src, name)
        params = sum(p.numel() for p in model.parameters())
        model.to(dev)
        res = {"route": route, "params": params, "build_seconds": round(time.time() - t0, 1),
               "report": {k: v for k, v in rep.items() if k not in ("copied",)}}
        for mode, amp in modes:
            res[mode] = compare(ref[mode], outputs(model, qs, pad, dev, amp))
        out["candidates"][name] = res
        print(name, json.dumps({k: res[k] for k in res if k in ("fp32", "bf16_autocast", "params")}), flush=True)
        del model
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
