"""Krun One v1.1 — EXACT parameter counts of the architecture candidates (analytic, cross-checked by instantiating
every module on the meta device), FLOPs per token and weight memory.

    uv run python scripts/v11/arch_budget.py      # -> results/v11/architecture_search/param_budget.{json,md}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from krun_one.v1 import arch  # noqa: E402
from krun_one.v11 import candidates  # noqa: E402
from krun_one.v11.model import KrunOneV11Model, count_parameters_v11  # noqa: E402

LENGTHS = (128, 512, 1024)


def row(spec, route: str) -> dict:
    a = count_parameters_v11(spec)
    with torch.device("meta"):
        m = KrunOneV11Model.from_spec(spec)
    inst = sum(p.numel() for p in m.parameters())
    assert inst == a["total"], (spec.name, inst, a["total"])
    e = spec.base.encoder
    fl = {L: arch.forward_flops_per_token(spec.base, L) for L in LENGTHS}
    ref = {L: arch.forward_flops_per_token(candidates.V1, L) for L in LENGTHS}
    return {"name": spec.name, "growth_from_v1": route, "notes": spec.base.notes, "spec": spec.to_dict(), "params": a,
            "params_instantiated": inst, "hidden": e.hidden_size, "layers": e.num_layers, "heads": e.num_heads,
            "head_dim": e.head_dim, "ffn": e.intermediate_size, "vocab": e.vocab_size,
            "head_layers": spec.base.decision.head_layers, "head_ffn_mult": spec.base.decision.head_ffn_mult,
            "cand_layers": spec.cand_layers, "gflops_per_token": {L: round(fl[L] / 1e9, 3) for L in LENGTHS},
            "flops_vs_v1": {L: round(fl[L] / ref[L], 3) for L in LENGTHS},
            "weights_gb_bf16": round(a["total"] * 2 / 1e9, 2), "weights_gb_fp32": round(a["total"] * 4 / 1e9, 2),
            "train_state_gb_adamw": round(a["total"] * 16 / 1e9, 1)}


def table(rows: list[dict]) -> list[str]:
    out = ["| candidate | total params | non-embedding | embeddings | encoder body | decision stack | hidden | layers | heads | "
           "FFN (GLU) | head | FLOPs/token vs v1 (512 tok) | weights bf16 / fp32 (GB) | init from v1 |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---|"]
    for r in rows:
        p = r["params"]
        head = f"{r['head_layers']}L ffn x{r['head_ffn_mult']}" + (f" + {r['cand_layers']} cand" if r["cand_layers"] else "")
        out.append(f"| {r['name']} | {p['total']:,} | {p['non_embedding']:,} | {p['encoder_embeddings']:,} | {p['encoder_body']:,} | "
                   f"{p['decision_total']:,} | {r['hidden']} | {r['layers']} | {r['heads']} | {r['ffn']} | {head} | "
                   f"{r['flops_vs_v1'][512]:.2f}x | {r['weights_gb_bf16']} / {r['weights_gb_fp32']} | {r['growth_from_v1']} |")
    return out


def main() -> None:
    enc = [row(s, g) for s, g in candidates.ENCODER_CANDIDATES]
    head = [row(s, g) for s, g in candidates.HEAD_CANDIDATES]
    out = ROOT / "results/v11/architecture_search"
    out.mkdir(parents=True, exist_ok=True)
    (out / "param_budget.json").write_text(json.dumps({"encoder_candidates": enc, "head_candidates": head}, indent=2))
    md = ["# Krun One v1.1 — parameter budget (computed, cross-checked against instantiated modules)", "",
          "## Encoder candidates", ""] + table(enc) + ["", "## Decision-head candidates (v1 encoder)", ""] + table(head) + [""]
    (out / "param_budget.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
