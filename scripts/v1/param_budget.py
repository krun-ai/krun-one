"""Krun One v1 — parameter budget of the candidate architectures (computed, not estimated).

For each spec in krun_one.v1.arch (reference v0.3, candidates, proxy):
* analytic parameter count, cross-checked against the real modules instantiated on the meta device;
* embedding vs non-embedding split (embeddings cost memory, not latency);
* dense forward FLOPs per token at 128/256/512/1024 tokens, relative to v0.3;
* weight memory (bf16 serving, fp32 artifact) and a training-state estimate (bf16 autocast + fp32 master/Adam).

    uv run python scripts/v1/param_budget.py        # -> results/v1/param_budget.{json,md}
"""

from __future__ import annotations

import json
from pathlib import Path

from krun_one.v1 import arch

ROOT = Path(__file__).resolve().parents[2]
LENGTHS = (128, 256, 512, 1024)


def main() -> None:
    specs = [arch.REFERENCE_V03] + arch.CANDIDATES + [arch.PROXY]
    ref = {L: arch.forward_flops_per_token(arch.REFERENCE_V03, L) for L in LENGTHS}
    rows = []
    for s in specs:
        a = arch.count_parameters(s)
        inst = arch.instantiate_counts(s)
        assert inst["total"] == a["total"], (s.name, inst, a)
        fl = {L: arch.forward_flops_per_token(s, L) for L in LENGTHS}
        rows.append({
            "name": s.name, "notes": s.notes, "spec": s.to_dict(), "params": a, "params_instantiated": inst["total"],
            "layers": s.encoder.num_layers, "hidden": s.encoder.hidden_size, "heads": s.encoder.num_heads,
            "head_dim": s.encoder.head_dim, "ffn": s.encoder.intermediate_size, "vocab": s.encoder.vocab_size,
            "gflops_per_token": {L: round(fl[L] / 1e9, 3) for L in LENGTHS},
            "flops_vs_v03": {L: round(fl[L] / ref[L], 3) for L in LENGTHS},
            "weights_gb_bf16": round(a["total"] * 2 / 1e9, 2), "weights_gb_fp32": round(a["total"] * 4 / 1e9, 2),
            "train_state_gb_adamw": round(a["total"] * 16 / 1e9, 1),
        })
    out = ROOT / "results/v1"
    out.mkdir(parents=True, exist_ok=True)
    (out / "param_budget.json").write_text(json.dumps(rows, indent=2))
    md = ["| spec | L | d | heads (hd) | FFN | vocab | total | embeddings | non-emb | decision stack | "
          "GFLOP/tok @256 | ×v0.3 @256 / @1024 | bf16 GB |", "|" + "---|" * 13]
    for r in rows:
        p = r["params"]
        md.append(f"| {r['name']} | {r['layers']} | {r['hidden']} | {r['heads']} ({r['head_dim']}) | {r['ffn']} | "
                  f"{r['vocab']:,} | {p['total']:,} | {p['encoder_embeddings']:,} | {p['non_embedding']:,} | "
                  f"{p['decision_total']:,} | {r['gflops_per_token'][256]} | {r['flops_vs_v03'][256]} / "
                  f"{r['flops_vs_v03'][1024]} | {r['weights_gb_bf16']} |")
    (out / "param_budget.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
