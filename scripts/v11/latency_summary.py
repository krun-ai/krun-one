"""Krun One v1.1 — latency / VRAM table of the candidates vs the owner's preferred gates (offline).

    uv run python scripts/v11/latency_summary.py     # -> results/v11/architecture_search/latency_rtx4090.md
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "results/v11/architecture_search"
d = json.loads((D / "latency_rtx4090.json").read_text())
budget = {r["name"]: r for part in json.loads((D / "param_budget.json").read_text()).values() for r in part}
W = ["typical_1q_k4_ctx48", "typical_1q_tools8_ctx48", "typical_4q_mixed_ctx96", "k_8_tools_ctx64", "k_32_tools_ctx64",
     "k_64_tools_ctx64", "k_128_tools_ctx64", "len_1024_k4", "stress_16q_tools32_ctx512", "throughput_32req_k4_ctx48"]
by = {(r["spec"], r["dtype"], r["workload"]): r for r in d["results"]}
specs = list(dict.fromkeys(r["spec"] for r in d["results"]))
md = [f"# Krun One v1.1 candidates — latency and memory ({d['gpu']}, torch {d['torch']}, random weights, {d['iters']} iterations, en-US)", "",
      "`total` = tokenize + collate + host→device + forward + merge/softmax (v1 harness `scripts/v1/latency_proxy.py`). "
      "The v1 shape (first row) is re-measured in the same process, so ratios are same-host.", "",
      "Preferred gates (owner): typical ≤ 25–30 ms, k=128 ≤ 60–70 ms.", ""]
for dtype, label in (("autocast", "fp32 weights + bf16 autocast (the production runtime numerics)"), ("bf16", "bf16 weights")):
    md += [f"## {label}: total p50 ms (forward p50 in brackets)", "",
           "| candidate | params | typical k4 | typical 8 tools | 4 questions | k=32 | k=64 | k=128 | 1024 tokens | 16q x 32 tools | 32 requests batch | typical gate | k128 gate |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for s in specs:
        g = lambda w: by[(s, dtype, w)]  # noqa: E731
        typ = max(g(w)["total_p50_ms"] for w in W[:3])
        k128 = g("k_128_tools_ctx64")["total_p50_ms"]
        gate = lambda v, lo, hi: "✅" if v <= lo else ("⚠️ within upper bound" if v <= hi else "❌")  # noqa: E731
        cells = [f"{g(w)['total_p50_ms']:.1f} ({g(w)['forward_p50_ms']:.1f})" for w in (W[0], W[1], W[2], W[4], W[5], W[6], W[7], W[8], W[9])]
        md.append(f"| {s} | {budget[s]['params']['total'] / 1e6:.0f}M | " + " | ".join(cells) + f" | {gate(typ, 25, 30)} | {gate(k128, 60, 70)} |")
    md.append("")
md += ["## Memory (VRAM in MiB; process RSS is the max over the whole sweep)", "",
       "| candidate | dtype | weights on GPU | peak typical | peak k=128 | peak 16q x 32 tools | peak 32-request batch |", "|---|---|---:|---:|---:|---:|---:|"]
for k, m in d["memory"].items():
    p = m["peak_vram_mib_by_workload"]
    md.append(f"| {k.split('/')[0]} | {k.split('/')[1]} | {m['weights_vram_mib']} | {p[W[1]]} | {p[W[6]]} | {p[W[8]]} | {p[W[9]]} |")
md += ["", f"Host process max RSS: {max(m['process_max_rss_mib'] for m in d['memory'].values())} MiB. "
       "An encoder has no KV cache: memory is weights + one forward's activations, which scale with padded tokens x sequences in the batch.", ""]
(D / "latency_rtx4090.md").write_text("\n".join(md))
print("\n".join(md))
