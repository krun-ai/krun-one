"""Summarize a latency_proxy.py JSON into markdown tables (worker-side model total p50 and forward p50).

    uv run python scripts/v1/summarize_latency.py results/v1/latency_rtx4090_sdpa.json > results/v1/latency_rtx4090_sdpa.md
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict


def main(path: str) -> None:
    d = json.load(open(path))
    rows = d["results"]
    specs = list(dict.fromkeys(r["spec"] for r in rows))
    wls = list(dict.fromkeys(r["workload"] for r in rows))
    ref = specs[0]
    print(f"# Latency proxy — {d['gpu']} (torch {d['torch']}, CUDA {d['cuda']}, transformers {d.get('transformers')}, "
          f"attn={d['attn']}, {d['iters']} iters)\n")
    print("Random-init weights (latency does not depend on values); real serializer + tokenizer; contexts MASSIVE, "
          "candidates real tool strings. `total` = tokenize + collate + H2D + forward + post (the model-specific part "
          "of the worker). Ratio = spec / v0.3 reference in the same mode.\n")
    for dtype in dict.fromkeys(r["dtype"] for r in rows):
        for loc in dict.fromkeys(r["locale"] for r in rows):
            t = {(r["spec"], r["workload"]): r for r in rows if r["dtype"] == dtype and r["locale"] == loc}
            print(f"## weights={dtype}, context={loc} — total p50 ms (p95) [ratio vs v0.3]\n")
            print("| workload | seqs×len (v1) | " + " | ".join(specs) + " |")
            print("|---|---|" + "---|" * len(specs))
            for w in wls:
                base = t.get((ref, w))
                cells = []
                for s in specs:
                    r = t.get((s, w))
                    if not r:
                        cells.append("—")
                        continue
                    ratio = f" [{r['total_p50_ms'] / base['total_p50_ms']:.2f}×]" if base and s != ref else ""
                    cells.append(f"{r['total_p50_ms']:.1f} ({r['total_p95_ms']:.1f}){ratio}")
                v1 = t.get((specs[1], w)) if len(specs) > 1 else base
                shape = f"{v1['sequences']}×{v1['padded_len']}" if v1 else ""
                print(f"| {w} | {shape} | " + " | ".join(cells) + " |")
            print()
    stage = defaultdict(dict)
    for r in rows:
        if r["dtype"] == "bf16" and r["locale"] == "pt-PT" and r["workload"] in ("typical_1q_k4_ctx48", "typical_4q_mixed_ctx96", "q_16_tools8_ctx256"):
            stage[(r["spec"], r["workload"])] = r
    print("## stage breakdown p50 ms (bf16 weights, pt-PT)\n")
    print("| spec | workload | tokenize | collate | h2d | forward | post | total |")
    print("|---|---|---|---|---|---|---|---|")
    for (s, w), r in stage.items():
        print(f"| {s} | {w} | {r['tokenize_p50_ms']:.2f} | {r['collate_p50_ms']:.2f} | {r['h2d_p50_ms']:.2f} | "
              f"{r['forward_p50_ms']:.2f} | {r['post_p50_ms']:.2f} | {r['total_p50_ms']:.2f} |")
    if d.get("peak_vram_mib"):
        print("\n## peak VRAM (MiB)\n")
        for k, v in d["peak_vram_mib"].items():
            print(f"- {k}: {v}")


if __name__ == "__main__":
    main(sys.argv[1])
