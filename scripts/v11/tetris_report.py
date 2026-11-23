"""Krun One v1.1 — Tetris representation diagnosis: top-1 per view (A raw, B structured state, C explicit consequence)
on the held-out states, for any evaluated model.   uv run python scripts/v11/tetris_report.py v1 diag_tetris_A_3k ...
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from krun_one.v11 import training as T  # noqa: E402

ex = {e["id"]: e for e in T.load_jsonl(ROOT / "data/v11/tetris_representation/test.jsonl")}
view = lambda e: [t for t in e["tags"] if t.startswith("tetris_")][0][-1]  # noqa: E731
res = {}
for m in sys.argv[1:]:
    p = ROOT / "results/v11/eval" / m / "tech_test.jsonl.gz"
    if not p.exists():
        continue
    acc, pairs, margin = {}, {}, {}
    for line in gzip.open(p, "rt"):
        r = json.loads(line)
        e = ex[r["id"]]
        z = np.asarray(r["z"])
        vals = np.array([c["value"] for c in e["candidates"]], dtype=float)
        key = (view(e), e["k"])
        acc.setdefault(key, []).append(int(z.argmax() == e["gold"]))
        a, b = np.triu_indices(len(z), 1)
        d = vals[a] != vals[b]
        pairs.setdefault(key, []).append(float((((z[a] - z[b]) * (vals[a] - vals[b])) > 0)[d].mean()))
        neg = [i for i in range(len(z)) if i != e["gold"]]
        margin.setdefault(key, []).append(float(z[e["gold"]] - z[max(neg, key=lambda i: vals[i])]))
    res[m] = {f"{v}/k{k}": {"n": len(x), "top1": float(np.mean(x)), "pairs_ordered": float(np.mean(pairs[(v, k)])),
                           "margin_logit": float(np.mean(margin[(v, k)]))} for (v, k), x in sorted(acc.items())}
out = ROOT / "results/v11/tetris_representation"
out.mkdir(parents=True, exist_ok=True)
(out / "tetris_views.json").write_text(json.dumps(res, indent=1))
cells = ["A/k8", "B/k8", "C/k8", "A/k16", "B/k16", "C/k16"]
md = ["# Tetris representation diagnosis — held-out states (chance: 12.5% at k = 8, 6.25% at k = 16)", "",
      "A = raw board; B = A + structured board features; C = A + explicit consequence of every placement.", "",
      "| model | " + " | ".join(f"{c} top-1" for c in cells) + " | pairs ordered A / B / C (k=8) |", "|---|" + "---:|" * len(cells) + "---|"]
for m, r in res.items():
    md.append(f"| {m} | " + " | ".join(f"{100 * r[c]['top1']:.1f}" if c in r else "—" for c in cells) + " | "
              + " / ".join(f"{100 * r[c]['pairs_ordered']:.0f}" if c in r else "—" for c in cells[:3]) + " |")
(out / "tetris_views.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
