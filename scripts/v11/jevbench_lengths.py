"""JevBench public items — AGGREGATE length / truncation statistics only (no item text is printed or stored).

    uv run python scripts/v11/jevbench_lengths.py            # writes results/v11/jevbench/lengths.json

``full_tokens`` = tokens of the whole request if nothing were truncated (state + instructions + options + specials);
``lost_tokens`` = full_tokens - tokens actually fed (1,024-token sequence, serializer truncation of the state).
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/v11"))

import jevbench_eval as J  # noqa: E402
from krun_one.v11 import harness  # noqa: E402

BUCKETS = [(0, 512), (513, 1024), (1025, 2048), (2049, 4096), (4097, 10**9)]


def bucket(n: int) -> str:
    for a, b in BUCKETS:
        if a <= n <= b:
            return f"{a}-{b}" if b < 10**9 else f">{a - 1}"
    raise ValueError(n)


def lengths() -> dict[str, dict]:
    ser = harness.serializer(ROOT / "artifacts/krun-one-v1-text-final")
    out = {}
    for it in J.items():
        q, _ = J.encode(ser, it)
        fed = harness.length(q)
        ctx = it["state"] if isinstance(it["state"], str) else json.dumps(it["state"], ensure_ascii=False, separators=(",", ":"))
        ctx_ids = len(ser.tok(ctx, add_special_tokens=False)["input_ids"])
        kept = sum(int(c.context_tokens) if hasattr(c, "context_tokens") else 0 for c in q.chunks)
        trunc = any(c.context_truncated for c in q.chunks)
        # the serializer keeps the head (question + options) and cuts the context: lost = context tokens that did not fit
        head = fed - (kept if kept else 0)
        full = fed if not trunc else None
        out[it["id"]] = {"tier": it["tier"], "family": it["family"], "type": it["question"]["type"], "fed": fed, "ctx_tokens": ctx_ids,
                         "truncated": trunc, "chunks": len(q.chunks), "state_is_json": not isinstance(it["state"], str),
                         "n_labels": len(it.get("labels") or [])}
    # full length: for truncated items = fed - (context kept) + ctx_tokens; the context kept is fed - (non-context tokens);
    # non-context tokens are measured by re-encoding the same request with an empty state
    for it in J.items():
        e = dict(it, state="")
        q0, _ = J.encode(ser, e)
        o = out[it["id"]]
        o["full_tokens"] = harness.length(q0) + o["ctx_tokens"] if o["chunks"] == 1 else None
        o["lost_tokens"] = max(0, (o["full_tokens"] or o["fed"]) - o["fed"]) if o["truncated"] else 0
    return out


def main() -> None:
    L = lengths()
    pub = json.loads((J.D / "per_task_v1.2.json").read_text())
    topic = {t["id"]: t["topic"] for t in pub["tasks"] if t["public"]}
    agg = defaultdict(list)
    for i, o in L.items():
        o["topic"] = topic.get(i)
        o["bucket"] = bucket(o["full_tokens"] or o["fed"])
        agg[o["tier"]].append(o)
    res = {"n": len(L), "by_tier": {}, "by_bucket": dict(Counter(o["bucket"] for o in L.values())),
           "hard_by_topic": {}, "types": dict(Counter(o["type"] for o in L.values())),
           "state_is_json": float(np.mean([o["state_is_json"] for o in L.values()])), "items": L}
    for t, v in agg.items():
        res["by_tier"][t] = {"n": len(v), "truncated": float(np.mean([o["truncated"] for o in v])),
                             "median_full": float(np.median([o["full_tokens"] or o["fed"] for o in v])),
                             "max_full": int(max(o["full_tokens"] or o["fed"] for o in v)),
                             "buckets": dict(Counter(o["bucket"] for o in v)), "types": dict(Counter(o["type"] for o in v))}
    tp = defaultdict(list)
    for o in L.values():
        if o["tier"] == "hard":
            tp[o["topic"]].append(o)
    for t, v in sorted(tp.items()):
        res["hard_by_topic"][t] = {"n": len(v), "truncated": float(np.mean([o["truncated"] for o in v])),
                                   "median_full": float(np.median([o["full_tokens"] or o["fed"] for o in v])),
                                   "max_full": int(max(o["full_tokens"] or o["fed"] for o in v)),
                                   "median_lost_if_truncated": float(np.median([o["lost_tokens"] for o in v if o["truncated"]] or [0])),
                                   "types": dict(Counter(o["type"] for o in v)), "median_labels": float(np.median([o["n_labels"] for o in v]))}
    (J.OUT / "lengths.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "items"}, indent=1))


if __name__ == "__main__":
    main()
