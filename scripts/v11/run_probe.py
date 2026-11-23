"""Krun One v1.1 R0 — zero-shot Technical Capacity Probe: dump the frozen v1's rank logits on every probe problem.

    python scripts/v11/run_probe.py --out results/v11/capacity_probe/v1_zero_shot/signals.jsonl.gz [--limit N --device cpu]

No training, no weight is written; the v1 artifact is only read (SHA256SUMS verified by the loader). Metrics are
computed offline from the dump by scripts/v11/analyze_probe.py.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from krun_one.v11 import harness, probe  # noqa: E402


def dump(model, ser, problems, out: Path, dev, extra: dict | None = None) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n = 0
    with gzip.open(out, "wt") as f:
        for cond in probe.CONDITIONS:
            qs = [harness.encode(ser, p, cond) for p in problems]
            zs = harness.rank_logits(model, qs, ser.tok.pad_token_id, dev)
            for p, q, z in zip(problems, qs, zs):
                f.write(json.dumps({"id": p["id"], "cond": cond, "z": [round(float(x), 5) for x in z],
                                    "tokens": harness.length(q), "chunks": len(q.chunks)}) + "\n")
                n += 1
    meta = {"rows": n, "seconds": round(time.time() - t0, 1), "device": str(dev), "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None, **(extra or {})}
    out.with_suffix("").with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default=str(ROOT / "artifacts/krun-one-v1-text-final"))
    ap.add_argument("--probe", default=str(ROOT / "data/v11/probe/probe.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, help="random subset (smoke)")
    args = ap.parse_args()
    from krun_one.v1.runtime import TextArtifact

    art = TextArtifact.load(args.artifact, device=args.device, dtype="bf16" if args.device == "cuda" else "fp32")
    problems = harness.load_problems(args.probe)
    if args.limit:
        problems = random.Random(0).sample(problems, args.limit)
    sha = (Path(args.artifact) / "SHA256SUMS").read_text().split()[0]
    print(json.dumps(dump(art.model, art.ser, problems, Path(args.out), art.device,
                          {"model": art.model_id, "artifact_first_sha256": sha, "n_problems": len(problems)})))


if __name__ == "__main__":
    main()
