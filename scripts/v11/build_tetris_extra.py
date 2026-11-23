"""Exploratory (NOT in the registered R0 protocol): a larger Tetris-only training file for the question "can the raw
state -> consequence computation be learned at all, and does model size change it?".

    uv run python scripts/v11/build_tetris_extra.py     # -> data/v11/probe/ltrain_tetris.jsonl (30,000 problems)
"""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/v11"))

import build_probe as bp  # noqa: E402
from krun_one.v11 import envs, harness  # noqa: E402

ser = harness.serializer(ROOT / "artifacts/krun-one-v1-text-final")
b = bp.Builder(ser, 20261002)
b.seen = {p["hash"] for f in ("probe", "ltrain") for p in harness.load_problems(bp.OUT / f"{f}.jsonl") if "hash" in p}
for k, n in ((8, 20000), (16, 10000)):
    b.fill(envs.gen_tetris, n, k, f"lx-tetris-k{k}", max_tries=2_000_000)
path = bp.OUT / "ltrain_tetris.jsonl"
path.write_text("".join(json.dumps(p) + "\n" for p in b.items))
print(len(b.items), hashlib.sha256(path.read_bytes()).hexdigest(), dict(b.dropped))
