#!/usr/bin/env python
"""On the pod: download the upstream Laya files at the exact commit of the local ./laya checkout and verify sha256.
Only reads from the Hub; the result is a read-only copy at /workspace/krun/laya (never modified)."""
import hashlib
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO, REV = "convaiinnovations/laya", "1c5edc17a7acd8701df6fc341c0d179f1c62c982"
FILES = {
    "model.safetensors": "891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c",
    "encoder/config.json": "bf3ab80598fdccf414855a2ce80f22859e4492d06ca8a62ddd1cfb63972f8979",
    "tokenizer/tokenizer.json": "6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30",
    "tokenizer/tokenizer_config.json": "50044de60daaa73df97d262e15a40d4faf0160e7d742df64b377877a1320dd12",
    "rl_agent_config.json": "ae287b56bbcf5f8c4f4541ae9dfd00c914c4c48b940b8398c3058af37ba92bbd",
    "rl_common.py": "8d83611d480c971d640a7b7d3aa2f2219c5e8455e9cc2329fd073681bd8be23e",
}
dst = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/krun/laya")
for f, want in FILES.items():
    out = dst / f
    if not out.exists() or hashlib.sha256(out.read_bytes()).hexdigest() != want:
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(hf_hub_download(REPO, f, revision=REV), out)
    got = hashlib.sha256(out.read_bytes()).hexdigest()
    print(("OK  " if got == want else "BAD ") + f, got)
    if got != want:
        sys.exit(1)
