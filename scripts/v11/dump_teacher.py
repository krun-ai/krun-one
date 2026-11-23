"""Krun One v1.1 — the frozen v1's own outputs on the replay sample (targets of the replay-distillation term O4).

    python scripts/v11/dump_teacher.py     # -> data/v11/corpus/teacher_replay.jsonl.gz  ({id, rank, apply})
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from krun_one.v11 import harness, training as T  # noqa: E402

ART = ROOT / "artifacts/krun-one-v1-text-final"


@torch.inference_mode()
def main() -> None:
    dev = torch.device("cuda")
    ser = harness.serializer(ART)
    model = harness.load_v1(ART).to(dev).eval()
    recs = T.load_jsonl(ROOT / "data/v11/corpus/train_replay.jsonl")
    qs = [ser.encode_record(r) for r in recs]
    n = 0
    with gzip.open(ROOT / "data/v11/corpus/teacher_replay.jsonl.gz", "wt") as f:
        for idx in harness.batches_by_length([harness.length(q) for q in qs], 24_000):
            m = harness.forward_questions(model, [qs[i] for i in idx], ser.tok.pad_token_id, dev)
            z, ap = m["rank"].float().cpu().numpy(), m["apply"].float().cpu().numpy()
            for j, i in enumerate(idx):
                k = qs[i].n_candidates
                f.write(json.dumps({"id": recs[i]["id"], "rank": [round(float(x), 4) for x in z[j, :k]],
                                    "apply": [round(float(x), 4) for x in ap[j, :k]]}) + "\n")
                n += 1
    print(json.dumps({"teacher_rows": n}))


if __name__ == "__main__":
    main()
