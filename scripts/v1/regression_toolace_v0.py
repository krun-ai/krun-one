"""Regression check (KrunBench v0 = regression only): ToolACE items of v0 val_tools / holdout_tools, v1 vs v0.3.
xLAM / xlam-irrelevance items are skipped (excluded source). Writes v1-format records + both signal dumps.

    CUDA_VISIBLE_DEVICES= python scripts/v1/regression_toolace_v0.py
"""
import gzip, json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v1"))
from krun_one.v1 import schema  # noqa: E402

import argparse  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", default=str(ROOT / "runs/v1/ft/krun-one-v1-ft"))
ap.add_argument("--out", default=str(ROOT / "runs/v1/ft/regression_toolace_v0"))
ap.add_argument("--device", default="cpu")
ap.add_argument("--skip-v03", action="store_true")
ap.add_argument("--tag", default="v1")
ap.add_argument("--weights", help="model.safetensors to use instead of <run>/final (Stage 2 milestones)")
ap.add_argument("--spec", help="spec.json when --weights has none next to it")
ap.add_argument("--fp32-scorer", action="store_true", help="diagnostic only: run the scorer MLP in fp32 outside autocast")
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
recs = []
for f in ("val_tools", "holdout_tools"):
    for l in open(ROOT / f"data/v0/{f}.jsonl"):
        d = json.loads(l)
        if d["provenance"].get("dataset") != "toolace":
            continue
        r = schema.from_krunbench_v0(d, split="validation")
        r["metadata"]["v0_file"] = f
        recs.append(r)
(out / "records.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))
print("records", len(recs), flush=True)
torch.set_num_threads(12)  # CPU path
# v1
from safetensors.torch import load_model  # noqa: E402
from krun_one.v1.arch import ModelSpec  # noqa: E402
from krun_one.v1.model import KrunOneV1Model  # noqa: E402
from krun_one.v1.serialization import Budget, Serializer  # noqa: E402
from krun_one.v1.training import dump_signals, encode_records  # noqa: E402
from train_decisions import tokenizer  # noqa: E402

run = Path(args.run)
spec = ModelSpec.from_dict(json.loads(Path(args.spec or (run / "final/spec.json")).read_text()))
m = KrunOneV1Model.from_spec(spec, attn_implementation="sdpa")
load_model(m, args.weights or str(run / "final/model.safetensors"))
if args.fp32_scorer:
    class _FP32(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            with torch.autocast(x.device.type, enabled=False):
                return self.inner(x.float())

    m.scorer = _FP32(m.scorer)
tok = tokenizer()
items = encode_records(Serializer(tok, Budget(1024, 896, max_candidates_per_chunk=64)), recs)
with torch.no_grad():
    dump_signals(m, items, recs, tok.pad_token_id, out / f"signals_{args.tag}.jsonl.gz", device=args.device, batch_tokens=8192)
print("v1 done", flush=True)
del m
if args.skip_v03:
    raise SystemExit(0)
# v0.3
from krun_one.model import KrunOne  # noqa: E402

k = KrunOne.from_pretrained(ROOT / "artifacts/krun-one-v0.3", device=args.device)
with gzip.open(out / "signals_v03.jsonl.gz", "wt") as fh, torch.no_grad():
    for i in range(0, len(recs), 16):
        batch = recs[i:i + 16]
        its = [k.encode(r["context"]["text"], [c["id"] for c in r["candidates"]],
                        {c["id"]: c.get("description") for c in r["candidates"]}, r["task"].get("instructions")) for r in batch]
        for r, z in zip(batch, k.logits(its)):
            ids = [c["id"] for c in r["candidates"]]
            fh.write(json.dumps({"id": r["id"], "answerable": r["answerable"], "gold": [ids.index(x) for x in r["target"]["labels"]],
                                 "rank": [float(x) for x in z[:len(ids)]], "slices": r["slices"], "k": len(ids),
                                 "v0_file": r["metadata"]["v0_file"]}) + "\n")
print("v03 done", flush=True)
