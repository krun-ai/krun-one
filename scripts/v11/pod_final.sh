#!/bin/bash
# Final evaluation of the selected candidates on THIS host: KrunTechBench v1 (protected — opened here for the registered
# comparison only), the Tetris representation test, the full v1 regression inputs and text latency. v1 is dumped on
# the same host first so every comparison is same-host.
#   MODELS="r2/r2_s1 r2/r2_s2" bash scripts/v11/pod_final.sh
set -u
K=/workspace/krun/experiments/krun_one; cd $K
export HF_HUB_CACHE=/workspace/hfcache TORCH_DISABLE_NATIVE_JIT=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=/workspace/logs; PY=.venv/bin/python
st() { echo "$1 $(date -u +%FT%TZ)" >> $L/status; }
TECH="data/v11/techbench/protected.jsonl data/v11/techbench/val.jsonl data/v11/tetris_representation/test.jsonl"
[ -f results/v11/eval/v1/tech_protected.jsonl.gz ] && [ -f results/v11/eval/v1/latency.json ] && [ -f results/v11/eval/v1/signals_val_eval.jsonl.gz ] || \
  $PY scripts/v11/eval_v11.py --ckpt v1 --out results/v11/eval/v1 --tech $TECH --regression --latency > $L/final_v1.log 2>&1
st "final v1 rc=$?"
[ -f results/v11/jevbench/v1_gpu.json ] || $PY scripts/v11/jevbench_eval.py --ckpt v1 --tag v1_gpu >> $L/final_v1.log 2>&1
for M in $MODELS; do
  T=$(basename $M)
  [ -f results/v11/eval/$T/latency.json ] && [ -f results/v11/eval/$T/tech_protected.jsonl.gz ] || \
    $PY scripts/v11/eval_v11.py --ckpt runs/v11/$M --out results/v11/eval/$T --tech $TECH --regression --latency > $L/final_$T.log 2>&1
  [ -f results/v11/jevbench/$T.json ] || $PY scripts/v11/jevbench_eval.py --ckpt runs/v11/$M --tag $T >> $L/final_$T.log 2>&1
  st "final $T rc=$?"
done
(cd results/v11/eval && find . -type f | sort | xargs sha256sum > $L/eval_sha256.txt)
st "FINAL_DONE $MODELS"
