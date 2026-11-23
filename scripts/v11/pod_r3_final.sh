#!/bin/bash
# R3 final evaluation on THIS host. v1 first (same-host reference for every comparison), then each model.
#   MODELS="r3f_s1 r3f_s2" [R2=1] bash scripts/v11/pod_r3_final.sh
# Per model: KrunTechBench v1 (labels restored) + the R3 extension (protected, opened only here), validation files, the
# Tetris views, the full v1 regression inputs, text latency / VRAM / RSS, JevBench public items (evaluation only).
# R2=1 also dumps the kept R2 seed (r2l4na_s1) on the NEW files only (its other dumps exist from R1 / R2).
set -u
K=/workspace/krun/experiments/krun_one; cd $K
export HF_HOME=/workspace/hf HF_HUB_CACHE=/workspace/hf/hub TORCH_DISABLE_NATIVE_JIT=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=/workspace/logs; PY=.venv/bin/python
st() { echo "$1 $(date -u +%FT%TZ)" >> $L/status; }
B=data/v11/techbench
TECH="$B/protected_v1_labeled.jsonl $B/protected_r3.jsonl $B/val_v1_labeled.jsonl $B/val_r3.jsonl data/v11/tetris_representation/test.jsonl"
[ -f results/v11/eval/v1/latency.json ] && [ -f results/v11/eval/v1/tech_protected_r3.jsonl.gz ] || \
  $PY scripts/v11/eval_v11.py --ckpt v1 --out results/v11/eval/v1 --tech $TECH --regression --latency > $L/final_v1.log 2>&1
st "final v1 rc=$?"
[ -f results/v11/jevbench/v1_r3host.json ] || $PY scripts/v11/jevbench_eval.py --ckpt v1 --tag v1_r3host >> $L/final_v1.log 2>&1
if [ "${R2:-0}" = 1 ]; then
  [ -f results/v11/eval/r2l4na_s1/tech_val_r3.jsonl.gz ] || $PY scripts/v11/eval_v11.py --ckpt runs/v11/r2/r2l4na_s1 --out results/v11/eval/r2l4na_s1 \
      --tech $B/protected_r3.jsonl $B/val_r3.jsonl $B/protected_v1_labeled.jsonl > $L/final_r2.log 2>&1
  st "final r2l4na_s1 rc=$?"
fi
for T in $MODELS; do
  D=runs/v11/r3/text/$T
  [ -f results/v11/eval/$T/latency.json ] && [ -f results/v11/eval/$T/tech_protected_r3.jsonl.gz ] || \
    $PY scripts/v11/eval_v11.py --ckpt $D --out results/v11/eval/$T --tech $TECH --regression --latency > $L/final_$T.log 2>&1
  [ -f results/v11/jevbench/$T.json ] || $PY scripts/v11/jevbench_eval.py --ckpt $D --tag $T >> $L/final_$T.log 2>&1
  st "final $T rc=$?"
done
(cd results/v11 && find eval jevbench -type f | sort | xargs sha256sum > $L/results_sha256.txt)
st "FINAL_DONE $MODELS"
