#!/bin/bash
# R3 text runs (configs/v11/r3_protocol.yaml): from the frozen v1, last 4 blocks, no adapter, objective O2, one pass.
#   JOBS="r3f_s1 r3r_s1" bash scripts/v11/pod_r3_text.sh        f = family-aware batching, r = random batching
# After each run: validation dumps (KrunTechVal v1 with labels, val_r3, KrunBench val_fit). Only the exported
# v1-architecture weights are kept (model_v1.safetensors + its sha256).
set -u
K=/workspace/krun/experiments/krun_one; cd $K
export HF_HOME=/workspace/hf HF_HUB_CACHE=/workspace/hf/hub TORCH_DISABLE_NATIVE_JIT=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=/workspace/logs; PY=.venv/bin/python
st() { echo "$1 $(date -u +%FT%TZ)" >> $L/status; }
for J in $JOBS; do
  case $J in *_s1) SEED=20261001 ;; *_s2) SEED=20261002 ;; *) st "bad job $J"; continue ;; esac
  case $J in r3f_*) BATCH=family ;; r3r_*) BATCH=random ;; *) st "bad job $J"; continue ;; esac
  D=runs/v11/r3/text/$J
  if [ ! -f $D/run.json ]; then
    $PY scripts/v11/train_v11.py --tag $J --track r3/text --config configs/v11/r3_protocol.yaml --mode last_n --last-n 4 --adapter-dim 0 \
        --objective O2 --seed $SEED --batching $BATCH --tech data/v11/r3/train_tech.jsonl --replay data/v11/r3/train_replay.jsonl \
        --aux-stats data/v11/r3/aux_stats.json --export-v1 > $L/train_$J.log 2>&1
  fi
  st "train $J rc=$?"
  if [ -f $D/model_v1.safetensors ]; then
    [ -f $D/model_v1.sha256 ] || sha256sum $D/model_v1.safetensors > $D/model_v1.sha256
    rm -f $D/model.safetensors
  fi
  [ -f results/v11/eval/$J/signals_val_fit.jsonl.gz ] || $PY scripts/v11/eval_v11.py --ckpt $D --out results/v11/eval/$J \
      --tech data/v11/techbench/val_v1_labeled.jsonl data/v11/techbench/val_r3.jsonl --regression --v-files val_fit > $L/eval_$J.log 2>&1
  st "eval $J rc=$?"
done
st "TEXT_DONE $JOBS"
