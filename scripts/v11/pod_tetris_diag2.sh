#!/bin/bash
# Tetris representation diagnostic at the REGISTERED budget (3,000 steps = several passes over the 20,000 training states
# of one view). The first runner trained one pass only (833 steps); those results are kept as the 833-step points.
set -u
K=/workspace/krun/experiments/krun_one; cd $K
export HF_HUB_CACHE=/workspace/hfcache TORCH_DISABLE_NATIVE_JIT=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=/workspace/logs; PY=.venv/bin/python; TR=data/v11/tetris_representation
st() { echo "$1 $(date -u +%FT%TZ)" >> $L/status; }
until grep -q TETRIS_DONE $L/status; do sleep 20; done
for V in B A; do
  T=diag_tetris_${V}_3k
  if [ ! -f results/v11/eval/$T/tech_test.jsonl.gz ]; then
    $PY scripts/v11/train_v11.py --tag $T --track diag --mode last_n --last-n 22 --objective O0 --adapter-dim 0 --seed 20261001 \
        --tech $TR/train_$V.jsonl --replay none --epochs 4 --max-steps 3000 --max-tokens 12288 --grad-ckpt > $L/$T.log 2>&1
    st "train $T rc=$?"
    $PY scripts/v11/eval_v11.py --ckpt runs/v11/diag/$T --out results/v11/eval/$T --tech $TR/test.jsonl > $L/eval_$T.log 2>&1
    st "eval $T rc=$?"
    rm -f runs/v11/diag/$T/model.safetensors
  fi
done
st TETRIS3K_DONE
