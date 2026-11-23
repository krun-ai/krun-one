#!/bin/bash
# Tetris representation diagnostic (configs/v11/r1_r2_protocol.yaml: tetris_representation.dedicated_diagnostic).
# Throwaway full fine-tunes of the frozen v1 on ONE view each; the weights are deleted after the evaluation dump.
set -u
K=/workspace/krun/experiments/krun_one; cd $K
export HF_HUB_CACHE=/workspace/hfcache TORCH_DISABLE_NATIVE_JIT=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=/workspace/logs; PY=.venv/bin/python; TR=data/v11/tetris_representation
st() { echo "$1 $(date -u +%FT%TZ)" >> $L/status; }
[ -f results/v11/eval/v1/tech_test.jsonl.gz ] || $PY scripts/v11/eval_v11.py --ckpt v1 --out results/v11/eval/v1 --tech $TR/test.jsonl > $L/eval_v1_tetris.log 2>&1
st "v1 tetris rc=$?"
run() {  # view steps
  T=diag_tetris_$1
  if [ ! -f results/v11/eval/$T/tech_test.jsonl.gz ]; then
    [ -f runs/v11/diag/$T/run.json ] || $PY scripts/v11/train_v11.py --tag $T --track diag --mode last_n --last-n 22 --objective O0 --adapter-dim 0 \
        --seed 20261001 --tech $TR/train_$1.jsonl --replay none --max-steps $2 --max-tokens 12288 --grad-ckpt > $L/$T.log 2>&1
    st "train $T rc=$?"
    $PY scripts/v11/eval_v11.py --ckpt runs/v11/diag/$T --out results/v11/eval/$T --tech $TR/test.jsonl > $L/eval_$T.log 2>&1
    st "eval $T rc=$?"
    rm -f runs/v11/diag/$T/model.safetensors
  fi
}
run C 1000
run B 3000
run A 3000
st TETRIS_DONE
