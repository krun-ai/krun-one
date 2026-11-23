#!/bin/bash
# R3 diagnostic (POST-REGISTRATION, not a candidate): can the capability curriculum be learned at all from the raw state?
# Throwaway runs trained ONLY on the capability examples of the R3 corpus, from the frozen v1, 3,000 steps:
#   diag_cap_l4    last 4 encoder blocks + head  (the R3 trainable set)
#   diag_cap_full  the whole encoder + head      (like the R0 / Tetris diagnostics)
# Evaluated on val_r3 (never on the protected extension). Weights are deleted.
set -u
K=/workspace/krun/experiments/krun_one; cd $K
export HF_HOME=/workspace/hf HF_HUB_CACHE=/workspace/hf/hub TORCH_DISABLE_NATIVE_JIT=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=/workspace/logs; PY=.venv/bin/python
st() { echo "$1 $(date -u +%FT%TZ)" >> $L/status; }
[ -f data/v11/r3/train_cap.jsonl ] || $PY - <<'PY'
import json
n = 0
with open("data/v11/r3/train_cap.jsonl", "w", encoding="utf-8") as out:
    for line in open("data/v11/r3/train_tech.jsonl", encoding="utf-8").read().split("\n"):
        if line and "capability" in json.loads(line)["tags"]:
            out.write(line + "\n"); n += 1
print("capability examples", n)
PY
for J in ${JOBS:-diag_cap_l4 diag_cap_full}; do
  case $J in diag_cap_l4) A="--last-n 4" ;; diag_cap_full) A="--last-n 22 --max-tokens 12288 --grad-ckpt" ;; esac
  if [ ! -f results/v11/eval/$J/tech_val_r3.jsonl.gz ]; then
    $PY scripts/v11/train_v11.py --tag $J --track diag --config configs/v11/r3_protocol.yaml --mode last_n $A --adapter-dim 0 --objective O2 \
        --seed 20261001 --batching family --tech data/v11/r3/train_cap.jsonl --replay none --aux-stats data/v11/r3/aux_stats.json \
        --epochs 4 --max-steps 3000 > $L/$J.log 2>&1
    st "train $J rc=$?"
    $PY scripts/v11/eval_v11.py --ckpt runs/v11/diag/$J --out results/v11/eval/$J --tech data/v11/techbench/val_r3.jsonl > $L/eval_$J.log 2>&1
    st "eval $J rc=$?"
    rm -f runs/v11/diag/$J/model.safetensors
  fi
done
st "DIAG_DONE"
