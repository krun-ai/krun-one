#!/bin/bash
# Pod A, phase 1: v1's outputs on the replay sample (distillation targets), the v1 baseline dumps on this host, and the
# registered objective pilot (O0..O4 on the R2 configuration, the same 25% subset, one seed). Pilot weights are deleted.
set -u
K=/workspace/krun/experiments/krun_one; cd $K
export HF_HUB_CACHE=/workspace/hfcache TORCH_DISABLE_NATIVE_JIT=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=/workspace/logs; PY=.venv/bin/python
st() { echo "$1 $(date -u +%FT%TZ)" >> $L/status; }
(nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 60 >> $L/gpu.csv &)
sha256sum -c inputs_v11_r12.sha256 > $L/inputs_check.log 2>&1 || { st "FAILED input checksums"; exit 1; }
st "inputs verified"
[ -f data/v11/corpus/teacher_replay.jsonl.gz ] || $PY scripts/v11/dump_teacher.py > $L/teacher.log 2>&1
st "teacher rc=$?"
[ -f results/v11/eval/v1/latency.json ] || $PY scripts/v11/eval_v11.py --ckpt v1 --out results/v11/eval/v1 \
    --tech data/v11/techbench/val.jsonl data/v11/tetris_representation/test.jsonl --regression --latency > $L/eval_v1.log 2>&1
st "eval v1 rc=$?"
for O in O0 O1 O2 O3 O4; do
  T=pilot_$O
  if [ ! -f results/v11/eval/$T/signals_val_fit.jsonl.gz ]; then
    [ -f runs/v11/pilot/$T/run.json ] || $PY scripts/v11/train_v11.py --tag $T --track pilot --mode last_n --last-n 2 --objective $O \
        --seed 20261001 --subset-frac 0.25 > $L/$T.log 2>&1
    st "train $T rc=$?"
    $PY scripts/v11/eval_v11.py --ckpt runs/v11/pilot/$T --out results/v11/eval/$T --tech data/v11/techbench/val.jsonl \
        --regression --v-files val_fit > $L/eval_$T.log 2>&1
    st "eval $T rc=$?"
    rm -f runs/v11/pilot/$T/model.safetensors
  fi
done
st PHASE1_DONE
