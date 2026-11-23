#!/bin/bash
# Multimodal M1.5c regression (nothing retrained): the registered fusion weights on top of each text core, same host.
#   CORES="v1:v1 r1_s1:runs/v11/r1/r1_s1 r2l4_s1:runs/v11/r2/r2l4_s1" bash scripts/v11/pod_mm_run.sh
set -u
K=/workspace/krun/experiments/krun_one; cd $K
export HF_HOME=/workspace/hf HF_HUB_CACHE=/workspace/hf/hub TORCH_DISABLE_NATIVE_JIT=1 PYTHONUNBUFFERED=1 \
       TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True KRUN_M15_CONFIG=configs/v1/m15c.yaml
L=/workspace/logs; PY=.venv/bin/python
st() { echo "$1 $(date -u +%FT%TZ)" >> $L/status_mm; }
for C in $CORES; do
  T=${C%%:*}; P=${C#*:}
  [ -f results/v11/multimodal/$T.json ] || $PY scripts/v11/mm_eval.py --ckpt $P --tag $T > $L/mm_$T.log 2>&1
  st "mm $T rc=$?"
done
st "MM_DONE $CORES"
