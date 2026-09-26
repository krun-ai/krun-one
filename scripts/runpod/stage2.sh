#!/bin/bash
# KRUN-005 stage 2: official runs A and B (auto-resume from the latest checkpoint on crash), then validation eval.
source /workspace/krun_env.sh
cd /workspace/krun/experiments/krun_one
L=/workspace/logs/krun005; PY=.venv/bin/python
for V in A B; do
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 10 > $L/train_${V}_gpu.csv &
  SMI=$!
  echo "train_${V}_start $(date -u +%FT%TZ)" >> $L/times.txt
  RESUME=()
  for attempt in 1 2 3 4; do
    echo "== attempt $attempt ${RESUME[*]} $(date -u +%FT%TZ)" >> $L/train_${V}.log
    $PY scripts/train.py --variant $V "${RESUME[@]}" >> $L/train_${V}.log 2>&1
    rc=$?; echo "== exit $rc" >> $L/train_${V}.log
    [ $rc -eq 0 ] && break
    ls -d runs/$V/checkpoint-*/trainer_state.json >/dev/null 2>&1 && RESUME=(--resume-from-checkpoint latest)
  done
  kill $SMI
  echo "train_${V}_end $(date -u +%FT%TZ) rc=$rc" >> $L/times.txt
  [ $rc -eq 0 ] && $PY scripts/evaluate.py --model runs/$V/final --tag krun_one_$V > $L/eval_krun_one_$V.log 2>&1
  echo "eval_${V}_end $(date -u +%FT%TZ) rc=$?" >> $L/times.txt
done
echo STAGE2_DONE >> $L/times.txt
