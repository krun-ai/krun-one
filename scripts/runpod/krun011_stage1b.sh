#!/bin/bash
# KRUN-011 stage 1b: continuation after the stage-1 shell was killed (SSH session closed) during M40_CE:
# resume M40_CE from its latest checkpoint (auto-resume), then M50_RPS; validation-only evals.
source /workspace/krun_env.sh
cd /workspace/krun/experiments/krun_one
L=/workspace/logs/krun011; PY=.venv/bin/python
t() { echo "$1 $(date -u +%FT%TZ)" >> $L/times.txt; }
t stage1b_start
for V in M40_CE M50_RPS; do
  t "train_${V}_start"
  RESUME=()
  ls -d runs/v02_$V/checkpoint-*/trainer_state.json >/dev/null 2>&1 && RESUME=(--resume-from-checkpoint latest)
  for attempt in 1 2 3 4; do
    echo "== attempt $attempt ${RESUME[*]} $(date -u +%FT%TZ)" >> $L/train_${V}.log
    $PY scripts/train_primitives.py --variant $V "${RESUME[@]}" >> $L/train_${V}.log 2>&1
    rc=$?; echo "== exit $rc" >> $L/train_${V}.log
    [ $rc -eq 0 ] && break
    ls -d runs/v02_$V/checkpoint-*/trainer_state.json >/dev/null 2>&1 && RESUME=(--resume-from-checkpoint latest)
  done
  t "train_${V}_end rc=$rc"
  if [ $rc -eq 0 ]; then
    rm -rf runs/v02_$V/checkpoint-*   # volume quota: resume state is not needed once final/ exists
    $PY scripts/evaluate.py --model runs/v02_$V/final --tag v02_$V > $L/eval_choice_$V.log 2>&1; t "eval_choice_$V rc=$?"
    $PY scripts/evaluate_primitives.py --model runs/v02_$V/final --tag v02_$V --no-test > $L/eval_prim_$V.log 2>&1; t "eval_prim_$V rc=$?"
  fi
done
t STAGE1_DONE
