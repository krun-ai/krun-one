#!/bin/bash
# KRUN-011 round 2 (v0.3): tests, baselines re-measured on primitives data v2 (Laya x2 budgets, v0),
# variants A and B of configs/krun_one_v0_3.yaml (auto-resume), validation-only evaluation. Test sets stay locked.
source /workspace/krun_env.sh
cd /workspace/krun/experiments/krun_one
L=/workspace/logs/krun011_r2; mkdir -p $L
PY=.venv/bin/python
D="--data-dir data/primitives_v2"
t() { echo "$1 $(date -u +%FT%TZ)" >> $L/times.txt; }
t stage2_start
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 10 > $L/gpu.csv &
SMI=$!
$PY -m pytest -q > $L/pytest.log 2>&1; t "pytest rc=$?"
$PY scripts/evaluate_primitives.py --model laya --tag laya_p2 $D > $L/eval_prim_laya_p2.log 2>&1; t "eval_prim_laya_p2 rc=$?"
$PY scripts/evaluate_primitives.py --model laya --tag laya_default_p2 --max-len 512 --head-max-len 192 $D > $L/eval_prim_laya_default_p2.log 2>&1; t "eval_prim_laya_default_p2 rc=$?"
$PY scripts/evaluate_primitives.py --model artifacts/krun-one-v0 --tag krun_one_v0_p2 $D > $L/eval_prim_v0_p2.log 2>&1; t "eval_prim_v0_p2 rc=$?"
for V in A B; do
  t "train_${V}_start"
  RESUME=()
  ls -d runs/v03_$V/checkpoint-*/trainer_state.json >/dev/null 2>&1 && RESUME=(--resume-from-checkpoint latest)
  for attempt in 1 2 3 4; do
    echo "== attempt $attempt ${RESUME[*]} $(date -u +%FT%TZ)" >> $L/train_${V}.log
    $PY scripts/train_primitives.py --config configs/krun_one_v0_3.yaml --variant $V "${RESUME[@]}" >> $L/train_${V}.log 2>&1
    rc=$?; echo "== exit $rc" >> $L/train_${V}.log
    [ $rc -eq 0 ] && break
    ls -d runs/v03_$V/checkpoint-*/trainer_state.json >/dev/null 2>&1 && RESUME=(--resume-from-checkpoint latest)
  done
  t "train_${V}_end rc=$rc"
  if [ $rc -eq 0 ]; then
    rm -rf runs/v03_$V/checkpoint-*
    $PY scripts/evaluate.py --model runs/v03_$V/final --tag v03_$V > $L/eval_choice_$V.log 2>&1; t "eval_choice_$V rc=$?"
    $PY scripts/evaluate_primitives.py --model runs/v03_$V/final --tag v03_$V $D --no-test > $L/eval_prim_$V.log 2>&1; t "eval_prim_$V rc=$?"
  fi
done
kill $SMI
t STAGE2_DONE
