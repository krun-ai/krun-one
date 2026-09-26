#!/bin/bash
# KRUN-011 stage 1 on RunPod (RTX 4090): tests, primitive baselines (Laya x2 budgets, Krun One v0),
# deterministic kill/resume test of the multi-primitive trainer, the 3 registered variants (auto-resume),
# validation-only evaluation of every variant (choice + noul + score). Test sets stay closed for candidates.
source /workspace/krun_env.sh
cd /workspace/krun/experiments/krun_one
L=/workspace/logs/krun011; mkdir -p $L
PY=.venv/bin/python
t() { echo "$1 $(date -u +%FT%TZ)" >> $L/times.txt; }
t stage1_start
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 10 > $L/gpu.csv &
SMI=$!
$PY -m pytest -q > $L/pytest.log 2>&1; t "pytest rc=$?"
$PY scripts/evaluate_primitives.py --model laya --tag laya > $L/eval_prim_laya.log 2>&1; t "eval_prim_laya rc=$?"
$PY scripts/evaluate_primitives.py --model laya --tag laya_default --max-len 512 --head-max-len 192 > $L/eval_prim_laya_default.log 2>&1; t "eval_prim_laya_default rc=$?"
$PY scripts/evaluate_primitives.py --model artifacts/krun-one-v0 --tag krun_one_v0 > $L/eval_prim_v0.log 2>&1; t "eval_prim_v0 rc=$?"
# resume test (deterministic kernels): continuous 60 steps vs kill -9 after checkpoint-40 + resume latest
T="scripts/train_primitives.py --variant M50_RPS --max-steps 60 --save-every 20 --log-every 10 --deterministic --gradient-checkpointing"
rm -rf runs/p_det_ref runs/p_det_res
$PY $T --run-name p_det_ref > $L/p_det_ref.log 2>&1; t "p_det_ref rc=$?"
$PY $T --run-name p_det_res > $L/p_det_res_part1.log 2>&1 &
P=$!
until [ -f runs/p_det_res/checkpoint-40/trainer_state.json ] || ! kill -0 $P 2>/dev/null; do sleep 0.5; done
kill -9 $P; t "p_det killed after checkpoint-40 ($(ls runs/p_det_res | tr '\n' ' '))"
$PY $T --run-name p_det_res --resume-from-checkpoint latest > $L/p_det_res_part2.log 2>&1; t "p_det_res part2 rc=$?"
$PY scripts/compare_checkpoints.py runs/p_det_ref/checkpoint-60 runs/p_det_res/checkpoint-60 > $L/p_det_compare.json 2>&1; t "compare rc=$?"
for V in M50_CE M40_CE M50_RPS; do
  t "train_${V}_start"
  RESUME=()
  for attempt in 1 2 3 4; do
    echo "== attempt $attempt ${RESUME[*]} $(date -u +%FT%TZ)" >> $L/train_${V}.log
    $PY scripts/train_primitives.py --variant $V "${RESUME[@]}" >> $L/train_${V}.log 2>&1
    rc=$?; echo "== exit $rc" >> $L/train_${V}.log
    [ $rc -eq 0 ] && break
    ls -d runs/v02_$V/checkpoint-*/trainer_state.json >/dev/null 2>&1 && RESUME=(--resume-from-checkpoint latest)
  done
  t "train_${V}_end rc=$rc"
  if [ $rc -eq 0 ]; then
    $PY scripts/evaluate.py --model runs/v02_$V/final --tag v02_$V > $L/eval_choice_$V.log 2>&1; t "eval_choice_$V rc=$?"
    $PY scripts/evaluate_primitives.py --model runs/v02_$V/final --tag v02_$V --no-test > $L/eval_prim_$V.log 2>&1; t "eval_prim_$V rc=$?"
  fi
done
kill $SMI
t STAGE1_DONE
