#!/bin/bash
# KRUN-011 stage 3: final (once) evaluation of the frozen selection v03_B, export, reload test, serving benchmark.
source /workspace/krun_env.sh
cd /workspace/krun/experiments/krun_one
L=/workspace/logs/krun011_r2; PY=.venv/bin/python
t() { echo "$1 $(date -u +%FT%TZ)" >> $L/times.txt; }
t stage3_start
$PY scripts/evaluate.py --model runs/v03_B/final --tag v03_B --final > $L/eval_choice_final_B.log 2>&1; t "eval_choice_final_B rc=$?"
$PY scripts/evaluate_primitives.py --model runs/v03_B/final --tag v03_B --data-dir data/primitives_v2 > $L/eval_prim_final_B.log 2>&1; t "eval_prim_final_B rc=$?"
$PY scripts/export_primitives.py --variant B --round v03 > $L/export.log 2>&1; t "export rc=$?"
$PY scripts/reload_primitives_test.py --artifact artifacts/krun-one-v0.3-candidate --eval-tag v03_B > $L/reload.log 2>&1; t "reload rc=$?"
$PY scripts/benchmark_primitives.py --artifact artifacts/krun-one-v0.3-candidate > $L/bench.log 2>&1; t "bench rc=$?"
t STAGE3_DONE
