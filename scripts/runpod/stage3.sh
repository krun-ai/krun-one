#!/bin/bash
# KRUN-005 stage 3: freeze decisions (validation only), final sets once, export, reload test, GPU serving benchmark.
source /workspace/krun_env.sh
cd /workspace/krun/experiments/krun_one
L=/workspace/logs/krun005; PY=.venv/bin/python
C=$($PY -c "import json;print(json.load(open('results/krun_one_decisions.json'))['chosen_config'])")
echo "decisions frozen: chosen=$C $(date -u +%FT%TZ)" >> $L/times.txt
$PY scripts/evaluate.py --model laya --tag laya --final > $L/final_laya.log 2>&1
$PY scripts/evaluate.py --model laya --tag laya_default --max-len 512 --head-max-len 192 --final > $L/final_laya_default.log 2>&1
$PY scripts/evaluate.py --model runs/$C/final --tag krun_one_$C --final > $L/final_krun_one.log 2>&1
echo "final evals done $(date -u +%FT%TZ)" >> $L/times.txt
$PY scripts/export_model.py --run runs/$C/final --eval results/eval_krun_one_$C.json --train-summary results/train_$C.json \
    --out artifacts/krun-one-v0 > $L/export.log 2>&1
echo "export rc=$? $(date -u +%FT%TZ)" >> $L/times.txt
$PY scripts/reload_test.py --artifact artifacts/krun-one-v0 --eval-tag krun_one_$C > $L/reload_test_gpu.log 2>&1
echo "reload rc=$? $(date -u +%FT%TZ)" >> $L/times.txt
$PY scripts/benchmark_serving.py --model artifacts/krun-one-v0 --tag krun_one_v0 --device cuda > $L/bench_gpu_krun.log 2>&1
$PY scripts/benchmark_serving.py --model laya --tag laya --device cuda > $L/bench_gpu_laya.log 2>&1
echo "bench done $(date -u +%FT%TZ)" >> $L/times.txt
# error analysis v0 done before unlocking (validation only)
echo STAGE3_DONE >> $L/times.txt
