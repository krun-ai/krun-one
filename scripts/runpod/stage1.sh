#!/bin/bash
# KRUN-005 stage 1 on RunPod: tests, Laya baseline, smoke + kill/resume validation.
source /workspace/krun_env.sh
cd /workspace/krun/experiments/krun_one
L=/workspace/logs/krun005; mkdir -p $L
PY=.venv/bin/python
T="scripts/train.py --variant B --max-steps 60 --save-every 20 --log-every 10"
echo "stage1_start $(date -u +%FT%TZ)" > $L/times.txt
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 5 > $L/stage1_gpu.csv &
SMI=$!
$PY -m pytest -q > $L/pytest.log 2>&1; echo "pytest rc=$? $(date -u +%FT%TZ)" >> $L/times.txt
$PY scripts/evaluate.py --model laya --tag laya > $L/eval_laya.log 2>&1; echo "eval_laya rc=$? $(date -u +%FT%TZ)" >> $L/times.txt
$PY scripts/evaluate.py --model laya --tag laya_default --max-len 512 --head-max-len 192 > $L/eval_laya_default.log 2>&1
echo "eval_laya_default rc=$? $(date -u +%FT%TZ)" >> $L/times.txt
# smoke (60 steps, config B = the heaviest) = reference for the resume test
$PY $T --run-name smoke_ref > $L/smoke_ref.log 2>&1; echo "smoke_ref rc=$? $(date -u +%FT%TZ)" >> $L/times.txt
# resume test: kill -9 right after checkpoint-40 is complete, then resume from latest
rm -rf runs/smoke_res
$PY $T --run-name smoke_res > $L/smoke_res_part1.log 2>&1 &
P=$!
until [ -f runs/smoke_res/checkpoint-40/trainer_state.json ] || ! kill -0 $P 2>/dev/null; do sleep 0.5; done
kill -9 $P; echo "killed part1 after checkpoint-40 $(date -u +%FT%TZ)" >> $L/times.txt
ls runs/smoke_res >> $L/times.txt
$PY $T --run-name smoke_res --resume-from-checkpoint latest > $L/smoke_res_part2.log 2>&1; echo "smoke_res part2 rc=$? $(date -u +%FT%TZ)" >> $L/times.txt
$PY scripts/compare_checkpoints.py runs/smoke_ref/checkpoint-60 runs/smoke_res/checkpoint-60 > $L/resume_compare.json 2>&1
echo "compare rc=$? $(date -u +%FT%TZ)" >> $L/times.txt
kill $SMI
echo STAGE1_DONE >> $L/times.txt
