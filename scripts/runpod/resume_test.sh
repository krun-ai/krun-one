#!/bin/bash
# Resume validation in deterministic mode: continuous 60 steps vs kill -9 after checkpoint-40 + resume latest.
source /workspace/krun_env.sh
cd /workspace/krun/experiments/krun_one
L=/workspace/logs/krun005; PY=.venv/bin/python
T="scripts/train.py --variant B --max-steps 60 --save-every 20 --log-every 10 --deterministic --gradient-checkpointing"
rm -rf runs/det_ref runs/det_ref2 runs/det_res
$PY $T --run-name det_ref > $L/det_ref.log 2>&1; echo "det_ref rc=$? $(date -u +%FT%TZ)" >> $L/times.txt
$PY $T --run-name det_ref2 > $L/det_ref2.log 2>&1; echo "det_ref2 rc=$? $(date -u +%FT%TZ)" >> $L/times.txt
$PY $T --run-name det_res > $L/det_res_part1.log 2>&1 &
P=$!
until [ -f runs/det_res/checkpoint-40/trainer_state.json ] || ! kill -0 $P 2>/dev/null; do sleep 0.5; done
kill -9 $P; echo "det killed part1 after checkpoint-40 $(date -u +%FT%TZ) ($(ls runs/det_res | tr '\n' ' '))" >> $L/times.txt
$PY $T --run-name det_res --resume-from-checkpoint latest > $L/det_res_part2.log 2>&1; echo "det_res part2 rc=$? $(date -u +%FT%TZ)" >> $L/times.txt
$PY scripts/compare_checkpoints.py runs/det_ref/checkpoint-60 runs/det_ref2/checkpoint-60 > $L/det_compare_ref_vs_ref2.json 2>&1
$PY scripts/compare_checkpoints.py runs/det_ref/checkpoint-60 runs/det_res/checkpoint-60 > $L/det_compare_ref_vs_resumed.json 2>&1
echo RESUME_TEST2_DONE >> $L/times.txt
