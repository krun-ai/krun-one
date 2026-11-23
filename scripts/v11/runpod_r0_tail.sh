#!/bin/bash
# Second half of the R0 pod session, started after runpod_r0.sh wrote DONE: candidate latency (reduced sweep), growth
# verification in fp32 + bf16 on the GPU, and the exploratory Tetris-raw runs (not in the registered protocol).
set -u
K=/workspace/krun/experiments/krun_one
cd $K
export HF_HUB_CACHE=/workspace/hfcache TORCH_DISABLE_NATIVE_JIT=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=/workspace/logs; PY=.venv/bin/python; P=results/v11/capacity_probe
st() { echo "$1 $(date -u +%FT%TZ)" >> $L/status2; }
until grep -q '^DONE' $L/status; do sleep 20; done
st "tail start"
[ -f results/v11/architecture_search/latency_rtx4090.json ] || timeout 2400 $PY scripts/v11/latency_v11.py --locales en-US --iters 30 \
   --workloads typical_1q_k4_ctx48,typical_1q_tools8_ctx48,typical_4q_mixed_ctx96,k_8_tools_ctx64,k_32_tools_ctx64,k_64_tools_ctx64,k_128_tools_ctx64,len_1024_k4,stress_16q_tools32_ctx512,throughput_32req_k4_ctx48 \
   --out results/v11/architecture_search/latency_rtx4090.json > $L/latency.log 2>&1
st "latency rc=$?"
[ -f results/v11/architecture_search/growth_verification_cuda.json ] || $PY scripts/v11/growth_verify.py --device cuda --n 400 > $L/growth.log 2>&1
st "growth rc=$?"
run() { T=$1; shift; [ -f $P/learnability/$T/run.json ] || $PY scripts/v11/learnability.py --tag $T "$@" > $L/learn_$T.log 2>&1; st "learn $T rc=$?"; }
X="--train-file data/v11/probe/ltrain_tetris.jsonl --only-cond raw --steps 3000"
run x_tetris_raw_k1final  --init k1-final    --train full $X
run x_tetris_raw_base     --init mmbert-base --train full $X
run x_tetris_raw_k1_0b    --init k1-0b       --train full $X
st TAILDONE
