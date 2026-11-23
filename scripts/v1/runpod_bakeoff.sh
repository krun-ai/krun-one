#!/bin/bash
# Answerability Bakeoff on a fresh RunPod pod (RTX 4090). Layout on the pod mirrors the workspace:
#   /root/krun/{experiments/krun_one, experiments/encoder_bakeoff, benchmarks/krunbench}, HF cache in /root/hfcache.
# Seeds are the OUTER loop so an interrupted run still has every variant for the first seeds.
# Writes /root/logs/status (one line per step) and ends with DONE; the caller copies runs/v1 back and terminates the pod.
set -u
cd /root/krun/experiments/krun_one
export HF_HUB_CACHE=/root/hfcache TORCH_DISABLE_NATIVE_JIT=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
L=/root/logs; mkdir -p $L
st() { echo "$1 $(date -u +%FT%TZ)" >> $L/status; }
st start
pip install -q uv > /dev/null 2>&1
uv sync --frozen > $L/uv.log 2>&1; st "uv_sync rc=$?"
PY=.venv/bin/python
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > $L/gpu.txt
$PY -c "import torch,transformers;print(torch.__version__,torch.version.cuda,transformers.__version__)" >> $L/gpu.txt
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 15 > $L/gpu.csv &
$PY -m pytest -q tests/v1 > $L/pytest.log 2>&1; st "pytest rc=$?"
# GPU checkpoint/resume smoke (bf16 kernels: functional, not bitwise)
$PY scripts/v1/train_proxy.py --variant J_joint --seed 1 --max-steps 30 --save-every 15 --run-name gpu_resume_smoke > $L/resume_a.log 2>&1; st "resume_smoke_a rc=$?"
$PY scripts/v1/train_proxy.py --variant J_joint --seed 1 --max-steps 45 --save-every 15 --run-name gpu_resume_smoke --resume-from-checkpoint latest > $L/resume_b.log 2>&1; st "resume_smoke_b rc=$?"
rm -rf runs/v1/gpu_resume_smoke
for s in ${SEEDS:-20260926 20260927 20260928}; do
  for v in A_margin B_null C_ans_head D_apply J_joint; do
    $PY scripts/v1/train_proxy.py --variant $v --seed $s > $L/${v}_s$s.log 2>&1; rc=$?
    st "train $v $s rc=$rc"
    rm -rf runs/v1/${v}_s$s/checkpoint-*        # keep final/, signals and metrics only
  done
done
# serving cost of the answerability additions ([NULL] + 2-output scorer + answer head) on the 879M candidate
$PY scripts/v1/latency_proxy.py --specs "krun-one-v0.3,K1-W1536-L22,K1-W1536-L22-plain" \
  --gemma-tokenizer "$(dirname $(ls $HF_HUB_CACHE/models--jhu-clsp--mmBERT-small/snapshots/abc*/tokenizer.json))" \
  --dtypes bf16 --locales pt-PT --out results/v1/latency_answerability_rtx4090.json > $L/latency.log 2>&1; st "latency rc=$?"
kill %1 2>/dev/null
st DONE
