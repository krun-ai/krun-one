#!/bin/bash
# Krun One v1 fine-tune on ONE 24 GB+ pod (RTX 4090 class). Idempotent: a restarted pod re-runs this script and
# resumes (training resumes from its latest atomic checkpoint; finished stages are skipped).
#   bash scripts/v1/runpod_finetune.sh [all|smoke|lr_smoke|train|post]
# Order: GPU health check (image torch, BEFORE any install) -> env -> input checksums -> tiny GPU smoke -> LR smoke ->
# full fine-tune -> V eval + calibration -> v0.3 baseline signals -> latency -> checksums -> DONE.
# The caller copies artifacts back, verifies checksums and TERMINATES the pod (success or failure).
set -u
MODE=${1:-all}
K=/workspace/krun/experiments/krun_one
cd $K
export HF_HUB_CACHE=/workspace/hfcache TORCH_DISABLE_NATIVE_JIT=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=/workspace/logs; mkdir -p $L results/v1/ft; PY=.venv/bin/python
st() { echo "$1 $(date -u +%FT%TZ)" >> $L/status; }
fail() { st "FAILED $1"; st DONE; exit 1; }
st "start mode=$MODE"
# 1. health check with the image's torch, before installing anything
python3 -c "import transformers" 2>/dev/null || python3 -m pip install -q --break-system-packages transformers > /dev/null 2>&1
python3 scripts/v1/gpu_preflight.py > $L/preflight_image.log 2>&1 || fail "gpu_preflight(image)"
st "gpu_preflight(image) ok"
# 2. env
command -v uv >/dev/null || pip install -q uv > /dev/null 2>&1
[ -x $PY ] || { UV_HTTP_TIMEOUT=600 uv sync --frozen > $L/uv.log 2>&1 || UV_HTTP_TIMEOUT=600 uv sync --frozen >> $L/uv.log 2>&1; st "uv_sync rc=$?"; }
# Blackwell (sm_120, e.g. RTX 5090) needs the cu130 build of the SAME torch version (the lock pins cu126)
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | cut -d. -f1)
if [ "${CAP:-0}" -ge 12 ] && ! $PY -c "import torch,sys;sys.exit(0 if torch.version.cuda.startswith('13') else 1)" 2>/dev/null; then
  UV_HTTP_TIMEOUT=600 uv pip install -q -p $PY "torch==2.14.0" --index-url https://download.pytorch.org/whl/cu130 >> $L/uv.log 2>&1
  st "torch cu130 swap rc=$?"
fi
$PY scripts/v1/gpu_preflight.py > $L/preflight_venv.log 2>&1 || fail "gpu_preflight(venv)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > $L/gpu.txt
GPU=$(head -1 $L/gpu.txt | cut -d, -f1 | sed 's/NVIDIA //; s/GeForce //; s/ //g' | tr A-Z a-z)
(nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 60 >> $L/gpu.csv &)
# 3. inputs: checksums (init tar, datasets), init extraction
sha256sum -c inputs.sha256 > $L/inputs_check.log 2>&1 || fail "input checksums"
INIT=/workspace/init/runs/v1/mlm/k1-continued-mlm-1b/milestone-0
[ -f $INIT/model.safetensors ] || { mkdir -p /workspace/init && tar xf runs/v1/mlm1b_import/k1_0b.tar -C /workspace/init || fail "extract init"; }
st "inputs verified"
$PY -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('jhu-clsp/mmBERT-small')" > /dev/null 2>&1
# 4. tiny GPU smoke: 40 steps of the real model on the real data (catches OOM / dtype / loader bugs in minutes)
if [ "$MODE" = all ] || [ "$MODE" = smoke ]; then
  [ -f results/v1/ft/gpu_smoke.json ] || {
    $PY scripts/v1/train_decisions.py train --init-dir $INIT --run-name gpu_smoke --max-steps 40 > $L/gpu_smoke.log 2>&1
    rc=$?; st "gpu_smoke rc=$rc"
    [ $rc -eq 0 ] || fail "gpu_smoke"                 # the marker below is written ONLY on success
    grep -h '"step"' $L/gpu_smoke.log | tail -2 > results/v1/ft/gpu_smoke.json; rm -rf runs/v1/ft/gpu_smoke/checkpoint-*; }
fi
# 5. LR smoke (registered rule, configs/v1/finetune.yaml lr_smoke)
if [ "$MODE" = all ] || [ "$MODE" = lr_smoke ]; then
  [ -f results/v1/ft/lr_smoke.json ] || { $PY scripts/v1/train_decisions.py lr_smoke --init-dir $INIT > $L/lr_smoke.log 2>&1; st "lr_smoke rc=$?"; }
  [ -f results/v1/ft/lr_smoke.json ] || fail "lr_smoke"
fi
# 6. full fine-tune (auto-resume)
RUN=runs/v1/ft/krun-one-v1-ft
if [ "$MODE" = all ] || [ "$MODE" = train ]; then
  LRE=$($PY -c "import json;print(json.load(open('results/v1/ft/lr_smoke.json'))['choice']['lr_encoder'])")
  LRH=$($PY -c "import json;print(json.load(open('results/v1/ft/lr_smoke.json'))['choice']['lr_head'])")
  # hard gate: the independent KrunBench v1 check (build_krunbench_v1.py --check, run by the caller) must report
  # LEAKAGE = 0 for THIS train.jsonl before the full run starts (waits up to 2 h for the flag)
  for i in $(seq 1 240); do [ -f leakage_ok.flag ] && break; sleep 30; done
  [ -f leakage_ok.flag ] && grep -q "$(sha256sum data/kd1/train.jsonl | cut -c1-64)" leakage_ok.flag || fail "leakage gate"
  if [ ! -f $RUN/final/model.safetensors ]; then
    RES=""; ls $RUN/checkpoint-*/trainer_state.json >/dev/null 2>&1 && RES="--resume-from-checkpoint latest"
    st "train start lr_encoder=$LRE lr_head=$LRH resume='$RES'"
    $PY scripts/v1/train_decisions.py train --init-dir $INIT --lr-encoder $LRE --lr-head $LRH $RES >> $L/train.log 2>&1
    st "train rc=$?"
  fi
  [ -f $RUN/final/model.safetensors ] || fail "train"
  rm -rf $RUN/checkpoint-*
fi
# 7. post: calibration + metrics (V only), v0.3 baseline on the same V files, latency, checksums
if [ "$MODE" = all ] || [ "$MODE" = post ]; then
  [ -f runs/v1/ft/v03_baseline/signals_val_ml.jsonl.gz ] || { $PY scripts/v1/dump_signals_v03.py --out runs/v1/ft/v03_baseline > $L/v03.log 2>&1; st "v03_signals rc=$?"; }
  $PY scripts/v1/kd1_evaluate.py --run $RUN --baseline runs/v1/ft/v03_baseline --tag v1_ft > $L/evaluate.log 2>&1; st "evaluate rc=$?"
  TOK=$(dirname $(ls $HF_HUB_CACHE/models--jhu-clsp--mmBERT-small/snapshots/*/tokenizer.json | head -1))
  [ -f results/v1/ft/latency_$GPU.json ] || { $PY scripts/v1/latency_proxy.py --specs krun-one-v0.3,K1-W1536-L22 --gemma-tokenizer $TOK \
      --out results/v1/ft/latency_$GPU.json > $L/latency.log 2>&1; st "latency rc=$?"; }
  (cd $RUN/final && sha256sum model.safetensors spec.json > SHA256SUMS)
fi
st DONE
