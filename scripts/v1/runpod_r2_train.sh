#!/bin/bash
# R2 full fine-tune on ONE pod (no LR smoke): health check -> env -> checksums -> leakage flag -> train -> V signals ->
# ToolACE regression signals -> checksums -> DONE. Idempotent/resumable. Caller copies artifacts and TERMINATES the pod.
#   RUN_NAME=krun-one-v1-lr1e5 LR_ENC=1e-5 LR_HEAD=5e-5 TRAIN=data/kd1/train.jsonl bash scripts/v1/runpod_r2_train.sh
set -u
K=/workspace/krun/experiments/krun_one
cd $K
export HF_HUB_CACHE=/workspace/hfcache TORCH_DISABLE_NATIVE_JIT=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=/workspace/logs; mkdir -p $L results/v1/ft; PY=.venv/bin/python
st() { echo "$1 $(date -u +%FT%TZ)" >> $L/status; }
fail() { st "FAILED $1"; st DONE; exit 1; }
st "start run=$RUN_NAME lr_enc=$LR_ENC lr_head=$LR_HEAD train=$TRAIN"
python3 -c "import transformers" 2>/dev/null || python3 -m pip install -q --break-system-packages transformers > /dev/null 2>&1
python3 scripts/v1/gpu_preflight.py > $L/preflight_image.log 2>&1 || fail "gpu_preflight(image)"
st "gpu_preflight(image) ok"
command -v uv >/dev/null || pip install -q uv > /dev/null 2>&1
[ -x $PY ] || { UV_HTTP_TIMEOUT=600 uv sync --frozen > $L/uv.log 2>&1 || UV_HTTP_TIMEOUT=600 uv sync --frozen >> $L/uv.log 2>&1; }
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | cut -d. -f1)
if [ "${CAP:-0}" -ge 12 ] && ! $PY -c "import torch,sys;sys.exit(0 if torch.version.cuda.startswith('13') else 1)" 2>/dev/null; then
  UV_HTTP_TIMEOUT=600 uv pip install -q -p $PY "torch==2.14.0" --index-url https://download.pytorch.org/whl/cu130 >> $L/uv.log 2>&1
fi
$PY scripts/v1/gpu_preflight.py > $L/preflight_venv.log 2>&1 || fail "gpu_preflight(venv)"
(nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 60 >> $L/gpu.csv &)
sha256sum -c ${INPUTS:-inputs.sha256} > $L/inputs_check.log 2>&1 || fail "input checksums"
grep -q "$(sha256sum $TRAIN | cut -c1-64)" leakage_ok.flag || fail "leakage gate (no LEAKAGE=0 flag for this train file)"
INIT=/workspace/init/runs/v1/mlm/k1-continued-mlm-1b/milestone-0
[ -f $INIT/model.safetensors ] || { mkdir -p /workspace/init && tar xf runs/v1/mlm1b_import/k1_0b.tar -C /workspace/init || fail "extract init"; }
$PY -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('jhu-clsp/mmBERT-small')" > /dev/null 2>&1
st "inputs verified"
RUN=runs/v1/ft/$RUN_NAME
if [ ! -f $RUN/final/model.safetensors ]; then
  RES=""; ls $RUN/checkpoint-*/trainer_state.json >/dev/null 2>&1 && RES="--resume-from-checkpoint latest"
  st "train start resume='$RES'"
  $PY scripts/v1/train_decisions.py train --config ${CONFIG:-configs/v1/finetune.yaml} --init-dir $INIT --run-name $RUN_NAME --lr-encoder $LR_ENC --lr-head $LR_HEAD \
      --train-file $TRAIN $RES >> $L/train_$RUN_NAME.log 2>&1
  st "train rc=$?"
fi
[ -f $RUN/final/model.safetensors ] || fail "train"
rm -rf $RUN/checkpoint-*
$PY scripts/v1/regression_toolace_v0.py --run $RUN --out $RUN/regression_toolace_v0 --device cuda --skip-v03 --tag $RUN_NAME \
    > $L/regression_$RUN_NAME.log 2>&1; st "toolace_regression rc=$?"
(cd $RUN/final && sha256sum model.safetensors spec.json > SHA256SUMS)
st DONE
