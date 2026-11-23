#!/bin/bash
# xLAM ablation probe: two identical short fine-tunes from the v1-lr1e5 model (with / without xLAM-sourced decisions).
set -u
K=/workspace/krun/experiments/krun_one; cd $K
export HF_HUB_CACHE=/workspace/hfcache TORCH_DISABLE_NATIVE_JIT=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=/workspace/logs; mkdir -p $L; PY=.venv/bin/python
st() { echo "$1 $(date -u +%FT%TZ)" >> $L/status; }
fail() { st "FAILED $1"; st DONE; exit 1; }
st start
python3 -c "import transformers" 2>/dev/null || python3 -m pip install -q --break-system-packages transformers > /dev/null 2>&1
python3 scripts/v1/gpu_preflight.py > $L/preflight_image.log 2>&1 || fail "gpu_preflight(image)"
command -v uv >/dev/null || pip install -q uv > /dev/null 2>&1
[ -x $PY ] || { UV_HTTP_TIMEOUT=600 uv sync --frozen > $L/uv.log 2>&1; }
$PY scripts/v1/gpu_preflight.py > $L/preflight_venv.log 2>&1 || fail "gpu_preflight(venv)"
sha256sum -c inputs_probe.sha256 > $L/inputs.log 2>&1 || fail "inputs"
$PY -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('jhu-clsp/mmBERT-small')" > /dev/null 2>&1
st "ready"
for V in with_xlam without_xlam; do
  R=runs/v1/ft/probe_$V
  [ -f $R/final/model.safetensors ] || { $PY scripts/v1/train_decisions.py train --init-full init_full/model.safetensors \
      --train-file data/kd1_r2/probe/$V.jsonl --run-name probe_$V --lr-encoder 1e-5 --lr-head 5e-5 > $L/train_$V.log 2>&1; st "train $V rc=$?"; }
  rm -rf $R/checkpoint-*
  $PY scripts/v1/regression_toolace_v0.py --run $R --out $R/regression_toolace_v0 --device cuda --skip-v03 --tag probe_$V > $L/reg_$V.log 2>&1
  st "regression $V rc=$?"
done
st DONE
