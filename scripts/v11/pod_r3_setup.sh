#!/bin/bash
# R3 pod setup: GPU health check, environment, input checksums.   MM=1 also installs the multimodal dependency groups.
set -u
K=/workspace/krun/experiments/krun_one; cd $K
export HF_HOME=/workspace/hf HF_HUB_CACHE=/workspace/hf/hub PYTHONUNBUFFERED=1
L=/workspace/logs; mkdir -p $L; PY=.venv/bin/python
st() { echo "$1 $(date -u +%FT%TZ)" >> $L/status; }
fail() { st "FAILED $1"; exit 1; }
st "setup start"
python3 -c "import transformers" 2>/dev/null || python3 -m pip install -q --break-system-packages transformers > /dev/null 2>&1
python3 scripts/v1/gpu_preflight.py > $L/preflight_image.log 2>&1 || fail "gpu_preflight(image)"
command -v uv >/dev/null || pip install -q uv > /dev/null 2>&1
G=""; [ "${MM:-0}" = 1 ] && G="--group m1 --group m15"
UV_HTTP_TIMEOUT=600 UV_LINK_MODE=copy uv sync --frozen $G > $L/uv.log 2>&1 || UV_HTTP_TIMEOUT=600 UV_LINK_MODE=copy uv sync --frozen $G >> $L/uv.log 2>&1 || fail "uv sync"
$PY scripts/v1/gpu_preflight.py > $L/preflight_venv.log 2>&1 || fail "gpu_preflight(venv)"
(nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 60 >> $L/gpu.csv &)
sha256sum -c inputs_v11_r3.sha256 > $L/inputs_check.log 2>&1 || fail "input checksums"
st "SETUP_OK"
