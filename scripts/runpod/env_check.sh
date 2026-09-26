#!/bin/bash
# Pod bootstrap for KRUN-005 (idempotent). Run from /workspace/krun/experiments/krun_one.
set -e
source /workspace/krun_env.sh
command -v uv >/dev/null || pip install -q uv
uv sync --frozen 2>&1 | tail -2
.venv/bin/python scripts/runpod/fetch_laya.py /workspace/krun/laya
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
.venv/bin/python -c "import torch,transformers;print(torch.__version__,transformers.__version__,torch.cuda.get_device_name(0))"
