#!/bin/bash
# Krun Decisions v1 GPU text jobs on ONE ~24 GB pod: health check FIRST (image torch, before any install) -> vLLM venv ->
# Qwen phase (describe, translate, translate_desc, pilot v2 generate) -> Granite phase (judge, pilot v2 verify).
# Idempotent + resumable (chunked outputs). Writes /workspace/kd1/status, ends with DONE. The caller copies out/ and
# pilot/ back, verifies checksums, and TERMINATES the pod (success or failure).
set -u
W=/workspace/kd1
cd $W
st() { echo "$1 $(date -u +%FT%TZ)" >> status; }
st start
python3 gpu_preflight.py > preflight.log 2>&1 || { st "FAILED preflight"; cat preflight.log >> status; st DONE; exit 1; }
st "preflight ok"
export PYTHONUNBUFFERED=1 VLLM_LOGGING_LEVEL=WARNING HF_HOME=/workspace/hf VLLM_USE_FLASHINFER_SAMPLER=0  # flashinfer JIT sampler rejects sm_120 (RTX 5090)
export PATH=$W/.venv/bin:$PATH   # vLLM JIT-compiles kernels at engine start and needs `ninja` on PATH
if [ ! -x .venv/bin/ninja ]; then
  pip install -q uv > /dev/null 2>&1
  [ -d .venv ] || uv venv -q -p 3.12 .venv
  for a in 1 2 3; do
    # driver CUDA < 13 (e.g. 12.8 / 570.x): pin a vLLM built for torch cu128; CUDA 13 hosts take the latest
    DRV=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+" | head -1)
    VL="vllm transformers"; [ "${DRV:-13}" -lt 13 ] && VL="vllm==0.11.0 transformers==4.56.2"
    UV_HTTP_TIMEOUT=600 uv pip install -q -p .venv/bin/python $VL hf_transfer ninja > setup.log 2>&1 && break
    st "setup attempt $a failed"; sleep 20
  done
fi
[ -x .venv/bin/ninja ] && .venv/bin/python -c "import vllm" || { st "FAILED setup"; st DONE; exit 1; }
st "setup ok"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 30 > gpu.csv 2>&1 &
T0=$(date +%s)
.venv/bin/python kd1_gpu_jobs.py qwen --work $W > qwen.log 2>&1; st "qwen rc=$?"
T1=$(date +%s)
rm -rf /workspace/hf/hub/models--Qwen--Qwen3-8B   # disk headroom before the second model
.venv/bin/python kd1_gpu_jobs.py granite --work $W > granite.log 2>&1; st "granite rc=$?"
T2=$(date +%s)
echo "{\"qwen_seconds\": $((T1 - T0)), \"granite_seconds\": $((T2 - T1))}" > timing.json
(cd $W && find out pilot -type f -name '*.jsonl' -o -name '*.json' | sort | xargs sha256sum > SHA256SUMS)
st DONE
