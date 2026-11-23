#!/bin/bash
# Synthetic pilot (2k) on one ~24 GB GPU pod: CUDA health check -> vLLM env -> generate (Qwen3-8B) -> verify
# (Granite-3.3-8B, blind). Finalize runs locally. Ends with DONE in /root/pilot/status; the caller copies results
# back and TERMINATES the pod.
set -u
cd /root/pilot
st() { echo "$1 $(date -u +%FT%TZ)" >> status; }
st start
export HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONUNBUFFERED=1 VLLM_LOGGING_LEVEL=WARNING
export PATH=/root/pilot/.venv/bin:$PATH   # vLLM JIT-compiles kernels at engine start and needs `ninja` on PATH
if [ ! -x .venv/bin/ninja ]; then
  pip install -q uv > /dev/null 2>&1
  [ -d .venv ] || uv venv -q -p 3.12 .venv
  uv pip install -q -p .venv/bin/python vllm hf_transfer transformers ninja > setup.log 2>&1; st "setup rc=$?"
fi
.venv/bin/python gpu_preflight.py > preflight.log 2>&1 || { st "FAILED preflight"; cat preflight.log >> status; st DONE; exit 1; }
st "preflight ok"
T0=$(date +%s)
.venv/bin/python synthetic_pilot.py generate --out . > generate.log 2>&1; st "generate rc=$?"
.venv/bin/python synthetic_pilot.py verify --out . > verify.log 2>&1; st "verify rc=$?"
echo "{\"pilot_seconds_after_setup\": $(( $(date +%s) - T0 ))}" > timing.json
st DONE
