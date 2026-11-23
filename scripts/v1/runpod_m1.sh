#!/bin/bash
# FREEZE checks + M1 on ONE pod (configs/v1/m1.yaml). Health check -> env -> input checksums -> FREEZE (reload bf16 +
# latency) -> M1 health (vision/audio/fusion forward, backward only through trainable params) -> data build (fetch,
# render, TTS in a separate venv) -> decisions -> teacher -> 30-step smoke (discarded) -> WAIT for the local KrunBench
# leakage flag (/workspace/leakage_ok) -> train -> dumps -> text-only regression -> latency -> checksums -> DONE.
# Idempotent / resumable. The caller copies the outputs and TERMINATES the pod.
set -u
K=/workspace/krun/experiments/krun_one
cd $K
export HF_HOME=/workspace/hf HF_HUB_CACHE=/workspace/hf/hub TORCH_DISABLE_NATIVE_JIT=1 PYTHONUNBUFFERED=1 \
       TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=/workspace/logs; mkdir -p $L results/v1/freeze results/v1/m1; PY=.venv/bin/python
RUN=runs/v1/m1/krun-one-v1-m1
st() { echo "$1 $(date -u +%FT%TZ)" >> $L/status; }
fail() { st "FAILED $1"; st DONE; exit 1; }
st "start"
python3 -c "import transformers" 2>/dev/null || python3 -m pip install -q --break-system-packages transformers > /dev/null 2>&1
python3 scripts/v1/gpu_preflight.py > $L/preflight_image.log 2>&1 || fail "gpu_preflight(image)"
st "gpu_preflight(image) ok"
command -v uv >/dev/null || pip install -q uv > /dev/null 2>&1
[ -x $PY ] || { UV_HTTP_TIMEOUT=600 uv sync --frozen --group m1 > $L/uv.log 2>&1 || UV_HTTP_TIMEOUT=600 uv sync --frozen --group m1 >> $L/uv.log 2>&1; }
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | cut -d. -f1)
TIDX=cu126
if [ "${CAP:-0}" -ge 12 ]; then
  TIDX=cu130
  $PY -c "import torch,sys;sys.exit(0 if torch.version.cuda.startswith('13') else 1)" 2>/dev/null || \
    UV_HTTP_TIMEOUT=600 uv pip install -q -p $PY --reinstall "torch==2.14.0+cu130" --index-url https://download.pytorch.org/whl/cu130 >> $L/uv.log 2>&1
fi
$PY scripts/v1/gpu_preflight.py > $L/preflight_venv.log 2>&1 || fail "gpu_preflight(venv)"
st "gpu_preflight(venv) ok"
(nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 60 >> $L/gpu.csv &)
sha256sum -c inputs_m1.sha256 > $L/inputs_check.log 2>&1 || fail "input checksums"
st "inputs verified"

# ---------------------------------------------------------------- A. FREEZE checks (GPU)
[ -f results/v1/freeze/reload_cuda_bf16.json ] || $PY scripts/v1/freeze_verify.py reload --device cuda --dtype bf16 \
    --out results/v1/freeze/reload_cuda_bf16.json > $L/freeze_reload.log 2>&1 || fail "freeze reload"
grep -q '"PASS": true' results/v1/freeze/reload_cuda_bf16.json || fail "freeze reload not PASS"
st "freeze reload ok"
[ -f results/v1/freeze/latency_rtx4090.json ] || $PY scripts/v1/freeze_verify.py latency --out results/v1/freeze/latency_rtx4090.json \
    > $L/freeze_latency.log 2>&1 || fail "freeze latency"
st "freeze latency done"

# ---------------------------------------------------------------- B. M1 health
$PY scripts/v1/train_m1.py health --out results/v1/m1/health.json > $L/m1_health.log 2>&1 || fail "m1 health"
st "m1 health ok"

# ---------------------------------------------------------------- data
[ -f data/m1/tts_jobs.jsonl ] || $PY scripts/v1/build_m1_data.py tts_jobs > $L/tts_jobs.log 2>&1
if [ ! -f $L/tts_done ]; then
  command -v espeak-ng >/dev/null || (apt-get update -qq && apt-get install -y -qq espeak-ng > $L/apt.log 2>&1)
  if [ ! -x /workspace/tts/bin/python ]; then
    uv venv -q -p 3.12 /workspace/tts > $L/tts_env.log 2>&1
    UV_HTTP_TIMEOUT=600 uv pip install -q -p /workspace/tts/bin/python "torch==2.14.0" --index-url https://download.pytorch.org/whl/$TIDX >> $L/tts_env.log 2>&1
    UV_HTTP_TIMEOUT=600 uv pip install -q -p /workspace/tts/bin/python "kokoro==0.9.4" soundfile scipy "misaki[en]" \
      "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl" >> $L/tts_env.log 2>&1
  fi
  /workspace/tts/bin/python scripts/v1/m1_tts.py --shard 99 --nshards 100 --limit 1 --budget-min 5 > $L/tts_probe.log 2>&1 || st "tts probe failed"
  st "tts env ready"
  for s in 0 1 2; do
    (cd $K && /workspace/tts/bin/python scripts/v1/m1_tts.py --shard $s --nshards 3 --budget-min ${TTS_MIN:-40} > $L/tts_$s.log 2>&1) &
  done
fi
[ -f $L/fetch_ok ] || { $PY scripts/v1/build_m1_data.py fetch > $L/fetch.log 2>&1 && touch $L/fetch_ok; } || fail "fetch"
st "fetch ok"
[ -f $L/render_ok ] || { $PY scripts/v1/build_m1_data.py render > $L/render.log 2>&1 && touch $L/render_ok; } || fail "render"
st "render ok"
wait
touch $L/tts_done; st "tts done $(cat data/m1/tts_manifest.*.jsonl 2>/dev/null | wc -l) clips"
[ -f data/m1/manifest.json ] || $PY scripts/v1/build_m1_data.py decisions > $L/decisions.log 2>&1 || fail "decisions"
st "decisions ok"
[ -f data/m1/teacher.jsonl.gz ] || $PY scripts/v1/train_m1.py teacher > $L/teacher.log 2>&1 || fail "teacher"
st "teacher ok"
if [ ! -f $L/smoke_ok ]; then
  $PY scripts/v1/train_m1.py train --max-steps 30 --run-name m1_smoke > $L/smoke.log 2>&1 || fail "gpu smoke"
  grep -q '"step": 1' $L/smoke.log || fail "smoke (no step)"
  rm -rf runs/v1/m1/m1_smoke; touch $L/smoke_ok; st "smoke ok"
fi
st "waiting for leakage flag"
while [ ! -f /workspace/leakage_ok ]; do sleep 20; done
grep -q "$(sha256sum data/m1/train.jsonl | cut -c1-64)" /workspace/leakage_ok || fail "leakage flag does not match train.jsonl"
st "leakage flag ok"

# ---------------------------------------------------------------- train + eval
if [ ! -f $RUN/train_result.json ]; then
  st "train start"
  $PY scripts/v1/train_m1.py train > $L/train.log 2>&1 || fail "train"
fi
st "train done"
[ -f $RUN/signals_val_eval.jsonl.gz ] || $PY scripts/v1/train_m1.py dump --run $RUN --ablate-lora > $L/dump.log 2>&1 || fail "dump"
st "dump ok"
[ -f results/v1/m1/text_regression.json ] || $PY scripts/v1/m1_serving_checks.py textreg --run $RUN > $L/textreg.log 2>&1 || fail "textreg"
st "textreg done"
[ -f results/v1/m1/latency_rtx4090.json ] || $PY scripts/v1/m1_serving_checks.py latency --run $RUN > $L/latency.log 2>&1 || fail "latency"
st "latency done"
(cd $RUN && sha256sum step_*/fusion.safetensors fusion_config.json train_result.json signals_*.jsonl.gz > SHA256SUMS)
st DONE
