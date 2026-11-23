#!/bin/bash
# Multimodal regression, step 1 (pod): rebuild the M1 / M1.5 media with the registered deterministic builders — the
# media-rebuild half of scripts/v1/runpod_m15c.sh, nothing trained. Step 2 is scripts/v11/mm_eval.py per text core.
set -u
K=/workspace/krun/experiments/krun_one; cd $K
export HF_HOME=/workspace/hf HF_HUB_CACHE=/workspace/hf/hub TORCH_DISABLE_NATIVE_JIT=1 PYTHONUNBUFFERED=1 \
       TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True KRUN_M15_CONFIG=configs/v1/m15c.yaml
L=/workspace/logs; PY=.venv/bin/python
st() { echo "$1 $(date -u +%FT%TZ)" >> $L/status_mm; }
fail() { st "FAILED $1"; exit 1; }
mkdir -p results/v11/multimodal
st "mm prepare start"
UV_HTTP_TIMEOUT=600 UV_LINK_MODE=copy uv sync --frozen --group m1 --group m15 > $L/uv_mm.log 2>&1 || UV_HTTP_TIMEOUT=600 UV_LINK_MODE=copy uv sync --frozen --group m1 --group m15 >> $L/uv_mm.log 2>&1 || fail "uv sync mm"
$PY scripts/v1/gpu_preflight.py > $L/preflight_mm.log 2>&1 || fail "gpu_preflight(mm env)"
command -v espeak-ng >/dev/null || (apt-get update -qq && apt-get install -y -qq espeak-ng > $L/apt.log 2>&1)
$PY scripts/v1/train_m15.py health --out results/v11/multimodal/health_pre.json > $L/health_pre.log 2>&1 || fail "health(pre)"
st "health(pre) ok"
if [ ! -f $L/tts_done ]; then
  if [ ! -x /workspace/tts/bin/python ]; then
    uv venv -q -p 3.12 /workspace/tts > $L/tts_env.log 2>&1
    UV_HTTP_TIMEOUT=600 uv pip install -q -p /workspace/tts/bin/python "torch==2.14.0" --index-url https://download.pytorch.org/whl/cu126 >> $L/tts_env.log 2>&1
    UV_HTTP_TIMEOUT=600 uv pip install -q -p /workspace/tts/bin/python "kokoro==0.9.4" soundfile scipy "misaki[en]" \
      "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl" >> $L/tts_env.log 2>&1
  fi
  for s in 0 1 2; do
    (cd $K && /workspace/tts/bin/python scripts/v1/m1_tts.py --jobs data/m1/tts_jobs_m15b.jsonl --shard $s --nshards 3 --budget-min 40 > $L/tts_$s.log 2>&1) &
  done
fi
[ -f $L/fetch_ok ] || { $PY scripts/v1/build_m1_data.py fetch > $L/fetch.log 2>&1 && touch $L/fetch_ok; } || fail "fetch"
st "fetch(m1) ok"
[ -f $L/fetch_new_ok ] || { $PY scripts/v1/build_m15_data.py fetch_new > $L/fetch_new.log 2>&1 && touch $L/fetch_new_ok; } || fail "fetch_new"
st "fetch(new) ok"
[ -f $L/render_ok ] || { $PY scripts/v1/build_m1_data.py render > $L/render.log 2>&1 && touch $L/render_ok; } || fail "render"
st "render ok"
[ -f $L/extract_ok ] || { $PY scripts/v1/build_m15_data.py extract_media > $L/extract.log 2>&1 && touch $L/extract_ok; } || fail "extract_media"
[ -f $L/audio_media_ok ] || { $PY scripts/v1/build_m15_data.py audio_media > $L/audio_media.log 2>&1 && touch $L/audio_media_ok; } || fail "audio_media"
st "media extracted"
wait
touch $L/tts_done; st "tts done $(cat data/m1/tts_manifest.*.jsonl 2>/dev/null | wc -l) clips"
$PY scripts/v1/train_m15.py health --out results/v11/multimodal/health.json > $L/health.log 2>&1 || fail "health"
st "MM_READY"
