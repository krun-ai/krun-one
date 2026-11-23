#!/bin/bash
# M1.5c on ONE pod (configs/v1/m15c.yaml; the M1.5b runner without the OCR sweep, whose choice is reused). Health -> env -> input checksums -> media rebuild (deterministic builders;
# TTS only for the kd1 clips of the M1.5 splits) -> OCR profile + optimization (GPU alone) -> M1.5b data (replay + new
# layout decisions + missing OCR) -> smoke -> WAIT for the local leakage flag -> train -> dumps (C, P = M1.5, A, B, O)
# -> text regression -> latency M1.5b AND M1.5 rerun on the same host -> checksums -> DONE. Idempotent / resumable.
set -u
K=/workspace/krun/experiments/krun_one
cd $K
export HF_HOME=/workspace/hf HF_HUB_CACHE=/workspace/hf/hub TORCH_DISABLE_NATIVE_JIT=1 PYTHONUNBUFFERED=1 \
       TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True KRUN_M15_CONFIG=configs/v1/m15c.yaml
L=/workspace/logs; mkdir -p $L results/v1/m15c; PY=.venv/bin/python
RUN=runs/v1/m15c/krun-one-v1-m15c
st() { echo "$1 $(date -u +%FT%TZ)" >> $L/status; }
fail() { st "FAILED $1"; st DONE; exit 1; }
st "start"
python3 -c "import transformers" 2>/dev/null || python3 -m pip install -q --break-system-packages transformers > /dev/null 2>&1
python3 scripts/v1/gpu_preflight.py > $L/preflight_image.log 2>&1 || fail "gpu_preflight(image)"
st "gpu_preflight(image) ok"
command -v uv >/dev/null || pip install -q uv > /dev/null 2>&1
[ -x $PY ] || { UV_HTTP_TIMEOUT=600 uv sync --frozen --group m1 --group m15 > $L/uv.log 2>&1 || UV_HTTP_TIMEOUT=600 uv sync --frozen --group m1 --group m15 >> $L/uv.log 2>&1; }
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | cut -d. -f1)
TIDX=cu126
[ "${CAP:-0}" -ge 12 ] && fail "Blackwell host: the pinned onnxruntime-gpu 1.22 / cu126 stack is only validated on CUDA 12 GPUs"
$PY scripts/v1/gpu_preflight.py > $L/preflight_venv.log 2>&1 || fail "gpu_preflight(venv)"
st "gpu_preflight(venv) ok"
command -v espeak-ng >/dev/null || (apt-get update -qq && apt-get install -y -qq espeak-ng > $L/apt.log 2>&1)
(nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 60 >> $L/gpu.csv &)
sha256sum -c inputs_m15c.sha256 > $L/inputs_check.log 2>&1 || fail "input checksums"
st "inputs verified"
$PY scripts/v1/train_m15.py health --out results/v1/m15c/health_pre.json > $L/health_pre.log 2>&1 || fail "health(pre)"
st "health(pre) ok"

# ---------------------------------------------------------------- media rebuild
if [ ! -f $L/tts_done ]; then
  if [ ! -x /workspace/tts/bin/python ]; then
    uv venv -q -p 3.12 /workspace/tts > $L/tts_env.log 2>&1
    UV_HTTP_TIMEOUT=600 uv pip install -q -p /workspace/tts/bin/python "torch==2.14.0" --index-url https://download.pytorch.org/whl/$TIDX >> $L/tts_env.log 2>&1
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
[ -f results/v1/m15b/ocr_optimization.json ] || fail "missing the M1.5b OCR optimization choice"
$PY scripts/v1/train_m15.py health --out results/v1/m15c/health.json > $L/health.log 2>&1 || fail "health"
st "health ok"
# ---------------------------------------------------------------- data
[ -f data/m15c/train_pre.jsonl ] || $PY scripts/v1/build_m15b_data.py build > $L/m15c_build.log 2>&1 || fail "m15c build"
[ -f data/m15c/ocr_extra.jsonl ] || $PY scripts/v1/build_m15b_data.py ocr_missing > $L/m15c_ocr.log 2>&1 || fail "m15c ocr_missing"
[ -f data/m15c/manifest.json ] || $PY scripts/v1/build_m15b_data.py finalize > $L/m15c_finalize.log 2>&1 || fail "m15c finalize"
cp -n data/m15/teacher.jsonl.gz data/m15c/teacher.jsonl.gz
st "m15c data ok"
if [ ! -f $L/smoke_ok ]; then
  $PY scripts/v1/train_m15.py train --max-steps 20 --run-name m15c_smoke > $L/smoke.log 2>&1 || fail "gpu smoke"
  rm -rf runs/v1/m15c/m15c_smoke; touch $L/smoke_ok; st "smoke ok"
fi
st "waiting for leakage flag"
while [ ! -f /workspace/leakage_ok ]; do sleep 20; done
grep -q "$(sha256sum data/m15c/train.jsonl | cut -c1-64)" /workspace/leakage_ok || fail "leakage flag does not match train.jsonl"
st "leakage flag ok"
[ -f $RUN/train_result.json ] || { st "train start"; $PY scripts/v1/train_m15.py train > $L/train.log 2>&1 || fail "train"; }
st "train done"
[ -f $RUN/signals_val_eval.jsonl.gz ] || $PY scripts/v1/train_m15.py dump --run $RUN > $L/dump.log 2>&1 || fail "dump"
st "dump ok"
[ -f results/v1/m15c/text_regression.json ] || $PY scripts/v1/m15_serving_checks.py textreg --run $RUN > $L/textreg.log 2>&1 || fail "textreg"
st "textreg done"
[ -f results/v1/m15c/latency_rtx4090.json ] || $PY scripts/v1/m15_serving_checks.py latency --run $RUN > $L/latency.log 2>&1 || fail "latency m15c"
st "latency m15c done"
[ -f results/v1/m15/latency_rtx4090_same_host_as_m15c.json ] || KRUN_M15_CONFIG=configs/v1/m15.yaml $PY scripts/v1/m15_serving_checks.py latency \
    --run runs/v1/m15/krun-one-v1-m15 --tag _same_host_as_m15c > $L/latency_m15.log 2>&1 || fail "latency m15 rerun"
st "latency m15 rerun done"
(cd $RUN && sha256sum step_*/m15.safetensors train_result.json signals_*.jsonl.gz > SHA256SUMS)
st DONE
