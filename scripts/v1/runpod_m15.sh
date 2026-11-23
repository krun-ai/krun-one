#!/bin/bash
# M1.5 on ONE pod (configs/v1/m15.yaml). Health (CUDA, text, vision, audio encoder, ASR decode, OCR smoke, fusion,
# trainable backward) -> env -> input checksums -> M1 media rebuild (deterministic builder) + new sources -> media
# verification -> OCR benchmark (registered rule picks the OCR) -> health with the chosen OCR -> selection -> ASR / OCR
# passes -> decisions -> teacher -> 30-step smoke -> WAIT for the local KrunBench leakage flag -> train -> dumps ->
# text-only regression -> latency -> checksums -> DONE. Idempotent / resumable. The caller copies outputs and TERMINATES.
set -u
K=/workspace/krun/experiments/krun_one
cd $K
export HF_HOME=/workspace/hf HF_HUB_CACHE=/workspace/hf/hub TORCH_DISABLE_NATIVE_JIT=1 PYTHONUNBUFFERED=1 \
       TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True DOCTR_CACHE_DIR=/workspace/doctr
L=/workspace/logs; mkdir -p $L results/v1/m15; PY=.venv/bin/python
RUN=runs/v1/m15/krun-one-v1-m15
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
if [ "${CAP:-0}" -ge 12 ]; then
  TIDX=cu130
  $PY -c "import torch,sys;sys.exit(0 if torch.version.cuda.startswith('13') else 1)" 2>/dev/null || \
    UV_HTTP_TIMEOUT=600 uv pip install -q -p $PY --reinstall "torch==2.14.0+cu130" "torchvision==0.29.0+cu130" --index-url https://download.pytorch.org/whl/cu130 >> $L/uv.log 2>&1
fi
$PY scripts/v1/gpu_preflight.py > $L/preflight_venv.log 2>&1 || fail "gpu_preflight(venv)"
st "gpu_preflight(venv) ok"
command -v tesseract >/dev/null || (apt-get update -qq && apt-get install -y -qq tesseract-ocr tesseract-ocr-por tesseract-ocr-spa tesseract-ocr-fra espeak-ng > $L/apt.log 2>&1)
(nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 60 >> $L/gpu.csv &)
sha256sum -c inputs_m15.sha256 > $L/inputs_check.log 2>&1 || fail "input checksums"
st "inputs verified"
# health #1 (OCR smoke with a shortlist OCR; repeated with the chosen OCR after the benchmark)
$PY scripts/v1/train_m15.py health --ocr rapidocr:v6 --out results/v1/m15/health_pre.json > $L/health_pre.log 2>&1 || fail "health(pre)"
st "health(pre) ok"

# ---------------------------------------------------------------- media rebuild (M1 builder) + new sources
[ -f data/m1/tts_jobs.jsonl ] || $PY scripts/v1/build_m1_data.py tts_jobs > $L/tts_jobs.log 2>&1
if [ ! -f $L/tts_done ]; then
  if [ ! -x /workspace/tts/bin/python ]; then
    uv venv -q -p 3.12 /workspace/tts > $L/tts_env.log 2>&1
    UV_HTTP_TIMEOUT=600 uv pip install -q -p /workspace/tts/bin/python "torch==2.14.0" --index-url https://download.pytorch.org/whl/$TIDX >> $L/tts_env.log 2>&1
    UV_HTTP_TIMEOUT=600 uv pip install -q -p /workspace/tts/bin/python "kokoro==0.9.4" soundfile scipy "misaki[en]" \
      "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl" >> $L/tts_env.log 2>&1
  fi
  for s in 0 1 2; do
    (cd $K && /workspace/tts/bin/python scripts/v1/m1_tts.py --shard $s --nshards 3 --budget-min ${TTS_MIN:-45} > $L/tts_$s.log 2>&1) &
  done
fi
[ -f $L/fetch_ok ] || { $PY scripts/v1/build_m1_data.py fetch > $L/fetch.log 2>&1 && touch $L/fetch_ok; } || fail "fetch"
st "fetch(m1) ok"
[ -f $L/fetch_new_ok ] || { $PY scripts/v1/build_m15_data.py fetch_new > $L/fetch_new.log 2>&1 && touch $L/fetch_new_ok; } || fail "fetch_new"
st "fetch(new) ok"
[ -f $L/render_ok ] || { $PY scripts/v1/build_m1_data.py render > $L/render.log 2>&1 && touch $L/render_ok; } || fail "render"
st "render ok"
[ -f $L/extract_ok ] || { $PY scripts/v1/build_m15_data.py extract_media > $L/extract.log 2>&1 && touch $L/extract_ok; } || fail "extract_media"
st "extract media ok"
# OCR benchmark on val_fit renders (GPU shared with TTS; latency re-measured later in the serving checks)
[ -f results/v1/m15/ocr_benchmark.json ] || $PY scripts/v1/m15_ocr_bench.py \
    --candidates "rapidocr:v6,rapidocr:v5-latin,doctr:db_mobilenet_v3_large+hub:Felix92/doctr-torch-parseq-multilingual-v1,tesseract" \
    > $L/ocr_bench.log 2>&1 || fail "ocr bench"
st "ocr bench ok: $($PY -c "import json;print(json.load(open('results/v1/m15/ocr_benchmark.json'))['chosen'])")"
wait
touch $L/tts_done; st "tts done $(cat data/m1/tts_manifest.*.jsonl 2>/dev/null | wc -l) clips"
[ -f data/m15/media_rebuild_check.json ] || $PY scripts/v1/build_m15_data.py verify_media > $L/verify_media.log 2>&1 || fail "verify_media"
st "media verified"
$PY scripts/v1/train_m15.py health --out results/v1/m15/health.json > $L/health.log 2>&1 || fail "health(chosen ocr)"
st "health ok"
[ -f data/m15/selected.jsonl ] || $PY scripts/v1/build_m15_data.py select > $L/select.log 2>&1 || fail "select"
st "select ok"
[ -f $L/asr_ok ] || { $PY scripts/v1/build_m15_data.py asr > $L/asr.log 2>&1 && touch $L/asr_ok; } || fail "asr"
st "asr ok"
if [ ! -f $L/ocr_ok ]; then
  for s in 0 1 2; do ($PY scripts/v1/build_m15_data.py ocr --shard $s --nshards 3 > $L/ocr_$s.log 2>&1; echo $? > $L/ocr_rc_$s) & done
  wait
  [ "$(cat $L/ocr_rc_0 $L/ocr_rc_1 $L/ocr_rc_2 | tr -d '\n')" = "000" ] || fail "ocr pass"
  touch $L/ocr_ok
fi
st "ocr ok"
[ -f data/m15/manifest.json ] || $PY scripts/v1/build_m15_data.py decisions > $L/decisions.log 2>&1 || fail "decisions"
st "decisions ok"
[ -f data/m15/teacher.jsonl.gz ] || $PY scripts/v1/train_m15.py teacher > $L/teacher.log 2>&1 || fail "teacher"
st "teacher ok"
if [ ! -f $L/smoke_ok ]; then
  $PY scripts/v1/train_m15.py train --max-steps 30 --run-name m15_smoke > $L/smoke.log 2>&1 || fail "gpu smoke"
  rm -rf runs/v1/m15/m15_smoke; touch $L/smoke_ok; st "smoke ok"
fi
st "waiting for leakage flag"
while [ ! -f /workspace/leakage_ok ]; do sleep 20; done
grep -q "$(sha256sum data/m15/train.jsonl | cut -c1-64)" /workspace/leakage_ok || fail "leakage flag does not match train.jsonl"
st "leakage flag ok"
[ -f $RUN/train_result.json ] || { st "train start"; $PY scripts/v1/train_m15.py train > $L/train.log 2>&1 || fail "train"; }
st "train done"
[ -f $RUN/signals_val_eval.jsonl.gz ] || $PY scripts/v1/train_m15.py dump --run $RUN > $L/dump.log 2>&1 || fail "dump"
st "dump ok"
[ -f results/v1/m15/text_regression.json ] || $PY scripts/v1/m15_serving_checks.py textreg --run $RUN > $L/textreg.log 2>&1 || fail "textreg"
st "textreg done"
[ -f results/v1/m15/latency_rtx4090.json ] || $PY scripts/v1/m15_serving_checks.py latency --run $RUN > $L/latency.log 2>&1 || fail "latency"
st "latency done"
(cd $RUN && sha256sum step_*/m15.safetensors train_result.json signals_*.jsonl.gz > SHA256SUMS)
st DONE
