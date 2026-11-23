#!/bin/bash
# R3 multimodal recovery (configs/v11/r3_m15c_realign.yaml, scripts/v11/mm_realign.py).
#   bash scripts/v11/pod_r3_mm.sh prepare                 media rebuild with the registered deterministic builders (+ synthetic speech)
#   CORE=r3f_s1 SEEDS="20260932 20260933" [EPOCHS=1] bash scripts/v11/pod_r3_mm.sh run
# prepare can run while text training uses the GPU (CPU + network; speech synthesis uses ~2 GB of VRAM).
set -u
K=/workspace/krun/experiments/krun_one; cd $K
export HF_HOME=/workspace/hf HF_HUB_CACHE=/workspace/hf/hub TORCH_DISABLE_NATIVE_JIT=1 PYTHONUNBUFFERED=1 \
       TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=/workspace/logs; PY=.venv/bin/python
st() { echo "$1 $(date -u +%FT%TZ)" >> $L/status_mm; }
fail() { st "FAILED $1"; exit 1; }
if [ "$1" = prepare ]; then
  export KRUN_M15_CONFIG=configs/v1/m15c.yaml
  mkdir -p results/v11/r3/multimodal
  st "mm prepare start"
  command -v espeak-ng >/dev/null || (apt-get update -qq && apt-get install -y -qq espeak-ng > $L/apt.log 2>&1)
  if [ ! -f $L/tts_env_ok ]; then      # R1 / R2 lesson: the unpinned install resolved to an unbuildable tokenizers and failed silently
    uv venv -q -p 3.12 /workspace/tts > $L/tts_env.log 2>&1
    UV_HTTP_TIMEOUT=600 uv pip install -q -p /workspace/tts/bin/python "torch==2.14.0" --index-url https://download.pytorch.org/whl/cu126 >> $L/tts_env.log 2>&1
    UV_HTTP_TIMEOUT=600 uv pip install -q -p /workspace/tts/bin/python "kokoro==0.9.4" soundfile scipy numpy "misaki[en]" "transformers>=4.44" "tokenizers>=0.20" \
      "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl" >> $L/tts_env.log 2>&1
    /workspace/tts/bin/python -c "import kokoro, numpy, soundfile, scipy" >> $L/tts_env.log 2>&1 || fail "tts env"
    touch $L/tts_env_ok
  fi
  if [ ! -f $L/tts_done ]; then
    for s in 0 1 2; do
      (/workspace/tts/bin/python scripts/v1/m1_tts.py --jobs data/m1/tts_jobs_m15b.jsonl --shard $s --nshards 3 --budget-min 90 > $L/tts_$s.log 2>&1) &
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
  touch $L/tts_done; st "tts done $(ls data/m1/media/tts 2>/dev/null | wc -l) clips"
  rm -rf data/m1/raw                       # re-downloadable cache; the pod volume quota is ~100 GB
  # every media file the train / validation splits reference must exist and be readable
  $PY scripts/v11/mm_media_check.py > $L/media_check.log 2>&1 || fail "media check: $(tail -1 $L/media_check.log)"
  st "MM_READY"
  exit 0
fi
export KRUN_M15_CONFIG=configs/v11/r3_m15c_realign.yaml
grep -q MM_READY $L/status_mm || fail "media not ready"
C=runs/v11/r3/text/$CORE
$PY scripts/v1/train_m15.py health --out results/v11/r3/multimodal/health.json > $L/mm_health.log 2>&1 || fail "health"
# same-host reference: the registered M1.5c fusion on the frozen v1 core (dump + latency)
[ -f results/v11/r3/multimodal/v1_host.json ] || { KRUN_M15_CONFIG=configs/v1/m15c.yaml $PY scripts/v11/mm_eval.py --ckpt v1 --tag v1 > $L/mm_v1.log 2>&1 && cp results/v11/multimodal/v1.json results/v11/r3/multimodal/v1_host.json; }
st "mm v1 reference rc=$?"
[ -f results/v11/r3/multimodal/latency_rtx4090_v1_m15c.json ] || KRUN_M15_CONFIG=configs/v1/m15c.yaml $PY scripts/v11/mm_realign.py latency --core v1 --tag v1_m15c \
    --run runs/v1/m15c/krun-one-v1-m15c > $L/mm_lat_v1.log 2>&1
st "mm v1 latency rc=$?"
[ -f data/v11/r3/mm/teacher_$CORE.jsonl.gz ] || $PY scripts/v11/mm_realign.py teacher --core $C > $L/mm_teacher_$CORE.log 2>&1
st "mm teacher $CORE rc=$?"
for SEED in $SEEDS; do
  T=${CORE}_m${SEED}_e${EPOCHS:-1}
  [ -f runs/v11/r3/multimodal/$T/train_result.json ] || $PY scripts/v11/mm_realign.py train --core $C --tag $T --seed $SEED --epochs ${EPOCHS:-1} > $L/mm_train_$T.log 2>&1
  st "mm train $T rc=$?"
  [ -f results/v11/r3/multimodal/$T.json ] || $PY scripts/v11/mm_realign.py dump --core $C --tag $T > $L/mm_dump_$T.log 2>&1
  st "mm dump $T rc=$?"
  [ -f results/v11/r3/multimodal/latency_rtx4090_$T.json ] || $PY scripts/v11/mm_realign.py latency --core $C --tag $T > $L/mm_lat_$T.log 2>&1
  st "mm latency $T rc=$?"
done
(cd runs/v11/r3/multimodal && find . -name '*.safetensors' -o -name 'train_result.json' | sort | xargs sha256sum > $L/mm_sha256.txt)
st "MM_DONE $CORE $SEEDS e${EPOCHS:-1}"
