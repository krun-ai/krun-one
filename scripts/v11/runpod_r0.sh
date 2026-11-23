#!/bin/bash
# Krun One v1.1 R0 on ONE pod: health check -> env -> input checksums -> zero-shot probe (frozen v1, read-only) ->
# candidate latency/VRAM (random weights) -> R0-L learnability diagnostic (throwaway weights, nothing saved) -> DONE.
# Idempotent: every step is skipped when its output exists. The caller copies results/v11 and TERMINATES the pod.
#   STEPS="probe latency learn" bash scripts/v11/runpod_r0.sh
set -u
K=/workspace/krun/experiments/krun_one
cd $K
export HF_HUB_CACHE=/workspace/hfcache TORCH_DISABLE_NATIVE_JIT=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=/workspace/logs; mkdir -p $L; PY=.venv/bin/python
STEPS=${STEPS:-"probe latency learn"}
st() { echo "$1 $(date -u +%FT%TZ)" >> $L/status; }
fail() { st "FAILED $1"; st DONE; exit 1; }
st "start steps='$STEPS'"
python3 -c "import transformers" 2>/dev/null || python3 -m pip install -q --break-system-packages transformers > /dev/null 2>&1
python3 scripts/v1/gpu_preflight.py > $L/preflight_image.log 2>&1 || fail "gpu_preflight(image)"
st "gpu_preflight(image) ok"
command -v uv >/dev/null || pip install -q --break-system-packages uv > /dev/null 2>&1
[ -x $PY ] || { UV_HTTP_TIMEOUT=600 uv sync --frozen > $L/uv.log 2>&1 || UV_HTTP_TIMEOUT=600 uv sync --frozen >> $L/uv.log 2>&1; }
$PY scripts/v1/gpu_preflight.py > $L/preflight_venv.log 2>&1 || fail "gpu_preflight(venv)"
st "gpu_preflight(venv) ok"
(nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 60 >> $L/gpu.csv &)
sha256sum -c inputs_v11_r0.sha256 > $L/inputs_check.log 2>&1 || fail "input checksums"
st "inputs verified"
P=results/v11/capacity_probe
case " $STEPS " in *" probe "*)
  [ -f $P/v1_zero_shot/signals.jsonl.gz ] || $PY scripts/v11/run_probe.py --out $P/v1_zero_shot/signals.jsonl.gz > $L/probe.log 2>&1
  st "probe rc=$?";; esac
case " $STEPS " in *" latency "*)
  [ -f results/v11/architecture_search/latency_rtx4090.json ] || timeout 3000 $PY scripts/v11/latency_v11.py \
      --out results/v11/architecture_search/latency_rtx4090.json > $L/latency.log 2>&1
  st "latency rc=$?";; esac
run() {  # $1 = tag, rest = learnability args
  T=$1; shift
  [ -f $P/learnability/$T/run.json ] || $PY scripts/v11/learnability.py --tag $T "$@" > $L/learn_$T.log 2>&1
  st "learn $T rc=$?"
}
case " $STEPS " in *" learn "*)
  run k1final_full       --init k1-final    --train full
  run k1final_head       --init k1-final    --train head
  run mmbert_base_full   --init mmbert-base --train full
  run k1_0b_full         --init k1-0b       --train full
  run mmbert_small_full  --init mmbert-small --train full
  run k1final_head_ci2   --init k1-final    --train head --cand-layers 2
  run k1final_head_L4    --init k1-final    --train head --head-extra 2
  run k1final_full_ci2   --init k1-final    --train full --cand-layers 2
  run mmbert_base_full_s2 --init mmbert-base --train full --seed 20261001
  run k1_0b_full_s2      --init k1-0b       --train full --seed 20261001
  ;; esac
(cd results/v11 && find . -type f \( -name '*.json' -o -name '*.gz' \) | sort | xargs sha256sum > /workspace/logs/results_v11.sha256)
st DONE
