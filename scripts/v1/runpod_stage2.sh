#!/bin/bash
# Stage 2 on ONE pod (configs/v1/stage2_protocol.yaml): health check (incl. VRAM smoke) -> env -> checksums -> leakage
# flag -> short GPU smoke (20 steps, discarded) -> Stage 2 training from the v1-tool candidate (milestones 25/50/75% +
# final) -> for EVERY checkpoint: V signals + ToolACE regression (bf16 = gate harness; fp32 scorer = diagnostic) ->
# v1-tool fp32-scorer diagnostic -> latency -> checksums -> DONE. Idempotent / resumable. The caller copies the outputs
# and TERMINATES the pod.
set -u
K=/workspace/krun/experiments/krun_one
cd $K
export HF_HUB_CACHE=/workspace/hfcache TORCH_DISABLE_NATIVE_JIT=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=/workspace/logs; mkdir -p $L; PY=.venv/bin/python
RUN_NAME=krun-one-v1-stage2; RUN=runs/v1/ft/$RUN_NAME; CFG=configs/v1/stage2.yaml
INIT=artifacts/krun-one-v1-tool-candidate
TRAIN=data/kd1_s2/train.jsonl
st() { echo "$1 $(date -u +%FT%TZ)" >> $L/status; }
fail() { st "FAILED $1"; st DONE; exit 1; }
st "start"
python3 -c "import transformers" 2>/dev/null || python3 -m pip install -q --break-system-packages transformers > /dev/null 2>&1
python3 scripts/v1/gpu_preflight.py > $L/preflight_image.log 2>&1 || fail "gpu_preflight(image)"
st "gpu_preflight(image) ok"
command -v uv >/dev/null || pip install -q uv > /dev/null 2>&1
[ -x $PY ] || { UV_HTTP_TIMEOUT=600 uv sync --frozen > $L/uv.log 2>&1 || UV_HTTP_TIMEOUT=600 uv sync --frozen >> $L/uv.log 2>&1; }
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | cut -d. -f1)
if [ "${CAP:-0}" -ge 12 ] && ! $PY -c "import torch,sys;sys.exit(0 if torch.version.cuda.startswith('13') else 1)" 2>/dev/null; then
  UV_HTTP_TIMEOUT=600 uv pip install -q -p $PY --reinstall "torch==2.14.0+cu130" --index-url https://download.pytorch.org/whl/cu130 >> $L/uv.log 2>&1
fi
$PY scripts/v1/gpu_preflight.py > $L/preflight_venv.log 2>&1 || fail "gpu_preflight(venv)"
st "gpu_preflight(venv) ok"
(nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 60 >> $L/gpu.csv &)
sha256sum -c inputs_s2.sha256 > $L/inputs_check.log 2>&1 || fail "input checksums"
grep -q "$(sha256sum $TRAIN | cut -c1-64)" leakage_ok.flag || fail "leakage gate (no LEAKAGE=0 flag for the Stage 2 file)"
$PY -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('jhu-clsp/mmBERT-small')" > /dev/null 2>&1
st "inputs verified"
# short GPU smoke: the real model / config / data for 20 steps (also builds the shared encoding cache); discarded
if [ ! -f $RUN/final/model.safetensors ] && ! ls $RUN/checkpoint-* >/dev/null 2>&1 && [ ! -f $L/smoke_ok ]; then
  $PY scripts/v1/train_decisions.py train --config $CFG --init-full $INIT/model.safetensors --run-name s2_smoke \
      --max-steps 20 > $L/smoke.log 2>&1 || fail "gpu smoke"
  grep -q '"step": 20' $L/smoke.log || fail "gpu smoke (no step 20)"
  rm -rf runs/v1/ft/s2_smoke; touch $L/smoke_ok; st "smoke ok"
fi
if [ ! -f $RUN/final/model.safetensors ]; then
  RES=""; ls $RUN/checkpoint-*/trainer_state.json >/dev/null 2>&1 && RES="--resume-from-checkpoint latest"
  st "train start resume='$RES'"
  $PY scripts/v1/train_decisions.py train --config $CFG --init-full $INIT/model.safetensors --run-name $RUN_NAME \
      $RES >> $L/train.log 2>&1
  st "train rc=$?"
fi
[ -f $RUN/final/model.safetensors ] || fail "train"
rm -rf $RUN/checkpoint-*
cp $RUN/final/spec.json $RUN/milestones/ 2>/dev/null
(cd $RUN/final && sha256sum model.safetensors spec.json > SHA256SUMS)
# final: V signals were written by the driver into $RUN; ToolACE regression (gate harness + fp32-scorer diagnostic)
reg() {  # $1 = weights dir (holds model.safetensors), $2 = tag
  [ -f $1/regression_toolace_v0/signals_$2.jsonl.gz ] || $PY scripts/v1/regression_toolace_v0.py --run $RUN --weights $1/model.safetensors \
      --spec $RUN/final/spec.json --out $1/regression_toolace_v0 --device cuda --skip-v03 --tag $2 >> $L/regression.log 2>&1
  [ -f $1/regression_toolace_v0/signals_$2_fp32.jsonl.gz ] || $PY scripts/v1/regression_toolace_v0.py --run $RUN --weights $1/model.safetensors \
      --spec $RUN/final/spec.json --out $1/regression_toolace_v0 --device cuda --skip-v03 --tag $2_fp32 --fp32-scorer >> $L/regression.log 2>&1
}
reg $RUN/final s2_final; st "final toolace rc=$?"
for M in $RUN/milestones/step_*; do
  S=$(basename $M)
  [ -f $M/eval_validation.json ] || $PY scripts/v1/train_decisions.py eval --config $CFG --run $M --weights $M/model.safetensors >> $L/eval_$S.log 2>&1
  st "eval $S rc=$?"
  reg $M s2_$S; st "toolace $S rc=$?"
  (cd $M && sha256sum model.safetensors > SHA256SUMS)
done
# v1-tool fp32-scorer diagnostic (the gate numbers of v1-tool stay the R2 bf16 ones)
mkdir -p runs/v1/ft/s2_v1tool_diag
[ -f runs/v1/ft/s2_v1tool_diag/regression_toolace_v0/signals_v1tool_fp32.jsonl.gz ] || $PY scripts/v1/regression_toolace_v0.py \
    --run $RUN --weights $INIT/model.safetensors --spec $INIT/spec.json --out runs/v1/ft/s2_v1tool_diag/regression_toolace_v0 \
    --device cuda --skip-v03 --tag v1tool_fp32 --fp32-scorer >> $L/regression.log 2>&1
st "v1-tool fp32 diag rc=$?"
TOK=$(dirname $(ls $HF_HUB_CACHE/models--jhu-clsp--mmBERT-small/snapshots/*/tokenizer.json | head -1))
[ -f results/v1/stage2/latency_stage2.json ] || { mkdir -p results/v1/stage2; timeout 1500 $PY scripts/v1/latency_proxy.py \
    --specs K1-W1536-L22 --dtypes bf16 --gemma-tokenizer $TOK --ref-tokenizer $TOK \
    --workloads typical_1q_k4_ctx48,typical_1q_tools8_ctx48,typical_4q_mixed_ctx96,k_32_tools_ctx64,k_64_tools_ctx64,k_128_tools_ctx64,stress_16q_tools32_ctx512 \
    --out results/v1/stage2/latency_stage2.json > $L/latency.log 2>&1; }
st "latency rc=$?"
st DONE
