#!/bin/bash
# Budgeted continued MLM, stage 1 (1B tokens) + 0B/1B decision probes, on ONE RTX 4090 (or benchmarked 3090) pod.
# Everything lives on the pod's persistent volume (/workspace) and every step is idempotent, so a pod restart just
# re-runs this script and resumes (MLM resumes from its latest atomic checkpoint). Writes /workspace/logs/status and
# ends with DONE; the caller copies artifacts back, verifies them, and TERMINATES the pod (success or failure).
set -u
K=/workspace/krun/experiments/krun_one
cd $K
export HF_HUB_CACHE=/workspace/hfcache TORCH_DISABLE_NATIVE_JIT=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=true PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=/workspace/logs; mkdir -p $L; PY=.venv/bin/python
st() { echo "$1 $(date -u +%FT%TZ)" >> $L/status; }
st start
command -v uv >/dev/null || pip install -q uv > /dev/null 2>&1
[ -x $PY ] || { uv sync --frozen > $L/uv.log 2>&1; st "uv_sync rc=$?"; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > $L/gpu.txt
GPU=$(head -1 $L/gpu.txt | cut -d, -f1 | sed 's/NVIDIA //; s/GeForce //; s/ //g' | tr A-Z a-z)   # e.g. rtx4090
$PY scripts/v1/gpu_preflight.py >> $L/gpu.txt 2>&1 || { st "FAILED gpu_preflight (CUDA unusable on this host)"; st DONE; exit 1; }
st "gpu_preflight ok"
(nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 60 >> $L/gpu.csv &)
# 1. corpus (4 languages in parallel; skipped when already verified)
if ! ls data/v1_mlm/manifest_fr.json >/dev/null 2>&1 || ! grep -q verification data/v1_mlm/manifest_fr.json; then
  for l in en pt es fr; do
    $PY scripts/v1/prepare_mlm_corpus.py --langs $l --protected results/v1/mlm_protected_texts.json --out data/v1_mlm > $L/corpus_$l.log 2>&1 &
  done
  wait; st "corpus done: $(grep -l verification data/v1_mlm/manifest_*.json 2>/dev/null | wc -l)/4 verified"
fi
[ $(grep -l verification data/v1_mlm/manifest_*.json 2>/dev/null | wc -l) -eq 4 ] || { st "FAILED corpus"; st DONE; exit 1; }
# raw parquet (~34 GB, datasets--* plus the shared blobs/ dir) is no longer needed once the tokenized corpus is verified;
# keep only models--* (mmBERT tokenizer/weights). Leaving blobs/ filled the 90 GB volume at the 300M checkpoint.
find $HF_HUB_CACHE -mindepth 1 -maxdepth 1 ! -name 'models--*' -exec rm -rf {} +
# 2. HyperClone mmBERT-base -> K1 (+1e-3 symmetry-breaking noise) = the 0B baseline and the MLM init
G=runs/v1/grow/k1_w1536_l22_t0_noise
[ -f $G/growth.json ] || { $PY scripts/v1/grow_encoder.py --source jhu-clsp/mmBERT-base --out $G --noise-std 1e-3 > $L/grow.log 2>&1; st "grow rc=$?"; }
[ -f $G/growth.json ] || { st "FAILED grow"; st DONE; exit 1; }
probe() {  # $1 init-dir $2 tag $3 seed
  R=runs/v1/probe/$2_s$3
  [ -f $R/val_signals.jsonl.gz ] && return 0
  $PY scripts/v1/train_proxy.py --variant A_margin --spec k1 --encoder-init $1 --lr-encoder 2e-5 --lr-head 1e-4 \
     --gradient-checkpointing --save-every 1000000000 --data-dir data/v1_probe --runs-dir runs/v1/probe --run-name $2_s$3 --seed $3 > $L/probe_$2_s$3.log 2>&1
  st "probe $2 seed $3 rc=$?"; rm -rf $R/checkpoint-* $R/final
}
# 3. decision probe on the 0B init (before the long run: catches probe bugs early)
for s in 20260926 20260927; do probe $G k1_0b $s; done
# 4. continued MLM 1B (auto-resume)
M=runs/v1/mlm/k1-continued-mlm-1b
if [ ! -f $M/milestone-1000000000/eval.json ]; then
  RES=""; ls $M/checkpoint-*/trainer_state.json >/dev/null 2>&1 && RES="--resume-from-checkpoint latest"
  $PY scripts/v1/continued_mlm.py --config configs/v1/continued_mlm_budget.yaml --run-dir $M $RES >> $L/mlm.log 2>&1
  st "mlm rc=$? resume='$RES'"
fi
[ -f $M/milestone-1000000000/eval.json ] || { st "FAILED mlm"; st DONE; exit 1; }
# 5. decision probe on the 1B checkpoint + registered gate
for s in 20260926 20260927; do probe $M/milestone-1000000000 k1_1b $s; done
$PY scripts/v1/probe_compare.py --mlm-run $M --probe-runs runs/v1/probe > $L/gate.log 2>&1; st "gate rc=$?"
# text-only latency before/after the multimodal scaffold on the production GPU class (a few minutes)
$PY scripts/v1/bench_text_regression.py --dtype bf16 --gemma-tokenizer "$(dirname $(ls $HF_HUB_CACHE/models--jhu-clsp--mmBERT-small/snapshots/*/tokenizer.json | head -1))" \
   --out results/v1/multimodal/text_regression_$GPU.json > $L/text_regression.log 2>&1; st "text_regression rc=$?"
st DONE
