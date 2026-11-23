#!/bin/bash
# Upload what the M1.5b pod needs (M1 fusion weights + M1 split files for the deterministic media rebuild): code + path deps, the FROZEN text artifact (model in 10 parallel 100 MB streams), the
# kd1 pool + fonts, the V files (text-only regression), then inputs_m15b.sha256 (runpod_m1.sh verifies it first).
# final_protected is NOT uploaded (the KrunBench leakage check runs locally).
#   bash scripts/v1/pod_upload_m1.sh <ssh_config> <host>
set -eu
CFG=$1; H=$2
E="ssh -F $CFG"
R=${KRUN_PROJECT_ROOT:-$(cd "$(dirname "$0")/../../../.." && pwd)}
K=/workspace/krun/experiments/krun_one
cd $R/experiments/krun_one
$E $H "mkdir -p /workspace/krun/benchmarks $K/{data/kd1,data/kd1_r2,data/krunbench_v1,data/m1,artifacts/krun-one-v1-text-final,results/v1}"
X="--exclude=.venv --exclude=__pycache__ --exclude=data --exclude=runs --exclude=results --exclude=artifacts --exclude=.pytest_cache"
rsync -rltz -e "$E" $X $R/benchmarks/krunbench $H:/workspace/krun/benchmarks/
rsync -rltz -e "$E" $X $R/experiments/encoder_bakeoff $H:/workspace/krun/experiments/
rsync -rltz -e "$E" $X ./ $H:$K/
rsync -rltz -e "$E" results/v1/latency_corpus.json $H:$K/results/v1/
rsync -rltz -e "$E" data/krunbench_v1/val_fit.jsonl data/krunbench_v1/val_eval.jsonl $H:$K/data/krunbench_v1/
rsync -rltz -e "$E" data/kd1/val_ml.jsonl $H:$K/data/kd1/
rsync -rltz -e "$E" data/kd1_r2/val_tools_v2.jsonl $H:$K/data/kd1_r2/
rsync -rltz -e "$E" data/m1/kd1_pool.jsonl data/m1/kd1_pool_stats.json data/m1/fonts data/m1/train.jsonl data/m1/val_fit.jsonl \
  data/m1/val_eval.jsonl data/m1/protected.jsonl data/m1/manifest.json $H:$K/data/m1/
$E $H "mkdir -p $K/runs/v1/m1/krun-one-v1-m1/step_009370"
rsync -rltz -e "$E" runs/v1/m1/krun-one-v1-m1/step_009370/fusion.safetensors $H:$K/runs/v1/m1/krun-one-v1-m1/step_009370/
$E $H "mkdir -p $K/data/m15 $K/runs/v1/m15/krun-one-v1-m15/step_006160 $K/results/v1/m15"
rsync -rltz -e "$E" data/m15/train.jsonl data/m15/val_fit.jsonl data/m15/val_eval.jsonl data/m15/protected.jsonl data/m15/asr.jsonl \
  data/m15/ocr.0.jsonl data/m15/ocr.1.jsonl data/m15/ocr.2.jsonl data/m15/teacher.jsonl.gz data/m15/manifest.json $H:$K/data/m15/
rsync -rltz -e "$E" data/m1/tts_jobs_m15b.jsonl $H:$K/data/m1/
rsync -rltz -e "$E" runs/v1/m15/krun-one-v1-m15/train_result.json $H:$K/runs/v1/m15/krun-one-v1-m15/
rsync -rltz -e "$E" runs/v1/m15/krun-one-v1-m15/step_006160/m15.safetensors $H:$K/runs/v1/m15/krun-one-v1-m15/step_006160/
rsync -rltz -e "$E" results/v1/m15/ocr_benchmark.json $H:$K/results/v1/m15/
A=artifacts/krun-one-v1-text-final
rsync -rltz -e "$E" --exclude=model.safetensors $A/ $H:$K/$A/
# model: parallel dd chunks
SZ=$(stat -c %s $A/model.safetensors); N=$(( (SZ + 104857599) / 104857600 ))
$E $H "truncate -s $SZ $K/$A/model.safetensors"
seq 0 $((N-1)) | xargs -P 10 -I{} sh -c "dd if=$A/model.safetensors bs=100M skip={} count=1 iflag=fullblock status=none | $E $H 'dd of=$K/$A/model.safetensors bs=100M seek={} count=1 iflag=fullblock conv=notrunc status=none'"
# the Stage 2 signal dumps of step_003299 (reload check on GPU compares against them)
S=runs/v1/ft/krun-one-v1-stage2/milestones/step_003299
$E $H "mkdir -p $K/$S $K/runs/v1/ft/krun-one-v1-stage2/final"
rsync -rltz -e "$E" $S/signals_val_eval.jsonl.gz $S/signals_val_fit.jsonl.gz $S/signals_val_ml.jsonl.gz $H:$K/$S/
rsync -rltz -e "$E" runs/v1/ft/krun-one-v1-stage2/final/spec.json $H:$K/runs/v1/ft/krun-one-v1-stage2/final/
$E $H "cd $K/$S && ln -sf ../../../../../../$A/model.safetensors model.safetensors"
sha256sum $A/model.safetensors $A/spec.json $A/calibration.json $A/decision_config.json $A/tokenizer/tokenizer.json \
  data/m1/kd1_pool.jsonl data/krunbench_v1/val_fit.jsonl data/krunbench_v1/val_eval.jsonl data/kd1/val_ml.jsonl \
  data/kd1_r2/val_tools_v2.jsonl configs/v1/m1.yaml configs/v1/m1_labels.yaml configs/v1/m15.yaml configs/v1/m15b.yaml \
  data/m15/train.jsonl data/m15/val_fit.jsonl data/m15/val_eval.jsonl runs/v1/m15/krun-one-v1-m15/step_006160/m15.safetensors \
  runs/v1/m1/krun-one-v1-m1/step_009370/fusion.safetensors data/m1/train.jsonl data/m1/val_fit.jsonl data/m1/val_eval.jsonl > inputs_m15b.sha256
grep -q 781e5c7b7006da630d6b77693de26bd886839b0f8f96673774f9ace136dc18b6 inputs_m15b.sha256 || { echo "artifact sha mismatch"; exit 1; }
rsync -rltz -e "$E" inputs_m15b.sha256 $H:$K/
echo uploaded
