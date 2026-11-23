#!/bin/bash
# Upload what the M1 pod needs: code + path deps, the FROZEN text artifact (model in 10 parallel 100 MB streams), the
# kd1 pool + fonts, the V files (text-only regression), then inputs_m1.sha256 (runpod_m1.sh verifies it first).
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
rsync -rltz -e "$E" data/m1/kd1_pool.jsonl data/m1/kd1_pool_stats.json data/m1/fonts $H:$K/data/m1/
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
  data/kd1_r2/val_tools_v2.jsonl configs/v1/m1.yaml configs/v1/m1_labels.yaml > inputs_m1.sha256
grep -q 781e5c7b7006da630d6b77693de26bd886839b0f8f96673774f9ace136dc18b6 inputs_m1.sha256 || { echo "artifact sha mismatch"; exit 1; }
rsync -rltz -e "$E" inputs_m1.sha256 $H:$K/
echo uploaded
