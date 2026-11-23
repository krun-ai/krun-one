#!/bin/bash
# Upload everything the fine-tune pod needs (code + path deps + inputs), then write inputs.sha256 on the pod side
# (runpod_finetune.sh verifies it before training).
#   bash scripts/v1/pod_upload_finetune.sh <ssh_config> <host>
set -eu
CFG=$1; H=$2
E="ssh -F $CFG"
R=${KRUN_PROJECT_ROOT:-$(cd "$(dirname "$0")/../../../.." && pwd)}
cd $R/experiments/krun_one
$E $H "mkdir -p /workspace/krun/benchmarks /workspace/krun/experiments/krun_one/{data/kd1,data/krunbench_v1,runs/v1/mlm1b_import,artifacts,results/v1}"
X="--exclude=.venv --exclude=__pycache__ --exclude=data --exclude=runs --exclude=results --exclude=artifacts --exclude=.pytest_cache"
rsync -rltz -e "$E" $X $R/benchmarks/krunbench $H:/workspace/krun/benchmarks/
rsync -rltz -e "$E" $X $R/experiments/encoder_bakeoff $H:/workspace/krun/experiments/
rsync -rltz -e "$E" $X ./ $H:/workspace/krun/experiments/krun_one/
rsync -rltz -e "$E" results/v1/latency_corpus.json $H:/workspace/krun/experiments/krun_one/results/v1/
rsync -rltz -e "$E" data/krunbench_v1/val_fit.jsonl data/krunbench_v1/val_eval.jsonl $H:/workspace/krun/experiments/krun_one/data/krunbench_v1/
rsync -rltz -e "$E" data/kd1/train.jsonl data/kd1/val_ml.jsonl $H:/workspace/krun/experiments/krun_one/data/kd1/
rsync -rltz -e "$E" artifacts/krun-one-v0.3 $H:/workspace/krun/experiments/krun_one/artifacts/
rsync -rltz -e "$E" runs/v1/mlm1b_import/k1_0b.tar $H:/workspace/krun/experiments/krun_one/runs/v1/mlm1b_import/
sha256sum runs/v1/mlm1b_import/k1_0b.tar data/kd1/train.jsonl data/kd1/val_ml.jsonl data/krunbench_v1/val_fit.jsonl \
  data/krunbench_v1/val_eval.jsonl artifacts/krun-one-v0.3/model.safetensors > inputs.sha256
grep -q 061160e3ff0a606b8ee4efd6ef657e5355af1bfff91048325745db5c0a987565 inputs.sha256 || { echo "k1_0b.tar sha mismatch"; exit 1; }
rsync -rltz -e "$E" inputs.sha256 $H:/workspace/krun/experiments/krun_one/
echo uploaded
