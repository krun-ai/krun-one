#!/bin/bash
# Upload what an R1/R2 pod needs. Idempotent (rsync); the frozen v1 artifact is uploaded once in 10 parallel streams.
#   bash scripts/v11/pod_upload_r12.sh <ssh_config> <host> [code|artifact|tetris|corpus|regression|all]
set -eu
CFG=$1; H=$2; WHAT=${3:-all}
E="ssh -F $CFG"
R=${KRUN_PROJECT_ROOT:-$(cd "$(dirname "$0")/../../../.." && pwd)}
K=/workspace/krun/experiments/krun_one
cd $R/experiments/krun_one
A=artifacts/krun-one-v1-text-final
X="--exclude=.venv --exclude=__pycache__ --exclude=data --exclude=runs --exclude=results --exclude=artifacts --exclude=.pytest_cache"
has() { [ "$WHAT" = all ] || [ "$WHAT" = "$1" ]; }
$E $H "mkdir -p /workspace/krun/benchmarks $K/{data/v11/corpus,data/v11/techbench,data/v11/tetris_representation,data/krunbench_v1,data/kd1,data/kd1_r2,$A,results/v1,results/v11,runs/v1/ft/regression_toolace_v0}"
if has code; then
  rsync -rltz -e "$E" $X $R/benchmarks/krunbench $H:/workspace/krun/benchmarks/
  rsync -rltz -e "$E" $X $R/experiments/encoder_bakeoff $H:/workspace/krun/experiments/
  rsync -rltz -e "$E" $X ./ $H:$K/
  rsync -rltz -e "$E" results/v1/latency_corpus.json $H:$K/results/v1/
fi
if has artifact; then
  rsync -rltz -e "$E" --exclude=model.safetensors $A/ $H:$K/$A/
  SZ=$(stat -c %s $A/model.safetensors); N=$(( (SZ + 104857599) / 104857600 ))
  $E $H "truncate -s $SZ $K/$A/model.safetensors"
  seq 0 $((N-1)) | xargs -P 10 -I{} sh -c "dd if=$A/model.safetensors bs=100M skip={} count=1 iflag=fullblock status=none | $E $H 'dd of=$K/$A/model.safetensors bs=100M seek={} count=1 iflag=fullblock conv=notrunc status=none'"
  $E $H "cd $K && sha256sum $A/model.safetensors | grep -q 781e5c7b7006da630d6b77693de26bd886839b0f8f96673774f9ace136dc18b6 && echo artifact-sha-ok"
fi
if has tetris; then rsync -rltz -e "$E" data/v11/tetris_representation/ $H:$K/data/v11/tetris_representation/; fi
if has corpus; then
  rsync -rltz -e "$E" data/v11/corpus/ $H:$K/data/v11/corpus/
  rsync -rltz -e "$E" data/v11/techbench/ $H:$K/data/v11/techbench/
fi
if has regression; then
  rsync -rltz -e "$E" data/krunbench_v1/val_fit.jsonl data/krunbench_v1/val_eval.jsonl $H:$K/data/krunbench_v1/
  rsync -rltz -e "$E" data/kd1/val_ml.jsonl $H:$K/data/kd1/
  rsync -rltz -e "$E" data/kd1_r2/val_tools_v2.jsonl $H:$K/data/kd1_r2/
  rsync -rltz -e "$E" runs/v1/ft/regression_toolace_v0/records.jsonl $H:$K/runs/v1/ft/regression_toolace_v0/
fi
echo "uploaded $WHAT"
