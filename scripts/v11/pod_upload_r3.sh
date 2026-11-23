#!/bin/bash
# Upload what an R3 pod needs. Idempotent; large weight files go in 10 parallel 100 MB streams and are sha-checked.
#   bash scripts/v11/pod_upload_r3.sh <ssh_config> <host> [code|artifact|corpus|regression|r2|mm|all]
set -eu
CFG=$1; H=$2; WHAT=${3:-all}
E="ssh -F $CFG"
R=${KRUN_PROJECT_ROOT:-$(cd "$(dirname "$0")/../../../.." && pwd)}
K=/workspace/krun/experiments/krun_one
cd $R/experiments/krun_one
A=artifacts/krun-one-v1-text-final
X="--exclude=.venv --exclude=__pycache__ --exclude=data --exclude=runs --exclude=results --exclude=artifacts --exclude=.pytest_cache"
has() { [ "$WHAT" = all ] || [ "$WHAT" = "$1" ]; }
big() {  # local file -> same relative path on the pod, parallel chunks + sha256
  F=$1; SZ=$(stat -c %s $F); N=$(( (SZ + 104857599) / 104857600 ))
  $E $H "mkdir -p $K/$(dirname $F) && truncate -s $SZ $K/$F"
  seq 0 $((N-1)) | xargs -P 10 -I{} sh -c "dd if=$F bs=100M skip={} count=1 iflag=fullblock status=none | timeout 900 $E $H 'dd of=$K/$F bs=100M seek={} count=1 iflag=fullblock conv=notrunc status=none'"
  L=$(sha256sum $F | cut -c1-64); G=$($E $H "sha256sum $K/$F | cut -c1-64")
  [ "$L" = "$G" ] && echo "sha-ok $F" || { echo "SHA MISMATCH $F"; exit 1; }
}
$E $H "mkdir -p /workspace/logs /workspace/krun/benchmarks $K/{data/v11/r3,data/v11/techbench,data/v11/tetris_representation,data/v11/external,data/krunbench_v1,data/kd1,data/kd1_r2,$A,results/v1,results/v11,runs/v1/ft/regression_toolace_v0}"
if has code; then
  rsync -rltz -e "$E" $X $R/benchmarks/krunbench $H:/workspace/krun/benchmarks/
  rsync -rltz -e "$E" $X $R/experiments/encoder_bakeoff $H:/workspace/krun/experiments/
  rsync -rltz -e "$E" $X ./ $H:$K/
  rsync -rltz -e "$E" results/v1/latency_corpus.json $H:$K/results/v1/
fi
if has artifact; then
  rsync -rltz -e "$E" --exclude=model.safetensors $A/ $H:$K/$A/
  big $A/model.safetensors
fi
if has corpus; then
  rsync -rltz -e "$E" data/v11/r3/train_tech.jsonl data/v11/r3/train_replay.jsonl data/v11/r3/aux_stats.json data/v11/r3/manifest.json $H:$K/data/v11/r3/
  rsync -rltz -e "$E" data/v11/techbench/ $H:$K/data/v11/techbench/
  rsync -rltz -e "$E" data/v11/tetris_representation/test.jsonl $H:$K/data/v11/tetris_representation/
  rsync -rltz -e "$E" data/v11/external/jevbench $H:$K/data/v11/external/
fi
if has regression; then
  rsync -rltz -e "$E" data/krunbench_v1/val_fit.jsonl data/krunbench_v1/val_eval.jsonl $H:$K/data/krunbench_v1/
  rsync -rltz -e "$E" data/kd1/val_ml.jsonl $H:$K/data/kd1/
  rsync -rltz -e "$E" data/kd1_r2/val_tools_v2.jsonl $H:$K/data/kd1_r2/
  rsync -rltz -e "$E" runs/v1/ft/regression_toolace_v0/records.jsonl $H:$K/runs/v1/ft/regression_toolace_v0/
fi
if has r2; then
  $E $H "mkdir -p $K/runs/v11/r2/r2l4na_s1"
  rsync -rltz -e "$E" runs/v11/r2/r2l4na_s1/spec.json runs/v11/r2/r2l4na_s1/run.json $H:$K/runs/v11/r2/r2l4na_s1/
  big runs/v11/r2/r2l4na_s1/model_v1.safetensors
fi
if has mm; then
  $E $H "mkdir -p $K/{data/m1,data/m15,data/m15c,results/v1/m15,results/v1/m15b,results/v1/m15c} $K/runs/v1/m1/krun-one-v1-m1/step_009370 $K/runs/v1/m15/krun-one-v1-m15/step_006160 $K/runs/v1/m15c/krun-one-v1-m15c/step_001169"
  rsync -rltz -e "$E" --exclude='tts_manifest.*' data/m1/ $H:$K/data/m1/
  rsync -rltz -e "$E" data/m15/ $H:$K/data/m15/
  rsync -rltz -e "$E" data/m15c/ $H:$K/data/m15c/
  rsync -rlt -e "$E" runs/v1/m1/krun-one-v1-m1/step_009370/fusion.safetensors $H:$K/runs/v1/m1/krun-one-v1-m1/step_009370/
  rsync -rlt -e "$E" runs/v1/m15/krun-one-v1-m15/step_006160/m15.safetensors $H:$K/runs/v1/m15/krun-one-v1-m15/step_006160/
  rsync -rlt -e "$E" runs/v1/m15/krun-one-v1-m15/train_result.json $H:$K/runs/v1/m15/krun-one-v1-m15/
  rsync -rlt -e "$E" runs/v1/m15c/krun-one-v1-m15c/step_001169/m15.safetensors $H:$K/runs/v1/m15c/krun-one-v1-m15c/step_001169/
  rsync -rlt -e "$E" runs/v1/m15c/krun-one-v1-m15c/train_result.json $H:$K/runs/v1/m15c/krun-one-v1-m15c/
  rsync -rltz -e "$E" results/v1/m15/ocr_benchmark.json $H:$K/results/v1/m15/
  rsync -rltz -e "$E" results/v1/m15b/ocr_optimization.json $H:$K/results/v1/m15b/
  rsync -rltz -e "$E" results/v1/m15c/m15c_eval.json $H:$K/results/v1/m15c/
fi
echo "uploaded $WHAT"
