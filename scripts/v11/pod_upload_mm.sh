#!/bin/bash
# Upload what the multimodal M1.5c regression needs on a pod that already has the code and the v1 artifact:
# the M1 / M1.5 / M1.5c split files (media is rebuilt on the pod by the registered deterministic builders),
# the frozen fusion weights and the OCR choice.   bash scripts/v11/pod_upload_mm.sh <ssh_config> <host>
set -eu
CFG=$1; H=$2; E="ssh -F $CFG"
K=/workspace/krun/experiments/krun_one
cd "$(dirname "$0")/../.."
$E $H "mkdir -p $K/{data/m1,data/m15,data/m15c,results/v1/m15,results/v1/m15b,results/v1/m15c} $K/runs/v1/m1/krun-one-v1-m1/step_009370 $K/runs/v1/m15/krun-one-v1-m15/step_006160 $K/runs/v1/m15c/krun-one-v1-m15c/step_001169"
rsync -rltz -e "$E" data/m1/ $H:$K/data/m1/
rsync -rltz -e "$E" data/m15/ $H:$K/data/m15/
rsync -rltz -e "$E" data/m15c/ $H:$K/data/m15c/
rsync -rlt -e "$E" runs/v1/m1/krun-one-v1-m1/step_009370/fusion.safetensors $H:$K/runs/v1/m1/krun-one-v1-m1/step_009370/
rsync -rlt -e "$E" runs/v1/m15/krun-one-v1-m15/step_006160/m15.safetensors $H:$K/runs/v1/m15/krun-one-v1-m15/step_006160/
rsync -rlt -e "$E" runs/v1/m15/krun-one-v1-m15/train_result.json $H:$K/runs/v1/m15/krun-one-v1-m15/
rsync -rlt -e "$E" runs/v1/m15c/krun-one-v1-m15c/step_001169/m15.safetensors runs/v1/m15c/krun-one-v1-m15c/train_result.json $H:$K/runs/v1/m15c/krun-one-v1-m15c/ 2>/dev/null || true
rsync -rlt -e "$E" runs/v1/m15c/krun-one-v1-m15c/step_001169/m15.safetensors $H:$K/runs/v1/m15c/krun-one-v1-m15c/step_001169/
rsync -rlt -e "$E" runs/v1/m15c/krun-one-v1-m15c/train_result.json $H:$K/runs/v1/m15c/krun-one-v1-m15c/
rsync -rltz -e "$E" results/v1/m15/ocr_benchmark.json $H:$K/results/v1/m15/
rsync -rltz -e "$E" results/v1/m15b/ocr_optimization.json $H:$K/results/v1/m15b/
rsync -rltz -e "$E" results/v1/m15c/m15c_eval.json $H:$K/results/v1/m15c/
echo uploaded-mm
