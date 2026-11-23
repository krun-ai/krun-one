#!/bin/bash
# Upload what the R0 pod needs: code + path deps, the FROZEN v1 text artifact (read-only use; model in 10 parallel
# 100 MB streams), the probe files, the latency corpus; then inputs_v11_r0.sha256 (runpod_r0.sh verifies it first).
#   bash scripts/v11/pod_upload_r0.sh <ssh_config> <host>
set -eu
CFG=$1; H=$2
E="ssh -F $CFG"
R=${KRUN_PROJECT_ROOT:-$(cd "$(dirname "$0")/../../../.." && pwd)}
K=/workspace/krun/experiments/krun_one
cd $R/experiments/krun_one
A=artifacts/krun-one-v1-text-final
$E $H "mkdir -p /workspace/krun/benchmarks $K/{data/v11/probe,$A,results/v1,results/v11}"
X="--exclude=.venv --exclude=__pycache__ --exclude=data --exclude=runs --exclude=results --exclude=artifacts --exclude=.pytest_cache"
rsync -rltz -e "$E" $X $R/benchmarks/krunbench $H:/workspace/krun/benchmarks/
rsync -rltz -e "$E" $X $R/experiments/encoder_bakeoff $H:/workspace/krun/experiments/
rsync -rltz -e "$E" $X ./ $H:$K/
rsync -rltz -e "$E" results/v1/latency_corpus.json $H:$K/results/v1/
rsync -rltz -e "$E" data/v11/probe/probe.jsonl data/v11/probe/ltrain.jsonl data/v11/probe/manifest.json $H:$K/data/v11/probe/
rsync -rltz -e "$E" --exclude=model.safetensors $A/ $H:$K/$A/
SZ=$(stat -c %s $A/model.safetensors); N=$(( (SZ + 104857599) / 104857600 ))
$E $H "truncate -s $SZ $K/$A/model.safetensors"
seq 0 $((N-1)) | xargs -P 10 -I{} sh -c "dd if=$A/model.safetensors bs=100M skip={} count=1 iflag=fullblock status=none | $E $H 'dd of=$K/$A/model.safetensors bs=100M seek={} count=1 iflag=fullblock conv=notrunc status=none'"
sha256sum $A/model.safetensors data/v11/probe/probe.jsonl data/v11/probe/ltrain.jsonl results/v1/latency_corpus.json > inputs_v11_r0.sha256
grep -q 781e5c7b7006da630d6b77693de26bd886839b0f8f96673774f9ace136dc18b6 inputs_v11_r0.sha256 || { echo "v1 text-final sha mismatch"; exit 1; }
rsync -rltz -e "$E" inputs_v11_r0.sha256 $H:$K/
echo uploaded
