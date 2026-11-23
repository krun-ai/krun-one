#!/bin/bash
# Upload for the packaging pod: code + tests + fixtures, the frozen text artifact (parallel chunks), the multimodal
# candidate artifact, the latency corpus; then inputs_mmpack.sha256.   bash scripts/v1/pod_upload_mmpack.sh <ssh_config> <host>
set -eu
CFG=$1; H=$2; E="ssh -F $CFG"; R=${KRUN_PROJECT_ROOT:-$(cd "$(dirname "$0")/../../../.." && pwd)}; K=/workspace/krun/experiments/krun_one
cd $R/experiments/krun_one
$E $H "mkdir -p /workspace/krun/benchmarks $K/artifacts/krun-one-v1-text-final $K/results/v1 $K/tests"
X="--exclude=.venv --exclude=__pycache__ --exclude=data --exclude=runs --exclude=results --exclude=artifacts --exclude=.pytest_cache"
rsync -rltz -e "$E" $X $R/benchmarks/krunbench $H:/workspace/krun/benchmarks/
rsync -rltz -e "$E" $X $R/experiments/encoder_bakeoff $H:/workspace/krun/experiments/
rsync -rltz -e "$E" $X ./ $H:$K/
$E $H "mkdir -p /workspace/krun/krun-api-app"; rsync -rltz -e "$E" $R/krun-api-app/openapi.json $H:/workspace/krun/krun-api-app/
rsync -rltz -e "$E" results/v1/latency_corpus.json $H:$K/results/v1/
A=artifacts/krun-one-v1-text-final
rsync -rltz -e "$E" --exclude=model.safetensors $A/ $H:$K/$A/
SZ=$(stat -c %s $A/model.safetensors); N=$(( (SZ + 104857599) / 104857600 ))
$E $H "truncate -s $SZ $K/$A/model.safetensors"
seq 0 $((N-1)) | xargs -P 10 -I{} sh -c "dd if=$A/model.safetensors bs=100M skip={} count=1 iflag=fullblock status=none | $E $H 'dd of=$K/$A/model.safetensors bs=100M seek={} count=1 iflag=fullblock conv=notrunc status=none'"
rsync -rltz -e "$E" artifacts/krun-one-v1-multimodal-candidate $H:$K/artifacts/
sha256sum $A/model.safetensors artifacts/krun-one-v1-multimodal-candidate/SHA256SUMS artifacts/krun-one-v1-multimodal-candidate/fusion/*.safetensors \
  tests/fixtures/multimodal/expected_cpu.json > inputs_mmpack.sha256
rsync -rltz -e "$E" inputs_mmpack.sha256 $H:$K/
echo uploaded
