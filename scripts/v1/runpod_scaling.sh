#!/bin/bash
# HyperCloning continued-MLM scaling, stage 1 (0 / 100M / 500M / 1B / 5B tokens). PREPARED, NOT RUN.
# Pre-conditions (checked): tokenizer license gate CLEARED; data/v1_mlm built (prepare_mlm_corpus.py);
# bakeoff decided -> DOWNSTREAM_VARIANT (loss variant for the decision fine-tunes, default A_margin).
# Recommended GPU: 1x H100 80GB (≈ 3.5e18 FLOPs per 1B tokens -> ~2.5 h/1B at 40% MFU); a 4090 is ~6x slower.
# The caller copies runs/v1/scaling + results back and TERMINATES the pod (success or failure).
set -u
cd /root/krun/experiments/krun_one
export HF_HUB_CACHE=/root/hfcache TORCH_DISABLE_NATIVE_JIT=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
L=/root/logs; mkdir -p $L; PY=.venv/bin/python
st() { echo "$1 $(date -u +%FT%TZ)" >> $L/status; }
V=${DOWNSTREAM_VARIANT:-A_margin}
st start
grep -q "GATE: CLEARED" docs/KRUN_ONE_V1_TOKENIZER_LICENSE.md || { st "BLOCKED license gate"; st DONE; exit 1; }
[ -f data/v1_mlm/manifest.json ] || { st "BLOCKED no corpus"; st DONE; exit 1; }
pip install -q uv > /dev/null 2>&1; uv sync --frozen > $L/uv.log 2>&1; st "uv_sync rc=$?"
$PY -m pytest -q tests/v1 > $L/pytest.log 2>&1; st "pytest rc=$?"
# G0: exact growth (0 tokens) + the non-grown reference
$PY scripts/v1/grow_encoder.py --source jhu-clsp/mmBERT-base --out runs/v1/grow/k1_w1536_l22_t0 > $L/grow.log 2>&1; st "grow rc=$?"
$PY scripts/v1/grow_encoder.py --source jhu-clsp/mmBERT-base --out runs/v1/grow/k1_w1536_l22_t0_noise --noise-std 1e-3 > $L/grow_noise.log 2>&1; st "grow_noise rc=$?"
REF=$($PY -c "from huggingface_hub import snapshot_download as s; print(s('jhu-clsp/mmBERT-base'))")
# G1: continued MLM with milestone checkpoints (auto-resume if the pod is restarted)
$PY scripts/v1/continued_mlm.py --config configs/v1/scaling.yaml --stage 1 --resume-from-checkpoint latest > $L/mlm.log 2>&1 \
  || $PY scripts/v1/continued_mlm.py --config configs/v1/scaling.yaml --stage 1 > $L/mlm.log 2>&1
st "mlm rc=$?"
R=runs/v1/scaling/k1-hyperclone-scaling
# downstream: same decision fine-tune recipe for every milestone + the mmBERT-base reference
ft() { $PY scripts/v1/train_proxy.py --variant $V --spec $1 --encoder-init $2 --lr-encoder 2e-5 --lr-head 1e-4 \
         --runs-dir runs/v1/scaling_downstream --run-name $3 > $L/ft_$3.log 2>&1; st "ft $3 rc=$?";
       rm -rf runs/v1/scaling_downstream/$3/checkpoint-* runs/v1/scaling_downstream/$3/final; }
ft mmbert-base-ref "$REF" ref_mmbert_base
for m in 0 100000000 500000000 1000000000 5000000000; do ft k1 $R/milestone-$m k1_t$m; done
$PY scripts/v1/scaling_summary.py > $L/summary.log 2>&1; st "summary rc=$?"
st DONE
