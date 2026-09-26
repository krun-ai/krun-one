#!/bin/bash
# KRUN-011 fixed regression suite: every future Krun One release must pass it before promotion.
#   bash scripts/regression_suite.sh <run dir with model.safetensors | artifact dir> <tag>
# 1. unit tests (serialization of choice / noul / score, type embeddings, mixed batches, typed answers, RPS loss)
# 2. Choice regression   — KrunBench val/test, val_hard, val_tools, holdout_tools, BFCL (scripts/evaluate.py)
# 3. Noul + Score        — data/primitives_v2 val (+ test only after results/primitives_decisions.json exists)
# 4. Mixed primitives    — MHS mixed requests: per-type quality + alone-vs-mixed invariance (in 3, with test)
# Then compare with the gates in configs/krun_one_v0_3.yaml: scripts/compare_primitives.py select v03
set -euo pipefail
MODEL=${1:?model dir}; TAG=${2:?tag}
cd "$(dirname "$0")/.."
PY=.venv/bin/python
$PY -m pytest -q tests/test_primitives.py tests/test_laya_equivalence.py
$PY scripts/evaluate.py --model "$MODEL" --tag "$TAG"
$PY scripts/evaluate_primitives.py --model "$MODEL" --tag "$TAG" --data-dir data/primitives_v2 --no-test
echo "regression suite done for $TAG — now: $PY scripts/compare_primitives.py select v03 (gates) "
