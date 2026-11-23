#!/bin/bash
# R1 / R2 main runs (configs/v11/r1_r2_protocol.yaml). One epoch each on the full mixture, the pilot-selected objective.
#   OBJ=O3 JOBS="r2_s1 r2_s2 r2na_s1 r2na_s2 hn_s1 hn_s2" bash scripts/v11/pod_phase2.sh
# After every run: KrunTechVal + KrunBench val_fit dumps (selection inputs). Weights are kept until the final evaluation.
set -u
K=/workspace/krun/experiments/krun_one; cd $K
export HF_HUB_CACHE=/workspace/hfcache TORCH_DISABLE_NATIVE_JIT=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=/workspace/logs; PY=.venv/bin/python; S1=20261001; S2=20261002
st() { echo "$1 $(date -u +%FT%TZ)" >> $L/status; }
train() {  # tag track args...
  T=$1; TR=$2; shift 2
  [ -f runs/v11/$TR/$T/run.json ] || $PY scripts/v11/train_v11.py --tag $T --track $TR --objective $OBJ "$@" > $L/train_$T.log 2>&1
  st "train $T rc=$?"
  [ -f results/v11/eval/$T/signals_val_fit.jsonl.gz ] || $PY scripts/v11/eval_v11.py --ckpt runs/v11/$TR/$T --out results/v11/eval/$T \
      --tech data/v11/techbench/val.jsonl --regression --v-files val_fit > $L/eval_$T.log 2>&1
  st "eval $T rc=$?"
}
hn() {  # seed-name seed
  N=$1; SEED=$2
  [ -f results/v11/eval/r2_$N/tech_train_tech.jsonl.gz ] || $PY scripts/v11/eval_v11.py --ckpt runs/v11/r2/r2_$N --out results/v11/eval/r2_$N \
      --tech data/v11/corpus/train_tech.jsonl > $L/mine_dump_$N.log 2>&1
  [ -f data/v11/corpus/hn_${N}_tech.jsonl ] || $PY scripts/v11/mine_hard.py --dump results/v11/eval/r2_$N/tech_train_tech.jsonl.gz --tag $N > $L/mine_$N.log 2>&1
  st "mine $N rc=$?"
  train r2hn_$N r2 --mode last_n --last-n 2 --seed $SEED --init-ckpt runs/v11/r2/r2_$N --tech data/v11/corpus/hn_${N}_tech.jsonl \
      --replay data/v11/corpus/hn_${N}_replay.jsonl --lr-scale 0.5
}
for J in $JOBS; do
  case $J in
    r1_s1) train r1_s1 r1 --mode head --seed $S1 ;;
    r1_s2) train r1_s2 r1 --mode head --seed $S2 ;;
    r2_s1) train r2_s1 r2 --mode last_n --last-n 2 --seed $S1 ;;
    r2_s2) train r2_s2 r2 --mode last_n --last-n 2 --seed $S2 ;;
    r2na_s1) train r2na_s1 r2 --mode last_n --last-n 2 --adapter-dim 0 --seed $S1 --export-v1 ;;
    r2na_s2) train r2na_s2 r2 --mode last_n --last-n 2 --adapter-dim 0 --seed $S2 --export-v1 ;;
    r2ci_s1) train r2ci_s1 r2 --mode last_n --last-n 2 --cand-layers 2 --seed $S1 ;;
    r2l4_s1) train r2l4_s1 r2 --mode last_n --last-n 4 --seed $S1 ;;
    r2l4_s2) train r2l4_s2 r2 --mode last_n --last-n 4 --seed $S2 ;;
    r2l4na_s1) train r2l4na_s1 r2 --mode last_n --last-n 4 --adapter-dim 0 --seed $S1 --export-v1 ;;   # post-registration addition
    r2l4na_s2) train r2l4na_s2 r2 --mode last_n --last-n 4 --adapter-dim 0 --seed $S2 --export-v1 ;;
    hn_s1) hn s1 $S1 ;;
    hn_s2) hn s2 $S2 ;;
  esac
done
st "PHASE2_DONE $JOBS"
