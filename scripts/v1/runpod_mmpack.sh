#!/bin/bash
# Multimodal packaging round on ONE pod (no training): preflight -> env -> input checksums -> health (every component,
# via the GPU reference run of the fixtures) -> fresh-process reload + golden tests (KRUN_MM_DEVICE=cuda) + contract tests
# -> CPU-vs-GPU decision agreement -> memory report (fresh process per configuration) -> latency -> DONE.
set -u
K=/workspace/krun/experiments/krun_one
cd $K
export HF_HOME=/workspace/hf HF_HUB_CACHE=/workspace/hf/hub TORCH_DISABLE_NATIVE_JIT=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
L=/workspace/logs; O=results/v1/multimodal_packaging; mkdir -p $L $O; PY=.venv/bin/python
st() { echo "$1 $(date -u +%FT%TZ)" >> $L/status; }
fail() { st "FAILED $1"; st DONE; exit 1; }
st start
python3 -c "import transformers" 2>/dev/null || python3 -m pip install -q --break-system-packages transformers > /dev/null 2>&1
python3 scripts/v1/gpu_preflight.py > $L/preflight_image.log 2>&1 || fail "gpu_preflight(image)"
command -v uv >/dev/null || pip install -q uv > /dev/null 2>&1
[ -x $PY ] || { UV_HTTP_TIMEOUT=600 uv sync --frozen --group m1 --group m15 --group dev > $L/uv.log 2>&1 || UV_HTTP_TIMEOUT=600 uv sync --frozen --group m1 --group m15 --group dev >> $L/uv.log 2>&1; }
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | cut -d. -f1)
[ "${CAP:-0}" -ge 12 ] && fail "Blackwell host: the pinned onnxruntime-gpu 1.22 / cu126 stack is only validated on CUDA 12 GPUs"
$PY scripts/v1/gpu_preflight.py > $L/preflight_venv.log 2>&1 || fail "gpu_preflight(venv)"
st "preflight ok"
sha256sum -c inputs_mmpack.sha256 > $L/inputs_check.log 2>&1 || fail "input checksums"
st "inputs verified"
# health = every component on the GPU (text, vision, OCR, ASR decode, acoustic, fusion) through the fixture reference run
[ -f tests/fixtures/multimodal/expected_cuda.json ] || $PY scripts/v1/make_multimodal_fixtures.py reference --device cuda > $L/reference_cuda.log 2>&1 || fail "health / GPU reference"
st "health ok (GPU reference of all fixtures)"
KRUN_MM_DEVICE=cuda $PY -m pytest -p no:warnings -rA tests/v1/test_multimodal_runtime.py tests/v1/test_multimodal_contract.py > $L/pytest_cuda.log 2>&1
grep -qE "[0-9]+ passed" $L/pytest_cuda.log && ! grep -qE "[0-9]+ failed|[0-9]+ error" $L/pytest_cuda.log || fail "pytest cuda (see pytest_cuda.log)"
st "reload + golden + contract tests ok (fresh process, cuda)"
$PY - > $O/cpu_vs_cuda.json <<'PYEOF'
import json
c=json.load(open('tests/fixtures/multimodal/expected_cpu.json'))['cases']; g=json.load(open('tests/fixtures/multimodal/expected_cuda.json'))['cases']
rows=[]
for n in c:
    for q,a in c[n]['answers'].items():
        b=g[n]['answers'][q]
        d=max([abs(a[f][i]-b[f][i]) for f in ('probabilities','scores') if f in a for i in a[f]]+[abs(a[f]-b[f]) for f in ('noul','score') if f in a]+[0])
        rows.append({'case':n,'q':q,'same_top':a.get('top_option')==b.get('top_option'),'same_decision':a.get('decision')==b.get('decision'),
                     'same_selected':a.get('selected')==b.get('selected'),'max_abs_prob_diff':d})
print(json.dumps({'n':len(rows),'all_same_top':all(r['same_top'] for r in rows),'all_same_decision':all(r['same_decision'] for r in rows),
                  'all_same_selected':all(r['same_selected'] for r in rows),'max_abs_prob_diff':max(r['max_abs_prob_diff'] for r in rows),'rows':rows},indent=1))
PYEOF
st "cpu vs cuda compared"
for c in text vision audio full; do $PY scripts/v1/multimodal_bench.py memory --config $c > $L/memory_$c.log 2>&1 || fail "memory $c"; done
st "memory ok"
$PY scripts/v1/multimodal_bench.py latency > $L/latency.log 2>&1 || fail "latency"
st "latency ok"
st DONE
