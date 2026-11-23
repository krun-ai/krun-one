"""CUDA health check. Exit 0 only if ALL pass; any failure -> the caller terminates the pod immediately.

Checks: nvidia-smi responds; torch.cuda.is_available(); compute capability >= 8.0 (bf16); a tiny CUDA tensor op
(bf16 matmul vs CPU reference); a tiny ModernBERT forward on CUDA that matches CPU; a tiny backward whose
gradients match CPU; a VRAM smoke (~80% of free memory written and read back).

Regression guard: on 2026-09-26 two RunPod Community RTX 4090 hosts (CUDA 13.0 / driver 580) listed the GPU in
nvidia-smi but failed CUDA init ("CUDA unknown error"); the first one silently trained on CPU for 25 min.

    python scripts/v1/gpu_preflight.py        # prints one PREFLIGHT line per check
"""

import subprocess
import sys


def check(name, fn):
    try:
        info = fn()
        print(f"PREFLIGHT OK {name}: {info}", flush=True)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"PREFLIGHT FAIL {name}: {type(e).__name__}: {e}", flush=True)
        return False


def main() -> int:
    ok = check("nvidia-smi", lambda: subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        check=True, capture_output=True, text=True, timeout=30).stdout.strip())
    import torch

    def avail():
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is False")
        return f"{torch.cuda.get_device_name(0)} torch {torch.__version__} cuda {torch.version.cuda}"

    ok &= check("cuda_available", avail)
    if not ok:
        return 1

    def capability():
        cap = torch.cuda.get_device_capability(0)
        if cap < (8, 0):
            raise RuntimeError(f"capability {cap} < 8.0 (no bf16)")
        return str(cap)

    def tensor_op():
        a, b = torch.randn(512, 512), torch.randn(512, 512)
        err = float(((a.cuda().bfloat16() @ b.cuda().bfloat16()).float().cpu() - a @ b).abs().max() / (a @ b).abs().max())
        if err > 0.05:
            raise RuntimeError(f"bf16 matmul rel err {err}")
        return f"bf16 rel err {err:.4f}"

    def tiny_forward():
        from transformers import ModernBertConfig, ModernBertModel

        cfg = ModernBertConfig(vocab_size=1000, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
                               num_attention_heads=2, pad_token_id=0, bos_token_id=2, cls_token_id=2, eos_token_id=1,
                               sep_token_id=1)
        torch.manual_seed(0)
        m = ModernBertModel(cfg).eval()
        ids = torch.randint(5, 1000, (2, 32))
        with torch.no_grad():
            ref = m(input_ids=ids).last_hidden_state
            got = m.cuda()(input_ids=ids.cuda()).last_hidden_state.cpu()
        err = float((ref - got).abs().max())
        if err > 1e-2:
            raise RuntimeError(f"forward mismatch {err}")
        return f"max |Δ| {err:.2e}"

    def tiny_backward():
        torch.manual_seed(0)
        lin = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.GELU(), torch.nn.Linear(64, 8))
        x, y = torch.randn(16, 64), torch.randint(0, 8, (16,))
        ref = [g.clone() for g in torch.autograd.grad(torch.nn.functional.cross_entropy(lin(x), y), lin.parameters())]
        lin = lin.cuda()
        got = torch.autograd.grad(torch.nn.functional.cross_entropy(lin(x.cuda()), y.cuda()), lin.parameters())
        err = max(float((a - b.cpu()).abs().max()) for a, b in zip(ref, got))
        if err > 1e-3:
            raise RuntimeError(f"backward mismatch {err}")
        return f"grad max |Δ| {err:.2e}"

    def vram_smoke():
        free, total = torch.cuda.mem_get_info()
        n = int(0.8 * free) // (256 * 2**20)   # fill ~80% of free VRAM in 256 MiB bf16 blocks, touch each one
        blocks = []
        for i in range(n):
            b = torch.empty(128 * 2**20, dtype=torch.bfloat16, device="cuda")
            b.fill_(i % 7 + 1)
            blocks.append(b)
        bad = sum(int(b[:: 2**16].float().sum().item() != (i % 7 + 1) * b[:: 2**16].numel()) for i, b in enumerate(blocks))
        gib = n * 0.25
        del blocks
        torch.cuda.empty_cache()
        if bad or n == 0:
            raise RuntimeError(f"VRAM smoke: {bad} bad blocks of {n}")
        return f"{gib:.1f} GiB written+verified of {total / 2**30:.1f} GiB"

    ok &= check("capability", capability)
    ok &= check("vram_smoke", vram_smoke)
    ok &= check("tensor_op", tensor_op)
    ok &= check("tiny_model_forward", tiny_forward)
    ok &= check("tiny_backward", tiny_backward)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
