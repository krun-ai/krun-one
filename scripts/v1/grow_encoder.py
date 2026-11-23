"""G0: grow mmBERT-base (768) into the K1-W1536-L22 encoder by HyperCloning and verify function preservation.

    uv run python scripts/v1/grow_encoder.py --source jhu-clsp/mmBERT-base --out runs/v1/grow/k1_w1536_l22_t0 \
        --noise-std 0.0            # 0 tokens checkpoint (exactly the source function)

Writes an HF ModernBertForMaskedLM directory + growth.json with: source revision, parameter counts, and the max
|Δ| between the source and grown MLM logits / hidden states on multilingual probe sentences (must be ~0 without
noise). The grown encoder config must equal krun_one.v1.arch.CANDIDATES[0].encoder (checked).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from krun_one.v1 import arch
from krun_one.v1.growth import widen_modernbert

PROBES = ["quero cancelar minha assinatura", "¿Puedo cambiar la fecha de mi vuelo?", "Je voudrais parler à un conseiller.",
          "Book a table for two tonight.", "O cliente pediu reembolso porque o produto veio quebrado."]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="jhu-clsp/mmBERT-base")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--noise-std", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=20260926)
    args = ap.parse_args()
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, ModernBertForMaskedLM

    src_dir = snapshot_download(args.source, revision=args.revision)
    tok = AutoTokenizer.from_pretrained(src_dir)
    src = ModernBertForMaskedLM.from_pretrained(src_dir, attn_implementation="eager", dtype=torch.float32).eval()
    dst = widen_modernbert(src, noise_std=args.noise_std, seed=args.seed).eval()
    want = arch.CANDIDATES[0].encoder
    c = dst.config
    shape = (c.hidden_size, c.num_hidden_layers, c.num_attention_heads, c.intermediate_size, c.vocab_size)
    assert shape == (want.hidden_size, want.num_layers, want.num_heads, want.intermediate_size, want.vocab_size), shape
    x = tok(PROBES, return_tensors="pt", padding=True)
    with torch.no_grad():
        a, b = src(**x), dst(**x)
        ha = src.model(**x).last_hidden_state
        hb = dst.model(**x).last_hidden_state
    m = x["attention_mask"].bool()
    report = {"source": args.source, "source_dir": src_dir, "noise_std": args.noise_std, "seed": args.seed,
              "params_source": sum(p.numel() for p in src.parameters()),
              "params_grown": sum(p.numel() for p in dst.parameters()),
              "max_abs_logit_diff": float((a.logits - b.logits)[m].abs().max()),
              "max_abs_hidden_diff": float(torch.maximum((hb[..., :ha.size(-1)] - ha).abs(),
                                                         (hb[..., ha.size(-1):] - ha).abs())[m].max()),
              "top1_agreement": float((a.logits.argmax(-1) == b.logits.argmax(-1))[m].float().mean())}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dst.save_pretrained(out)
    tok.save_pretrained(out)
    (out / "growth.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
