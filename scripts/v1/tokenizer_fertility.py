"""Krun One v1 — tokenizer fertility: ModernBERT (v0.3) vs Gemma-2/mmBERT tokenizer on EN/PT/ES/FR.

Tokens per decision drive latency roughly linearly for an encoder, so a multilingual vocabulary that
needs fewer tokens for PT/ES/FR is a latency lever, not only a quality one.

Data (all local, CC-BY-4.0): MASSIVE test utterances en-US / pt-PT / es-ES / fr-FR (parallel: same ids),
official xLAM/ToolACE tool strings from data/v0/val_tools.jsonl (English). Evaluation data only; nothing trained.

    uv run python scripts/v1/tokenizer_fertility.py
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
HUB = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
MASSIVE_REV = "ed58ac423a2f4121720918bf5301577edce4ffd3"
LOCALES = ["en-US", "pt-PT", "es-ES", "fr-FR"]
TOKENIZERS = {
    "modernbert (krun-one-v0.3)": ROOT / "artifacts/krun-one-v0.3/tokenizer",
    "gemma2-256k (mmBERT)": next(iter(glob.glob(str(HUB / "models--jhu-clsp--mmBERT-small/snapshots/abc3262*")))),
}


def massive(locale: str) -> dict[str, str]:
    t = pq.read_table(next(iter(glob.glob(str(HUB / f"datasets--AmazonScience--massive/snapshots/{MASSIVE_REV}/{locale}/test/*.parquet")))))
    return dict(zip(t.column("id").to_pylist(), t.column("utt").to_pylist()))


def tool_descriptions(n: int = 3000) -> list[str]:
    """Distinct official "name: description" tool strings from the (already opened) v0 validation file."""
    seen: dict[str, None] = {}
    with open(ROOT / "data/v0/val_tools.jsonl") as f:
        for line in f:
            d = json.loads(line)
            for name, desc in (d.get("option_descriptions") or {}).items():
                if desc:
                    seen.setdefault(f"{name}: {desc}")
            if len(seen) >= n:
                break
    return list(seen)[:n]


def main() -> None:
    utts = {loc: massive(loc) for loc in LOCALES}
    common = sorted(set.intersection(*(set(u) for u in utts.values())))
    tools = tool_descriptions()
    report: dict = {"massive_revision": MASSIVE_REV, "n_parallel_utterances": len(common), "n_tool_descriptions": len(tools),
                    "tokenizers": {}}
    for name, path in TOKENIZERS.items():
        tok = AutoTokenizer.from_pretrained(str(path))
        r = {"vocab_size": len(tok), "mask": tok.mask_token, "cls": tok.cls_token, "sep": tok.sep_token, "by_locale": {}}
        for loc in LOCALES:
            texts = [utts[loc][i] for i in common]
            n_tok = np.array([len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts])
            n_words = np.array([max(1, len(t.split())) for t in texts])
            n_chars = np.array([len(t) for t in texts])
            r["by_locale"][loc] = {"tokens_mean": float(n_tok.mean()), "tokens_p95": float(np.percentile(n_tok, 95)),
                                   "tokens_per_word": float(n_tok.sum() / n_words.sum()),
                                   "chars_per_token": float(n_chars.sum() / n_tok.sum())}
        tt = np.array([len(tok(t, add_special_tokens=False)["input_ids"]) for t in tools])
        r["tool_description"] = {"tokens_mean": float(tt.mean()), "tokens_p95": float(np.percentile(tt, 95)),
                                 "share_over_48": float((tt > 48).mean())}
        # " " + option (Laya renders options with a leading space): does it add a stray token with Metaspace?
        a = tok(" cancel subscription", add_special_tokens=False)["input_ids"]
        b = tok("cancel subscription", add_special_tokens=False)["input_ids"]
        r["leading_space_probe"] = {"with_space": tok.convert_ids_to_tokens(a), "without_space": tok.convert_ids_to_tokens(b)}
        report["tokenizers"][name] = r
    base, new = report["tokenizers"].values()
    report["ratio_gemma_over_modernbert"] = {loc: new["by_locale"][loc]["tokens_mean"] / base["by_locale"][loc]["tokens_mean"]
                                             for loc in LOCALES}
    report["ratio_gemma_over_modernbert"]["tool_description"] = new["tool_description"]["tokens_mean"] / base["tool_description"]["tokens_mean"]
    out = ROOT / "results/v1/tokenizer_fertility.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
