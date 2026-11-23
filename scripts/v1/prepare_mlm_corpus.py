"""Budgeted continued-MLM corpus (~1B tokens, EN 35 / PT 25 / ES 20 / FR 20), quality > quantity.

Sources and licenses: configs/v1/sources.yaml (mlm_sources) and docs/KRUN_ONE_V1_CORPUS_PROVENANCE.md. FineWiki is
EXCLUDED (owner decision 2026-09-26) and is not replaced token-for-token. Repo revisions are pinned below.

Slices (token quotas per language; a slice that has less text contributes what it has — never upsampled):
  web        EN FineWeb-Edu sample-10BT · PT/ES/FR FineWeb2-HQ (top-10% quality)            ODC-By 1.0 + CC ToU
  technical  FinePDFs-edu (manuals, specs, documentation)                                  ODC-By 1.0 + CC ToU
  forums     FineWeb-2 documents whose URL looks like a forum/community (informal text)     ODC-By 1.0 + CC ToU
  support    Bitext customer support (EN), OpenAssistant oasst1+oasst2 conversations        CDLA-Sharing-1.0 / Apache-2.0
  tools      ToolACE tool schemas (JSON) + dialogue turns (EN)                               Apache-2.0

Documents of all slices of a language are shuffled together before sharding (so training does not see slices in
blocks). The first documents of the web slice (not used for training) form the per-language held-out set.

Decontamination (documents dropped, counts in the manifest): 8-word shingles of every protected context with >= 8
words, and the full word sequence of protected contexts with 6-7 words; protected = KrunBench v0 test (incl. BFCL,
HWU64), holdout_tools, primitives test sets, and the decision-probe VALIDATION set (so the 1B-vs-0B probe is fair).
A verification pass then decodes random training windows and searches them for protected contexts (>= 30 chars).

    python scripts/v1/prepare_mlm_corpus.py --protected results/v1/mlm_protected_texts.json --out data/v1_mlm
    python scripts/v1/prepare_mlm_corpus.py --dump-protected results/v1/mlm_protected_texts.json   # locally, once
"""

from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from krun_one.v1.leakage import normalize

ROOT = Path(__file__).resolve().parents[2]
HUB = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
REV = {"HuggingFaceFW/fineweb-edu": "87f09149ef", "epfml/FineWeb2-HQ": "c0c06e94fd", "HuggingFaceFW/finepdfs-edu": "9cfabe2127",
       "HuggingFaceFW/fineweb-2": "af9c13333e", "bitext/Bitext-customer-support-llm-chatbot-training-dataset": "430d1a89bd",
       "OpenAssistant/oasst1": "fdf72ae082", "OpenAssistant/oasst2": "179dd21fc5", "Team-ACE/ToolACE": "6bda777c88"}
FW2 = {"pt": "por_Latn", "es": "spa_Latn", "fr": "fra_Latn"}
QUOTA = {  # tokens (Gemma tokenizer); FineWiki's former share is intentionally NOT reassigned
    "en": {"web": 250e6, "technical": 40e6, "support": 25e6, "tools": 10e6},
    "pt": {"web": 190e6, "technical": 25e6, "forums": 12e6, "support": 3e6},
    "es": {"web": 150e6, "technical": 20e6, "forums": 10e6, "support": 5e6},
    "fr": {"web": 150e6, "technical": 20e6, "forums": 10e6, "support": 5e6},
}
OASST_LANG = {"en": {"en"}, "pt": {"pt-BR", "pt"}, "es": {"es"}, "fr": {"fr"}}
FORUM_RE = re.compile(r"forum|/t/|thread|topic|viewtopic|comunidade|/foro|foros|discussion|/community|/answers/|perguntas|preguntas", re.I)
SHARD_TOKENS = 100_000_000


# ------------------------------------------------------------------------------------------------ protected
def dump_protected(path: Path) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("bpd", ROOT / "scripts/v1/build_proxy_data.py")
    bpd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bpd)
    texts = [r["context"]["text"] for r in bpd.protected_records()]
    for f in [ROOT / "data/v1_probe/validation.jsonl", *sorted(glob.glob(str(ROOT / "data/krunbench_v1/*.jsonl"))),
              *glob.glob(str(ROOT.parents[1] / "benchmarks/krunbench/v1/**/*.jsonl"), recursive=True)]:
        if Path(f).exists():
            with open(f) as fh:
                texts += [(d.get("context") or {}).get("text") or d.get("text") for d in map(json.loads, fh)]
    texts = sorted({t for t in texts if t})
    path.write_text(json.dumps({"n": len(texts), "sha256": hashlib.sha256("\n".join(texts).encode()).hexdigest(),
                                "texts": texts}, ensure_ascii=False))
    print(f"protected texts: {len(texts)} -> {path}")


class Decontaminator:
    def __init__(self, texts: list[str], n: int = 8, min_words: int = 6, min_chars_verify: int = 30):
        self.n = n
        self.sets: dict[int, set[int]] = {n: set()}
        self.long = []
        for t in texts:
            w = normalize(t).split()
            if len(w) >= n:
                for i in range(len(w) - n + 1):
                    self.sets[n].add(hash(" ".join(w[i:i + n])))
            elif len(w) >= min_words:
                self.sets.setdefault(len(w), set()).add(hash(" ".join(w)))
            nt = normalize(t)
            if len(nt) >= min_chars_verify:
                self.long.append(nt)

    def dirty(self, doc: str) -> bool:
        w = normalize(doc).split()
        for n, s in self.sets.items():
            for i in range(len(w) - n + 1):
                if hash(" ".join(w[i:i + n])) in s:
                    return True
        return False


# ------------------------------------------------------------------------------------------------ sources
def hf_files(repo: str, include) -> list[str]:
    from huggingface_hub import HfApi

    return sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset", revision=REV[repo]) if include(f))


def fetch(repo: str, f: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, f, repo_type="dataset", revision=REV[repo])


def parquet_texts(repo: str, files: list[str], url_filter=None):
    for f in files:
        pf = pq.ParquetFile(fetch(repo, f))
        cols = ["text"] + (["url"] if url_filter else [])
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=cols)
            texts = t.column("text").to_pylist()
            if url_filter:
                urls = t.column("url").to_pylist()
                texts = [x for x, u in zip(texts, urls) if u and url_filter.search(u)]
            yield from texts


def slice_texts(lang: str, sl: str):
    if sl == "web":
        if lang == "en":
            yield from parquet_texts("HuggingFaceFW/fineweb-edu", hf_files("HuggingFaceFW/fineweb-edu", lambda f: f.startswith("sample/10BT/")))
        else:
            yield from parquet_texts("epfml/FineWeb2-HQ", hf_files("epfml/FineWeb2-HQ", lambda f: f.startswith(FW2[lang] + "/")))
    elif sl == "technical":
        code = "eng_Latn" if lang == "en" else FW2[lang]
        yield from parquet_texts("HuggingFaceFW/finepdfs-edu",
                                 hf_files("HuggingFaceFW/finepdfs-edu", lambda f: f.startswith(f"data/{code}/train/")))
    elif sl == "forums":
        # at most 2 raw FineWeb-2 files (~4.8 GB each) per language: the slice takes what they contain, never more
        yield from parquet_texts("HuggingFaceFW/fineweb-2",
                                 hf_files("HuggingFaceFW/fineweb-2", lambda f: f.startswith(f"data/{FW2[lang]}/train/"))[:2], FORUM_RE)
    elif sl == "support":
        if lang == "en":
            import csv

            with open(fetch("bitext/Bitext-customer-support-llm-chatbot-training-dataset",
                            "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"), newline="") as fh:
                for row in csv.DictReader(fh):
                    yield f"Customer: {row.get('instruction', '')}\nAgent: {row.get('response', '')}"
        for repo, f in (("OpenAssistant/oasst1", "2023-04-12_oasst_ready.messages.jsonl.gz"),
                        ("OpenAssistant/oasst2", "2023-11-05_oasst2_ready.messages.jsonl.gz")):
            with gzip.open(fetch(repo, f), "rt") as fh:
                for line in fh:
                    m = json.loads(line)
                    if m.get("lang") in OASST_LANG[lang] and m.get("text"):
                        yield m["text"]
    elif sl == "tools":
        data = json.loads(Path(fetch("Team-ACE/ToolACE", "data.json")).read_text())
        for item in data:
            yield str(item.get("system", ""))[:20000]      # tool schemas (JSON) live in the system prompt
            for turn in item.get("conversations", []):
                if turn.get("value"):
                    yield str(turn["value"])


# ------------------------------------------------------------------------------------------------ build
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data/v1_mlm"))
    ap.add_argument("--protected", default=str(ROOT / "results/v1/mlm_protected_texts.json"))
    ap.add_argument("--dump-protected")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply all quotas (smoke tests)")
    ap.add_argument("--heldout", type=float, default=2e6)
    ap.add_argument("--langs", default="en,pt,es,fr")
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument("--slices", default=None, help="comma list to restrict slices (smoke tests)")
    ap.add_argument("--tokenizer", default=None)
    args = ap.parse_args()
    if args.dump_protected:
        dump_protected(Path(args.dump_protected))
        return
    from transformers import AutoTokenizer

    tok_dir = args.tokenizer or str(Path(sorted(glob.glob(str(HUB / "models--jhu-clsp--mmBERT-small/snapshots/*/tokenizer.json")))[0]).parent)
    tok = AutoTokenizer.from_pretrained(tok_dir)
    prot = json.loads(Path(args.protected).read_text())
    deco = Decontaminator(prot["texts"])
    out = Path(args.out)
    manifest = {"tokenizer": tok_dir, "protected": {"n": prot["n"], "sha256": prot["sha256"]}, "revisions": REV,
                "excluded": {"finewiki": "excluded_for_v1_initial_training (share-alike review)"}, "languages": {}}
    rng = random.Random(args.seed)
    mf_name = f"manifest_{'_'.join(args.langs.split(','))}.json"   # one per process: languages can build in parallel
    for lang in args.langs.split(","):
        t0 = time.time()
        docs: list[np.ndarray] = []
        held: list[int] = []
        st: dict = {}
        for sl, q in QUOTA[lang].items():
            if args.slices and sl not in args.slices.split(","):
                continue
            quota, got, n_docs, dropped, need_held = int(q * args.scale), 0, 0, 0, int(args.heldout * args.scale) if sl == "web" else 0
            batch: list[str] = []

            def flush():
                nonlocal got, n_docs, need_held
                for ids in tok(batch, add_special_tokens=False)["input_ids"]:
                    ids = ids + [tok.eos_token_id]
                    if need_held > 0:
                        held.extend(ids)
                        need_held -= len(ids)
                        continue
                    if got >= quota:
                        break
                    docs.append(np.asarray(ids, dtype=np.uint32))
                    got += len(ids)
                    n_docs += 1
                batch.clear()

            for text in slice_texts(lang, sl):
                if not text or len(text) < 50:
                    continue
                if deco.dirty(text):
                    dropped += 1
                    continue
                batch.append(text)
                if len(batch) >= 256:
                    flush()
                    if got >= quota:
                        break
            if batch and got < quota:
                flush()
            st[sl] = {"tokens": got, "docs": n_docs, "dropped_decontamination": dropped, "quota": quota}
            print(lang, sl, st[sl], f"{time.time() - t0:.0f}s", flush=True)
        rng.shuffle(docs)
        d = out / lang
        d.mkdir(parents=True, exist_ok=True)
        flat = np.concatenate(docs) if docs else np.zeros(0, dtype=np.uint32)
        shards = {}
        for i in range(0, len(flat), SHARD_TOKENS):
            p = d / f"train-{i // SHARD_TOKENS:05d}.npy"
            np.save(p, flat[i:i + SHARD_TOKENS])
            shards[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        np.save(d / "heldout.npy", np.asarray(held, dtype=np.uint32))
        manifest["languages"][lang] = {"slices": st, "train_tokens": int(len(flat)), "heldout_tokens": len(held),
                                       "shards": shards, "seconds": round(time.time() - t0)}
        (out / mf_name).write_text(json.dumps(manifest, indent=2))
    manifest["verification"] = verify(out, tok, deco, args.langs.split(","), rng)
    (out / mf_name).write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["verification"]))


def verify(out: Path, tok, deco: Decontaminator, langs: list[str], rng: random.Random, windows: int = 500) -> dict:
    """Independent check: decode random training windows and look for protected contexts (>= 30 chars) as substrings."""
    hits, checked = [], 0
    for lang in langs:
        files = sorted((out / lang).glob("train-*.npy"))
        if not files:
            continue
        arrs = [np.load(f, mmap_mode="r") for f in files]
        for _ in range(windows):
            a = arrs[rng.randrange(len(arrs))]
            if len(a) < 2048:
                continue
            s = rng.randrange(len(a) - 1024)
            text = normalize(tok.decode(np.asarray(a[s:s + 1024]).tolist()))
            checked += 1
            for t in deco.long:
                if t in text:
                    hits.append({"lang": lang, "context": t[:80]})
                    break
    return {"windows_checked": checked, "protected_hits": len(hits), "examples": hits[:5]}


if __name__ == "__main__":
    main()
