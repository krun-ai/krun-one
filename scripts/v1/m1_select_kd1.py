"""M1: pick the spoken / screenshot pool from Krun Decisions v1 TRAIN (already LEAKAGE = 0 vs KrunBench v1).

Short single-utterance contexts (8..180 chars, no newline) in EN / PT / ES / FR, k <= 32; no toxicity sources (their
texts are not rendered or spoken); stratified reservoir per (type, family, language) with fixed quotas.

    python scripts/v1/m1_select_kd1.py  -> data/m1/kd1_pool.jsonl + data/m1/kd1_pool_stats.json
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXCLUDE_SOURCES = {"civil_comments", "mhs", "measuring_hate_speech", "multiwoz"}
LANGS = {"en", "pt", "es", "fr"}
QUOTA = {("choice", "intent"): 5200, ("choice", "tool"): 1500, ("noul", None): 1500, ("score", None): 1400,
         ("multi", None): 1100}          # per language -> x4 languages ~ 43k


def main() -> None:
    rng = random.Random(20260929)
    res: dict = defaultdict(list)
    seen: Counter = Counter()
    src_counts: Counter = Counter()
    with open(ROOT / "data/kd1/train.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            t, fam = r["task"]["type"], r["task"].get("family")
            key = (t, fam if t == "choice" else None)
            if key not in QUOTA:
                continue
            src = r["id"].split("/")[2]
            if any(s in src for s in EXCLUDE_SOURCES):
                continue
            ctx = r["context"]["text"]
            lang = r["context"]["language"].split("-")[0]
            if lang not in LANGS or not (8 <= len(ctx) <= 180) or "\n" in ctx or len(r["candidates"]) > 32:
                continue
            if r["context"].get("code_switching"):
                continue
            k = key + (lang,)
            seen[k] += 1
            q = QUOTA[key]
            if len(res[k]) < q:
                res[k].append(r)
            else:
                j = rng.randrange(seen[k])
                if j < q:
                    res[k][j] = r
    out = ROOT / "data/m1"
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out / "kd1_pool.jsonl", "w", encoding="utf-8") as f:
        for k in sorted(res, key=str):
            for r in res[k]:
                src_counts[r["id"].split("/")[2]] += 1
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
    stats = {"n": n, "per_stratum": {"/".join(map(str, k)): len(v) for k, v in sorted(res.items(), key=str)},
             "eligible_per_stratum": {"/".join(map(str, k)): v for k, v in sorted(seen.items(), key=str)},
             "sources": dict(src_counts.most_common())}
    (out / "kd1_pool_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
