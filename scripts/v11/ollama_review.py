"""Quality review of generated decisions with an Ollama model (a REVIEWER, never a label source).

    uv run python scripts/v11/ollama_review.py --file data/v11/techbench/val_r3.jsonl --per-slice 24 --model gpt-oss:120b-cloud --tag r3a

The model sees what Krun sees in the raw rendering (instructions, context, candidate names) and answers with JSON.
Every disagreement with the programmatic label is written out for a manual look: a disagreement means either the model
is wrong (the label is computed, it stays) or the item is ambiguous / badly worded (the GENERATOR gets fixed).
Provenance per row: teacher_model, prompt_version, generation_type=quality_review, verification_method=programmatic label.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROMPT_VERSION = "r3-review-1"
SYSTEM = ("You are a careful reviewer of decision problems. Work the problem out exactly (do the arithmetic, follow every rule as "
          "written) and answer with JSON only.")


def prompt(e: dict) -> tuple[str, dict, int]:
    qt = e.get("qtype", "choice")
    if qt == "noul":
        return (f"Context:\n{e['context']}\n\nStatement: {e['instructions']}\n\nIs the statement true? Answer with JSON only: {{\"answer\": true}} or "
                f"{{\"answer\": false}}"), {"type": "object", "properties": {"answer": {"type": "boolean"}}, "required": ["answer"]}, int(e["gold"])
    lines = "\n".join(f"{i}. {c['name']}" for i, c in enumerate(e["candidates"], 1))
    none = "" if qt == "score" else "\n0. none of the candidates qualifies"
    gold = (e["gold"] + 1) if e["answerable"] else 0
    return (f"Task: {e['instructions']}\n\nContext:\n{e['context']}\n\nCandidates:\n{lines}{none}\n\nAnswer with JSON only: {{\"choice\": <number>}}",
            {"type": "object", "properties": {"choice": {"type": "integer"}}, "required": ["choice"]}, gold)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--model", default="gpt-oss:120b-cloud")
    ap.add_argument("--per-slice", type=int, default=24)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--think", default="high")
    ap.add_argument("--conc", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    ex = [json.loads(x) for x in open(ROOT / a.file, encoding="utf-8").read().split("\n") if x]
    ex = [e for e in ex if e["render"] == "raw" and e.get("qtype", "choice") != "multi"]
    by = defaultdict(list)
    for e in ex:
        by[(e["slice"], e.get("qtype", "choice"))].append(e)
    rng = random.Random(a.seed)
    sample = [e for k in sorted(by) for e in rng.sample(by[k], min(len(by[k]), a.per_slice))]
    out = ROOT / "results/v11/r3/quality_review"
    out.mkdir(parents=True, exist_ok=True)
    rows, lock = [], threading.Lock()
    think = {"false": False, "true": True}.get(a.think, a.think)

    def one(e):
        p, schema, gold = prompt(e)
        body = {"model": a.model, "stream": False, "think": think, "format": schema,
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}],
                "options": {"temperature": 0, "num_predict": 4096, "seed": 20261201}}
        ans, err = None, None
        for attempt in range(3):
            try:
                req = urllib.request.Request("http://localhost:11434/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
                d = json.load(urllib.request.urlopen(req, timeout=300))
                j = json.loads((d.get("message") or {}).get("content") or "{}")
                ans = int(j["answer"]) if "answer" in j else j.get("choice")
                err = None
                break
            except Exception as x:  # noqa: BLE001
                err = f"{type(x).__name__}: {x}"
                time.sleep(3 * (attempt + 1))
        r = {"id": e["id"], "slice": e["slice"], "track": e["track"], "qtype": e.get("qtype", "choice"), "gold": gold, "answer": ans,
             "agree": ans == gold, "error": err, "teacher_model": a.model, "prompt_version": PROMPT_VERSION, "generation_type": "quality_review",
             "verification_method": "programmatic label (arithmetic / calendar / graph walk / rule interpreter)", "think": a.think}
        with lock:
            rows.append(r)

    with ThreadPoolExecutor(a.conc) as pool:
        list(pool.map(one, sample))
    (out / f"review_{a.tag}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    byid = {e["id"]: e for e in sample}
    agg = defaultdict(Counter)
    for r in rows:
        agg[(r["slice"], r["qtype"])]["n"] += 1
        agg[(r["slice"], r["qtype"])]["agree"] += bool(r["agree"])
        agg[(r["slice"], r["qtype"])]["error"] += r["error"] is not None
    summ = {f"{s}/{q}": dict(c) for (s, q), c in sorted(agg.items())}
    dis = [{**r, "instructions": byid[r["id"]]["instructions"], "context": byid[r["id"]]["context"],
            "candidates": [c["name"] + " || " + c["feat"] for c in byid[r["id"]]["candidates"]]} for r in rows if not r["agree"] and not r["error"]]
    (out / f"summary_{a.tag}.json").write_text(json.dumps({"model": a.model, "prompt_version": PROMPT_VERSION, "file": a.file, "n": len(rows),
                                                          "agreement": sum(bool(r["agree"]) for r in rows) / max(1, len(rows)), "by_slice": summ}, indent=1))
    (out / f"disagreements_{a.tag}.json").write_text(json.dumps(dis, indent=1, ensure_ascii=False))
    print(json.dumps({"n": len(rows), "agree": sum(bool(r["agree"]) for r in rows), "errors": sum(r["error"] is not None for r in rows)}))
    for k, c in summ.items():
        print(k, c)


if __name__ == "__main__":
    main()
