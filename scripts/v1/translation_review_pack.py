"""KrunBench v1 final_protected translation jobs -> human-review pack (PROTECTED data: never training, never published).

    uv run python scripts/v1/translation_review_pack.py --gpu-out data/krunbench_v1/translation_review/gpu_out
    -> data/krunbench_v1/translation_review/{review_items.jsonl, batches/batch_XXX.jsonl, sampling_plan.json, README.md}
    Review tool: open scripts/v1/translation_review.html in a browser (local file; loads a batch, exports decisions).

A machine draft (Qwen3-8B) + an automatic judge (Granite-3.3-8B) + rule flags are ONLY triage signals. Every item still
needs a human verdict (approve / edit / reject) before any final_protected PT/ES/FR slice exists. An automatic review is
never reported as a human one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ISSUES = ["meaning_changed", "wrong_regional_variant", "unnatural", "untranslated_or_mixed", "entity_or_number_error",
          "label_leak", "grammar_or_typo", "other"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-out", default=str(ROOT / "data/krunbench_v1/translation_review/gpu_out"))
    ap.add_argument("--batch-size", type=int, default=200)
    args = ap.parse_args()
    out = ROOT / "data/krunbench_v1/translation_review"
    jobs = {json.loads(l)["key"]: json.loads(l) for l in open(out / "gpu_jobs/translate.jsonl")}
    g = Path(args.gpu_out)
    drafts = {r["key"]: r for f in sorted(g.glob("translate.part-*.jsonl")) for r in map(json.loads, open(f))}
    judge = {r["key"]: r.get("verdict") or {} for f in sorted(g.glob("judge.part-*.jsonl")) for r in map(json.loads, open(f))}
    src = {}
    for l in open(ROOT / "data/krunbench_v1/final_protected.jsonl"):
        r = json.loads(l)
        src[r["id"]] = r
    try:
        from lingua import Language, LanguageDetectorBuilder

        lid = LanguageDetectorBuilder.from_languages(Language.ENGLISH, Language.PORTUGUESE, Language.SPANISH,
                                                     Language.FRENCH).build()
        code = {"ENGLISH": "en", "PORTUGUESE": "pt", "SPANISH": "es", "FRENCH": "fr"}
    except Exception:  # noqa: BLE001
        lid = None
    items = []
    for key, j in jobs.items():
        d = drafts.get(key, {})
        draft = (d.get("output") or "").strip()
        v = judge.get(key, {})
        flags = []
        if not draft:
            flags.append("no_draft")
        else:
            if set(re.findall(r"\d+", j["text"])) - set(re.findall(r"\d+", draft)):
                flags.append("number_changed")
            if not 0.4 <= len(draft) / max(1, len(j["text"])) <= 2.6:
                flags.append("length_ratio")
            if lid and len(draft.split()) >= 3:
                l = lid.detect_language_of(draft)
                if l is None or code.get(l.name) != j["target_locale"][:2]:
                    flags.append("language_id")
            try:
                if int(v.get("adequacy") or 0) < 4:
                    flags.append("judge_low_adequacy")
                if int(v.get("fluency") or 0) < 3:
                    flags.append("judge_low_fluency")
            except (TypeError, ValueError):
                flags.append("judge_unparseable")
        s = src.get(j["oid"].rsplit("/", 1)[0], {})
        gold = ", ".join(s.get("target", {}).get("labels") or []) or "(none / unanswerable)"
        items.append({"id": j["oid"], "key": key, "locale": j["target_locale"], "source_text": j["text"], "draft": draft,
                      "judge": v, "auto_flags": flags, "source_dataset": (s.get("provenance") or {}).get("source_dataset"),
                      "task": (s.get("task") or {}).get("type"), "gold_label": gold,
                      "human": {"status": "pending", "verdict": None, "issues": [], "corrected": None, "reviewer": None}})
    # review priority: flagged first, then a stratified random order (locale x source)
    items.sort(key=lambda x: (0 if x["auto_flags"] else 1, hashlib.sha256(x["key"].encode()).hexdigest()))
    (out / "batches").mkdir(exist_ok=True)
    with open(out / "review_items.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    by = defaultdict(list)
    for it in items:
        by[(it["locale"], it["source_dataset"])].append(it)
    # batch k takes items round-robin over strata, so every batch is a stratified sample
    order, strata = [], [v for _, v in sorted(by.items())]
    i = 0
    while any(strata):
        for s in strata:
            if s:
                order.append(s.pop(0))
        i += 1
    for b in range(0, len(order), args.batch_size):
        with open(out / f"batches/batch_{b // args.batch_size:03d}.jsonl", "w") as f:
            for it in order[b:b + args.batch_size]:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
    plan = {"n_items": len(items), "batch_size": args.batch_size, "n_batches": (len(order) + args.batch_size - 1) // args.batch_size,
            "by_locale": dict(Counter(it["locale"] for it in items)),
            "by_source": dict(Counter(it["source_dataset"] for it in items)),
            "auto_flagged": sum(1 for it in items if it["auto_flags"]),
            "flags": dict(Counter(f for it in items for f in it["auto_flags"])),
            "issue_categories": ISSUES,
            "acceptance_rule": "a PT/ES/FR final_protected slice exists only for items with human verdict approve/edit; "
                               "reject -> item dropped. A slice is reported only if >= 300 items per locale are human-approved.",
            "suggested_first_pass": "batches 000-005 (1,200 items, stratified) to estimate the error rate per locale; "
                                    "if > 10% edits in a locale, review that locale fully"}
    (out / "sampling_plan.json").write_text(json.dumps(plan, indent=1))
    print(json.dumps(plan, indent=1))


if __name__ == "__main__":
    main()
