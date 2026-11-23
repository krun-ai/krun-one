"""Krun Decisions v1 — dataset stats + tool data numbers (-> results/v1/kd1_stats.{json,md}).

    uv run python scripts/v1/kd1_stats.py [--data data/kd1]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def lang_bucket(l: str) -> str:
    if l == "pt-BR":
        return "PT-BR"
    if l == "pt-PT":
        return "PT-PT"
    p = l.split("-")[0]
    return {"en": "EN", "es": "ES", "fr": "FR", "mul": "MUL"}.get(p, l)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/kd1"))
    ap.add_argument("--train", default="train.jsonl")
    ap.add_argument("--skip", type=int, default=0, help="skip the first N lines (e.g. the candidate-1 base inside the R2 file)")
    ap.add_argument("--val", default="val_ml.jsonl")
    ap.add_argument("--tag", default="kd1_stats")
    args = ap.parse_args()
    d = Path(args.data)
    C = defaultdict(Counter)
    n = 0
    chars = 0
    tools_gold, tools_all, tool_queries, tool_domains = set(), set(), set(), Counter()
    minimal_sets = set()
    for li, l in enumerate(open(d / args.train)):
        if li < args.skip:
            continue
        r = json.loads(l)
        n += 1
        md = r["metadata"]
        cat, org, lang = md["category"], md["origin"], r["context"]["language"]
        lb = lang_bucket(lang)
        C["category"][cat] += 1
        C["origin"][org] += 1
        C["lang"][lb] += 1
        C["locale"][lang] += 1
        C["stage"][md["stage"]] += 1
        C["type"][r["task"]["type"]] += 1
        C["cat_origin"][f"{cat}|{org}"] += 1
        C["cat_lang"][f"{cat}|{lb}"] += 1
        C["source"][r["provenance"]["source_dataset"]] += 1
        C["tool_source_all"][str(md.get("tool_source"))] += 1
        if md.get("tool_action"):
            C["tool_action"][md["tool_action"]] += 1
        C["difficulty"][r["difficulty"]] += 1
        C["answerable"][str(r["answerable"])] += 1
        if r["task"]["type"] in ("choice", "multi"):
            k = len(r["candidates"])
            kb = min((b for b in (2, 4, 8, 16, 32, 64, 128) if k <= b), default=128)
            C["k_bucket"][f"<={kb}"] += 1
        for s in r["slices"]:
            C["slices"][s] += 1
        if lb != "EN":
            C["non_en_schema"][r["schema_language"]] += 1
            C["non_en_cross"][str(r["cross_language"])] += 1
        C["pair"][f"{lb}->{r['schema_language']}"] += 1
        if md.get("ood_type"):
            C["ood_type"][md["ood_type"]] += 1
        if r.get("minimal_pair"):
            minimal_sets.add(r["minimal_pair"]["set_id"])
            C["minimal_pair_relation"][r["minimal_pair"].get("relation", "?")] += 1
        if r["provenance"].get("synthetic"):
            C["synthetic_transformation"][r["provenance"]["synthetic"]["transformation"]] += 1
        chars += len(r["context"]["text"]) + sum(len(c.get("description") or c["id"]) for c in r["candidates"])
        if r["task"].get("family") == "tool" and r["task"]["type"] in ("choice", "multi"):
            src = md.get("tool_source", "?")
            C["tool_source"][src] += 1
            C["tool_source_origin"][f"{src}|{org}"] += 1
            C["tool_lang"][lb] += 1
            if "near_duplicate" in r["slices"]:
                C["tool_flags"]["near_duplicate"] += 1
            if r["target"]["labels"] and r["task"]["type"] == "choice":
                import re as _re
                tok = lambda x: set(_re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", x).lower().replace("_", " ").replace("-", " ").split())  # noqa: E731
                g = tok(r["target"]["labels"][0])
                if any(len(g & tok(c["id"])) / max(1, len(g)) >= 0.5 for c in r["candidates"] if c["id"] != r["target"]["labels"][0]):
                    C["tool_flags"]["description_dependent"] += 1
            if any(c.get("language") not in (None, "en") for c in r["candidates"]):
                C["tool_flags"]["translated_schema"] += 1
            if not r["answerable"]:
                C["tool_flags"]["ood"] += 1
            if src == "toolace":
                tool_queries.add(r["semantic_group_id"])
                for c in r["candidates"]:
                    tools_all.add((c["id"], c.get("description")))
                for g in r["target"]["labels"]:
                    tools_gold.add(g)
    stats = {"n": n, "approx_tokens": int(chars / 3.6), **{k: dict(v.most_common()) for k, v in C.items()},
             "minimal_pair_sets": len(minimal_sets),
             "toolace": {"unique_query_groups": len(tool_queries), "unique_gold_tools": len(tools_gold),
                         "unique_candidate_tools": len(tools_all)}}
    val = Counter()
    for l in open(d / args.val):
        r = json.loads(l)
        val[f"{r['metadata'].get('val_part')}|{lang_bucket(r['context']['language'])}|{r['task']['family']}"] += 1
    stats["val_ml"] = dict(sorted(val.items()))
    out = ROOT / "results/v1"
    (out / f"{args.tag}.json").write_text(json.dumps(stats, indent=1))

    def pct(c, tot):
        return f"{100 * c / max(1, tot):.1f}%"

    md = [f"# Krun Decisions v1 — stats ({n:,} training decisions)\n"]
    for key, title in (("category", "Task bucket"), ("origin", "Origin"), ("lang", "Context language"),
                       ("locale", "Locale"), ("stage", "Curriculum stage"), ("type", "Question type"),
                       ("k_bucket", "Candidate count (choice/multi)"), ("ood_type", "OOD type"),
                       ("difficulty", "Difficulty"), ("source", "Source dataset")):
        md += [f"\n## {title}\n", "| value | decisions | share |", "|---|---:|---:|"]
        tot = sum(C[key].values())
        md += [f"| {k} | {v:,} | {pct(v, tot)} |" for k, v in C[key].most_common()]
    non_en = sum(C["non_en_cross"].values())
    md += ["\n## Cross-language\n",
           f"Non-EN decisions: {non_en:,}. Of these, **{pct(C['non_en_cross'].get('True', 0), non_en)} are cross-language** "
           f"and **{pct(C['non_en_schema'].get('en', 0), non_en)} use an EN schema**.\n",
           "| context → schema | decisions |", "|---|---:|"] + [f"| {k} | {v:,} |" for k, v in C["pair"].most_common(20)]
    md += ["\n## Category × origin\n", "| category | origin | decisions |", "|---|---|---:|"]
    md += [f"| {k.split('|')[0]} | {k.split('|')[1]} | {v:,} |" for k, v in sorted(C["cat_origin"].items())]
    md += ["\n## Category × language\n", "| category | language | decisions |", "|---|---|---:|"]
    md += [f"| {k.split('|')[0]} | {k.split('|')[1]} | {v:,} |" for k, v in sorted(C["cat_lang"].items())]
    md += ["\n## Slices\n", "| slice | decisions |", "|---|---:|"] + [f"| {k} | {v:,} |" for k, v in C["slices"].most_common()]
    tt = sum(C["tool_source"].values())
    md += ["\n## Tool routing block\n", f"Tool decisions (family tool, choice/multi): {tt:,}. "
           f"ToolACE unique query groups {stats['toolace']['unique_query_groups']:,}, unique gold tools "
           f"{stats['toolace']['unique_gold_tools']:,}, unique candidate tools {stats['toolace']['unique_candidate_tools']:,}.\n",
           "| tool source | origin | decisions |", "|---|---|---:|"]
    md += [f"| {k.split('|')[0]} | {k.split('|')[1]} | {v:,} |" for k, v in sorted(C["tool_source_origin"].items())]
    md += ["", "| flag | decisions | share of tool decisions |", "|---|---:|---:|"]
    md += [f"| {k} | {v:,} | {pct(v, tt)} |" for k, v in C["tool_flags"].most_common()]
    md += ["", "| language | tool decisions |", "|---|---:|"] + [f"| {k} | {v:,} |" for k, v in C["tool_lang"].most_common()]
    md += [f"\nMinimal-pair sets: {len(minimal_sets):,} ({dict(C['minimal_pair_relation'])}).",
           f"Synthetic (LLM) transformations: {dict(C['synthetic_transformation'])}.",
           f"Approx. text tokens (chars/3.6, context + candidates): {stats['approx_tokens']:,}.",
           "\n## Validation (val_ml)\n", "| part / language / family | decisions |", "|---|---:|"]
    md += [f"| {k} | {v:,} |" for k, v in sorted(val.items())]
    (out / f"{args.tag}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:40]))


if __name__ == "__main__":
    main()
