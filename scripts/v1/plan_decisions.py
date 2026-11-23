"""Krun Decisions v1: capacity-aware plan for 1M / 2M / 3M (+ training cost from measured throughput).

For each scale and category, demand = target x mix; supply is filled in this order, each bounded by capacity x caps:
  REAL (each real context once) -> DERIVED (caps per context) -> MT (EN contexts x locales, derived per MT context)
  -> RULES (share) -> LLM (<= llm_max_share of the category). Unfilled demand is reported as SHORT (never padded).
Outputs results/v1/decisions_plan.{json,md}, including context/schema language pair statistics.

    uv run python scripts/v1/plan_decisions.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
P = yaml.safe_load((ROOT / "configs/v1/decisions_plan.yaml").read_text())
PRICE_4090_COMMUNITY = 0.34


def real_capacity():
    ci = P["capacity"]["intent"]
    en_intent = ci["banking77"] + ci["massive"] // 4 + ci["clinc150_train"] + ci["snips_train"] + ci["bitext"]
    return {
        "choice_intent": {"real": sum(ci.values()), "en_real": en_intent, "fam": "intent"},
        "tool_routing": {"real": P["capacity"]["tool"]["toolace_single_call"], "en_real": P["capacity"]["tool"]["toolace_single_call"], "fam": "tool"},
        "noul": {"real": P["capacity"]["noul_contexts"], "en_real": 0, "fam": "noul"},
        "score": {"real": P["capacity"]["score_contexts"], "en_real": 0, "fam": "score"},
        "multi_label": {"real": 0, "en_real": 0, "fam": "multi"},          # built from pairs of intent contexts (DERIVED)
        "ood": {"real": 0, "en_real": 0, "fam": "ood"},                    # gold-removed / cross-task (DERIVED) + near-OOD (LLM)
        "minimal_pairs": {"real": 0, "en_real": 0, "fam": "minimal_pairs"},
    }


def fill(cat: str, demand: int, cap: dict) -> dict:
    c = P["caps"]
    fam = cap["fam"]
    row = {"demand": demand, "real": 0, "derived": 0, "mt": 0, "rules": 0, "llm": 0}
    left = demand
    take = lambda n: (min(n, left))  # noqa: E731
    if fam in ("intent", "tool", "noul", "score"):
        row["real"] = take(cap["real"]); left -= row["real"]
        row["derived"] = take(cap["real"] * (c["derived_per_context"][fam] - 1)); left -= row["derived"]
    if fam in ("intent", "tool"):
        mt_ctx = int(cap["en_real"] * c["mt_contexts_share"][fam] * c["mt_locales"])
        row["mt"] = take(mt_ctx * c["derived_per_mt_context"][fam]); left -= row["mt"]
    if fam == "multi":
        intent_real = real_capacity()["choice_intent"]["real"]
        row["derived"] = take(int(min(demand * (1 - c["llm_max_share"]["multi"]), intent_real * 2))); left -= row["derived"]
    if fam == "ood":
        row["derived"] = take(int(demand * (1 - c["llm_max_share"]["ood"]))); left -= row["derived"]
    row["rules"] = take(int(demand * c["rules_share"])) if fam != "minimal_pairs" else 0; left -= row["rules"]
    llm_cap = int(demand * c["llm_max_share"].get(fam if fam not in ("intent", "tool") else fam, 0.1))
    row["llm"] = take(llm_cap); left -= row["llm"]
    row["short"] = left
    row["supplied"] = demand - left
    return row


def lang_pairs(n: int) -> dict:
    L = P["languages"]
    xl = P["overlays"]["cross_language_share_of_non_en"]
    tr = P["overlays"]["translated_schema_share_of_non_en"]
    out = {}
    for lang, share in L.items():
        k = n * share
        if lang == "en":
            out["en->en"] = int(k)
        else:
            out[f"{lang}->en (cross-language)"] = int(k * xl)
            out[f"{lang}->{lang} (translated schema)"] = int(k * tr)
            out[f"{lang}->en-labels (humanized EN labels)"] = int(k * (1 - xl - tr))
    return out


def main() -> None:
    caps = real_capacity()
    pilot = ROOT / "results/v1/synthetic_pilot/report.json"
    llm_cost = json.loads(pilot.read_text())["cost_per_100k_accepted"] if pilot.exists() else P["llm_generation"]["cost_per_100k_accepted_usd"]
    b = {}
    for f in glob.glob(str(ROOT / "results/v1/bench_training/bench_training_4090.json")):
        s = json.loads(Path(f).read_text())["summary"]
        b = {"decision_tok_s": s["decision_best"]["real_tokens_per_s"]}
    res = {"llm_cost_per_100k_accepted": llm_cost, "llm_cost_source": "pilot" if pilot.exists() else "estimate", "scales": {}}
    md = [f"LLM cost per 100k accepted: ${llm_cost:.2f} ({res['llm_cost_source']}); training on RTX 4090 community "
          f"(${PRICE_4090_COMMUNITY}/h, {b.get('decision_tok_s', 0):,.0f} real tok/s measured).\n"]
    for sname, sc in P["scales"].items():
        rows = {cat: fill(cat, int(sc["decisions"] * share), caps[cat]) for cat, share in P["mix"].items()}
        tot = {k: sum(r[k] for r in rows.values()) for k in ("demand", "real", "derived", "mt", "rules", "llm", "short", "supplied")}
        tokens = sum(rows[c]["supplied"] * P["measured"]["tokens_per_decision"][c] for c in rows)
        train_tok = tokens / P["measured"]["padding_efficiency"] * sc["epochs"]
        hours = tokens * sc["epochs"] / max(1, b.get("decision_tok_s", 13381)) / 3600
        hn = {c: int(rows[c]["supplied"] * P["overlays"]["hard_negative_share"].get(c, 0)) for c in rows}
        res["scales"][sname] = {"rows": rows, "total": tot, "hard_negatives": hn,
                                "language_pairs": lang_pairs(tot["supplied"]),
                                "storage_gb": round(tot["supplied"] * P["measured"]["bytes_per_record"] / 1e9, 2),
                                "padded_training_tokens": int(train_tok), "train_hours_4090": round(hours, 1),
                                "train_usd": round(hours * PRICE_4090_COMMUNITY, 2),
                                "llm_usd": round(tot["llm"] / 1e5 * llm_cost, 2),
                                "mt_usd": round(tot["mt"] / 1e5 * P["mt_generation"]["cost_per_100k_usd"], 2)}
        r = res["scales"][sname]
        md += [f"## {sname}: target {sc['decisions']:,} → supplied {tot['supplied']:,} (short {tot['short']:,})\n",
               "| category | demand | REAL | DERIVED | MT | RULES | LLM | short | hard negatives |", "|---|" + "---|" * 8]
        for c, x in rows.items():
            md.append(f"| {c} | {x['demand']:,} | {x['real']:,} | {x['derived']:,} | {x['mt']:,} | {x['rules']:,} | {x['llm']:,} | "
                      f"{x['short']:,} | {hn[c]:,} |")
        t = tot
        md.append(f"| **total** | {t['demand']:,} | {t['real']:,} | {t['derived']:,} | {t['mt']:,} | {t['rules']:,} | {t['llm']:,} | {t['short']:,} | {sum(hn.values()):,} |")
        md += ["", f"Storage {r['storage_gb']} GB jsonl · padded training tokens {r['padded_training_tokens'] / 1e9:.2f}B · "
                   f"train {r['train_hours_4090']} h ≈ ${r['train_usd']} · LLM ${r['llm_usd']} · MT ${r['mt_usd']}", ""]
    # language pairs of the 3M-equivalent supplied total
    lp = res["scales"]["S3M"]["language_pairs"]
    md += ["## Context → schema language pairs (S3M supplied)\n", "| pair | decisions |", "|---|---|"]
    md += [f"| {k} | {v:,} |" for k, v in lp.items()]
    (ROOT / "results/v1/decisions_plan.json").write_text(json.dumps(res, indent=2))
    (ROOT / "results/v1/decisions_plan.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
