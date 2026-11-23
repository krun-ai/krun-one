"""Krun Decisions v1 — synthetic pilot (2,000 items max): generator -> blind independent verifier -> filters.

Stages (the GPU stages run in separate processes so each model gets the whole GPU):
  prepare   (CPU)  sample REAL source decisions from data/v1_probe/train.jsonl (never validation/protected), build
                   generation prompts per transformation            -> <out>/prompts.jsonl
  generate  (GPU)  Qwen/Qwen3-8B (Apache-2.0) via vLLM, thinking off   -> <out>/generated.jsonl (+ throughput)
  verify    (GPU)  ibm-granite/granite-3.3-8b-instruct (Apache-2.0, different family) answers each candidate decision
                   BLIND (never sees the intended label or the generator's text about it)  -> <out>/verified.jsonl
  finalize  (CPU)  schema, language, naturalness, minimal-pair edit bounds, dedup (exact + near), leakage vs
                   protected, acceptance = verifier agrees with the intended label  -> accepted.jsonl, rejected.jsonl,
                   report.json
Typos are RULE-generated (no LLM) but still go through the blind verifier.

    python scripts/v1/synthetic_pilot.py prepare --n 2000 --out results/v1/synthetic_pilot
    python scripts/v1/synthetic_pilot.py generate --out results/v1/synthetic_pilot          # GPU
    python scripts/v1/synthetic_pilot.py verify --out results/v1/synthetic_pilot            # GPU
    python scripts/v1/synthetic_pilot.py finalize --out results/v1/synthetic_pilot --gpu-price 0.34
"""

from __future__ import annotations

import argparse
import difflib
import json
import random
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GENERATOR = {"model": "Qwen/Qwen3-8B", "license": "Apache-2.0"}
VERIFIER = {"model": "ibm-granite/granite-3.3-8b-instruct", "license": "Apache-2.0"}
PROMPT_VERSION = "kd1-pilot-v1"
MIX = {"minimal_pair": 400, "near_ood": 300, "regional": 250, "counterfactual": 200, "hard_negative": 200,
       "slang": 200, "typo": 200, "code_switching": 150, "ambiguity": 100}
LANG_NAME = {"en": "English", "pt-BR": "Brazilian Portuguese", "pt-PT": "European Portuguese", "es-ES": "Spanish (Spain)",
             "es-419": "Latin American Spanish", "fr-FR": "French (France)", "fr-CA": "Canadian French"}
REGIONAL = {"pt-PT": "pt-BR", "es-ES": "es-419", "fr-FR": "fr-CA"}


def jl(p):
    with open(p) as f:
        return [json.loads(l) for l in f]


def cand_line(c):
    return f"- {c['id']}: {c['description']}" if c.get("description") else f"- {c['id']}"


def gen_prompt(kind: str, src: dict, extra: dict) -> str:
    ctx, lang = src["context"]["text"], src["context"]["language"]
    lname = LANG_NAME.get(lang, LANG_NAME.get(lang.split("-")[0], "English"))
    cands = "\n".join(cand_line(c) for c in src["candidates"])
    head = ("You create evaluation data for a routing classifier. Return ONLY a JSON object {\"text\": \"...\"} "
            "with a single user message, no explanations.\n\n")
    if kind == "minimal_pair":
        return head + (f"Original message ({lname}): \"{ctx}\"\nIt currently means: {extra['gold']}.\n"
                       f"Make the SMALLEST possible edit (change a few words) so that the message clearly means "
                       f"\"{extra['target']}\" instead. Same language, same style, natural.")
    if kind == "counterfactual":
        return head + (f"Available functions:\n{cands}\n\nOriginal request: \"{ctx}\" (it calls {extra['gold']}).\n"
                       f"Rewrite the request, keeping its topic and entities where possible, so that the correct "
                       f"function becomes {extra['target']}. Natural user wording.")
    if kind == "hard_negative":
        return ("You create evaluation data for a routing classifier. Return ONLY a JSON object "
                "{\"name\": \"snake_case_function_name\", \"description\": \"one sentence\"}.\n\n"
                f"User request: \"{ctx}\"\nCorrect function: {extra['gold']}: {extra['gold_desc']}\n"
                "Invent ONE different function from the same domain whose name and description look similar, "
                "but which would be the WRONG function for this request (it does something else).")
    if kind == "near_ood":
        return head + (f"Options:\n{cands}\n\nWrite a realistic request ({lname}) from the SAME domain/topic as these "
                       f"options that NONE of the options can handle. It must not be answerable by any option.")
    if kind == "ambiguity":
        return head + (f"Options:\n{cands}\n\nWrite a realistic, short request ({lname}) that could reasonably mean "
                       f"EITHER \"{extra['a']}\" OR \"{extra['b']}\", and no other option.")
    if kind == "slang":
        return head + (f"Rewrite this message in very informal, colloquial {lname} (slang, abbreviations typical of "
                       f"chat), keeping exactly the same meaning: \"{ctx}\"")
    if kind == "regional":
        tgt = LANG_NAME[extra["target_lang"]]
        return head + (f"Rewrite this {lname} message as a native speaker of {tgt} would naturally say it (regional "
                       f"vocabulary and grammar), keeping exactly the same meaning: \"{ctx}\"")
    if kind == "code_switching":
        return head + (f"Rewrite this {lname} message the way a bilingual speaker would, mixing in a few ENGLISH "
                       f"words or phrases naturally (code-switching), keeping exactly the same meaning: \"{ctx}\"")
    raise ValueError(kind)


KEYS = "qwertyuiopasdfghjklzxcvbnm"
NEAR = {c: KEYS[max(0, i - 1)] + KEYS[min(len(KEYS) - 1, i + 1)] for i, c in enumerate(KEYS)}


def rule_typos(text: str, rng: random.Random) -> str:
    chars = list(text)
    n = max(1, len(chars) // 15)
    for _ in range(n):
        i = rng.randrange(len(chars))
        c = chars[i].lower()
        op = rng.random()
        if c in NEAR and op < 0.4:
            chars[i] = rng.choice(NEAR[c])
        elif op < 0.6 and i + 1 < len(chars):
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
        elif op < 0.8:
            chars[i] = ""
        else:
            chars[i] = chars[i] * 2
    out = "".join(chars)
    return re.sub(r"[áàâã]", "a", out) if rng.random() < 0.3 else out   # accent stripping, common in chat


# ------------------------------------------------------------------------------------------------ prepare
def prepare(args) -> None:
    rng = random.Random(args.seed)
    train = [r for r in jl(ROOT / "data/v1_probe/train.jsonl") if r["split"] == "train" and r["task"]["type"] == "choice"
             and r["answerable"] and "permutation" not in r.get("slices", [])]
    intents = [r for r in train if r["task"].get("family") == "intent" and len(r["candidates"]) >= 4]
    tools = [r for r in train if r["task"].get("family") == "tool" and len(r["candidates"]) >= 3]
    by_lang = defaultdict(list)
    for r in intents:
        by_lang[r["context"]["language"]].append(r)
    scale = args.n / sum(MIX.values())
    items = []

    def add(kind, src, extra, prompt=None, text=None):
        items.append({"pid": f"{kind}/{len(items)}", "kind": kind, "source_id": src["id"], "source": src, "extra": extra,
                      "prompt": prompt, "rule_text": text, "seed": args.seed})

    for kind, n in MIX.items():
        n = int(round(n * scale))
        for _ in range(n):
            if kind in ("minimal_pair", "near_ood", "ambiguity", "slang", "code_switching", "typo"):
                pool = intents if kind in ("minimal_pair", "ambiguity", "typo") or rng.random() < 0.7 else tools
                if kind == "code_switching":
                    pool = [r for r in intents if not r["context"]["language"].startswith("en")]
                src = rng.choice(pool)
            elif kind == "regional":
                src = rng.choice(by_lang[rng.choice(list(REGIONAL))])
            else:
                src = rng.choice(tools)
            gold = src["target"]["labels"][0]
            others = [c["id"] for c in src["candidates"] if c["id"] != gold]
            extra = {"gold": gold}
            if kind in ("minimal_pair", "counterfactual"):
                extra["target"] = others[0] if kind == "counterfactual" else rng.choice(others[:3])
            if kind == "hard_negative":
                extra["gold_desc"] = next(c.get("description") or c["id"] for c in src["candidates"] if c["id"] == gold)
            if kind == "ambiguity":
                extra.update(a=gold, b=others[0])
            if kind == "regional":
                extra["target_lang"] = REGIONAL[src["context"]["language"]]
            if kind == "typo":
                add(kind, src, extra, text=rule_typos(src["context"]["text"], rng))
            else:
                add(kind, src, extra, prompt=gen_prompt(kind, src, extra))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "prompts.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(json.dumps({"prompts": len(items), "by_kind": dict(Counter(i["kind"] for i in items))}))


# ------------------------------------------------------------------------------------------------ GPU stages
def _llm(model: str):
    from vllm import LLM

    return LLM(model=model, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.88, seed=20260926)


def _parse_json(s: str) -> dict | None:
    m = re.search(r"\{.*\}", s, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def generate(args) -> None:
    from vllm import SamplingParams

    out = Path(args.out)
    items = jl(out / "prompts.jsonl")
    todo = [it for it in items if it["prompt"]]
    llm = _llm(GENERATOR["model"])
    sp = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=160, seed=20260926)
    msgs = [[{"role": "user", "content": it["prompt"]}] for it in todo]
    t0 = time.time()
    res = llm.chat(msgs, sp, chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
    dt = time.time() - t0
    out_tok = sum(len(r.outputs[0].token_ids) for r in res)
    in_tok = sum(len(r.prompt_token_ids) for r in res)
    for it, r in zip(todo, res):
        it["raw"] = r.outputs[0].text
        it["parsed"] = _parse_json(it["raw"])
    for it in items:
        if it["rule_text"] is not None:
            it["parsed"] = {"text": it["rule_text"]}
    with open(out / "generated.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    (out / "generate_stats.json").write_text(json.dumps({"model": GENERATOR, "seconds": dt, "prompt_tokens": in_tok,
                                                         "output_tokens": out_tok, "output_tokens_per_s": out_tok / dt,
                                                         "requests": len(todo)}, indent=2))
    print("generated", len(todo), f"{dt:.0f}s", f"{out_tok / dt:.0f} out tok/s")


def candidate_decision(it: dict) -> dict | None:
    """The decision the verifier must answer, built from the generated item (intended label kept aside)."""
    src, p, k = it["source"], it.get("parsed") or {}, it["kind"]
    cands = [dict(c) for c in src["candidates"]]
    if k == "hard_negative":
        if not p.get("name") or not p.get("description"):
            return None
        gold = it["extra"]["gold"]
        new = {"id": re.sub(r"\W+", "_", str(p["name"]).strip())[:60], "label": str(p["name"])[:60],
               "description": str(p["description"])[:400], "language": "en"}
        if new["id"] in {c["id"] for c in cands}:
            return None
        gold_c = next(c for c in cands if c["id"] == gold)
        return {"text": src["context"]["text"], "candidates": [gold_c, new], "intended": [gold], "new_tool": new}
    text = p.get("text")
    if not text or not isinstance(text, str):
        return None
    if k in ("minimal_pair", "counterfactual"):
        intended = [it["extra"]["target"]]
    elif k == "near_ood":
        intended = []
    elif k == "ambiguity":
        intended = sorted([it["extra"]["a"], it["extra"]["b"]])
    else:
        intended = [it["extra"]["gold"]]
    return {"text": text.strip(), "candidates": cands, "intended": intended}


def verifier_prompt(dec: dict, kind: str) -> str:
    opts = "\n".join(cand_line(c) for c in dec["candidates"])
    multi = kind == "ambiguity"
    ask = ("List EVERY option that is a reasonable interpretation of the message" if multi
           else "Choose the ONE option that the message asks for, or NONE if no option applies")
    return ("You are a careful annotator. " + ask + ". Then report the language of the message as a BCP-47 tag "
            "(e.g. en, pt-BR, pt-PT, es-ES, es-419, fr-FR, fr-CA) and rate how natural it sounds for a real user "
            "(1 = broken/unnatural, 5 = perfectly natural).\n"
            "Return ONLY JSON: {\"answer\": " + ("[\"option_id\", ...] or []" if multi else "\"option_id\" or \"NONE\"")
            + ", \"language\": \"tag\", \"naturalness\": 1-5}\n\n"
            f"Message: \"{dec['text']}\"\n\nOptions:\n{opts}")


def verify(args) -> None:
    from vllm import SamplingParams

    out = Path(args.out)
    items = jl(out / "generated.jsonl")
    todo = []
    for it in items:
        it["decision"] = candidate_decision(it)
        if it["decision"]:
            todo.append(it)
    llm = _llm(VERIFIER["model"])
    sp = SamplingParams(temperature=0.0, max_tokens=80)
    t0 = time.time()
    res = llm.chat([[{"role": "user", "content": verifier_prompt(it["decision"], it["kind"])}] for it in todo], sp,
                   use_tqdm=False)
    dt = time.time() - t0
    for it, r in zip(todo, res):
        it["verifier_raw"] = r.outputs[0].text
        it["verifier"] = _parse_json(r.outputs[0].text)
    with open(out / "verified.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    (out / "verify_stats.json").write_text(json.dumps({"model": VERIFIER, "seconds": dt, "requests": len(todo),
                                                       "prompt_tokens": sum(len(r.prompt_token_ids) for r in res),
                                                       "output_tokens": sum(len(r.outputs[0].token_ids) for r in res)}, indent=2))
    print("verified", len(todo), f"{dt:.0f}s")


# ------------------------------------------------------------------------------------------------ finalize
def finalize(args) -> None:
    from krun_one.v1 import leakage, schema

    out = Path(args.out)
    items = jl(out / "verified.jsonl")
    prot = json.loads((ROOT / "results/v1/mlm_protected_texts.json").read_text())["texts"]
    prot_recs = [{"id": f"p{i}", "context": {"text": t}} for i, t in enumerate(prot)]
    near_idx = leakage.NearDuplicateIndex(threshold=0.9)
    # one batched leakage pass over every generated text (the protected index is built once)
    cands = [{"id": it["pid"], "context": {"text": it["decision"]["text"]}} for it in items if it.get("decision")]
    rep = leakage.check_leakage(cands, prot_recs, near_threshold=0.9)
    leaked = {v["id"] for k in ("exact", "near") for v in rep.violations.get(k, [])}
    seen_norm, accepted, rejected = set(), [], []
    reasons = Counter()
    for it in items:
        k, dec, v = it["kind"], it.get("decision"), it.get("verifier") or {}
        why = None
        target_lang = it["extra"].get("target_lang") or it["source"]["context"]["language"]
        if dec is None:
            why = "generation_unparseable"
        elif not v:
            why = "verifier_unparseable"
        else:
            ans = v.get("answer")
            got = sorted(ans) if isinstance(ans, list) else ([] if str(ans).upper() == "NONE" else [str(ans)])
            if got != sorted(dec["intended"]):
                why = "semantic_label_disagreement"
            elif k == "minimal_pair" and not (0.55 <= difflib.SequenceMatcher(None, it["source"]["context"]["text"].lower(),
                                                                                dec["text"].lower()).ratio() < 0.98):
                why = "bad_minimal_pair_edit_distance"
            elif int(v.get("naturalness") or 0) < 3 and k != "typo":
                why = "unnatural_text"
            elif k != "code_switching" and str(v.get("language", "")).split("-")[0] != target_lang.split("-")[0]:
                why = "language_error"
            elif k == "regional" and str(v.get("language", "")) != target_lang:
                why = "regional_variant_not_recognized"
        if why is None:
            norm = leakage.normalize(dec["text"])
            if norm in seen_norm or norm == leakage.normalize(it["source"]["context"]["text"]) and k not in ("hard_negative",):
                why = "duplicate"
            elif k != "hard_negative" and near_idx.query(dec["text"]):
                why = "near_duplicate"
        if why is None and it["pid"] in leaked:
            why = "leakage_violation"
        rec = None
        if why is None:
            rec = to_record(it, dec, v)
            errs = schema.validate(rec)
            if errs:
                why = "schema_violation"
                it["schema_errors"] = errs
        if why:
            reasons[why] += 1
            it["rejection_reason"] = why
            rejected.append(it)
        else:
            seen_norm.add(leakage.normalize(dec["text"]))
            if k != "hard_negative":
                near_idx.add(it["pid"], dec["text"])
            accepted.append(rec)
    gs = json.loads((out / "generate_stats.json").read_text())
    vs = json.loads((out / "verify_stats.json").read_text())
    gpu_s = gs["seconds"] + vs["seconds"] + args.overhead_s
    cost = gpu_s / 3600 * args.gpu_price
    by_kind = defaultdict(lambda: Counter())
    for it in rejected:
        by_kind[it["kind"]]["rejected"] += 1
    for r in accepted:
        by_kind[r["provenance"]["synthetic"]["transformation"]]["accepted"] += 1
    report = {"generated": len(items), "accepted": len(accepted), "rejected": len(rejected),
              "acceptance_rate": len(accepted) / max(1, len(items)), "rejection_reasons": dict(reasons),
              "by_transformation": {k: dict(v) | {"rate": v["accepted"] / max(1, v["accepted"] + v["rejected"])}
                                    for k, v in sorted(by_kind.items())},
              "generator": GENERATOR, "verifier": VERIFIER, "prompt_version": PROMPT_VERSION,
              "gpu": args.gpu_name, "gpu_price_per_hour": args.gpu_price,
              "generator_output_tokens_per_s": gs["output_tokens_per_s"], "generation_seconds": gs["seconds"],
              "verification_seconds": vs["seconds"], "overhead_seconds_included": args.overhead_s,
              "cost_usd": cost, "cost_per_100k_generated": cost / max(1, len(items)) * 1e5,
              "cost_per_100k_accepted": cost / max(1, len(accepted)) * 1e5}
    for name, recs in (("accepted", accepted), ("rejected", rejected)):
        with open(out / f"{name}.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def to_record(it: dict, dec: dict, v: dict) -> dict:
    from krun_one.v1 import schema

    src, k = it["source"], it["kind"]
    lang = v.get("language") if k == "code_switching" else (it["extra"].get("target_lang") or src["context"]["language"])
    rec = json.loads(json.dumps(src))
    rec["id"] = f"kd1pilot/{it['pid']}"
    rec["context"] = {"text": dec["text"], "language": "mul" if k == "code_switching" else lang,
                      "code_switching": k == "code_switching", "register": "slang" if k == "slang" else None,
                      "noise": ["typo"] if k == "typo" else []}
    rec["candidates"] = dec["candidates"]
    rec["target"] = {"labels": dec["intended"] if k != "ambiguity" else [], "distribution":
                     ({l: 0.5 for l in dec["intended"]} if k == "ambiguity" else None)}
    if k == "ambiguity":
        rec["task"] = dict(rec["task"], type="multi")
        rec["target"] = {"labels": dec["intended"], "distribution": None}
    rec["answerable"] = bool(rec["target"]["labels"])
    rec["slices"] = sorted(set(rec.get("slices", [])) - {"permutation"} | {
        "minimal_pair": {"minimal_pair"}, "counterfactual": {"counterfactual"}, "near_ood": {"ood_near"},
        "ambiguity": {"ambiguous"}, "slang": {"slang"}, "regional": {"regional"}, "code_switching": {"code_switching"},
        "typo": {"typo"}, "hard_negative": {"near_duplicate"}}[k])
    rec["minimal_pair"] = ({"set_id": f"mp:{src['id']}", "member": "b", "contrast": f"{it['extra']['gold']}->{it['extra']['target']}"}
                           if k == "minimal_pair" else None)
    rec["candidate_set"] = dict(rec.get("candidate_set") or {}, strategy="hard_v2" if k == "hard_negative" else
                                (rec.get("candidate_set") or {}).get("strategy", "native"), k=len(dec["candidates"]))
    # same split unit as the source (never leaks across splits); new semantic group (different meaning / new text)
    rec["semantic_group_id"] = f"syn:{it['pid']}"
    rec["provenance"] = dict(rec["provenance"], transforms=list(rec["provenance"].get("transforms", [])) + [f"synthetic:{k}"],
                             synthetic={"teacher": GENERATOR["model"] if k != "typo" else "rule:typos-v1",
                                        "prompt_id": f"{PROMPT_VERSION}/{k}", "verifier": VERIFIER["model"],
                                        "verdict": "accepted", "source_example_id": src["id"], "transformation": k,
                                        "language": lang, "seed": it["seed"],
                                        "verifier_output": v, "generated_utc": datetime.now(timezone.utc).isoformat()})
    rec["metadata"] = {"source_text": src["context"]["text"], **({"new_tool": dec["new_tool"]} if k == "hard_negative" else {})}
    return schema.derive_fields(rec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "generate", "verify", "finalize"])
    ap.add_argument("--out", default=str(ROOT / "results/v1/synthetic_pilot"))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument("--gpu-price", type=float, default=0.34)
    ap.add_argument("--gpu-name", default="unknown")
    ap.add_argument("--overhead-s", type=float, default=0.0, help="setup/model-load seconds to include in the cost")
    args = ap.parse_args()
    if args.n > 2000:
        raise SystemExit("pilot is capped at 2,000 items this round")
    {"prepare": prepare, "generate": generate, "verify": verify, "finalize": finalize}[args.stage](args)


if __name__ == "__main__":
    main()
