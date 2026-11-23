"""Krun Decisions v1 — synthetic pilot v2 (2,000 LLM items; prompt kd1-pilot-v2). CPU stages only; the GPU stages
(generate with Qwen3-8B, blind verify with Granite-3.3-8B) run in scripts/v1/kd1_gpu_jobs.py on RunPod.

Changes vs pilot v1 (results/v1/synthetic_pilot/REPORT.md, 42.6% acceptance):
* the generator always sees the FULL candidate list with 3 train examples per option, and must return null when
  the brief is impossible (counted as a rejection: ``generator_abstained``);
* near-OOD asks for a different need of the same domain + a one-line reason, and is checked against the full set;
* minimal pairs: target = one of the 3 hardest neighbours of the gold (TF-IDF over label names + train examples);
  explicit edit budget; edit-ratio + word-change bounds in finalize;
* ambiguity -> ``underspecified`` (the shared object is named, the discriminating action is not) with examples;
* code-switching: rule check (language ID must find the source language AND English segments; instruction-leak
  words rejected); slang: language ID must match the source language; regional rewriting is NOT an LLM task any more;
* typo items are no longer in the pilot (rules origin, gold correct by construction; not LLM synthetic);
* language quota over the pilot: EN 35 / PT 25 / ES 20 / FR 20 (PT = pt-BR BANKING77 + pt-PT MASSIVE).

    uv run python scripts/v1/synthetic_pilot_v2.py prepare --out results/v1/synthetic_pilot_v2
    (RunPod) python kd1_gpu_jobs.py pilot_generate / pilot_verify
    uv run python scripts/v1/synthetic_pilot_v2.py finalize --out results/v1/synthetic_pilot_v2 --gpu-price ... --gpu-name ...
"""

from __future__ import annotations

import argparse
import difflib
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GENERATOR = {"model": "Qwen/Qwen3-8B", "license": "Apache-2.0"}
VERIFIER = {"model": "ibm-granite/granite-3.3-8b-instruct", "license": "Apache-2.0"}
PROMPT_VERSION = "kd1-pilot-v2"
MIX = {"minimal_pair": 500, "near_ood": 400, "hard_negative": 300, "counterfactual": 250, "code_switching": 200,
       "slang": 200, "underspecified": 150}
LANG_QUOTA = {"en": 0.35, "pt": 0.25, "es": 0.20, "fr": 0.20}
LANG_NAME = {"en-US": "English", "pt-BR": "Brazilian Portuguese", "pt-PT": "European Portuguese", "es-ES": "Spanish",
             "es-419": "Latin American Spanish", "fr-FR": "French", "fr-CA": "Canadian French", "es": "Spanish",
             "fr": "French"}
TOOL_LOCALE = {"en": "en-US", "pt": "pt-BR", "es": "es", "fr": "fr"}   # generated text: no regional claim beyond pt-BR
LEAK_WORDS = re.compile(r"\b(english|inglês|ingles|inglés|anglais|portuguese|português|portugues|spanish|espagnol|"
                        r"español|espanhol|french|francês|francés|français|code[- ]?switch)\b", re.I)
NOISY_LABELS = {"general_quirky"}   # MASSIVE catch-all with known label noise (pilot v1 audit): never source/target
DOMAIN = {"banking77": "online banking support", "clinc150": "general virtual assistant", "massive": "voice assistant",
          "snips": "voice assistant", "bitext": "e-commerce customer support"}


def jl(p):
    with open(p) as f:
        return [json.loads(l) for l in f]


def humanize(x: str) -> str:
    return " ".join(re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", x).replace("_", " ").split()).lower()


class LabelSpace:
    """Per-catalog label neighbours: char TF-IDF of humanized names + word TF-IDF of train examples (RRF)."""

    def __init__(self, cat: str, examples: dict[str, list[str]]):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.cat, self.labels = cat, sorted(examples)
        self.examples = examples
        a = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit_transform([humanize(l) for l in self.labels])
        b = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True).fit_transform(
            [" ".join(examples[l][:200]) for l in self.labels])
        import numpy as np

        sa, sb = (a @ a.T).toarray(), (b @ b.T).toarray()
        ra = np.argsort(np.argsort(-sa, 1), 1)
        rb = np.argsort(np.argsort(-sb, 1), 1)
        self.fused = 1 / (10 + ra) + 1 / (10 + rb)
        self.pos = {l: i for i, l in enumerate(self.labels)}

    def neighbours(self, label: str, n: int) -> list[str]:
        import numpy as np

        order = np.argsort(-self.fused[self.pos[label]])
        return [self.labels[j] for j in order if self.labels[j] != label and self.labels[j] not in NOISY_LABELS][:n]


def lang_bucket(lang: str) -> str:
    return lang.split("-")[0]


def option_block(labels, space: LabelSpace, rng, n_ex=3):
    lines = []
    for l in labels:
        ex = rng.sample(space.examples[l], min(n_ex, len(space.examples[l])))
        lines.append(f"- {l}: e.g. " + " | ".join(f'"{e}"' for e in ex))
    return "\n".join(lines)


def tool_block(tools):
    return "\n".join(f"- {n}: {d}" for n, d in tools)


HEAD = ("You write test messages for a multilingual intent/function routing classifier. Answer with ONLY a JSON object, "
        "no explanations outside it.\n\n")


def prompt_for(kind: str, it: dict, rng) -> str:
    src, lang = it["source_item"], it["lang"]
    lname = LANG_NAME[lang]
    if kind in ("hard_negative", "counterfactual") or (kind == "near_ood" and src.get("tools")):
        tools = src["tools"]
        if kind == "hard_negative":
            gd = dict(tools)[it["gold"]]
            return ("You write test data for a function-routing classifier. Answer with ONLY a JSON object "
                    '{"name": "snake_case_or_camelCase_name", "description": "one sentence"}.\n\n'
                    f'User request: "{src["text"]}"\nCorrect function: {it["gold"]}: {gd}\n\n'
                    "Invent ONE new function from the same API domain whose name and description look similar to the "
                    "correct one, but which does something DIFFERENT, so it would be the WRONG function for this exact "
                    "request. Do not invent a synonym, alias or more general version of the correct function.")
        if kind == "counterfactual":
            return (HEAD + f"Available functions:\n{tool_block(tools)}\n\nOriginal request: \"{src['text']}\" "
                    f"(correct function: {it['gold']}).\n"
                    f"Rewrite the request in {lname} so that the ONLY correct function becomes {it['target']}: {dict(tools)[it['target']]}\n"
                    "Keep the topic and reuse entities where they fit the new function; give the values it needs. "
                    "It must be a natural user request that no other listed function fulfils. "
                    'Return {"text": "..."} or {"text": null} if impossible.')
        return (HEAD + f"Available functions:\n{tool_block(tools)}\n\n"
                f"Write a realistic user request in {lname} from the SAME domain as these functions that NONE of them can "
                "fulfil (a different need, not a paraphrase of any function; not answerable by combining them). "
                'Return {"text": "...", "why_none": "one short reason"}.')
    space = it["space"]
    opts = option_block(it["candidates"], space, rng)
    dom = DOMAIN.get(src["catalog"], "assistant")
    if kind == "minimal_pair":
        return (HEAD + f"Domain: {dom}. Options (intent labels with example messages):\n{opts}\n\n"
                f"Original message ({lname}): \"{src['text']}\" -> intent {it['gold']}.\n"
                f"Make the SMALLEST edit (change, add or remove at most 4 words) so that the new message clearly has "
                f"intent {it['target']} and NO other option applies. Keep the same language ({lname}), the same "
                "entities and style. The result must be something a real user would write. "
                'Return {"text": "..."} or {"text": null} if no small edit can do it.')
    if kind == "near_ood":
        return (HEAD + f"Domain: {dom}. Options (intent labels with example messages):\n{opts}\n\n"
                f"Write one realistic user message in {lname}, in the same domain, whose need is NOT covered by ANY "
                "of these options (a different request; not small talk; not a paraphrase of any option). "
                'Return {"text": "...", "why_none": "one short reason in English"}.')
    if kind == "underspecified":
        a, b = it["pair"]
        return (HEAD + f"Domain: {dom}. Options (intent labels with example messages):\n{opts}\n\n"
                f"Write one short, realistic user message in {lname} that is UNDERSPECIFIED: it names the thing the user "
                f"is dealing with but not what they want, so it could equally mean {a} or {b}, and no other option. "
                'Return {"text": "..."} or {"text": null} if impossible.')
    if kind == "slang":
        return (HEAD + f"Rewrite this {lname} message in very informal chat style of {lname} (colloquial words, "
                f"abbreviations), keeping exactly the same meaning and request: \"{src['text']}\"\n"
                'Return {"text": "..."}.')
    if kind == "code_switching":
        return (HEAD + f"Rewrite this {lname} message the way a bilingual user types: mostly {lname}, with 1 to 3 "
                f"ENGLISH words or short phrases mixed in naturally, same meaning and request: \"{src['text']}\"\n"
                'Do not mention languages. Return {"text": "..."}.')
    raise ValueError(kind)


# ------------------------------------------------------------------------------------------------ prepare
def prepare(args) -> None:
    rng = random.Random(args.seed)
    ctx = [c for c in jl(ROOT / "data/kd1/contexts.jsonl") if c["split"] == "train"]
    ex = defaultdict(lambda: defaultdict(list))
    for c in ctx:
        if c["label"] and c["lang"] == "en-US":
            ex[c["catalog"]][c["label"]].append(c["text"])
    spaces = {cat: LabelSpace(cat, dict(v)) for cat, v in ex.items() if len(v) >= 8}
    intents = defaultdict(list)                        # lang bucket -> items
    for c in ctx:
        if c["label"] and c["catalog"] in spaces and c["label"] not in NOISY_LABELS:
            intents[lang_bucket(c["lang"])].append(c)
    tools = [c for c in ctx if c["source"] == "toolace" and c.get("gold_tool") and len(c["tools"]) >= 2]
    items = []

    def add(kind, src, lang, **kw):
        it = {"pid": f"{args.tag}{kind}/{len(items)}", "kind": kind, "lang": lang, "source_item": src, **kw}
        it["prompt"] = prompt_for(kind, it, rng)
        # what the blind verifier sees for each option (never the intended label): tools -> official description,
        # intents -> 2 train examples (label ids alone are ambiguous, e.g. BANKING77)
        if src.get("tools"):
            it["verifier_options"] = {n: d for n, d in src["tools"]}
        else:
            sp = it["space"]
            it["verifier_options"] = {l: "e.g. " + " | ".join(f'"{e}"' for e in rng.sample(sp.examples[l], min(2, len(sp.examples[l]))))
                                      for l in it["candidates"]}
        it.pop("space", None)
        items.append(it)

    def langs(n, allow_en=True):
        q = {k: v for k, v in LANG_QUOTA.items() if allow_en or k != "en"}
        tot = sum(q.values())
        out = []
        for k, v in q.items():
            out += [k] * int(round(n * v / tot))
        rng.shuffle(out)
        return out[:n] + ["en"] * max(0, n - len(out))

    def intent_set(src, k=8, drop_gold=False):
        sp = spaces[src["catalog"]]
        neigh = sp.neighbours(src["label"], k)
        cands = neigh[: k - 1] + ([] if drop_gold else [src["label"]])
        if drop_gold:
            cands = neigh[:k]
        rng.shuffle(cands)
        return sp, cands

    for kind, n in MIX.items():
        n = int(round(n * args.n / 2000))                 # pilot = 2,000; a scale run keeps the pilot's mix
        if kind in ("hard_negative", "counterfactual"):
            for lb in (["en"] * n if kind == "hard_negative" else langs(n)):   # new tool texts are EN schema
                src = rng.choice(tools)
                others = [t for t, _ in src["tools"] if t != src["gold_tool"]]
                kw = {"gold": src["gold_tool"], "candidates": [t for t, _ in src["tools"]]}
                if kind == "counterfactual":
                    kw["target"] = rng.choice(others)
                add(kind, src, TOOL_LOCALE[lb], **kw)
            continue
        n_tool = int(n * 0.3) if kind == "near_ood" else 0
        for lb in langs(n_tool):
            src = rng.choice(tools)
            add(kind, src, TOOL_LOCALE[lb], gold=None, candidates=[t for t, _ in src["tools"]])
        for lb in langs(n - n_tool, allow_en=kind != "code_switching"):
            src = rng.choice(intents[lb])
            if kind == "near_ood":
                sp, cands = intent_set(src, 8, drop_gold=True)
                add(kind, src, src["lang"], space=sp, gold=None, candidates=cands)
            elif kind == "minimal_pair":
                sp, cands = intent_set(src, 8)
                tgt = rng.choice(sp.neighbours(src["label"], 3))
                if tgt not in cands:
                    cands[cands.index(next(c for c in cands if c != src["label"]))] = tgt
                add(kind, src, src["lang"], space=sp, gold=src["label"], target=tgt, candidates=cands)
            elif kind == "underspecified":
                sp, cands = intent_set(src, 8)
                b = sp.neighbours(src["label"], 1)[0]
                add(kind, src, src["lang"], space=sp, gold=None, pair=sorted([src["label"], b]), candidates=cands)
            else:  # slang, code_switching
                sp, cands = intent_set(src, 8)
                add(kind, src, src["lang"], space=sp, gold=src["label"], candidates=cands)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "prompts.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(json.dumps({"prompts": len(items), "by_kind": dict(Counter(i["kind"] for i in items)),
                      "by_lang": dict(Counter(lang_bucket(i["lang"]) for i in items))}, indent=1))


# ------------------------------------------------------------------------------------------------ finalize
def _lid():
    from lingua import Language, LanguageDetectorBuilder

    return LanguageDetectorBuilder.from_languages(Language.ENGLISH, Language.PORTUGUESE, Language.SPANISH,
                                                  Language.FRENCH).build()


LID_CODE = {"ENGLISH": "en", "PORTUGUESE": "pt", "SPANISH": "es", "FRENCH": "fr"}


def lang_of(lid, text: str) -> str | None:
    r = lid.detect_language_of(text)
    return LID_CODE.get(r.name) if r else None


def code_switch_ok(lid, text: str, base: str) -> bool:
    segs = lid.detect_multiple_languages_of(text)
    got = {LID_CODE.get(s.language.name) for s in segs}
    if base in got and "en" in got:
        return True
    # short English insertions are often below the segment detector's resolution: accept a base-language text that
    # contains >= 1 common English word not shared with the base language
    words = set(re.findall(r"[a-zA-Z']+", text.lower()))
    common_en = {"please", "refund", "ok", "okay", "cool", "weekend", "meeting", "deadline", "cash", "card", "email",
                 "update", "check", "cancel", "delivery", "top", "up", "account", "transfer", "payment", "online",
                 "app", "playlist", "alarm", "remind", "reminder", "music", "play", "song", "weather", "news", "booking",
                 "order", "tracking", "shopping", "list", "call", "chat", "link", "password", "login", "feedback", "fee",
                 "fees", "delay", "asap", "thanks", "help", "the", "my", "for", "and", "with", "now", "set", "turn", "off",
                 "on", "lights", "light", "timer", "meeting", "recipe", "podcast", "ticket", "train", "flight"}
    return lang_of(lid, text) == base and bool(words & common_en)


def finalize(args) -> None:
    from krun_one.v1 import leakage, schema

    out = Path(args.out)
    items = jl(out / "verified.jsonl")
    lid = _lid()
    prot = json.loads((ROOT / "results/v1/mlm_protected_texts.json").read_text())["texts"]
    kb = [json.loads(l)["context"]["text"] for s in ("dev", "val_fit", "val_eval", "final_protected")
          for l in open(ROOT / f"data/krunbench_v1/{s}.jsonl")]
    val_ml = [c["text"] for c in jl(ROOT / "data/kd1/contexts.jsonl") if c["split"] == "val_ml"]
    prot_recs = [{"id": f"p{i}", "context": {"text": t}} for i, t in enumerate(prot + kb + val_ml)]
    dec_recs = [{"id": it["pid"], "context": {"text": it["decision"]["text"]}} for it in items if it.get("decision")]
    rep = leakage.check_leakage(dec_recs, prot_recs, near_threshold=0.9)
    leaked = {v["id"] for k in ("exact", "near") for v in rep.violations.get(k, [])}
    src_norm = {leakage.normalize(c["text"]) for c in jl(ROOT / "data/kd1/contexts.jsonl") if c["split"] == "train"}
    near_idx = leakage.NearDuplicateIndex(threshold=0.9)
    seen, accepted, rejected, reasons = set(), [], [], Counter()
    for it in items:
        k, dec, v = it["kind"], it.get("decision"), it.get("verifier") or {}
        base = lang_bucket(it["lang"])
        why = None
        if it.get("abstained"):
            why = "generator_abstained"
        elif dec is None:
            why = "generation_unparseable"
        elif not v:
            why = "verifier_unparseable"
        else:
            ans = v.get("answer")
            got = sorted(ans) if isinstance(ans, list) else ([] if str(ans).upper() in ("NONE", "NULL", "") else [str(ans)])
            txt = dec["text"]
            if got != sorted(dec["intended"]):
                why = "semantic_label_disagreement"
            elif k == "minimal_pair":
                a, b = it["source_item"]["text"].lower(), txt.lower()
                ratio = difflib.SequenceMatcher(None, a, b).ratio()
                wa, wb = a.split(), b.split()
                changed = sum(1 for op in difflib.SequenceMatcher(None, wa, wb).get_opcodes() if op[0] != "equal"
                              for _ in range(max(op[2] - op[1], op[4] - op[3])))
                if not (0.5 <= ratio < 0.99) or changed > 6:
                    why = "bad_minimal_pair_edit_distance"
            if why is None and k != "hard_negative":
                if int(v.get("naturalness") or 0) < 3:
                    why = "unnatural_phrasing"
                elif LEAK_WORDS.search(txt) and k == "code_switching":
                    why = "instruction_leakage"
                elif k == "code_switching":
                    if not code_switch_ok(lid, txt, base):
                        why = "code_switching_not_detected"
                elif lang_of(lid, txt) not in (base, None) and len(txt.split()) >= 3:
                    why = "language_error"
        if why is None:
            norm = leakage.normalize(dec["text"])
            if k != "hard_negative" and (norm in seen or norm in src_norm):
                why = "duplicate"
            elif k != "hard_negative" and near_idx.query(dec["text"]):
                why = "duplicate"
            elif it["pid"] in leaked and k != "hard_negative":
                why = "leakage"
        rec = None
        if why is None:
            rec = to_record(it, dec, v)
            errs = schema.validate(rec)
            if errs:
                why = "schema_error"
                it["schema_errors"] = errs
        if why:
            reasons[why] += 1
            it["rejection_reason"] = why
            rejected.append(it)
        else:
            seen.add(leakage.normalize(dec["text"]))
            if k != "hard_negative":
                near_idx.add(it["pid"], dec["text"])
            accepted.append(rec)
    by_kind = defaultdict(Counter)
    by_lang = defaultdict(Counter)
    for it in rejected:
        by_kind[it["kind"]]["rejected"] += 1
        by_lang[lang_bucket(it["lang"])]["rejected"] += 1
    for r in accepted:
        by_kind[r["provenance"]["synthetic"]["transformation"]]["accepted"] += 1
        by_lang[lang_bucket(r["metadata"]["pilot_lang"])]["accepted"] += 1
    rate = lambda c: round(c["accepted"] / max(1, c["accepted"] + c["rejected"]), 4)  # noqa: E731
    timing = json.loads((out / "timing.json").read_text()) if (out / "timing.json").exists() else {}
    gpu_s = timing.get("generate_seconds", 0) + timing.get("verify_seconds", 0)
    acc = len(accepted) / max(1, len(items))
    report = {"prompt_version": PROMPT_VERSION, "generated": len(items), "accepted": len(accepted), "rejected": len(rejected),
              "acceptance_rate": round(acc, 4), "rejection_reasons": dict(reasons.most_common()),
              "by_transformation": {k: dict(v) | {"rate": rate(v)} for k, v in sorted(by_kind.items())},
              "by_language": {k: dict(v) | {"rate": rate(v)} for k, v in sorted(by_lang.items())},
              "generator": GENERATOR, "verifier": VERIFIER, "gpu": args.gpu_name, "gpu_price_per_hour": args.gpu_price,
              "timing": timing, "compute_cost_usd": round(gpu_s / 3600 * args.gpu_price, 4),
              "gate": {"target": 0.65, "no_scale_below": 0.55,
                       "verdict": "PASS" if acc >= 0.65 else ("MANUAL_REVIEW" if acc >= 0.55 else "FAIL")}}
    for name, recs in (("accepted", accepted), ("rejected", rejected)):
        with open(out / f"{name}.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


SLICE_OF = {"minimal_pair": ["minimal_pair"], "counterfactual": ["counterfactual"], "near_ood": ["ood_near"],
            "underspecified": ["ambiguous"], "slang": ["slang"], "code_switching": ["code_switching"],
            "hard_negative": ["near_duplicate"]}


def to_record(it: dict, dec: dict, v: dict) -> dict:
    from krun_one.v1 import schema

    src, k = it["source_item"], it["kind"]
    tools = dict(src["tools"]) if src.get("tools") else None
    cands = []
    for cid in dec["candidate_ids"]:
        if k == "hard_negative" and cid == dec["new_tool"]["id"]:
            cands.append({"id": cid, "label": cid, "description": dec["new_tool"]["description"], "language": "en"})
        else:
            cands.append({"id": cid, "label": cid, "description": tools.get(cid) if tools else None, "language": "en"})
    labels = list(dec["intended"])
    qtype = "multi" if k == "underspecified" else "choice"
    family = "tool" if tools else "intent"
    ins = ("Select every option that could be what the user wants." if qtype == "multi" else
           "Choose the function that should be called to fulfil the user request." if family == "tool" else
           "Choose the option that best describes the intent of the user message.")
    lang = "mul" if k == "code_switching" else it["lang"]
    rec = {"schema_version": schema.SCHEMA_VERSION, "id": f"kd1syn/{PROMPT_VERSION}/{it['pid']}", "split": "train",
           "task": {"type": qtype, "instructions": ins, "instructions_language": "en", "family": family},
           "context": {"text": dec["text"], "language": lang, "code_switching": k == "code_switching",
                       "register": "informal" if k == "slang" else None, "noise": []},
           "candidates": cands, "target": {"labels": labels, "distribution": None}, "answerable": bool(labels),
           "semantic_group_id": f"syn:{it['pid']}", "split_group_id": src["split_group"],
           "candidate_set": {"strategy": "hard_v2", "k": len(cands)}, "slices": SLICE_OF[k],
           "minimal_pair": ({"set_id": f"mp:{src['source']}:{src['oid']}", "member": "b",
                             "contrast": f"{it['gold']}->{it['target']}", "relation": "label_flip_minimal_edit",
                             "anchor_text": src["text"]} if k == "minimal_pair" else None),
           "provenance": {"source_dataset": src["source"], "revision": "see configs/v1/sources.yaml",
                          "license": "derived; see source", "original_split": "train", "original_id": src["oid"],
                          "transforms": [f"synthetic:{k}"],
                          "synthetic": {"teacher": GENERATOR["model"], "prompt_id": f"{PROMPT_VERSION}/{k}",
                                        "verifier": VERIFIER["model"], "verdict": "accepted",
                                        "source_example_id": f"{src['source']}:{src['oid']}", "transformation": k,
                                        "verifier_output": v, "generated_utc": datetime.now(timezone.utc).isoformat()}},
           "metadata": {"source_text": src["text"], "pilot_lang": it["lang"],
                        **({"new_tool": dec["new_tool"]} if k == "hard_negative" else {}),
                        **({"why_none": dec.get("why_none")} if dec.get("why_none") else {})}}
    return schema.derive_fields(rec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "finalize"])
    ap.add_argument("--out", default=str(ROOT / "results/v1/synthetic_pilot_v2"))
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--n", type=int, default=2000, help="2,000 = pilot; larger ONLY after the pilot gate passed")
    ap.add_argument("--tag", default="", help="pid prefix for a scale run (e.g. s1/)")
    ap.add_argument("--gpu-price", type=float, default=0.34)
    ap.add_argument("--gpu-name", default="unknown")
    args = ap.parse_args()
    if args.stage == "prepare" and args.n > 2000:
        rep = json.loads((ROOT / "results/v1/synthetic_pilot_v2/report.json").read_text())
        if rep["gate"]["verdict"] != "PASS":
            raise SystemExit(f"synthetic scale refused: pilot v2 gate is {rep['gate']['verdict']}")
    {"prepare": prepare, "finalize": finalize}[args.stage](args)


if __name__ == "__main__":
    main()
