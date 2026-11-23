"""Krun Decisions v1 — stage 2: build the ~1M training decisions + the multilingual validation set (val_ml).

    uv run python scripts/v1/build_decisions_v1.py [--scale 1.0] [--gpu-out data/kd1/gpu_out] [--synthetic <accepted.jsonl> ...]

Inputs: data/kd1/contexts.jsonl (stage 1), GPU outputs (translations + judge, intent descriptions, translated
descriptions), data/primitives_v2 (noul/score, xLAM-derived rows dropped), synthetic accepted files, KrunBench v1.
Outputs (data/kd1/): train.jsonl, val_ml.jsonl, build_manifest.json (+ results/v1/kd1_stats.{json,md} via kd1_stats.py).

Quality > quantity: every quota is a ceiling. A category that cannot be filled from its capped pools is reported SHORT
(caps = max decisions per context per generator). LEAKAGE must be 0 (KrunBench v1 protected manifest incl. unseen
tools, KrunBench v1 validation, val_ml); violators are dropped and the check is re-run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from krun_one.v1 import leakage, schema
from krun_one.v1.corpus import sources as S
from krun_one.v1.corpus.decisions import CONNECTORS, INSTR, Builder, pick_k, prim, stage_of
from krun_one.v1.corpus.mining import EmbeddingCache, LabelSpace, ToolPool, humanize, rng_for
from krun_one.v1.tools import TrainToolIndex

ROOT = Path(__file__).resolve().parents[2]
SEED = 20260927
MT_TAG = "mt:Qwen/Qwen3-8B:kd1-mt-v1+judge:ibm-granite/granite-3.3-8b-instruct"
LANGS = ("en", "pt", "es", "fr")
# per-category language targets (non-EN boosted where native/MT supply exists, because noul/score are EN-only)
QUOTA = {
    "choice_intent": {"en": 70_000, "pt": 70_000, "es": 55_000, "fr": 55_000},
    "tool_toolace": {"en": 65_000, "pt": 30_000, "es": 25_000, "fr": 25_000},
    "tool_multiwoz": {"en": 25_000},
    "tool_intent_fn": {"en": 25_000, "pt": 45_000, "es": 30_000, "fr": 30_000},
    "noul_primitives": {"en": 65_000},
    "noul_xl": {"pt": 12_000, "es": 9_000, "fr": 9_000},
    "noul_multiwoz": {"en": 5_000},
    "score_primitives": {"en": 60_000},
    "score_civil": {"en": 20_000},
    "multi_civil": {"en": 18_000},
    "multi_mhs": {"en": 10_000},
    "multi_concat": {"en": 7_000, "pt": 7_000, "es": 7_000, "fr": 7_000},
    "multi_partial": {"en": 1_500, "pt": 1_500, "es": 1_500, "fr": 1_500},
    "multi_empty": {"en": 1_500, "pt": 1_500, "es": 1_500, "fr": 1_500},
    "ood_intent_gold_removed": {"en": 20_000, "pt": 20_000, "es": 15_000, "fr": 15_000},
    "ood_tool_gold_removed": {"en": 15_000, "pt": 5_000, "es": 5_000, "fr": 5_000},
    "ood_multiwoz_gold_removed": {"en": 8_000},
    "ood_intent_fn_gold_removed": {"en": 5_000, "pt": 6_000, "es": 4_500, "fr": 4_500},
    "ood_native_irrelevance": {"en": 4_000, "pt": 1_500, "es": 1_500, "fr": 1_500},
    "ood_cross_domain": {"en": 12_000, "pt": 9_000, "es": 7_000, "fr": 7_000},
    "ood_cross_task": {"en": 8_000, "pt": 5_000, "es": 3_500, "fr": 3_500},
    "minimal_pairs_mined": {"en": 8_000, "pt": 6_000, "es": 4_000, "fr": 4_000},
}
CAPS = {"choice_intent": 3, "tool_toolace": 14, "tool_multiwoz": 2, "tool_intent_fn": 2, "noul_xl": 2, "multi_concat": 2,
        "ood_intent_gold_removed": 2, "ood_tool_gold_removed": 5, "ood_multiwoz_gold_removed": 1,
        "ood_intent_fn_gold_removed": 1, "ood_native_irrelevance": 8, "ood_cross_domain": 1, "ood_cross_task": 1}
TYPO_SHARE = 0.06
TRANSLATED_SCHEMA_SHARE = 0.2      # of non-EN intent-as-function / tool decisions (when a translation exists)
CROSS_DOMAIN = {"banking77": ["massive", "snips", "multiwoz"], "bitext": ["massive", "snips"],
                "massive": ["banking77", "bitext"], "snips": ["banking77", "bitext"]}


def jl(p):
    with open(p) as f:
        for l in f:
            yield json.loads(l)


def h01(*parts) -> float:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16) / 16**12


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ------------------------------------------------------------------------------------------------ GPU outputs
def load_gpu(gpu_out: Path):
    parts = lambda job: [r for f in sorted(gpu_out.glob(f"{job}.part-*.jsonl")) for r in jl(f)]  # noqa: E731
    judge = {r["key"]: (r.get("verdict") or {}) for r in parts("judge")}
    from lingua import Language, LanguageDetectorBuilder

    lid = LanguageDetectorBuilder.from_languages(Language.ENGLISH, Language.PORTUGUESE, Language.SPANISH,
                                                 Language.FRENCH).build()
    code = {"ENGLISH": "en", "PORTUGUESE": "pt", "SPANISH": "es", "FRENCH": "fr"}
    stats = Counter()

    def ok(r) -> str | None:
        out, src = (r.get("output") or "").strip(), r["text"]
        v = judge.get(r["key"])
        if not out:
            return "empty"
        if v is None:
            return "not_judged"
        try:
            a, fl = int(v.get("adequacy") or 0), int(v.get("fluency") or 0)
        except (TypeError, ValueError):
            return "judge_unparseable"
        if a < 4 or fl < 3:
            return "judge_rejected"
        if not 0.4 <= len(out) / max(1, len(src)) <= 2.6:
            return "length_ratio"
        if set(re.findall(r"\{\{[^}]+\}\}", src)) != set(re.findall(r"\{\{[^}]+\}\}", out)):
            return "placeholder_changed"
        if set(re.findall(r"\d+", src)) - set(re.findall(r"\d+", out)):
            return "number_changed"
        if len(out.split()) >= 3:
            l = lid.detect_language_of(out)
            if l is None or code.get(l.name) != prim(r["target_locale"]):
                return "language_id"
        return None

    tr = {}
    for r in parts("translate"):
        why = ok(r)
        stats[f"translate/{r['split']}/{'accepted' if why is None else why}"] += 1
        if why is None:
            tr[r["key"]] = r["output"].strip()
    desc_en = {}
    for r in parts("describe"):
        if r.get("output"):
            desc_en[(r["catalog"], r["label"])] = r["output"].strip()
    import yaml

    overrides = yaml.safe_load((ROOT / "configs/v1/intent_desc_overrides.yaml").read_text())
    overridden = set()
    for cat, labs in overrides.items():
        for lab, d in labs.items():
            desc_en[(cat, lab)] = d
            overridden.add((cat, lab))
    stats["intent_desc_overridden"] = len(overridden)
    desc_tr = {}
    for r in parts("translate_desc"):
        why = ok(r) if r["key"] in judge else "not_judged"
        stats[f"translate_desc/{'accepted' if why is None else why}"] += 1
        if why is None:
            if r["key"].startswith("intent::"):
                cat, lab = r["key"][len("intent::"):].split(">")[0].split("::")
                if (cat, lab) not in overridden:          # translation of a text the review replaced: dropped
                    desc_tr[(cat, lab, r["target_locale"])] = r["output"].strip()
            else:
                desc_tr[(r["name"], r["text"], r["target_locale"])] = r["output"].strip()
    return tr, desc_en, desc_tr, stats


# ------------------------------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data/kd1"))
    ap.add_argument("--gpu-out", default=str(ROOT / "data/kd1/gpu_out"))
    ap.add_argument("--synthetic", nargs="*", default=[str(ROOT / "results/v1/synthetic_pilot_v2/accepted.jsonl")])
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every quota (smoke: 0.01)")
    args = ap.parse_args()
    out = Path(args.out)
    rng_master = random.Random(SEED)
    sp_ = load_module("sp", ROOT / "scripts/v1/synthetic_pilot.py")        # rule_typos
    ppd = load_module("ppd", ROOT / "scripts/prepare_primitives_data.py")
    manifest: dict = {"seed": SEED, "scale": args.scale, "quota": QUOTA, "caps": CAPS}

    # ---------------------------------------------------------------- contexts (+ MT)
    ctx = list(jl(out / "contexts.jsonl"))
    src_by_key = {f"{c['source']}:{c['oid']}": c for c in ctx}
    tr, desc_en, desc_tr, mt_stats = load_gpu(Path(args.gpu_out)) if Path(args.gpu_out).exists() else ({}, {}, {}, Counter())
    manifest["mt_filter"] = dict(sorted(mt_stats.items()))
    mw_schema = S.multiwoz_schema()
    for svc, sv in mw_schema.items():
        for name, d in sv["intents"].items():
            desc_en[("multiwoz", name)] = d
    mt_ctx, val_mt = [], []
    for key, text in tr.items():
        base, loc = key.rsplit(">", 1)
        if base.startswith("kbv1:"):
            val_mt.append((base[len("kbv1:"):], loc, text))
            continue
        c = src_by_key.get(base)
        if c is None or c["split"] != "train":
            continue
        m = dict(c, text=text, lang=loc, oid=f"{c['oid']}>{loc}", origin="mt", mining_text=c["text"],
                 transforms=[MT_TAG])
        mt_ctx.append(m)
    train_ctx = [dict(c, origin="real") for c in ctx if c["split"] == "train"] + mt_ctx
    # protected tools (KrunBench v1 unseen tools): never gold, never candidate
    man = json.loads((ROOT / "data/krunbench_v1/protected_manifest.json").read_text())
    pidx, pnames = TrainToolIndex(), set()
    for t in man["unseen_tools"]:
        pidx.add(t["name"], t["description"])
        pnames.add(leakage.normalize(t["name"]))

    def protected(name, desc):
        return leakage.normalize(name) in pnames or pidx.status(name, desc) != "unseen"

    dropped_tools = Counter()
    keep = []
    for c in train_ctx:
        if c.get("tools"):
            tl = [tuple(t) for t in c["tools"]]
            bad = {t for t in tl if protected(*t)}
            if bad:
                if c.get("gold_tool") in {t[0] for t in bad} or set(c.get("calls") or []) & {t[0] for t in bad}:
                    dropped_tools["context_gold_protected"] += 1
                    continue
                dropped_tools["native_tool_removed"] += len(bad)
                c = dict(c, tools=[t for t in tl if t not in bad])
                if len(c["tools"]) < 1:
                    continue
        keep.append(c)
    train_ctx = keep
    manifest["protected_tool_filter"] = dict(dropped_tools)

    # ---------------------------------------------------------------- embeddings, label spaces, tool pool
    emb = EmbeddingCache(out / "emb_e5.npz")
    ex = defaultdict(lambda: defaultdict(list))
    for c in train_ctx:
        if c["label"] and c["lang"] == "en-US" and c.get("origin") == "real":
            ex[c["catalog"]][c["label"]].append((c["text"], c["split_group"]))
    tool_set = sorted({tuple(t) for c in train_ctx if c.get("tools") for t in c["tools"] if t[1]})
    need = [t for c in train_ctx if c.get("origin") == "real" for t in [c["text"]]] + [f"{n}: {d}" for n, d in tool_set]
    need += [c["text"] for c in ctx if c["split"] == "val_ml"]
    need += [f"{n}: {d}" for n, d in mw_schema_intents_list(mw_schema)]
    emb.ensure(need)
    spaces = {cat: LabelSpace(cat, list(v), dict(v), emb) for cat, v in ex.items() if len(v) >= 5}
    tools = ToolPool(tool_set, emb)
    B = Builder(emb, spaces, tools, desc_en, desc_tr, SEED)
    B.protected_names = set(pnames)
    fn_ok = {cat for cat in spaces if all((cat, l) in desc_en for l in spaces[cat].labels)}
    manifest["intent_as_function_catalogs"] = sorted(fn_ok)

    mw_intents = mw_schema_intents(mw_schema)
    by = defaultdict(lambda: defaultdict(list))   # kind -> lang -> contexts
    for c in train_ctx:
        L = prim(c["lang"])
        if c["label"] and c["catalog"] in spaces and not c["meta"].get("oos"):
            by["intent"][L].append(c)
            if c["catalog"] in fn_ok:
                by["intent_fn"][L].append(c)
        if c["source"] == "toolace" and c.get("gold_tool"):
            by["tool"][L].append(c)
        if c["source"] == "toolace" and c.get("calls") == []:
            by["tool_none"][L].append(c)
        if c["source"] == "toolace" and c.get("calls"):
            by["tool_multi"][L].append(c)
        if c["source"] == "clinc150" and c["meta"].get("oos"):
            by["clinc_oos"][L].append(c)
        if c["source"] == "multiwoz":        # keep only intents defined by the v2.2 schema (e.g. drops "find_taxi")
            c["labels"] = [l for l in c["labels"] if l in mw_intents]
        if c["source"] == "multiwoz" and len(c["labels"]) == 1:
            by["mw1"][L].append(c)
        if c["source"] == "multiwoz" and len(c["labels"]) >= 2:
            by["mw2"][L].append(c)
    manifest["pools"] = {k: {L: len(v) for L, v in d.items()} for k, d in by.items()}

    records: list[dict] = []
    fill_report = {}

    def fill(name: str, pool_by_lang: dict, make, quota_by_lang=None):
        q = quota_by_lang or QUOTA[name]
        cap = CAPS.get(name, 1)
        rep = {}
        for L, n in q.items():
            n = int(round(n * args.scale))
            pool = list(pool_by_lang.get(L, []))
            random.Random(f"{SEED}|{name}|{L}").shuffle(pool)
            got = []
            for cyc in range(cap):
                for c in pool:
                    if len(got) >= n:
                        break
                    r = make(c, rng_for(SEED, name, c["source"], c["oid"], cyc))
                    if r:
                        got += r if isinstance(r, list) else [r]
                if len(got) >= n:
                    break
            records.extend(got[:n])
            rep[L] = {"quota": n, "made": len(got[:n]), "short": max(0, n - len(got)), "pool": len(pool)}
        fill_report[name] = rep
        print(name, {L: (v["made"], v["short"]) for L, v in rep.items()}, flush=True)

    def schema_loc_for(c, rng):
        """translated schema for ~20% of non-EN decisions (same locale; PT->ES / EN->PT variants for a few)."""
        L = prim(c["lang"])
        u = rng.random()
        if L != "en" and u < TRANSLATED_SCHEMA_SHARE:
            return c["lang"] if c["lang"] in ("pt-BR", "es-419", "fr-FR") else {"pt": "pt-BR", "es": "es-419", "fr": "fr-FR"}[L]
        if L == "pt" and u < TRANSLATED_SCHEMA_SHARE + 0.03:
            return "es-419"                     # PT context -> ES schema
        if L == "en" and u < 0.05:
            return "pt-BR"                      # EN context -> PT schema
        if L == "fr" and u < TRANSLATED_SCHEMA_SHARE + 0.03:
            return None
        return None

    # ---------------------------------------------------------------- choice / intent
    fill("choice_intent", by["intent"], lambda c, r: B.intent_choice(c, r))
    # ---------------------------------------------------------------- tool routing
    fill("tool_toolace", by["tool"], lambda c, r: B.tool_choice(c, r, schema_loc=schema_loc_for(c, r)))
    fill("tool_multiwoz", by["mw1"], lambda c, r: B.multiwoz_choice(c, mw_schema_intents(mw_schema), r))
    def intent_fn(c, r):
        loc = schema_loc_for(c, r)
        return B.intent_choice(c, r, category="tool_routing", mode="fn_tr" if loc else "fn", schema_loc=loc, tag="fn")
    fill("tool_intent_fn", by["intent_fn"], intent_fn)
    # ---------------------------------------------------------------- OOD
    fill("ood_intent_gold_removed", by["intent"], lambda c, r: B.intent_choice(c, r, gold_removed=True, hard=r.random() < 0.8,
                                                                                 tag="gold_removed"))
    fill("ood_tool_gold_removed", by["tool"], lambda c, r: B.tool_choice(c, r, gold_removed=True, schema_loc=schema_loc_for(c, r),
                                                                         tag="gold_removed"))
    fill("ood_multiwoz_gold_removed", by["mw1"], lambda c, r: B.multiwoz_choice(c, mw_schema_intents(mw_schema), r,
                                                                                gold_removed=True, tag="gold_removed"))
    fill("ood_intent_fn_gold_removed", by["intent_fn"], lambda c, r: B.intent_choice(
        c, r, gold_removed=True, mode="fn", hard=True, tag="fn_gold_removed"))
    irr = {L: by["tool_none"].get(L, []) + by["clinc_oos"].get(L, []) for L in LANGS}

    def native_irr(c, r):
        if c["source"] == "toolace":
            return B.tool_native_irrelevance(c, r, schema_loc=schema_loc_for(c, r))
        sp = spaces["clinc150"]                         # CLINC out-of-scope: human-labelled "none of the 150 intents"
        k = pick_k(r, len(sp.labels))
        labs = sp.negatives(None, k, B.ctx_emb(c.get("mining_text") or c["text"]), r, hard=r.random() < 0.5)
        return B.rec(c, qtype="choice", family="intent", instr_family="intent",
                     candidates=[B.label_cand("clinc150", l, "bare") for l in labs], labels=[],
                     slices=["native_irrelevance", "ood_far"], strategy="native", category="ood", tag="oos", rng=r,
                     extra_md={"ood_type": "native_irrelevance"})
    fill("ood_native_irrelevance", irr, native_irr)

    def cross_domain(c, r):
        targets = [t for t in CROSS_DOMAIN.get(c["catalog"], []) if t in spaces or t == "multiwoz"]
        if not targets:
            return None
        tgt = str(r.choice(targets))
        e = B.ctx_emb(c.get("mining_text") or c["text"])
        own = float(spaces[c["catalog"]].centroids[spaces[c["catalog"]].pos[c["label"]]] @ e)
        if tgt == "multiwoz":
            cands = [{"id": n, "label": n, "description": d, "language": "en"} for n, d in mw_schema_intents_list(mw_schema)]
            sims = B.emb.get([f"{x['id']}: {x['description']}" for x in cands]) @ e
            fam, ins = "tool", "tool"
        else:
            sp = spaces[tgt]
            k = pick_k(r, len(sp.labels))
            labs = list(r.choice(sp.labels, size=k, replace=False))
            sims = sp.centroids[[sp.pos[l] for l in labs]] @ e
            cands = [B.label_cand(tgt, l, "bare") for l in labs]
            fam, ins = "intent", "intent"
        if float(sims.max()) > min(0.85, own - 0.05):
            return None                       # a target label may actually apply: never risk a false "none"
        return B.rec(c, qtype="choice", family=fam, instr_family=ins, candidates=cands, labels=[],
                     slices=["ood_far"], strategy="random", category="ood", tag=f"xdomain_{tgt}", rng=r,
                     extra_md={"ood_type": "cross_domain", "target_catalog": tgt, "max_sim": round(float(sims.max()), 3),
                               **({"tool_source": "multiwoz"} if tgt == "multiwoz" else {})})
    fill("ood_cross_domain", by["intent"], cross_domain)

    def cross_task(c, r):
        e = B.ctx_emb(c.get("mining_text") or c["text"])
        own = float(spaces[c["catalog"]].centroids[spaces[c["catalog"]].pos[c["label"]]] @ e)
        k = pick_k(r, 32)
        idx = r.choice(len(tools.tools), size=k * 3, replace=False)
        cand = [tools.tools[i] for i in idx]
        sims = tools.E[idx] @ e
        keep = [t for t, s in zip(cand, sims) if s < min(0.80, own - 0.05)]
        names, sel = set(), []
        for t in keep:
            if t[0] not in names:
                names.add(t[0])
                sel.append(t)
            if len(sel) == k:
                break
        if len(sel) < 2:
            return None
        return B.rec(c, qtype="choice", family="tool", instr_family="tool", candidates=[B.tool_cand(t) for t in sel],
                     labels=[], slices=["cross_task", "ood_far"], strategy="random", category="ood", tag="xtask", rng=r,
                     extra_md={"ood_type": "cross_task", "tool_source": "toolace_pool"})
    fill("ood_cross_task", by["intent"], cross_task)

    # ---------------------------------------------------------------- multi-label
    fill("multi_concat", by["intent"], lambda c, r: concat(B, c, by["intent"][prim(c["lang"])], r))
    partial_pool = {L: [c for c in by["intent"][L] if c["catalog"] in {"massive", "snips", "banking77", "bitext"}][:20000]
                    for L in LANGS}
    fill("multi_partial", by["intent"], lambda c, r: partial(B, c, partial_pool, r))
    fill("multi_empty", by["intent"], lambda c, r: empty_multi(B, c, r))
    for c in by["tool_multi"].get("en", []) + by["tool_multi"].get("pt", []) + by["tool_multi"].get("es", []) + by["tool_multi"].get("fr", []):
        for cyc in range(2):
            r = rng_for(SEED, "tool_multi", c["oid"], cyc)
            records.append(B.tool_multi(c, r, schema_loc=schema_loc_for(c, r)))
    for c in by["mw2"].get("en", []):
        records.append(B.multiwoz_multi(c, mw_schema_intents(mw_schema), rng_for(SEED, "mw2", c["oid"])))
    records.extend(civil_multi(B, ppd, int(18_000 * args.scale)))
    records.extend(mhs_multi(B, ppd, int(10_000 * args.scale)))

    # ---------------------------------------------------------------- noul / score
    records.extend(primitives(B, "noul", int(65_000 * args.scale)))
    fill("noul_xl", by["intent_fn"], lambda c, r: noul_xl(B, c, r))
    fill("noul_multiwoz", by["mw1"], lambda c, r: noul_mw(B, c, mw_schema, r))
    records.extend(primitives(B, "score", int(60_000 * args.scale)))
    records.extend(civil_score(B, ppd, int(20_000 * args.scale)))

    # ---------------------------------------------------------------- minimal pairs (mined natural contrasts)
    fill("minimal_pairs_mined", mined_pairs(by["intent"], spaces), lambda c, r: minimal_pair_rec(B, c, r))

    # ---------------------------------------------------------------- rule typos + synthetic
    typo = []
    for rr in records:
        if rr["task"]["type"] == "choice" and rr["metadata"]["category"] in ("choice_intent", "tool_routing") and \
                h01("typo", rr["id"]) < TYPO_SHARE and not rr["metadata"].get("dialogue"):
            t = json.loads(json.dumps(rr))
            t["id"] = rr["id"] + "/typo"
            t["context"] = dict(t["context"], text=sp_.rule_typos(rr["context"]["text"], random.Random(t["id"])), noise=["typo"])
            t["slices"] = sorted(set(t["slices"]) | {"typo"})
            t["provenance"]["transforms"] = t["provenance"]["transforms"] + ["rule:typos-v1"]
            t["metadata"]["origin"] = "rules"
            t = schema.derive_fields(t)
            t["metadata"]["stage"] = stage_of(t)
            typo.append(t)
    records.extend(typo)
    syn_dropped = 0
    for p in args.synthetic:
        if Path(p).exists():
            for r in jl(p):
                if any(c.get("description") and protected(c.get("label") or c["id"], c["description"]) for c in r["candidates"]):
                    syn_dropped += 1          # pilot sources predate the protected-tool filter
                    continue
                records.append(synthetic_rec(r))
    manifest["synthetic_dropped_protected_tools"] = syn_dropped

    records = [r for r in records if r]
    # ---------------------------------------------------------------- validation (val_ml)
    val = build_val_ml(B, ctx, val_mt, mw_schema)

    # ---------------------------------------------------------------- validate + leakage (drop + re-check)
    errs = Counter(e for r in records + val for e in schema.validate(r))
    if errs:
        raise SystemExit(f"schema errors: {dict(errs.most_common(10))}")
    kb = [r for s in ("dev", "val_fit", "val_eval", "final_protected") for r in jl(ROOT / f"data/krunbench_v1/{s}.jsonl")]
    rep = leakage.check_leakage(records, kb + val, near_threshold=0.9)
    drop = {v["id"] for k in ("exact", "near", "group", "id") for v in rep.violations.get(k, [])}
    manifest["leakage_dropped_pass1"] = {k: len(v) for k, v in rep.violations.items()}
    records = [r for r in records if r["id"] not in drop]
    rep2 = leakage.check_leakage(records, kb + val, near_threshold=0.9)
    grp = leakage.check_split_groups(records + val)
    manifest["leakage_pass2"] = rep2.summary()["violations"]
    manifest["split_groups"] = grp.summary()["violations"]
    tool_viol = Counter()
    for r in records + val:
        for c in r["candidates"]:
            if c.get("description") and protected(c.get("label") or c["id"], c["description"]):
                tool_viol[r["split"]] += 1
    manifest["protected_tool_violations"] = dict(tool_viol)
    if tool_viol.get("train"):
        raise SystemExit(f"protected unseen tools on the training side: {dict(tool_viol)}")
    if not rep2.ok or not grp.ok:
        raise SystemExit(f"LEAKAGE != 0 after drop: {rep2.summary()} {grp.summary()}")
    rng_master.shuffle(records)
    for name, rs in (("train", records), ("val_ml", val)):
        with open(out / f"{name}.jsonl", "w") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest["fill"] = fill_report
    manifest["n_train"], manifest["n_val_ml"] = len(records), len(val)
    manifest["sha256"] = {n: hashlib.sha256((out / f"{n}.jsonl").read_bytes()).hexdigest() for n in ("train", "val_ml")}
    (out / "build_manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    print(json.dumps({k: manifest[k] for k in ("n_train", "n_val_ml", "leakage_pass2", "split_groups")}, indent=1))


# ================================================================================================ helpers
def mw_schema_intents(mw) -> dict:
    return {n: {"description": d, "service": s} for s, v in mw.items() for n, d in v["intents"].items()}


def mw_schema_intents_list(mw) -> list:
    return [(n, d) for s, v in mw.items() for n, d in v["intents"].items()]


CONCAT_POOLS: dict = {}


def concat(B, a, pool, r):
    key = (id(pool), a["catalog"])
    if key not in CONCAT_POOLS:                   # (setdefault would rebuild the list on every call)
        CONCAT_POOLS[key] = [c for c in pool if c["catalog"] == a["catalog"]]
    same = CONCAT_POOLS[key]
    b = same[int(r.integers(len(same)))]
    if b["catalog"] != a["catalog"] or b["label"] == a["label"] or b["split_group"] == a["split_group"]:
        return None
    if B.spaces[a["catalog"]].s_cent[B.spaces[a["catalog"]].pos[a["label"]], B.spaces[a["catalog"]].pos[b["label"]]] > 0.9:
        return None                                   # near-synonymous intents: "both apply" would be dubious
    return B.intent_concat(a, b, r)


def partial(B, a, by, r):
    """a + an utterance from a DISJOINT catalog (not in the candidate set): only a's label applies."""
    tgt = CROSS_DOMAIN.get(a["catalog"])
    if not tgt:
        return None
    pool = by[prim(a["lang"])]
    for _ in range(8):
        b = pool[int(r.integers(len(pool)))]
        if b["catalog"] in tgt:
            return B.intent_concat(a, None, r, partial_from=b)
    return None


def empty_multi(B, c, r):
    sp = B.spaces[c["catalog"]]
    k = min(pick_k(r, len(sp.labels) - 1, kmin=4), len(sp.labels) - 1)
    labs = sp.negatives(c["label"], k, B.ctx_emb(c.get("mining_text") or c["text"]), r, hard=True)
    return B.rec(c, qtype="multi", family="intent", instr_family="multi",
                 candidates=[B.label_cand(c["catalog"], l, "bare") for l in labs], labels=[],
                 slices=["gold_removed", "ood_near"], strategy="hard_v2", category="ood", tag="multi_empty", rng=r,
                 extra_md={"ood_type": "gold_removed_multi"})


NOUL_TPL = {"en": "The user wants to: {d}", "pt": "O usuário quer: {d}", "es": "El usuario quiere: {d}",
            "fr": "L'utilisateur veut : {d}"}


def noul_xl(B, c, r):
    """Cross-language proposition: PT/ES/FR context, EN proposition built from the intent description (80%), or the
    translated description (20%). 50% gold (true), 50% hardest neighbour (false)."""
    sp = B.spaces[c["catalog"]]
    true = r.random() < 0.5
    lab = c["label"] if true else sp.negatives(c["label"], 1, B.ctx_emb(c.get("mining_text") or c["text"]), r, hard=True)[0]
    L = prim(c["lang"])
    loc = c["lang"] if c["lang"] in ("pt-BR", "es-419", "fr-FR") else {"pt": "pt-BR", "es": "es-419", "fr": "fr-FR"}[L]
    d_tr = B.desc_tr.get((c["catalog"], lab, loc))
    if d_tr and r.random() < 0.2:
        prop, il = NOUL_TPL[L].format(d=d_tr.rstrip(".")), L
    else:
        prop, il = NOUL_TPL["en"].format(d=B.intent_desc[(c["catalog"], lab)].rstrip(".")), "en"
    return B.rec(c, qtype="noul", family="proposition", instr_family="intent", instructions=prop, ins_lang=il,
                 candidates=[{"id": "false"}, {"id": "true"}], labels=["true" if true else "false"],
                 slices=[] if true else ["near_duplicate"], strategy="native", category="noul", tag="noul_xl", rng=r,
                 extra_md={"proposition_label": lab})


SLOT_Q = "The user asks for the {s}."


def noul_mw(B, c, mw, r):
    from krun_one.v1.corpus.sources import render_dialogue

    svc_slots = {k: v for s in mw.values() for k, v in s["slots"].items()}
    req = [s for s in c["meta"]["requested"] if s in svc_slots]
    if req and r.random() < 0.5:
        slot, y = req[int(r.integers(len(req)))], True
    else:
        svc = next((s for s, v in mw.items() if c["labels"][0] in v["intents"]), None)
        cands = [k for k in mw.get(svc, {}).get("slots", {}) if k not in c["meta"]["requested"]]
        if not cands:
            return None
        slot, y = cands[int(r.integers(len(cands)))], False
    return B.rec(c, qtype="noul", family="proposition", instr_family="intent",
                 instructions=SLOT_Q.format(s=svc_slots[slot].rstrip(".")), ins_lang="en",
                 candidates=[{"id": "false"}, {"id": "true"}], labels=["true" if y else "false"], slices=[],
                 strategy="native", category="noul", tag="noul_mw_requested", rng=r, ctx_text=render_dialogue(c),
                 extra_md={"dialogue": True, "slot": slot})


def primitives(B, kind, n):
    rows = [r for r in jl(ROOT / f"data/primitives_v2/{kind}_train.jsonl")
            if r["source"] not in ("xlam", "xlam_irrelevance", "boolq", "sst5")]
    random.Random(f"{SEED}prim{kind}").shuffle(rows)
    out = []
    for d in rows[:n]:
        rec = schema.from_primitive_v0(d, split="train")
        rec["id"] = "kd1/" + rec["id"]
        rec["metadata"].update({"category": kind, "origin": "derived" if d["source"] in ("banking77", "massive", "toolace")
                                else "real", "catalog": d["source"], "k": len(rec["candidates"])})
        rec["metadata"]["stage"] = stage_of(rec)
        out.append(rec)
    return out


CIVIL_MULTI = ["toxicity", "insult", "obscene", "threat", "sexual_explicit"]       # identity_attack stays unseen
CIVIL_MULTI_TEXT = {"toxicity": "rude or disrespectful", "insult": "insults someone", "obscene": "obscene or profane language",
                    "threat": "threatens harm or violence", "sexual_explicit": "sexually explicit content"}
TOX_LEVELS = ["not toxic at all", "slightly toxic", "moderately toxic", "very toxic", "extremely toxic"]


def civil_train(ppd):
    import pandas as pd
    from huggingface_hub import hf_hub_download

    fs = ["data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet"]
    df = pd.concat([pd.read_parquet(hf_hub_download("google/civil_comments", f, repo_type="dataset",
                                                    revision="f2970eb3a55777454c94069077cc8d9b5866312d")) for f in fs],
                   ignore_index=True)
    df["row"] = np.arange(len(df))
    return df[df.text.str.len().between(20, 800)]


def civil_ctx(row):
    return {"source": "civil_comments", "oid": f"train-{row.row}", "orig_split": "train", "text": row.text,
            "lang": "en-US", "group": f"civil:train-{row.row}", "split_group": f"civil:train-{row.row}",
            "catalog": "civil_comments", "license": "CC0-1.0", "revision": "f2970eb3a55777454c94069077cc8d9b5866312d"}


def civil_multi(B, ppd, n):
    df = civil_train(ppd)
    rng = np.random.default_rng(SEED)
    tox = df[df.toxicity >= 0.5]
    clean = df[df.toxicity < 0.1]
    sel = pd_concat([tox.iloc[rng.permutation(len(tox))[: int(n * 0.85)]], clean.iloc[rng.permutation(len(clean))[: n - int(n * 0.85)]]])
    out = []
    for _, row in sel.iterrows():
        r = rng_for(SEED, "civil_multi", row.row)
        labs = [a for a in CIVIL_MULTI if row[a] >= 0.5]
        cands = [{"id": a, "label": a, "description": CIVIL_MULTI_TEXT[a], "language": "en"} for a in CIVIL_MULTI]
        out.append(B.rec(civil_ctx(row), qtype="multi", family="proposition", instr_family="multi",
                         instructions="Which of these properties does the comment have? Select all that apply.",
                         ins_lang="en", candidates=cands, labels=labs, slices=[], strategy="fixed",
                         category="multi_label", tag="civil_multi", rng=r, origin="derived"))
    return out


def pd_concat(x):
    import pandas as pd

    return pd.concat(x)


def civil_score(B, ppd, n):
    df = civil_train(ppd)
    rng = np.random.default_rng(SEED + 1)
    bins = [0, 0.1, 0.3, 0.5, 0.7, 1.01]
    out = []
    for lvl in range(5):
        d = df[(df.toxicity >= bins[lvl]) & (df.toxicity < bins[lvl + 1])]
        for _, row in d.iloc[rng.permutation(len(d))[: n // 5]].iterrows():
            r = rng_for(SEED, "civil_score", row.row)
            out.append(B.rec(civil_ctx(row), qtype="score", family="rating", instr_family="intent",
                             instructions="How toxic is this comment?", ins_lang="en",
                             candidates=[{"id": str(i), "label": x} for i, x in enumerate(TOX_LEVELS)], labels=[str(lvl)],
                             slices=[], strategy="fixed", category="score", tag="civil_score", rng=r, origin="derived",
                             extra_md={"toxicity": float(row.toxicity)}))
    return out


def mhs_multi(B, ppd, n):
    cfg = {"sources": {"mhs": {"hf_id": "ucberkeley-dlab/measuring-hate-speech",
                               "revision": "5468f6e118396646b02a2f691e771f6b6d9502ea", "license": "CC-BY-4.0"}},
           "seed": 20261101, "mhs_split": {"train": 0.8, "val": 0.1}}
    comments = [c for c in ppd.load_mhs(cfg) if c["split"] == "train"]
    random.Random(SEED).shuffle(comments)
    out = []
    cands = [{"id": t, "label": t, "description": ppd.MHS_TARGET_TEXT[t], "language": "en"} for t in ppd.MHS_TARGETS]
    for c in comments[:n]:
        ctx = {"source": "mhs", "oid": str(c["cid"]), "orig_split": "train", "text": c["text"], "lang": "en-US",
               "group": f"mhs:{c['cid']}", "split_group": f"mhs:{c['cid']}", "catalog": "mhs", "license": "CC-BY-4.0",
               "revision": "5468f6e118396646b02a2f691e771f6b6d9502ea"}
        r = rng_for(SEED, "mhs_multi", c["cid"])
        out.append(B.rec(ctx, qtype="multi", family="proposition", instr_family="multi",
                         instructions="Which groups of people does this comment target? Select all that apply.",
                         ins_lang="en", candidates=cands, labels=list(c["targets"]), slices=[], strategy="fixed",
                         category="multi_label", tag="mhs_targets", rng=r, origin="derived"))
    return out


def mined_pairs(intent_by_lang, spaces):
    """Natural minimal pairs: EN train utterances of the same catalog, word-Jaccard >= 0.6, different labels.
    Every language version of a member (MASSIVE parallel rows, BANKING77 PT-BR, MT) inherits the pair."""
    en = [c for c in intent_by_lang.get("en", []) if c.get("origin") == "real"]
    by_cat = defaultdict(list)
    for c in en:
        by_cat[c["catalog"]].append(c)
    pairs = {}
    for cat, cs in by_cat.items():
        inv = defaultdict(list)
        toks = [set(re.findall(r"\w+", c["text"].lower())) for c in cs]
        for i, t in enumerate(toks):
            for w in t:
                inv[w].append(i)
        for i, t in enumerate(toks):
            if len(t) < 3:
                continue
            cand = {j for w in t if len(inv[w]) < 400 for j in inv[w] if j > i}   # share >= 1 non-stopword
            for j in cand:
                if cs[j]["label"] == cs[i]["label"]:
                    continue
                jac = len(t & toks[j]) / len(t | toks[j])
                if jac >= 0.6:
                    sid = f"mpm:{cat}:{min(cs[i]['group'], cs[j]['group'])}|{max(cs[i]['group'], cs[j]['group'])}"
                    pairs.setdefault(cs[i]["group"], (sid, cs[j]["label"], "a"))
                    pairs.setdefault(cs[j]["group"], (sid, cs[i]["label"], "b"))
    out = defaultdict(list)
    for L, cs in intent_by_lang.items():
        for c in cs:
            if c["group"] in pairs:
                out[L].append(dict(c, _mp=pairs[c["group"]]))
    return out


def minimal_pair_rec(B, c, r):
    sid, other, member = c["_mp"]
    sp = B.spaces[c["catalog"]]
    k = min(8, len(sp.labels))
    negs = sp.negatives(c["label"], k - 2, B.ctx_emb(c.get("mining_text") or c["text"]), r, hard=True, exclude=[other])
    labs = negs + [c["label"], other]
    r.shuffle(labs)
    rec = B.rec(c, qtype="choice", family="intent", instr_family="intent",
                candidates=[B.label_cand(c["catalog"], l, "bare") for l in labs], labels=[c["label"]],
                slices=["minimal_pair"], strategy="hard_v2", category="choice_intent", tag="mp_mined", rng=r)
    rec["minimal_pair"] = {"set_id": sid, "member": member, "contrast": f"{c['label']}<->{other}",
                           "relation": "natural_contrast_mined", "difficulty": "hard", "transformation": "none (real pair)"}
    return rec


SYN_CAT = {"minimal_pair": "choice_intent", "counterfactual": "tool_routing", "hard_negative": "tool_routing",
           "near_ood": "ood", "underspecified": "multi_label", "slang": "choice_intent", "code_switching": "choice_intent"}


def synthetic_rec(r):
    t = r["provenance"]["synthetic"]["transformation"]
    cat = SYN_CAT.get(t, "choice_intent")
    if r["task"]["family"] == "tool" and cat == "choice_intent":
        cat = "tool_routing"
    r["metadata"].update({"category": cat, "origin": "llm", "k": len(r["candidates"]),
                          **({"ood_type": "near_ood_synthetic"} if t == "near_ood" else {})})
    if t == "minimal_pair" and r.get("minimal_pair"):
        r["minimal_pair"].setdefault("difficulty", "hard")
        r["minimal_pair"].setdefault("transformation", "llm_minimal_edit")
    r["metadata"]["stage"] = stage_of(r)
    return r


# ------------------------------------------------------------------------------------------------ val_ml
def build_val_ml(B, ctx, val_mt, mw):
    """Multilingual V set (tuning / calibration / reporting; never training): MASSIVE validation (4 locales),
    BANKING77 test EN + PT-BR, MultiWOZ dev, and MT (pt-BR / es-419 / fr-FR, EN schema) of KrunBench v1 validation
    ToolACE / CLINC / Bitext items. fit/eval by split_group hash (bakeoff salt)."""
    bk = load_module("bk", ROOT / "scripts/v1/answerability_bakeoff.py")
    out = []
    vctx = [dict(c, origin="real", rec_split="validation") for c in ctx if c["split"] == "val_ml"]
    rng = random.Random(SEED + 7)
    b77 = [c for c in vctx if c["source"] in ("banking77", "banking77_pt_br")]
    keep_rows = set(rng.sample(sorted({c["oid"] for c in b77}), 1500))
    for c in vctx:
        r = rng_for(SEED, "val", c["source"], c["oid"])
        if c["source"] in ("banking77", "banking77_pt_br") and c["oid"] not in keep_rows:
            continue
        if c["source"] == "multiwoz":
            c = dict(c, labels=[l for l in c["labels"] if l in mw_schema_intents(mw)])
            if len(c["labels"]) != 1 or r.random() > 0.2:
                continue
            out.append(B.multiwoz_choice(c, mw_schema_intents(mw), r, gold_removed=r.random() < 0.15, tag="val"))
            continue
        if c["catalog"] not in B.spaces or c["label"] not in B.spaces[c["catalog"]].pos:
            continue
        u = r.random()
        if u < 0.15:
            out.append(B.intent_choice(c, r, gold_removed=True, tag="val_gr"))
        elif u < 0.40 and all((c["catalog"], l) in B.intent_desc for l in B.spaces[c["catalog"]].labels):
            out.append(B.intent_choice(c, r, category="tool_routing", mode="fn", tag="val_fn"))
        else:
            out.append(B.intent_choice(c, r, tag="val"))
    kb = {r["id"]: r for s in ("dev", "val_fit", "val_eval") for r in jl(ROOT / f"data/krunbench_v1/{s}.jsonl")}
    for rid, loc, text in val_mt:
        src = kb.get(rid)
        if src is None:
            continue
        t = json.loads(json.dumps(src))
        t["id"] = f"kd1val/mt/{rid}/{loc}"
        t["context"] = dict(t["context"], text=text, language=loc)
        t["provenance"]["transforms"] = t["provenance"]["transforms"] + [MT_TAG]
        part = {"val_fit": "fit", "val_eval": "eval", "dev": "dev"}[src["metadata"]["kbv1_split"]]
        t["metadata"] = dict(t.get("metadata") or {}, category="mt_of_krunbench_val", origin="mt", val_part=part)
        t["semantic_group_id"] = src["semantic_group_id"]
        out.append(schema.derive_fields(t))
    out = [o for o in out if o]
    # permutation consistency: 4 latin-square rotations of ~800 answerable choice decisions (slice "permutation";
    # kd1_evaluate uses the rotated copies ONLY for the consistency metric)
    base = [o for o in out if o["task"]["type"] == "choice" and o["answerable"] and len(o["candidates"]) >= 4]
    for o in random.Random(SEED + 11).sample(base, min(800, len(base))):
        o.setdefault("metadata", {})["v0_generation"] = {"consistency_group": o["id"]}
        k = len(o["candidates"])
        for rot in (1, 2, 3):
            t = json.loads(json.dumps(o))
            sh = rot * k // 4
            t["candidates"] = o["candidates"][sh:] + o["candidates"][:sh]
            t["id"] = f"{o['id']}/rot{rot}"
            t["slices"] = sorted(set(t["slices"]) | {"permutation"})
            out.append(t)
    for o in out:
        o["split"] = "validation"
        o["metadata"].setdefault("val_part", "fit" if bk.is_fit(o["split_group_id"]) else "eval")
        o["id"] = o["id"].replace("kd1/", "kd1val/", 1)
    return out


if __name__ == "__main__":
    sys.exit(main())
