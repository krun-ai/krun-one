#!/usr/bin/env python
"""KRUN-011: build noul / score training + evaluation data (and the mixed-primitive eval requests).

    .venv/bin/python scripts/prepare_primitives_data.py

Outputs (data/ is gitignored; everything is reproducible from configs/primitives_data.yaml + pinned HF revisions):
    data/primitives_v1/{noul,score}_{train,val,test}.jsonl   one question per record (schema krun.primitive.v0)
    data/primitives_v1/mixed_test.jsonl                     multi-primitive requests on one context (MHS test)
    data/primitives_v1/manifest.json                        config, sources, counts, sha256
    results/primitives_dataset_stats.json                   same counts (versioned)

Record schema (krun.primitive.v0):
    id, split (train|val|test), qtype (noul|score|choice), source, suite, context, instructions,
    criteria (noul, optional {true,false}), levels (score), options/option_descriptions (choice),
    target (int: noul 0/1, score level index, choice option index), group_id, provenance{...}, metadata{...}

Split hygiene: noul items derived from intents/tools reuse the Krun One v0 split of their context (v0 train ->
train, val_hard/val_tools -> val, KrunBench test/holdout_tools -> test), so no context crosses splits between
primitives. HelpSteer2 is split by prompt, MHS by comment_id, Civil Comments by its official splits.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from krunbench.splits import stable_fraction  # noqa: E402

SCHEMA = "krun.primitive.v0"


def hf_file(cfg, key, fname):
    from huggingface_hub import hf_hub_download
    s = cfg["sources"][key]
    return hf_hub_download(s["hf_id"], fname, repo_type="dataset", revision=s["revision"])


def rng_for(*parts) -> np.random.Generator:
    h = hashlib.sha256("\x1f".join(map(str, parts)).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def prov(cfg, key, original_split, original_id, **extra):
    s = cfg["sources"].get(key, {})
    return {"dataset": key, "hf_id": s.get("hf_id"), "revision": s.get("revision"), "license": s.get("license"),
            "original_split": original_split, "original_id": str(original_id), **extra}


def rec(split, qtype, source, suite, context, instructions, target, group_id, provenance, *, criteria=None,
        levels=None, metadata=None, id_suffix=""):
    rid = f"prim/{split}/{qtype}/{source}/{suite}/{provenance['original_id']}{id_suffix}"
    r = {"schema_version": SCHEMA, "id": rid, "split": split, "qtype": qtype, "source": source, "suite": suite,
         "context": context, "instructions": instructions, "target": int(target), "group_id": group_id,
         "provenance": provenance, "metadata": metadata or {}}
    if qtype == "noul":
        r["criteria"] = criteria
    if qtype == "score":
        r["levels"] = list(levels)
    return r


def cap(items, n, seed, tag):
    if n is None or len(items) <= n:
        return items
    idx = rng_for(seed, "cap", tag).permutation(len(items))[:n]
    return [items[i] for i in sorted(idx)]


def read_jsonl(p):
    with open(p) as f:
        for line in f:
            yield json.loads(line)


# ============================================================================================ noul: intents
INTENT_CRITERIA = {"true": "the message is about this", "false": "the message is about something else"}
TOOL_CRITERIA = {"true": "calling this function fulfils the request", "false": "this function does not fit the request"}
TOOL_INS = "Should the function `{name}` be called to fulfil this request? Function description: {desc}"


def intent_contexts(cfg):
    """Unique (dataset, config, original_id) intent contexts with their v0 split."""
    v0 = PROJECT / cfg["krun_one_v0_dir"]
    kb = PROJECT / cfg["krunbench_dir"]
    out = {}
    for split, files in (("train", [v0 / "train.jsonl"]), ("val", [v0 / "val_hard.jsonl"]),
                         ("test", [kb / "banking77" / "test.jsonl", kb / "massive" / "test.jsonl"])):
        for f in files:
            for d in read_jsonl(f):
                p = d["provenance"]
                if p["dataset"] not in ("banking77", "massive") or not p.get("original_label"):
                    continue
                key = (p["dataset"], p["config"], p["original_id"], split)
                out.setdefault(key, {"split": split, "dataset": p["dataset"], "config": p["config"],
                                     "original_id": p["original_id"], "context": d["context"], "label": p["original_label"],
                                     "language": d.get("language"), "group_id": p["group_id"], "hf_id": p["hf_id"],
                                     "revision": p["revision"], "original_split": p["original_split"]})
    return list(out.values())


def intent_neighbours(cfg, n):
    cnt = defaultdict(Counter)
    for d in read_jsonl(PROJECT / cfg["krun_one_v0_dir"] / "train.jsonl"):
        if d["task"] == "intent_routing" and d["suite"] == "core" and d["metadata"].get("negatives") == "hard_v1":
            for o in d["options"]:
                if o != d["answer"]:
                    cnt[d["answer"]][o] += 1
    return {g: [l for l, _ in c.most_common(n)] for g, c in cnt.items()}


def build_noul_intents(cfg, stats):
    c = cfg["noul"]["intent"]
    props_all = yaml.safe_load((ROOT / c["propositions"]).read_text())
    overlaps = props_all.pop("overlaps")
    props = {ds: v for ds, v in props_all.items()}
    label_ds = {lab: ds for ds, v in props.items() for lab in v}
    neigh = intent_neighbours(cfg, c["neighbours"])
    ctxs = intent_contexts(cfg)
    by_label = defaultdict(list)
    for x in ctxs:
        by_label[(x["split"], x["label"])].append(x)
    seed = cfg["seed"]
    out = defaultdict(list)
    for x in ctxs:
        ds, lab, split = x["dataset"], x["label"], x["split"]
        r = rng_for(seed, "noul-intent", x["group_id"], x["config"], split)
        pv = {"hf_id": x["hf_id"], "revision": x["revision"], "license": "CC-BY-4.0", "dataset": ds,
              "original_split": x["original_split"], "original_id": f"{x['config'] or '-'}/{x['original_id']}",
              "original_label": lab, "transform": "krun_one/noul_intent/1"}
        meta = {"language": x["language"]}

        def mk(suite, prop_label, target):
            use_crit = r.random() < cfg["noul"]["criteria_fraction"]
            return rec(split, "noul", ds, suite, x["context"], props[label_ds[prop_label]][prop_label], target,
                       x["group_id"], {**pv, "proposition": prop_label}, criteria=INTENT_CRITERIA if use_crit else None,
                       metadata={**meta, "proposition_id": f"intent:{prop_label}", "kind": suite}, id_suffix=f"/{prop_label}")

        if lab in props.get(ds, {}):
            out[split].append(mk("positive", lab, 1))
        # hard negative: a confusable label's proposition (never one that may be true for this context)
        cands = [n for n in neigh.get(lab, []) if n in props.get(ds, {}) and n != lab and lab not in overlaps.get(n, [])
                 and n not in overlaps.get(lab, [])]
        if cands:
            out[split].append(mk("hard_negative", cands[int(r.integers(len(cands)))], 0))
        if r.random() < c["per_context"]["cross_domain"]:
            other = "massive" if ds == "banking77" else "banking77"
            labs = sorted(props[other])
            out[split].append(mk("cross_domain", labs[int(r.integers(len(labs)))], 0))
    unseen = {lab for lab in label_ds if stable_fraction(seed, "unseen-intent-prop", lab) < c.get("unseen_fraction", 0.0)}
    for split in list(out):
        if unseen:
            for r_ in out[split]:
                r_["metadata"]["unseen_proposition"] = r_["provenance"]["proposition"] in unseen
            if split == "train":
                out[split] = [r_ for r_ in out[split] if not r_["metadata"]["unseen_proposition"]]
        out[split] = cap(out[split], c["cap"][split], seed, f"noul-intent-{split}")
        stats[f"noul/intent/{split}"] = dict(Counter(i["suite"] for i in out[split]))
    stats["noul/intent/unseen_propositions"] = sorted(unseen)
    return out


# ============================================================================================== noul: tools
def build_noul_tools(cfg, stats):
    c = cfg["noul"]["tool"]
    v0 = PROJECT / cfg["krun_one_v0_dir"]
    seed = cfg["seed"]
    out = defaultdict(list)
    # Seen = anywhere in the choice training file, as gold OR distractor, exactly or by equivalent description
    # (krun_one.v1.tools). Fixes the v0.3 bug: the flag was read from `metadata.unseen_tool`, a key the choice data
    # never writes (it writes `tool_seen`, about the gold), so the noul unseen-tool slice was always empty.
    from krun_one.v1.tools import TrainToolIndex  # noqa: PLC0415
    tool_index = TrainToolIndex().add_v0_decisions(read_jsonl(v0 / "train.jsonl"))
    for split, f in (("train", "train.jsonl"), ("val", "val_tools.jsonl"), ("test", "holdout_tools.jsonl")):
        seen = set()
        for d in read_jsonl(v0 / f):
            if d["task"] != "tool_routing" or d["suite"] not in ("core", "ood_native"):
                continue
            p = d["provenance"]
            if p["group_id"] in seen:
                continue
            seen.add(p["group_id"])
            r = rng_for(seed, "noul-tool", p["group_id"])
            desc = d["option_descriptions"] or {}
            lic = "Apache-2.0" if p["dataset"] == "toolace" else "CC-BY-4.0"
            pv = {"hf_id": p["hf_id"], "revision": p["revision"], "license": lic, "dataset": p["dataset"],
                  "original_split": p["original_split"], "original_id": p["original_id"], "transform": "krun_one/noul_tool/1"}

            def mk(suite, name, target):
                use_crit = r.random() < cfg["noul"]["criteria_fraction"]
                return rec(split, "noul", p["dataset"], suite, d["context"], TOOL_INS.format(name=name, desc=desc.get(name) or ""),
                           target, p["group_id"], {**pv, "tool": name}, criteria=TOOL_CRITERIA if use_crit else None,
                           metadata={"proposition_id": f"tool:{name}", "kind": suite,
                                     "tool_status": tool_index.status(name, desc.get(name)),
                                     "unseen_tool": tool_index.is_unseen(name, desc.get(name)),
                                     "gold_tool_seen": d["metadata"].get("tool_seen")}, id_suffix=f"/{name}")

            if d["answer"]:
                out[split].append(mk("positive", d["answer"], 1))
                natives = [o for o in d["options"] if o != d["answer"]]
                if natives:
                    out[split].append(mk("hard_negative", natives[int(r.integers(len(natives)))], 0))
            else:
                out[split].append(mk("hard_negative_irrelevance", d["options"][int(r.integers(len(d["options"])))], 0))
    for split in out:
        out[split] = cap(out[split], c["cap"][split], seed, f"noul-tool-{split}")
        stats[f"noul/tool/{split}"] = dict(Counter(i["suite"] for i in out[split]))
    return out


# ====================================================================================== noul: civil comments
CIVIL_PROPS = {
    "toxicity": "Is this comment rude, disrespectful or unreasonable enough to make someone want to leave the discussion?",
    "insult": "Does this comment insult someone?",
    "obscene": "Does this comment contain obscene or profane language?",
    "threat": "Does this comment threaten someone with harm or violence?",
    "identity_attack": "Does this comment attack people because of their identity, such as race, religion, gender or sexual orientation?",
    "sexual_explicit": "Does this comment contain sexually explicit content?",
}
CIVIL_CRITERIA = {"true": "the comment clearly has this property", "false": "the comment does not have this property"}


def build_noul_civil(cfg, stats):
    c = cfg["noul"]["civil_comments"]
    seed = cfg["seed"]
    files = {"train": ["data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet"],
             "val": ["data/validation-00000-of-00001.parquet"], "test": ["data/test-00000-of-00001.parquet"]}
    out = defaultdict(list)
    for split, fs in files.items():
        df = pd.concat([pd.read_parquet(hf_file(cfg, "civil_comments", f)) for f in fs], ignore_index=True)
        df["row"] = np.arange(len(df))
        df = df[df.text.str.strip().str.len() > 0]
        quota = c["per_proposition"] if split == "train" else c["eval_per_proposition"]
        toxic = df.toxicity >= c["threshold"]
        for prop, ins in CIVIL_PROPS.items():
            if split == "train" and prop in c["unseen_propositions"]:
                continue
            pos = df[df[prop] >= c["threshold"]]
            if prop == "toxicity":
                hard = df[(df.toxicity > 0) & (df.toxicity <= c["hard_negative_max"])]
            else:
                hard = df[toxic & (df[prop] < c["hard_negative_max"])]
            easy = df[df.toxicity == 0]
            for suite, pool, target, n in (("positive", pos, 1, quota["positive"]), ("hard_negative", hard, 0, quota["hard_negative"]),
                                           ("cross_domain", easy, 0, quota["cross_domain"])):
                r = rng_for(seed, "civil", split, prop, suite)
                sel = pool.iloc[r.permutation(len(pool))[:n]] if len(pool) > n else pool
                for _, row in sel.iterrows():
                    rr = rng_for(seed, "civil-crit", split, prop, row.row)
                    oid = f"{split}-{row.row}"
                    out[split].append(rec(split, "noul", "civil_comments", suite, row.text, ins, target, f"civil:{oid}",
                                          prov(cfg, "civil_comments", {"val": "validation"}.get(split, split), oid,
                                               transform="krun_one/noul_civil/1", rater_fraction=round(float(row[prop]), 4)),
                                          criteria=CIVIL_CRITERIA if rr.random() < cfg["noul"]["criteria_fraction"] else None,
                                          metadata={"proposition_id": f"civil:{prop}", "kind": suite,
                                                    "unseen_proposition": prop in c["unseen_propositions"]},
                                          id_suffix=f"/{prop}"))
        stats[f"noul/civil_comments/{split}"] = dict(Counter(i["suite"] for i in out[split]))
    return out


# ============================================================================================ MHS (noul+score)
MHS_ITEMS = {  # 5-point ordinal items, higher = more hateful (Kennedy et al. 2020). Level texts written by us.
    "sentiment": ("What is the overall sentiment of this comment towards the group it mentions?",
                  ["Strongly positive", "Somewhat positive", "Neutral", "Somewhat negative", "Strongly negative"]),
    "respect": ("How respectful is this comment towards the group it mentions?",
                ["Strongly respectful", "Somewhat respectful", "Neither respectful nor disrespectful",
                 "Somewhat disrespectful", "Strongly disrespectful"]),
    "insult": ("How insulting is this comment towards the group it mentions?",
               ["Not insulting at all", "Slightly insulting", "Somewhat insulting", "Clearly insulting", "Extremely insulting"]),
    "humiliate": ("How much does this comment humiliate or demean the group it mentions?",
                  ["Not humiliating at all", "Slightly humiliating", "Somewhat humiliating", "Clearly humiliating",
                   "Extremely humiliating"]),
    "status": ("How strongly does this comment claim the group it mentions is inferior to others?",
               ["Claims the group is superior or equal", "Slight hint of superiority", "No claim either way",
                "Implies the group is inferior", "Explicitly states the group is inferior"]),
    "dehumanize": ("How strongly does this comment deny the humanity of the group it mentions, for example by comparing it to animals or disease?",
                   ["Not dehumanizing at all", "Slightly dehumanizing", "Somewhat dehumanizing", "Clearly dehumanizing",
                    "Extremely dehumanizing"]),
    "violence": ("How strongly does this comment call for or threaten violence against the group it mentions?",
                 ["No call for violence at all", "Very indirect hint of violence", "Ambiguous or possible call for violence",
                  "Clear call for violence", "Explicit and strong call for violence"]),
    "genocide": ("How strongly does this comment support killing all or most members of the group it mentions?",
                 ["No support for killing at all", "Very indirect hint", "Ambiguous or possible support",
                  "Clear support for mass killing", "Explicit and strong call for mass killing"]),
    "attack_defend": ("Does this comment defend or attack the group it mentions?",
                      ["Strongly defends the group", "Somewhat defends the group", "Neither defends nor attacks",
                       "Somewhat attacks the group", "Strongly attacks the group"]),
}
MHS_HATE_INS = "Is this comment hate speech against a group of people?"
MHS_TARGETS = ["race", "religion", "origin", "gender", "sexuality", "age", "disability", "politics"]
MHS_TARGET_TEXT = {"race": "race or ethnicity", "religion": "religion", "origin": "national origin or citizenship",
                   "gender": "gender", "sexuality": "sexual orientation", "age": "age", "disability": "disability",
                   "politics": "political ideology"}


def mhs_split(cfg, cid):
    f = stable_fraction(cfg["seed"], "mhs-split", cid)
    s = cfg["mhs_split"]
    return "train" if f < s["train"] else "val" if f < s["train"] + s["val"] else "test"


def median_half_up(v):
    return int(np.floor(np.median(v) + 0.5))


def load_mhs(cfg):
    df = pd.read_parquet(hf_file(cfg, "mhs", "measuring-hate-speech.parquet"))
    g = df.groupby("comment_id")
    comments = []
    for cid, grp in g:
        text = grp.text.iloc[0]
        if not isinstance(text, str) or not text.strip():
            continue
        items = {k: [int(x) for x in grp[k].dropna()] for k in list(MHS_ITEMS) + ["hatespeech"]}
        tg = [t for t in MHS_TARGETS if f"target_{t}" in grp and grp[f"target_{t}"].mean() > 0.5]
        comments.append({"cid": int(cid), "text": text, "n_annot": len(grp), "items": items, "targets": tg,
                         "split": mhs_split(cfg, int(cid))})
    return comments


def build_mhs(cfg, comments, stats):
    seed = cfg["seed"]
    noul, score = defaultdict(list), defaultdict(list)
    hc = cfg["noul"]["mhs_hatespeech"]
    for c in comments:
        split, oid = c["split"], str(c["cid"])
        pv = prov(cfg, "mhs", "train", oid, n_annotators=c["n_annot"])
        hs = c["items"]["hatespeech"]
        if hs:
            m = float(np.mean(hs))
            y = 1 if m >= hc["yes_if_mean_ge"] else 0 if m <= hc["no_if_mean_le"] else None
            if y is not None:
                noul[split].append(rec(split, "noul", "mhs", "positive" if y else "negative", c["text"], MHS_HATE_INS, y,
                                       f"mhs:{oid}", {**pv, "transform": "krun_one/noul_mhs/1"},
                                       metadata={"proposition_id": "mhs:hatespeech", "kind": "positive" if y else "negative",
                                                 "annotations": hs}, id_suffix="/hatespeech"))
        for item, (ins, levels) in MHS_ITEMS.items():
            v = c["items"][item]
            if not v:
                continue
            score[split].append(rec(split, "score", "mhs", item, c["text"], ins, median_half_up(v), f"mhs:{oid}",
                                    {**pv, "transform": "krun_one/score_mhs/1"}, levels=levels,
                                    metadata={"attribute": item, "annotations": v}, id_suffix=f"/{item}"))
    for split in list(noul):
        noul[split] = cap(noul[split], hc["cap"][split], seed, f"mhs-noul-{split}")
        stats[f"noul/mhs/{split}"] = dict(Counter(i["target"] for i in noul[split]))
    for split in list(score):
        score[split] = cap(score[split], cfg["score"]["mhs"]["cap"][split], seed, f"mhs-score-{split}")
        stats[f"score/mhs/{split}"] = {"n": len(score[split]), "targets": dict(Counter(i["target"] for i in score[split]))}
    return noul, score


def build_mixed(cfg, comments, stats):
    """One request per MHS test comment: choice (target group) + noul (hate speech) + score (insult, sentiment)."""
    out = []
    hc = cfg["noul"]["mhs_hatespeech"]
    for c in comments:
        if c["split"] != "test" or len(c["targets"]) != 1 or not c["items"]["hatespeech"]:
            continue
        m = float(np.mean(c["items"]["hatespeech"]))
        y = 1 if m >= hc["yes_if_mean_ge"] else 0 if m <= hc["no_if_mean_le"] else None
        if y is None or not c["items"]["insult"] or not c["items"]["sentiment"]:
            continue
        r = rng_for(cfg["seed"], "mixed", c["cid"])
        opts = [c["targets"][0]] + [t for t in r.permutation(MHS_TARGETS).tolist() if t != c["targets"][0]][:3]
        opts = [opts[i] for i in r.permutation(len(opts))]
        qs = {
            "target_group": {"type": "choice", "instructions": "Which group of people is this comment mainly about?",
                             "options": opts, "option_descriptions": {o: MHS_TARGET_TEXT[o] for o in opts},
                             "target": opts.index(c["targets"][0])},
            "hate_speech": {"type": "noul", "instructions": MHS_HATE_INS, "target": y},
            "insult": {"type": "score", "instructions": MHS_ITEMS["insult"][0], "levels": MHS_ITEMS["insult"][1],
                       "target": median_half_up(c["items"]["insult"])},
            "sentiment": {"type": "score", "instructions": MHS_ITEMS["sentiment"][0], "levels": MHS_ITEMS["sentiment"][1],
                          "target": median_half_up(c["items"]["sentiment"])},
        }
        out.append({"schema_version": "krun.mixed.v0", "id": f"mixed/mhs/{c['cid']}", "context": c["text"], "questions": qs,
                    "provenance": prov(cfg, "mhs", "train", c["cid"], transform="krun_one/mixed_mhs/1")})
    out = cap(out, 1000, cfg["seed"], "mixed")
    stats["mixed/test"] = len(out)
    return out


# =================================================================================================== score: HS2
HS2 = {  # instructions + level texts written by us from the HelpSteer2 attribute definitions (dataset card)
    "helpfulness": ("How helpful is the response to the user's prompt overall?",
                    ["Not helpful at all: useless or wrong for the request", "Slightly helpful: mostly misses what was asked",
                     "Partially helpful: addresses the request with significant gaps or problems",
                     "Mostly helpful: addresses the request with minor room for improvement",
                     "Fully helpful: completely satisfies the request"]),
    "correctness": ("How correct and complete is the response: does it include all pertinent facts without errors?",
                    ["Completely incorrect or missing the pertinent facts", "Mostly incorrect, with major errors or omissions",
                     "Partially correct, with several errors or omissions", "Mostly correct, with minor errors or omissions",
                     "Fully correct and complete"]),
    "coherence": ("How coherent is the response: is it consistent and clearly expressed?",
                  ["Incoherent: cannot be understood", "Mostly incoherent: hard to follow",
                   "Somewhat coherent: parts are unclear or inconsistent", "Mostly coherent: clear with minor issues",
                   "Perfectly coherent and clear"]),
    "complexity": ("How much expertise is needed to write this response?",
                   ["Basic: anyone with basic language skills could write it", "Simple: high-school level",
                    "Intermediate: some college-level knowledge", "Advanced: requires specialized knowledge",
                    "Expert: requires deep domain expertise"]),
    "verbosity": ("How verbose is the response relative to what the prompt asks for?",
                  ["Very succinct", "Rather short", "Appropriate length", "Rather long", "Very verbose"]),
}


def hs2_context(prompt, response, pc, rc):
    p = prompt if len(prompt) <= pc else "…" + prompt[-pc:]
    r = response if len(response) <= rc else response[:rc] + "…"
    return f"Prompt:\n{p}\n\nResponse:\n{r}"


def build_hs2(cfg, stats):
    c = cfg["score"]["helpsteer2"]
    seed = cfg["seed"]
    out = defaultdict(list)
    for fname, official in (("train.jsonl.gz", "train"), ("validation.jsonl.gz", "validation")):
        rows = [json.loads(l) for l in gzip.open(hf_file(cfg, "helpsteer2", fname))]
        for i, row in enumerate(rows):
            gid = "hs2:" + hashlib.sha256(row["prompt"].encode()).hexdigest()[:16]
            split = "test" if official == "validation" else ("val" if stable_fraction(seed, "hs2", gid) < c["val_fraction"] else "train")
            ctx = hs2_context(row["prompt"], row["response"], c["prompt_chars"], c["response_chars"])
            for a in c["attributes"]:
                ins, levels = HS2[a]
                out[split].append(rec(split, "score", "helpsteer2", a, ctx, ins, row[a], gid,
                                      prov(cfg, "helpsteer2", official, f"{official}-{i}", transform="krun_one/score_hs2/1"),
                                      levels=levels, metadata={"attribute": a}, id_suffix=f"/{a}"))
    # official validation prompts must not occur in train/val (HelpSteer2 pairs share prompts within a split)
    test_g = {r["group_id"] for r in out["test"]}
    for s in ("train", "val"):
        before = len(out[s])
        out[s] = [r for r in out[s] if r["group_id"] not in test_g]
        stats[f"score/helpsteer2/{s}/dropped_prompt_overlap_with_test"] = before - len(out[s])
    for split in list(out):
        out[split] = cap(out[split], c["cap"][split], seed, f"hs2-{split}")
        stats[f"score/helpsteer2/{split}"] = {"n": len(out[split]),
                                             "by_attribute": dict(Counter(r["suite"] for r in out[split]))}
    return out


# ======================================================================================== external eval-only
SST5_LEVELS = ["Very negative", "Negative", "Neutral", "Positive", "Very positive"]


def build_sst5(cfg, stats):
    out = []
    for i, d in enumerate(read_jsonl(hf_file(cfg, "sst5", "test.jsonl"))):
        out.append(rec("test", "score", "sst5", "external", d["text"], "What is the sentiment of this movie review sentence?",
                       d["label"], f"sst5:test-{i}", prov(cfg, "sst5", "test", f"test-{i}", transform="krun_one/score_sst5/1"),
                       levels=SST5_LEVELS, metadata={"attribute": "sentiment", "eval_only": True}))
    stats["score/sst5/test"] = {"n": len(out), "targets": dict(Counter(r["target"] for r in out))}
    return out


def build_boolq(cfg, stats):
    df = pd.read_parquet(hf_file(cfg, "boolq", "data/validation-00000-of-00001.parquet"))
    out = []
    for i, row in df.iterrows():
        q = row.question.strip()
        q = q[0].upper() + q[1:] + ("" if q.endswith("?") else "?")
        out.append(rec("test", "noul", "boolq", "external", row.passage, q, int(bool(row.answer)), f"boolq:val-{i}",
                       prov(cfg, "boolq", "validation", f"val-{i}", transform="krun_one/noul_boolq/1"),
                       metadata={"proposition_id": "boolq", "kind": "external", "eval_only": True}))
    out = cap(out, cfg["noul"]["boolq_test_cap"], cfg["seed"], "boolq")
    stats["noul/boolq/test"] = dict(Counter(r["target"] for r in out))
    return out


# ============================================================================= noul v2: MHS items + HelpSteer2 quality
MHS_NOUL = {  # proposition per Likert item; median >= 3 -> true, <= 1 -> false, 2 (neutral / unsure) -> not used
    "sentiment": "Is this comment negative towards the group it mentions?",
    "respect": "Is this comment disrespectful towards the group it mentions?",
    "insult": "Does this comment insult the group it mentions?",
    "humiliate": "Does this comment humiliate or demean the group it mentions?",
    "status": "Does this comment claim that the group it mentions is inferior to others?",
    "dehumanize": "Does this comment deny the humanity of the group it mentions, for example by comparing it to animals or disease?",
    "violence": "Does this comment call for or threaten violence against the group it mentions?",
    "genocide": "Does this comment support killing all or most members of the group it mentions?",
    "attack_defend": "Does this comment attack the group it mentions?",
}
HS2_NOUL = {  # attribute -> (proposition, true if rating in, false if rating in); middle ratings are not used
    "helpfulness": ("Does the response fully satisfy the user's request?", {4}, {0, 1, 2}),
    "correctness": ("Is the response factually correct and complete?", {4}, {0, 1, 2}),
    "coherence": ("Is the response clear and coherent?", {4}, {0, 1, 2}),
    "complexity": ("Does writing this response require specialized or expert knowledge?", {3, 4}, {0, 1}),
    "verbosity": ("Is the response longer or more detailed than the prompt calls for?", {3, 4}, {0, 1}),
}


def build_noul_mhs_items(cfg, comments, stats):
    c = cfg["noul"]["mhs_items"]
    out = defaultdict(list)
    for cm in comments:
        split, oid = cm["split"], str(cm["cid"])
        for item, ins in MHS_NOUL.items():
            v = cm["items"][item]
            if not v:
                continue
            med = median_half_up(v)
            y = 1 if med >= 3 else 0 if med <= 1 else None
            unseen = item in c["unseen_propositions"]
            if y is None or (split == "train" and unseen):
                continue
            out[split].append(rec(split, "noul", "mhs_items", "positive" if y else "negative", cm["text"], ins, y,
                                  f"mhs:{oid}", prov(cfg, "mhs", "train", oid, transform="krun_one/noul_mhs_items/1",
                                                     n_annotators=cm["n_annot"]),
                                  metadata={"proposition_id": f"mhs:{item}", "kind": "positive" if y else "negative",
                                            "annotations": v, "unseen_proposition": unseen}, id_suffix=f"/noul-{item}"))
    for split in list(out):
        out[split] = cap(out[split], c["cap"][split], cfg["seed"], f"mhs-items-noul-{split}")
        stats[f"noul/mhs_items/{split}"] = dict(Counter(f"{r['metadata']['proposition_id']}={r['target']}" for r in out[split]))
    return out


def build_noul_hs2(cfg, stats):
    c = cfg["noul"]["helpsteer2"]
    sc = cfg["score"]["helpsteer2"]
    seed = cfg["seed"]
    out = defaultdict(list)
    for fname, official in (("train.jsonl.gz", "train"), ("validation.jsonl.gz", "validation")):
        rows = [json.loads(l) for l in gzip.open(hf_file(cfg, "helpsteer2", fname))]
        for i, row in enumerate(rows):
            gid = "hs2:" + hashlib.sha256(row["prompt"].encode()).hexdigest()[:16]
            split = "test" if official == "validation" else ("val" if stable_fraction(seed, "hs2", gid) < sc["val_fraction"] else "train")
            ctx = hs2_context(row["prompt"], row["response"], sc["prompt_chars"], sc["response_chars"])
            for a, (ins, pos, neg) in HS2_NOUL.items():
                y = 1 if row[a] in pos else 0 if row[a] in neg else None
                unseen = a in c["unseen_propositions"]
                if y is None or (split == "train" and unseen):
                    continue
                out[split].append(rec(split, "noul", "helpsteer2", "positive" if y else "negative", ctx, ins, y, gid,
                                      prov(cfg, "helpsteer2", official, f"{official}-{i}", transform="krun_one/noul_hs2/1"),
                                      metadata={"proposition_id": f"hs2:{a}", "kind": "positive" if y else "negative",
                                                "rating": row[a], "unseen_proposition": unseen}, id_suffix=f"/noul-{a}"))
    for split in list(out):
        out[split] = cap(out[split], c["cap"][split], seed, f"hs2-noul-{split}")
        stats[f"noul/helpsteer2/{split}"] = dict(Counter(f"{r['metadata']['proposition_id']}={r['target']}" for r in out[split]))
    return out


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "primitives_data.yaml"))
    cfg = yaml.safe_load(Path(ap.parse_args().config).read_text())
    out_dir = PROJECT / cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    noul, score = defaultdict(list), defaultdict(list)
    for part in (build_noul_intents(cfg, stats), build_noul_tools(cfg, stats), build_noul_civil(cfg, stats)):
        for s, v in part.items():
            noul[s] += v
    print("noul intents/tools/civil done", flush=True)
    comments = load_mhs(cfg)
    mn, ms = build_mhs(cfg, comments, stats)
    for s, v in mn.items():
        noul[s] += v
    if "mhs_items" in cfg["noul"]:  # v2
        for s, v in build_noul_mhs_items(cfg, comments, stats).items():
            noul[s] += v
    if "helpsteer2" in cfg["noul"]:  # v2
        for s, v in build_noul_hs2(cfg, stats).items():
            noul[s] += v
    for s, v in ms.items():
        score[s] += v
    for s, v in build_hs2(cfg, stats).items():
        score[s] += v
    score["test"] += build_sst5(cfg, stats)
    noul["test"] += build_boolq(cfg, stats)
    mixed = build_mixed(cfg, comments, stats)

    # split hygiene: a group_id may appear in exactly one split
    for name, d in (("noul", noul), ("score", score)):
        where = defaultdict(set)
        for s, rows in d.items():
            for r in rows:
                where[r["group_id"]].add(s)
        bad = [g for g, ss in where.items() if len(ss) > 1]
        if bad:
            raise SystemExit(f"{name}: {len(bad)} groups in more than one split, e.g. {bad[:3]}")
        ids = [r["id"] for rows in d.values() for r in rows]
        if len(ids) != len(set(ids)):
            raise SystemExit(f"{name}: duplicate ids: {[i for i, c in Counter(ids).items() if c > 1][:5]}")
    files = {}
    for name, d in (("noul", noul), ("score", score)):
        for s in ("train", "val", "test"):
            p = out_dir / f"{name}_{s}.jsonl"
            write_jsonl(p, d[s])
            files[p.name] = {"n": len(d[s]), "sha256": sha(p),
                             "by_source": dict(Counter(r["source"] for r in d[s])),
                             "by_target": dict(Counter(r["target"] for r in d[s])),
                             "by_suite": dict(Counter(f"{r['source']}:{r['suite']}" for r in d[s]))}
    p = out_dir / "mixed_test.jsonl"
    write_jsonl(p, mixed)
    files[p.name] = {"n": len(mixed), "sha256": sha(p)}
    manifest = {"config": cfg, "files": files, "stats": stats}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (ROOT / "results" / cfg.get("stats_file", "primitives_dataset_stats.json")).write_text(json.dumps(manifest, indent=2) + "\n")
    for k, v in files.items():
        print(k, v["n"], v.get("by_source"))


if __name__ == "__main__":
    main()
