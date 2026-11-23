"""``krun.decision.v1`` — one record format for choice / noul / score / multi / OOD decisions.

Formal description and rationale: docs/KRUN_DECISIONS_V1_SPEC.md. This module is the executable version:
``validate(rec)`` returns a list of error strings (empty = valid) and ``from_krunbench_v0`` /
``from_primitive_v0`` convert the existing Krun One data so v0.3's corpus is the seed of Krun Decisions v1.

Record (fields marked * are required)::

    schema_version*   "krun.decision.v1"
    id*               globally unique
    split*            train | validation | test | holdout | protected
    task*             {type*: choice|noul|score|multi, instructions, instructions_language, criteria (noul),
                       domain, family: intent|tool|proposition|rating|...}
    context*          {text*, language* (BCP-47: en-US, pt-BR, pt-PT, es-ES, es-419, fr-FR, fr-CA, und, mul),
                       code_switching: bool, register: formal|informal|slang, noise: [typo, asr, ...],
                       OPTIONAL media (multimodal-ready; absent = text-only record, unchanged):
                       images:    [{ref*, mime*, sha256, width, height}]
                       documents: [{ref*, mime*, document_type, pages, text_extractable, visual_required}]
                       audio:     [{ref*, mime*, duration_s, language, transcript}]}
    modalities        optional derived list, e.g. ["text", "image"]; image_count, document_type, audio_language,
                       audio_duration, visual_required may be set in metadata (never required)
    candidates*       [{id*, label, description, language}]  — ordered for score (level i = index i);
                       noul: exactly [{"id": "false"}, {"id": "true"}]
    target*           {labels*: [candidate ids], distribution: {id: p} | null}
    answerable*       bool — choice: labels non-empty; multi: labels non-empty; noul/score: always true
    schema_language*  language of the schema side (instructions + candidates), BCP-47 primary tag or "mul"
    cross_language*   bool: context language != schema language (primary subtags; "mul" contexts count as cross)
    difficulty*       easy | medium | hard (derived: negative strategy, OOD kind, near-duplicates, minimal pairs)
    semantic_group_id* same meaning (translations, paraphrases of one source item)
    split_group_id*   the unit that must never cross splits (>= semantic group; minimal-pair sets, source item)
    candidate_set     {strategy* (= negative_strategy): native|pool|hard_v1|hard_v2|random|mixed|fixed, k,
                       hard_negative_ids, removed_gold}
    slices            [tags] e.g. cross_language, unseen_tool, unseen_label, near_duplicate, minimal_pair, typo,
                       slang, code_switching, ood_near, ood_far, native_irrelevance, gold_removed, permutation
    minimal_pair      {set_id, member, contrast} | null
    provenance*       {source_dataset*, hf_id, revision*, license*, original_split, original_id,
                       transforms: [..], synthetic: null | {teacher*, prompt_id*, verifier*, verdict*,
                       source_example_id*, transformation*, generated_utc}}
    metadata          free-form
"""

from __future__ import annotations

import re
from typing import Any, Mapping

SCHEMA_VERSION = "krun.decision.v1"
QTYPES = ("choice", "noul", "score", "multi")
SPLITS = ("train", "validation", "test", "holdout", "protected")
LANG_RE = re.compile(r"^(und|mul|[a-z]{2,3}(-[A-Z]{2}|-[0-9]{3})?)$")
FIRST_CLASS_LOCALES = ("en-US", "en-GB", "pt-BR", "pt-PT", "es-ES", "es-419", "es-MX", "fr-FR", "fr-CA")
SLICES = {"cross_language", "same_language", "unseen_tool", "unseen_label", "unseen_domain", "unseen_instruction",
          "near_duplicate", "minimal_pair", "typo", "slang", "code_switching", "regional", "ood_near", "ood_far",
          "native_irrelevance", "gold_removed", "cross_task", "permutation", "counterfactual", "ambiguous"}
DIFFICULTIES = ("easy", "medium", "hard")
SYNTHETIC_REQUIRED = ("teacher", "prompt_id", "verifier", "verdict", "source_example_id", "transformation")


def validate(rec: Mapping[str, Any]) -> list[str]:
    e: list[str] = []

    def need(obj, key, where):
        if not isinstance(obj, Mapping) or key not in obj or obj[key] is None:
            e.append(f"{where}.{key} missing")
            return False
        return True

    if rec.get("schema_version") != SCHEMA_VERSION:
        e.append("schema_version must be krun.decision.v1")
    for k in ("id", "split", "task", "context", "candidates", "target", "answerable", "semantic_group_id",
              "split_group_id", "provenance"):
        need(rec, k, "record")
    if e:
        return e
    if rec["split"] not in SPLITS:
        e.append(f"split {rec['split']!r} not in {SPLITS}")
    task, ctx, cands, tgt = rec["task"], rec["context"], rec["candidates"], rec["target"]
    t = task.get("type")
    if t not in QTYPES:
        e.append(f"task.type {t!r} not in {QTYPES}")
        return e
    if need(ctx, "text", "context") and not str(ctx["text"]).strip():
        e.append("context.text empty")
    if need(ctx, "language", "context") and not LANG_RE.match(str(ctx["language"])):
        e.append(f"context.language {ctx['language']!r} is not BCP-47 (xx, xx-YY, xx-419, und, mul)")
    if task.get("instructions_language") and not LANG_RE.match(str(task["instructions_language"])):
        e.append("task.instructions_language is not BCP-47")
    if not isinstance(cands, list) or len(cands) < 2:
        e.append("candidates: need >= 2")
        return e
    ids = [c.get("id") for c in cands]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        e.append("candidates: ids must be unique non-empty strings")
    for c in cands:
        if c.get("language") and not LANG_RE.match(str(c["language"])):
            e.append(f"candidate {c.get('id')!r}: language not BCP-47")
    labels = tgt.get("labels")
    if not isinstance(labels, list) or any(x not in ids for x in labels):
        e.append("target.labels must be a list of candidate ids")
        labels = []
    if len(set(labels)) != len(labels):
        e.append("target.labels has duplicates")
    dist = tgt.get("distribution")
    if dist is not None:
        if set(dist) - set(ids) or abs(sum(dist.values()) - 1.0) > 1e-4 or any(v < 0 for v in dist.values()):
            e.append("target.distribution must be a probability distribution over candidate ids")
    ans = rec["answerable"]
    if t == "choice":
        if len(labels) > 1:
            e.append("choice: at most one label")
        if ans != (len(labels) == 1):
            e.append("choice: answerable must equal (one label)")
    elif t == "multi":
        if ans != (len(labels) > 0):
            e.append("multi: answerable must equal (labels non-empty)")
    elif t == "noul":
        if ids != ["false", "true"]:
            e.append('noul: candidates must be [{"id": "false"}, {"id": "true"}]')
        if len(labels) != 1 or not ans:
            e.append("noul: exactly one label and answerable=true")
        if not str(task.get("instructions") or "").strip():
            e.append("noul: instructions (the proposition) required")
    elif t == "score":
        if not 2 <= len(cands) <= 16:
            e.append("score: 2..16 levels")
        if len(labels) != 1 or not ans:
            e.append("score: exactly one label and answerable=true")
        if not str(task.get("instructions") or "").strip():
            e.append("score: instructions required")
    for k in ("schema_language", "cross_language", "difficulty"):
        need(rec, k, "record")
    if rec.get("schema_language") is not None and not LANG_RE.match(str(rec["schema_language"])):
        e.append("schema_language is not BCP-47")
    if rec.get("cross_language") is not None and not isinstance(rec["cross_language"], bool):
        e.append("cross_language must be a bool")
    if rec.get("difficulty") is not None and rec["difficulty"] not in DIFFICULTIES:
        e.append(f"difficulty must be one of {DIFFICULTIES}")
    cs = rec.get("candidate_set")
    if not isinstance(cs, Mapping) or not cs.get("strategy"):
        e.append("candidate_set.strategy (negative strategy) missing")
    elif isinstance(rec.get("cross_language"), bool) and rec.get("schema_language") and ctx.get("language"):
        if rec["cross_language"] != _is_cross(ctx["language"], rec["schema_language"]):
            e.append("cross_language inconsistent with context.language / schema_language")
    e.extend(validate_media(ctx))
    if rec.get("modalities") is not None:
        mods = rec["modalities"]
        if not isinstance(mods, list) or set(mods) - MODALITIES or "text" not in mods:
            e.append(f"modalities must be a list over {sorted(MODALITIES)} including 'text'")
        elif set(mods) != set(derive_modalities(ctx)):
            e.append("modalities inconsistent with context media")
    for s in rec.get("slices") or []:
        if s not in SLICES:
            e.append(f"unknown slice {s!r}")
    prov = rec["provenance"]
    for k in ("source_dataset", "revision", "license"):
        need(prov, k, "provenance")
    syn = prov.get("synthetic") if isinstance(prov, Mapping) else None
    if syn is not None:
        for k in SYNTHETIC_REQUIRED:
            need(syn, k, "provenance.synthetic")
        if syn.get("verdict") not in (None, "accepted") and rec["split"] != "protected":
            e.append("synthetic example without an accepted verifier verdict")
    mp = rec.get("minimal_pair")
    if mp is not None:
        for k in ("set_id", "member"):
            need(mp, k, "minimal_pair")
    return e


# ------------------------------------------------------------------------------------------------ media (optional)
MODALITIES = {"text", "image", "document", "audio"}
_MEDIA = {"images": ("image/",), "documents": ("application/", "text/", "image/"), "audio": ("audio/",)}


def validate_media(ctx: Mapping[str, Any]) -> list[str]:
    """Optional multimodal fields. A record without them is a text record and is valid exactly as before."""
    e: list[str] = []
    for key, prefixes in _MEDIA.items():
        items = ctx.get(key)
        if items is None:
            continue
        if not isinstance(items, list):
            e.append(f"context.{key} must be a list")
            continue
        for i, it in enumerate(items):
            if not isinstance(it, Mapping) or not it.get("ref") or not it.get("mime"):
                e.append(f"context.{key}[{i}] needs ref and mime")
                continue
            if not str(it["mime"]).startswith(prefixes):
                e.append(f"context.{key}[{i}].mime {it['mime']!r} not valid for {key}")
            if key == "audio":
                if it.get("duration_s") is not None and not float(it["duration_s"]) > 0:
                    e.append(f"context.audio[{i}].duration_s must be > 0")
                if it.get("language") and not LANG_RE.match(str(it["language"])):
                    e.append(f"context.audio[{i}].language not BCP-47")
    return e


def derive_modalities(ctx: Mapping[str, Any]) -> list[str]:
    mods = ["text"]
    for key, name in (("images", "image"), ("documents", "document"), ("audio", "audio")):
        if ctx.get(key):
            mods.append(name)
    return mods


# ------------------------------------------------------------------------------------------------ derived fields
def _primary(lang: str | None) -> str:
    return (lang or "und").split("-")[0]


def _is_cross(context_lang: str, schema_lang: str) -> bool:
    c, s = _primary(context_lang), _primary(schema_lang)
    return c == "mul" or s == "mul" or c != s


def derive_fields(rec: dict) -> dict:
    """Fill schema_language / cross_language / difficulty and the same_/cross_language slice from the record."""
    task = rec["task"]
    langs = {_primary(c.get("language")) for c in rec["candidates"] if c.get("language")}
    if task["type"] in ("noul", "score") or not langs:
        langs = {_primary(task.get("instructions_language") or "en")}
    rec["schema_language"] = langs.pop() if len(langs) == 1 else "mul"
    rec["cross_language"] = _is_cross(rec["context"]["language"], rec["schema_language"])
    sl = set(rec.get("slices") or []) - {"cross_language", "same_language"}
    sl.add("cross_language" if rec["cross_language"] else "same_language")
    rec["slices"] = sorted(sl)
    strat = str((rec.get("candidate_set") or {}).get("strategy") or "")
    if sl & {"near_duplicate", "minimal_pair", "ood_near", "native_irrelevance", "gold_removed", "counterfactual"} \
            or "hard" in strat or "native" in strat:
        rec["difficulty"] = "hard"
    elif strat.startswith("random") or sl & {"ood_far", "cross_task"}:
        rec["difficulty"] = "easy"
    else:
        rec["difficulty"] = "medium"
    return rec


# ------------------------------------------------------------------------------------------------ converters
_SPLIT_MAP = {"train": "train", "validation": "validation", "val": "validation", "test": "test"}
LICENSES = {  # as recorded by the v0/v0.3 pipelines (results/dataset_stats.json, primitives manifests)
    "banking77": "CC-BY-4.0", "massive": "CC-BY-4.0", "xlam": "CC-BY-4.0", "toolace": "Apache-2.0",
    "xlam_irrelevance": "CC-BY-4.0", "helpsteer2": "CC-BY-4.0", "mhs": "CC-BY-4.0", "mhs_items": "CC-BY-4.0",
    "civil_comments": "CC0-1.0", "bfcl": "Apache-2.0", "hwu64": "UNCONFIRMED", "boolq": "CC-BY-SA-3.0",
    "sst5": "UNCONFIRMED",
}
_LOCALE = {"en-US": "en-US", "pt-PT": "pt-PT", "es-ES": "es-ES", "fr-FR": "fr-FR", None: "en-US"}


def _suite_slices(suite: str | None, kind: str | None) -> list[str]:
    m = {"ood_label_removed": ["ood_near", "gold_removed"], "ood_cross_dataset": ["ood_far", "cross_task"],
         "ood_native": ["native_irrelevance"], "permutation": ["permutation"], "tool_removed": ["gold_removed"],
         "label_removed": ["ood_near", "gold_removed"], "cross_dataset": ["ood_far"], "cross_task": ["cross_task"],
         "native_irrelevance": ["native_irrelevance"]}
    return sorted(set(m.get(suite or "", []) + m.get(kind or "", [])))


def from_krunbench_v0(d: Mapping[str, Any], split: str | None = None) -> dict[str, Any]:
    """krunbench.decision.v0 (choice; intents and tools) -> krun.decision.v1."""
    prov = d.get("provenance") or {}
    ds = prov.get("dataset") or "unknown"
    desc = d.get("option_descriptions") or {}
    lang = _LOCALE.get(d.get("language"), d.get("language") or "en-US")
    group = prov.get("group_id") or d["id"]
    meta = dict(d.get("metadata") or {})
    slices = _suite_slices(d.get("suite"), meta.get("ood_kind") or meta.get("kind"))
    if meta.get("tool_seen") is False:
        slices.append("unseen_tool")
    return derive_fields({
        "schema_version": SCHEMA_VERSION,
        "id": "v1/" + d["id"],
        "split": split or _SPLIT_MAP.get(d.get("split", "train"), "train"),
        "task": {"type": "choice", "instructions": d.get("instructions"), "instructions_language": "en",
                 "family": "tool" if d.get("task") == "tool_routing" else "intent"},
        "context": {"text": d["context"], "language": lang},
        "candidates": [{"id": o, "label": o, "description": desc.get(o), "language": "en"} for o in d["options"]],
        "target": {"labels": [d["answer"]] if d.get("answerable") else [], "distribution": None},
        "answerable": bool(d.get("answerable")),
        "semantic_group_id": group,
        "split_group_id": group,
        "candidate_set": {"strategy": meta.get("negatives") or (d.get("generation") or {}).get("negative_strategy")
                          or "unknown", "k": len(d["options"])},
        "slices": sorted(set(slices)),
        "minimal_pair": None,
        "provenance": {"source_dataset": ds, "hf_id": prov.get("hf_id"), "revision": prov.get("revision"),
                       "license": LICENSES.get(ds, "UNCONFIRMED"), "original_split": prov.get("original_split"),
                       "original_id": prov.get("original_id"), "transforms": ["from:krunbench.decision.v0"],
                       "synthetic": None},
        "metadata": {"v0_suite": d.get("suite"), "v0_generation": d.get("generation"), **meta},
    })


def from_primitive_v0(d: Mapping[str, Any], split: str | None = None) -> dict[str, Any]:
    """krun.primitive.v0 (noul / score) -> krun.decision.v1."""
    prov = d.get("provenance") or {}
    meta = dict(d.get("metadata") or {})
    q = d["qtype"]
    if q == "noul":
        cands = [{"id": "false"}, {"id": "true"}]
        labels = ["true" if int(d["target"]) == 1 else "false"]
    else:
        cands = [{"id": str(i), "label": lv} for i, lv in enumerate(d["levels"])]
        labels = [str(int(d["target"]))]
    lang = _LOCALE.get(meta.get("language"), meta.get("language") or "en-US")
    group = d.get("group_id") or d["id"]
    slices = ["unseen_label"] if meta.get("unseen_proposition") else []
    slices += _suite_slices(None, meta.get("kind"))
    if prov.get("tool"):  # noul tool propositions: keep which tool the proposition is about
        meta["tool"] = prov["tool"]
        ins = str(d.get("instructions") or "")
        if "Function description:" in ins:
            meta["tool_description"] = ins.split("Function description:", 1)[1].strip()
    if meta.get("unseen_tool"):
        slices.append("unseen_tool")
    return derive_fields({
        "schema_version": SCHEMA_VERSION,
        "id": "v1/" + d["id"],
        "split": split or _SPLIT_MAP.get(d.get("split", "train"), "train"),
        "task": {"type": q, "instructions": d.get("instructions"), "instructions_language": "en",
                 "criteria": d.get("criteria"), "family": "proposition" if q == "noul" else "rating"},
        "context": {"text": d["context"], "language": lang},
        "candidates": cands,
        "target": {"labels": labels, "distribution": None},
        "answerable": True,
        "semantic_group_id": group,
        "split_group_id": group,
        "candidate_set": {"strategy": {"hard_negative": "native", "hard_negative_irrelevance": "native",
                                       "cross_domain": "random"}.get(meta.get("kind"), "fixed"), "k": len(cands)},
        "slices": sorted(set(slices)),
        "minimal_pair": None,
        "provenance": {"source_dataset": d.get("source") or prov.get("dataset"), "hf_id": prov.get("hf_id"),
                       "revision": prov.get("revision"), "license": prov.get("license") or LICENSES.get(d.get("source"), "UNCONFIRMED"),
                       "original_split": prov.get("original_split"), "original_id": prov.get("original_id"),
                       "transforms": ["from:krun.primitive.v0"], "synthetic": None},
        "metadata": {"v0_suite": d.get("suite"), **meta},
    })
