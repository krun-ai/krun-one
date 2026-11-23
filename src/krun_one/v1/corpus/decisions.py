"""Krun Decisions v1 — decision generators (context items -> krun.decision.v1 records).

Every record carries ``metadata``:
  category  choice_intent | tool_routing | noul | score | multi_label | ood      (the sampler's task buckets)
  origin    real | derived | mt | rules | llm       (never mixed: a decision on an MT context is "mt", etc.)
  stage     D1 | D2 | D3                              (curriculum; see stage_of)
  catalog / tool_source / k / ood_type / near_duplicates
Origins: ``real`` = the first (native) decision on an unchanged real context; further decisions on the same real
context (other k, hard negatives, gold removed, propositions, concatenations, cross-dataset sets) are ``derived``.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any, Sequence

import numpy as np

from .. import schema
from .mining import EmbeddingCache, LabelSpace, ToolPool, humanize, rng_for

K_DIST = {2: 0.10, 4: 0.20, 8: 0.25, 16: 0.22, 32: 0.13, 64: 0.08, 128: 0.02}
LANG3 = {"en": "en", "pt": "pt", "es": "es", "fr": "fr"}
INSTR = {
    "intent": {"en": ["Choose the option that best describes the intent of the user message.",
                      "Which intent does this message express?",
                      "Classify the user's request into one of the intents.",
                      "Pick the intent that matches what the user wants."],
               "pt": ["Escolha a opção que melhor descreve a intenção da mensagem do usuário."],
               "es": ["Elige la opción que mejor describe la intención del mensaje del usuario."],
               "fr": ["Choisissez l'option qui décrit le mieux l'intention du message de l'utilisateur."]},
    "tool": {"en": ["Choose the function that should be called to fulfil the user request.",
                    "Which tool should handle this request?",
                    "Select the API function to call for this user message.",
                    "Route the request to the right function."],
             "pt": ["Escolha a função que deve ser chamada para atender ao pedido do usuário."],
             "es": ["Elige la función que se debe llamar para atender la solicitud del usuario."],
             "fr": ["Choisissez la fonction à appeler pour répondre à la demande de l'utilisateur."]},
    "dialogue": {"en": ["Given the conversation, choose the function the assistant should call for the user's last message.",
                        "Which function does the user's latest turn require?"],
                 "pt": ["Dada a conversa, escolha a função que o assistente deve chamar para a última mensagem do usuário."],
                 "es": ["Dada la conversación, elige la función que el asistente debe llamar para el último mensaje del usuario."],
                 "fr": ["Compte tenu de la conversation, choisissez la fonction que l'assistant doit appeler pour le dernier message de l'utilisateur."]},
    "multi": {"en": ["Select every option that applies to the user message.",
                     "Which of these intents are present in the message? Select all that apply."],
              "pt": ["Selecione todas as opções que se aplicam à mensagem do usuário."],
              "es": ["Selecciona todas las opciones que se aplican al mensaje del usuario."],
              "fr": ["Sélectionnez toutes les options qui s'appliquent au message de l'utilisateur."]},
    "multi_tool": {"en": ["Select every function that should be called to fulfil the user request."],
                   "pt": ["Selecione todas as funções que devem ser chamadas para atender ao pedido."],
                   "es": ["Selecciona todas las funciones que se deben llamar para atender la solicitud."],
                   "fr": ["Sélectionnez toutes les fonctions à appeler pour répondre à la demande."]},
}
CONNECTORS = {"en": [". Also, ", " and also ", ". Then "], "pt": [". Além disso, ", " e também ", ". Depois, "],
              "es": [". Además, ", " y también ", ". Luego, "], "fr": [". Aussi, ", " et aussi ", ". Ensuite, "]}
PROB_PRIMARY = {"en-US": "en", "pt-BR": "pt", "pt-PT": "pt", "es-ES": "es", "es-419": "es", "fr-FR": "fr"}


def prim(lang: str) -> str:
    return (lang or "en").split("-")[0]


def pick_k(rng: np.random.Generator, kmax: int, kmin: int = 2, dist: dict = K_DIST) -> int:
    ks = [k for k in dist if kmin <= k <= kmax]
    if not ks:
        return max(kmin, min(kmax, 2))
    p = np.array([dist[k] for k in ks])
    k = int(rng.choice(ks, p=p / p.sum()))
    return k


def stage_of(rec: dict) -> str:
    sl, md = set(rec["slices"]), rec["metadata"]
    if (not rec["answerable"] or sl & {"minimal_pair", "counterfactual", "ambiguous", "ood_near"} or md["origin"] == "llm"
            or len(rec["candidates"]) >= 128 or md.get("near_duplicates", 0) >= 2 or md.get("partial")):
        return "D3"
    if (rec["cross_language"] and rec["task"]["type"] in ("choice", "multi")) or "typo" in sl or \
            (rec["candidate_set"]["strategy"].startswith("hard") and len(rec["candidates"]) >= 8) or md.get("dialogue"):
        return "D2"
    return "D1"


class Builder:
    def __init__(self, emb: EmbeddingCache, spaces: dict[str, LabelSpace], tools: ToolPool,
                 intent_desc: dict[tuple[str, str], str], desc_tr: dict[tuple[str, str, str], str], seed: int):
        self.emb, self.spaces, self.tools, self.seed = emb, spaces, tools, seed
        self.intent_desc, self.desc_tr = intent_desc, desc_tr
        self.n_real_used: set[str] = set()
        self.protected_names: set[str] = set()   # normalized names of KrunBench v1 protected unseen tools
        self.counter = defaultdict(int)

    # -------------------------------------------------------------------------------------------- record
    def rec(self, c: dict, *, qtype: str, family: str, instr_family: str, candidates: list[dict], labels: list[str],
            slices: Sequence[str], strategy: str, category: str, tag: str, rng, ctx_text: str | None = None,
            extra_md: dict | None = None, instructions: str | None = None, criteria=None, origin: str | None = None,
            distribution=None, ins_lang: str | None = None) -> dict:
        ids = [x["id"] for x in candidates]
        if qtype in ("choice", "multi") and (len(ids) < 2 or len(set(ids)) != len(ids)):
            return None                         # degenerate candidate set: never emitted
        lang = c["lang"]
        if instructions is None:
            il = ins_lang or ("en" if rng.random() < 0.8 or prim(lang) == "en" else prim(lang))
            instructions = str(rng.choice(INSTR[instr_family].get(il) or INSTR[instr_family]["en"]))
            ins_lang = il
        if origin is None:
            if c.get("origin") == "mt":
                origin = "mt"
            else:
                key = f"{c['source']}:{c['oid']}"
                origin = "derived" if key in self.n_real_used else "real"
                self.n_real_used.add(key)
        self.counter[(category, tag)] += 1
        rid = f"kd1/{category}/{c['source']}/{c['oid']}/{tag}/{self.counter[(category, tag)]}"
        rec = {"schema_version": schema.SCHEMA_VERSION, "id": rid, "split": c.get("rec_split", "train"),
               "task": {"type": qtype, "instructions": instructions, "instructions_language": ins_lang or "en",
                        "criteria": criteria, "family": family},
               "context": {"text": ctx_text or c["text"], "language": lang, "code_switching": False,
                           "register": None, "noise": []},
               "candidates": candidates, "target": {"labels": labels, "distribution": distribution},
               "answerable": bool(labels) if qtype in ("choice", "multi") else True,
               "semantic_group_id": c["group"], "split_group_id": c["split_group"],
               "candidate_set": {"strategy": strategy, "k": len(candidates)}, "slices": sorted(set(slices)),
               "minimal_pair": None,
               "provenance": {"source_dataset": c["source"], "revision": c.get("revision", "see configs/v1/sources.yaml"),
                              "license": c.get("license", "see configs/v1/sources.yaml"), "original_split": c["orig_split"],
                              "original_id": c["oid"], "transforms": list(c.get("transforms", [])) + [f"kd1:{category}:{tag}"],
                              "synthetic": None},
               "metadata": {"category": category, "origin": origin, "catalog": c.get("catalog"), "k": len(candidates),
                            **(extra_md or {})}}
        rec = schema.derive_fields(rec)
        rec["metadata"]["stage"] = stage_of(rec)
        return rec

    # -------------------------------------------------------------------------------------------- helpers
    def ctx_emb(self, text: str) -> np.ndarray:
        return self.emb.get([text])[0]

    def fn_id(self, label: str) -> str:
        """Function name of an intent rendered as a tool. A name equal to a KrunBench v1 protected unseen tool
        (e.g. Bitext `review`) gets a suffix, so protected tool names never appear on the training side."""
        from ..leakage import normalize

        return label + "_request" if normalize(label) in self.protected_names else label

    def label_cand(self, catalog: str, label: str, mode: str, loc: str | None = None) -> dict:
        """mode: bare (id only) | fn (function + EN description) | fn_tr (function + translated description)."""
        if mode == "bare":
            return {"id": label, "label": label, "description": None, "language": "en"}
        fid = self.fn_id(label)
        d_en = self.intent_desc.get((catalog, label))
        if mode == "fn_tr" and loc and (catalog, label, loc) in self.desc_tr:
            return {"id": fid, "label": fid, "description": self.desc_tr[(catalog, label, loc)], "language": loc}
        return {"id": fid, "label": fid, "description": d_en, "language": "en"}

    def tool_cand(self, t: tuple[str, str], loc: str | None = None) -> dict:
        if loc and (t[0], t[1], loc) in self.desc_tr:
            return {"id": t[0], "label": t[0], "description": self.desc_tr[(t[0], t[1], loc)], "language": loc}
        return {"id": t[0], "label": t[0], "description": t[1], "language": "en"}

    # -------------------------------------------------------------------------------------------- intents
    def intent_choice(self, c: dict, rng, *, category="choice_intent", gold_removed=False, mode="bare",
                      schema_loc: str | None = None, k: int | None = None, hard: bool | None = None,
                      tag="core") -> dict | None:
        sp = self.spaces.get(c["catalog"])
        if sp is None or c["label"] not in sp.pos:
            return None
        hard = (rng.random() < 0.65) if hard is None else hard
        kmax = len(sp.labels) - (1 if gold_removed else 0)
        k = min(k or pick_k(rng, kmax), kmax)
        e = self.ctx_emb(c.get("mining_text") or c["text"])
        negs = sp.negatives(c["label"], k - (0 if gold_removed else 1), e, rng, hard=hard)
        labs = negs if gold_removed else negs + [c["label"]]
        rng.shuffle(labs)
        if mode != "bare" and any((c["catalog"], l) not in self.intent_desc for l in labs):
            mode = "bare"
        cands = [self.label_cand(c["catalog"], l, mode, schema_loc) for l in labs]
        fam = "tool" if mode != "bare" else "intent"
        sl = ["gold_removed", "ood_near"] if gold_removed else []
        gold_id = c["label"] if mode == "bare" else self.fn_id(c["label"])
        return self.rec(c, qtype="choice", family=fam, instr_family="tool" if fam == "tool" else "intent",
                        candidates=cands, labels=[] if gold_removed else [gold_id], slices=sl,
                        strategy="hard_v2" if hard else "random", category="ood" if gold_removed else category,
                        tag=tag, rng=rng, extra_md={"ood_type": "gold_removed"} if gold_removed else
                        ({"tool_source": "intent_as_function"} if fam == "tool" else None))

    # -------------------------------------------------------------------------------------------- tools
    def tool_choice(self, c: dict, rng, *, gold_removed=False, schema_loc: str | None = None, k: int | None = None,
                    hard: bool | None = None, tag="core") -> dict | None:
        native = [tuple(t) for t in c["tools"]]
        gold = next((t for t in native if t[0] == c["gold_tool"]), None) if c.get("gold_tool") else None
        if gold is None and not gold_removed and c.get("gold_tool"):
            return None
        hard = (rng.random() < 0.7) if hard is None else hard
        k = k or pick_k(rng, 128)
        base = [t for t in native if t != gold]
        need = k - len(base) - (0 if gold_removed or gold is None else 1)
        e = self.ctx_emb(c.get("mining_text") or c["text"])
        pool, nd = self.tools.negatives(gold, e, max(0, need), rng, exclude_names={t[0] for t in native}, hard=hard)
        chosen = base + pool
        if not gold_removed and gold is not None:
            chosen = chosen + [gold]
        if len(chosen) < 2:
            return None
        rng.shuffle(chosen)
        cands = [self.tool_cand(t, schema_loc) for t in chosen]
        labels = [] if gold_removed or gold is None else [gold[0]]
        sl = (["gold_removed", "ood_near"] if gold_removed else []) + (["near_duplicate"] if nd else [])
        return self.rec(c, qtype="choice", family="tool", instr_family="tool", candidates=cands, labels=labels,
                        slices=sl, strategy=("native+hard_v2" if hard else "native+random"),
                        category="ood" if gold_removed else "tool_routing", tag=tag, rng=rng,
                        extra_md={"tool_source": "toolace", "near_duplicates": nd,
                                  **({"ood_type": "gold_removed"} if gold_removed else {})})

    def tool_native_irrelevance(self, c: dict, rng, schema_loc=None, tag="native") -> dict | None:
        native = [tuple(t) for t in c["tools"]]
        k = pick_k(rng, 64, kmin=2)
        e = self.ctx_emb(c.get("mining_text") or c["text"])
        extra, _ = self.tools.negatives(None, e, max(0, k - len(native)), rng, exclude_names={t[0] for t in native},
                                        hard=True) if rng.random() < 0.5 else ([], 0)
        chosen = native + extra
        rng.shuffle(chosen)
        # pool tools near the query could actually fulfil it: only native lists are guaranteed irrelevant -> keep
        # added tools only if they are less similar to the query than the least similar native tool
        if extra:
            s = {t: float(self.tools.doc_emb(t) @ e) for t in chosen if t in self.tools.pos}
            floor = min((s.get(t, 1.0) for t in native), default=1.0)
            chosen = [t for t in chosen if t in native or s.get(t, 1.0) < floor]
        if len(chosen) < 2:
            return None
        return self.rec(c, qtype="choice", family="tool", instr_family="tool",
                        candidates=[self.tool_cand(t, schema_loc) for t in chosen], labels=[],
                        slices=["native_irrelevance"], strategy="native", category="ood", tag=tag, rng=rng,
                        extra_md={"tool_source": "toolace", "ood_type": "native_irrelevance"})

    def tool_multi(self, c: dict, rng, schema_loc=None) -> dict | None:
        native = [tuple(t) for t in c["tools"]]
        return self.rec(c, qtype="multi", family="tool", instr_family="multi_tool",
                        candidates=[self.tool_cand(t, schema_loc) for t in native], labels=list(c["calls"]),
                        slices=[], strategy="native", category="multi_label", tag="toolace_multi", rng=rng,
                        extra_md={"tool_source": "toolace"})

    # -------------------------------------------------------------------------------------------- MultiWOZ
    def multiwoz_choice(self, c: dict, mw: dict, rng, *, gold_removed=False, tag="route") -> dict | None:
        from .sources import render_dialogue

        intents = [(n, v["description"]) for n, v in mw.items()]
        gold = c["labels"][0]
        others = [t for t in intents if t[0] != gold]
        k = int(rng.integers(2, len(intents) + (0 if gold_removed else 1)))
        chosen = list(rng.permutation(len(others)))[: k - (0 if gold_removed else 1)]
        cands = [others[i] for i in chosen] + ([] if gold_removed else [(gold, mw[gold]["description"])])
        rng.shuffle(cands)
        cands = [{"id": n, "label": n, "description": d, "language": "en"} for n, d in cands]
        text = render_dialogue(c)
        return self.rec(c, qtype="choice", family="tool", instr_family="dialogue" if c["history"] else "tool",
                        candidates=cands, labels=[] if gold_removed else [gold],
                        slices=["gold_removed", "ood_near"] if gold_removed else [], strategy="random",
                        category="ood" if gold_removed else "tool_routing", tag=tag, rng=rng, ctx_text=text,
                        extra_md={"tool_source": "multiwoz", "dialogue": bool(c["history"]),
                                  **({"ood_type": "gold_removed"} if gold_removed else {})})

    def multiwoz_multi(self, c: dict, mw: dict, rng) -> dict:
        from .sources import render_dialogue

        cands = [{"id": n, "label": n, "description": v["description"], "language": "en"} for n, v in mw.items()]
        rng.shuffle(cands)
        return self.rec(c, qtype="multi", family="tool", instr_family="multi_tool", candidates=cands,
                        labels=list(c["labels"]), slices=[], strategy="native", category="multi_label", tag="mw_multi",
                        rng=rng, ctx_text=render_dialogue(c), extra_md={"tool_source": "multiwoz", "dialogue": True})

    # -------------------------------------------------------------------------------------------- multi (intents)
    def intent_concat(self, a: dict, b: dict | None, rng, *, partial_from: dict | None = None) -> dict | None:
        """a (+ b) utterances joined with a language connector; labels {a, b}; partial: b comes from another catalog
        and is NOT in the candidate set (only a applies). a alone with gold removed -> empty set (unanswerable)."""
        sp = self.spaces.get(a["catalog"])
        if sp is None:
            return None
        L = prim(a["lang"])
        conn = str(rng.choice(CONNECTORS[L]))
        second = b or partial_from
        text = a["text"].rstrip(" .?!") + (conn + second["text"][0].lower() + second["text"][1:] if second else "")
        gold = [a["label"]] + ([b["label"]] if b and b["label"] != a["label"] else [])
        k = min(pick_k(rng, len(sp.labels), kmin=4), len(sp.labels))
        e = self.ctx_emb(a.get("mining_text") or a["text"])
        negs = sp.negatives(a["label"], k - len(gold), e, rng, hard=True, exclude=gold)
        labs = negs + gold
        rng.shuffle(labs)
        c = dict(a, group=a["group"] + "+" + (second["group"] if second else ""))
        return self.rec(c, qtype="multi", family="intent", instr_family="multi",
                        candidates=[self.label_cand(a["catalog"], l, "bare") for l in labs], labels=gold,
                        slices=[], strategy="hard_v2", category="multi_label", tag="concat" if not partial_from else "partial",
                        rng=rng, ctx_text=text, origin="derived" if a.get("origin") != "mt" else "mt",
                        extra_md={"partial": bool(partial_from), "second_group": second["group"] if second else None})
