"""Krun One v1.1 R1 / R2 training: items, objective and trainable sets.

Two kinds of training items share one batch stream:

* ``tech``   an example of the technical corpus (one problem in one rendering; see scripts/v11/build_corpus.py);
* ``replay`` a ``krun.decision.v1`` record of the data v1 was trained on, optionally with the frozen v1's own logits.

Objective (weights come from the registered config, never from this file)::

    L = L_current                                  v1's objective, unchanged, on every item
      + l_pair   * L_pairwise                      tech choice: log(1 + exp(-(z_gold - z_neg))), hard negatives weighted 1,
                                                   the others ``easy_neg_weight``
      + l_margin * L_margin                        tech choice: max(0, margin - (z_gold - z_hardest_negative))
      + l_value  * (L_Q_util + 0.5 L_Q_cons + 0.5 L_V)   auxiliary heads, training only:
                                                   Q util = outcome rank of each candidate in [0, 1] (BCE with logits)
                                                   Q cons = z-scored numeric consequence of each action   (Huber)
                                                   V      = z-scored numeric facts about the state         (Huber)
      + l_kd     * L_replay_kd                     replay: KL(v1 distribution || model) (multi: BCE to v1's sigmoid)

Trainable sets: ``head`` (R1: type embedding, decision head, scorer, answer head, adapter, interaction block, aux
heads; the encoder is frozen) and ``last_n`` (R2: the same + the last n encoder blocks and the final norm).
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..v1.arch import QTYPES_V1
from ..v1.losses import LossConfig, decision_loss
from ..v1.model import chunk_index, collate_chunks, merge_chunks

QD, VD = 4, 4                                     # aux dims: 1 util + QD consequences per candidate; VD state facts
V1_LOSS = LossConfig(unanswerable_mode="oe", w_oe=0.5, w_ans=0.0, w_apply=0.25, w_multi=1.0)   # configs/v1/stage2.yaml


@dataclass(frozen=True)
class Objective:
    l_pair: float = 0.0
    l_margin: float = 0.0
    margin: float = 3.0
    l_value: float = 0.0
    l_kd: float = 0.0
    hard_negatives: int = 3
    easy_neg_weight: float = 0.25


# ------------------------------------------------------------------------------------------------ items
def cons_keys(examples: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, list[str]]]:
    """Per slice: the (sorted, capped) consequence keys and state keys that the aux heads predict."""
    q, v = defaultdict(set), defaultdict(set)
    for e in examples:
        for c in e["candidates"]:
            q[e["slice"]].update((c.get("cons") or {}).keys())
        v[e["slice"]].update((e.get("state") or {}).keys())
    return {"q": {s: sorted(k)[:QD] for s, k in q.items()}, "v": {s: sorted(k)[:VD] for s, k in v.items()}}


def cons_stats(examples: Sequence[Mapping[str, Any]], keys: dict) -> dict[str, dict[str, list[float]]]:
    """mean / std per (slice, key), fitted on the training examples only."""
    acc = defaultdict(list)
    for e in examples:
        for c in e["candidates"]:
            for k in keys["q"].get(e["slice"], []):
                if k in (c.get("cons") or {}):
                    acc[("q", e["slice"], k)].append(c["cons"][k])
        for k in keys["v"].get(e["slice"], []):
            if k in (e.get("state") or {}):
                acc[("v", e["slice"], k)].append(e["state"][k])
    return {"|".join(k): [float(np.mean(v)), float(np.std(v) + 1e-6)] for k, v in acc.items()}


def options(e: Mapping[str, Any]) -> list[dict]:
    if e["render"] == "raw":
        return [{"id": c["name"]} for c in e["candidates"]]
    return [{"id": c["name"], "description": c["feat"]} for c in e["candidates"]]


def encode_tech(ser, e: Mapping[str, Any]):
    qt = e.get("qtype", "choice")
    if qt == "noul":
        return ser.encode("noul", e["context"], [{"id": "false"}, {"id": "true"}], e["instructions"])
    return ser.encode(qt, e["context"], options(e), e["instructions"])


def tech_item(ser, e: Mapping[str, Any], keys: dict | None = None, stats: dict | None = None) -> dict:
    q = encode_tech(ser, e)
    k = len(e["candidates"])
    qt = e.get("qtype", "choice")
    tgt = np.zeros(k, dtype=np.float32)
    if qt == "multi":
        tgt[e["gold_multi"]] = 1.0
    elif e["answerable"]:
        if e["gold"] is None:      # tgt[None] would silently set every entry (the R1 / R2 corpus had this for swe_involved)
            raise ValueError(f"answerable item without a gold index: {e['id']}")
        tgt[e["gold"]] = 1.0
    vals = np.array([-1.0 if c["value"] is None else float(c["value"]) for c in e["candidates"]], dtype=np.float32)
    util = np.clip(vals, 0, None)
    util = util / util.max() if util.max() > 0 else util
    qc, qm = np.zeros((k, QD), np.float32), np.zeros((k, QD), np.float32)
    vv, vm = np.zeros(VD, np.float32), np.zeros(VD, np.float32)
    if keys is not None and stats is not None:
        for j, key in enumerate(keys["q"].get(e["slice"], [])):
            mu_sd = stats.get(f"q|{e['slice']}|{key}")
            for i, c in enumerate(e["candidates"]):
                if mu_sd and key in (c.get("cons") or {}):
                    qc[i, j], qm[i, j] = (c["cons"][key] - mu_sd[0]) / mu_sd[1], 1.0
        for j, key in enumerate(keys["v"].get(e["slice"], [])):
            mu_sd = stats.get(f"v|{e['slice']}|{key}")
            if mu_sd and key in (e.get("state") or {}):
                vv[j], vm[j] = (e["state"][key] - mu_sd[0]) / mu_sd[1], 1.0
    for c in q.chunks:
        c.ids = np.asarray(c.ids, dtype=np.int32)
    fam = e.get("family") or {}
    return {"id": e["id"], "kind": "tech", "q": q, "target": tgt, "qtype": QTYPES_V1[qt], "answerable": bool(e["answerable"]),
            "family": f"{fam['id']}|{e['render']}" if fam.get("size", 1) > 1 else None,
            "length": sum(len(c.ids) for c in q.chunks), "weight": float(e.get("weight", 1.0)), "values": vals,
            "aux": qt == "choice" and len(q.chunks) == 1, "util": util.astype(np.float32), "qc": qc, "qm": qm, "v": vv, "vm": vm}


def replay_item(ser, rec: Mapping[str, Any], teacher: Mapping[str, Any] | None = None) -> dict:
    q = ser.encode_record(rec)
    ids = [c["id"] for c in rec["candidates"]]
    tgt = np.zeros(len(ids), dtype=np.float32)
    for lab in rec["target"]["labels"]:
        tgt[ids.index(lab)] = 1.0
    for c in q.chunks:
        c.ids = np.asarray(c.ids, dtype=np.int32)
    it = {"id": rec["id"], "kind": "replay", "q": q, "target": tgt, "qtype": QTYPES_V1[rec["task"]["type"]],
          "answerable": bool(rec["answerable"]), "length": sum(len(c.ids) for c in q.chunks), "weight": 1.0}
    if teacher is not None:
        it["t_rank"] = np.asarray(teacher["rank"], dtype=np.float32)
        it["t_apply"] = np.asarray(teacher["apply"], dtype=np.float32) if teacher.get("apply") is not None else None
    return it


# ------------------------------------------------------------------------------------------------ batch + loss
def forward_batch(model, batch: Sequence[dict], pad_id: int, dev, amp: bool = True):
    qs = [it["q"] for it in batch]
    flat, where = chunk_index(qs)
    b = {k: v.to(dev, non_blocking=True) for k, v in collate_chunks(flat, pad_id).items()}
    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp and dev.type == "cuda"):
        out = model(**b)
    merged = merge_chunks(out, where, len(qs), [q.n_candidates for q in qs])
    first_row = {}
    for qi, row, _ in where:
        first_row.setdefault(qi, row)
    return out, merged, [first_row[i] for i in range(len(qs))]


def _pad(batch, key, kmax, fill=0.0, tail=()):
    x = torch.full((len(batch), kmax, *tail), fill)
    for i, it in enumerate(batch):
        v = it.get(key)
        if v is not None:
            x[i, : len(v)] = torch.from_numpy(np.asarray(v, dtype=np.float32))
    return x


def loss_fn(out, merged, rows, batch: Sequence[dict], obj: Objective, dev) -> tuple[torch.Tensor, dict[str, float]]:
    kmax = merged["rank"].size(1)
    tgt = _pad(batch, "target", kmax).to(dev)
    qtype = torch.tensor([it["qtype"] for it in batch], device=dev)
    ans = torch.tensor([it["answerable"] for it in batch], device=dev)
    w = torch.tensor([it["weight"] for it in batch], device=dev)
    loss, parts = decision_loss(merged, qtype, tgt, ans, V1_LOSS, sample_weight=w)
    parts = {"current": parts["loss"]}
    z = merged["rank"].float()
    mask = merged["mask"]
    is_choice = qtype == QTYPES_V1["choice"]
    tech = torch.tensor([it["kind"] == "tech" for it in batch], device=dev)
    sel = tech & is_choice & ans
    if (obj.l_pair or obj.l_margin) and sel.any():
        vals = _pad(batch, "values", kmax, fill=-2.0).to(dev)                 # -1 infeasible, -2 padding / replay
        gold = tgt.argmax(-1)
        zg = z.gather(1, gold[:, None])
        neg = mask & (tgt < 0.5)
        negv = vals.masked_fill(~neg, -3.0)
        order = negv.argsort(-1, descending=True)
        rank_pos = torch.empty_like(order).scatter_(1, order, torch.arange(kmax, device=dev)[None, :].expand_as(order))
        wneg = torch.where(rank_pos < obj.hard_negatives, 1.0, obj.easy_neg_weight) * neg.float()
        d = zg - z
        if obj.l_pair:
            lp = (F.softplus(-d) * wneg).sum(-1) / wneg.sum(-1).clamp(min=1e-6)
            lp = (lp * w)[sel].mean()
            loss = loss + obj.l_pair * lp
            parts["pair"] = float(lp.detach())
        if obj.l_margin:
            hardest = d.masked_fill(~neg, 1e4).gather(1, order[:, :1])[:, 0]  # margin against the best-outcome negative
            lm = (F.relu(obj.margin - hardest) * w)[sel].mean()
            loss = loss + obj.l_margin * lm
            parts["margin"] = float(lm.detach())
    if obj.l_value and "q" in out:
        aux = [i for i, it in enumerate(batch) if it["kind"] == "tech" and it.get("aux")]
        if aux:
            r = torch.tensor([rows[i] for i in aux], device=dev)
            sub = [batch[i] for i in aux]
            ka = out["q"].size(1)
            qo = out["q"][r]                                                   # [A, K, 1 + QD]
            m = torch.zeros(len(aux), ka, device=dev)
            for n, it in enumerate(sub):
                m[n, : len(it["target"])] = 1.0
            util = _pad(sub, "util", ka).to(dev)
            lu = (F.binary_cross_entropy_with_logits(qo[..., 0], util, reduction="none") * m).sum() / m.sum().clamp(min=1)
            qc, qm = _pad(sub, "qc", ka, tail=(QD,)).to(dev), _pad(sub, "qm", ka, tail=(QD,)).to(dev)
            lc = (F.huber_loss(qo[..., 1:], qc, reduction="none") * qm).sum() / qm.sum().clamp(min=1)
            vt = torch.from_numpy(np.stack([it["v"] for it in sub])).to(dev)
            vm = torch.from_numpy(np.stack([it["vm"] for it in sub])).to(dev)
            lv = (F.huber_loss(out["v"][r], vt, reduction="none") * vm).sum() / vm.sum().clamp(min=1)
            la = lu + 0.5 * lc + 0.5 * lv
            loss = loss + obj.l_value * la
            parts.update(q_util=float(lu.detach()), q_cons=float(lc.detach()), v=float(lv.detach()))
    if obj.l_kd:
        kd_idx = [i for i, it in enumerate(batch) if it.get("t_rank") is not None]
        if kd_idx:
            idx = torch.tensor(kd_idx, device=dev)
            sub = [batch[i] for i in kd_idx]
            tr = _pad(sub, "t_rank", kmax, fill=-1e4).to(dev)
            mk = mask[idx]
            zs = z[idx].masked_fill(~mk, -1e4)
            single = qtype[idx] != QTYPES_V1["multi"]
            pt = torch.softmax(tr.masked_fill(~mk, -1e4), -1)
            kl = (pt * (torch.log(pt.clamp_min(1e-9)) - torch.log_softmax(zs, -1))).sum(-1)
            lk = kl[single].mean() if single.any() else z.sum() * 0
            if (~single).any() and "apply" in merged:
                ta = _pad(sub, "t_apply", kmax).to(dev)
                bce = F.binary_cross_entropy_with_logits(merged["apply"][idx].float(), torch.sigmoid(ta), reduction="none")
                lk = lk + ((bce * mk.float()).sum(-1) / mk.sum(-1).clamp(min=1))[~single].mean()
            loss = loss + obj.l_kd * lk
            parts["kd"] = float(lk.detach())
    parts["loss"] = float(loss.detach())
    return loss, parts


# ------------------------------------------------------------------------------------------------ trainable sets
def set_trainable(model, mode: str, last_n: int = 0) -> dict[str, int]:
    """mode 'head' (R1) or 'last_n' (R2). Returns parameter counts per group."""
    enc = model.encoder
    for p in enc.parameters():
        p.requires_grad_(False)
    unfrozen = []
    if mode == "last_n" and last_n > 0:
        unfrozen = list(enc.layers[-last_n:]) + [enc.final_norm]
        for m in unfrozen:
            for p in m.parameters():
                p.requires_grad_(True)
    elif mode not in ("head", "last_n"):
        raise ValueError(mode)
    enc_ids = {id(p) for p in enc.parameters()}
    new = {id(p) for name in ("adapter", "cand", "q_head", "v_head") if getattr(model, name, None) is not None
           for p in getattr(model, name).parameters()}
    groups = {"encoder": [p for p in enc.parameters() if p.requires_grad],
              "head": [p for p in model.parameters() if id(p) not in enc_ids and id(p) not in new],
              "new": [p for p in model.parameters() if id(p) in new]}
    return groups


def family_batches(lengths: Sequence[int], families: Sequence[str | None], max_tokens: int, max_items: int, rng: np.random.Generator,
                   chunk: int = 4096) -> list[list[int]]:
    """Like ``batches`` but a counterfactual family (same base, different constraint, different decision) is never split:
    all of its members land in the SAME batch, next to each other. Items without a family are units of one."""
    units: dict[str, list[int]] = {}
    for i, f in enumerate(families):
        units.setdefault(f if f is not None else f"#{i}", []).append(i)
    us = list(units.values())
    ln = np.asarray(lengths)
    order = rng.permutation(len(us))
    out, s = [], 0
    while s < len(order):
        part, n = [], 0
        while s < len(order) and n < chunk:
            part.append(us[order[s]])
            n += len(part[-1])
            s += 1
        part.sort(key=lambda u: int(ln[u].max()))
        win, cur, mx = [], [], 0
        for u in part:
            new = max(mx, int(ln[u].max()))
            if cur and (new * (len(cur) + len(u)) > max_tokens or len(cur) + len(u) > max_items):
                win.append(cur)
                cur, new = [], int(ln[u].max())
            cur += [int(i) for i in u]
            mx = new
        if cur:
            win.append(cur)
        out += [win[j] for j in rng.permutation(len(win))]
    return [out[j] for j in rng.permutation(len(out))]


def batches(lengths: Sequence[int], max_tokens: int, max_items: int, rng: np.random.Generator, chunk: int = 4096) -> list[list[int]]:
    """One epoch of length-bucketed batches under a padded-token budget (every item exactly once)."""
    order = rng.permutation(len(lengths))
    ln = np.asarray(lengths)
    out = []
    for s in range(0, len(order), chunk):
        part = order[s:s + chunk]
        part = part[np.argsort(ln[part], kind="stable")]
        win, cur, mx = [], [], 0
        for i in part:
            new = max(mx, int(ln[i]))
            if cur and (new * (len(cur) + 1) > max_tokens or len(cur) >= max_items):
                win.append(cur)
                cur, new = [], int(ln[i])
            cur.append(int(i))
            mx = new
        if cur:
            win.append(cur)
        out += [win[j] for j in rng.permutation(len(win))]
    return [out[j] for j in rng.permutation(len(out))]


def lr_at(step: int, total: int, warmup: int) -> float:
    return min(1.0, (step + 1) / max(1, warmup)) * max(0.0, (total - step) / max(1, total))


def load_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8").read().split("\n") if line]


def n_params(ps) -> int:
    return int(sum(p.numel() for p in ps))


__all__ = ["Objective", "tech_item", "replay_item", "forward_batch", "loss_fn", "set_trainable", "batches", "family_batches", "lr_at",
           "load_jsonl", "cons_keys", "cons_stats", "encode_tech", "options", "n_params", "QD", "VD", "math"]
