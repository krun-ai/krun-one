"""Krun One v1 objectives (per question, after merge_chunks).

Notation: rank logits z (K), applicability logits a (K), null logit z0, answerability logit g.

choice   answerable     : CE(z, gold)                               — Laya/v0.3 objective, unchanged
         unanswerable   : variant "oe"   -> w_oe * CE(z, uniform)   — v0.3 baseline (outlier exposure)
                          variant "null" -> CE([z; z0], null)       — K+1 softmax, gold = the [NULL] slot
                          variant "head" -> no rank loss            — rank is only "which is best"
         all            : BCE(g, answerable) * w_ans                — answerability head ("is any applicable")
                          BCE(a, one-hot gold) * w_apply            — applicability, shared with multi
multi                   : BCE(a, multi-hot) + BCE(g, any-positive) * w_ans
noul                    : CE over [false, true]                     — == BCE on z_true - z_false (v0.3)
score                   : CE(z, level) + w_rps * RPS + optional Gaussian soft-ordinal target (sigma)

For "null" the answerable case also uses the K+1 softmax so z0 is calibrated against the candidates.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .arch import QTYPES_V1
from .model import NEG


@dataclass(frozen=True)
class LossConfig:
    unanswerable_mode: str = "head"   # oe | null | head
    w_oe: float = 0.5                 # v0.3's unanswerable_weight
    w_ans: float = 0.5
    w_apply: float = 0.25
    w_multi: float = 1.0
    w_rps: float = 0.0                # v0.3 evidence: RPS gave no ranking gain (PRIMITIVES_REPORT §5)
    score_sigma: float = 0.0          # 0 = one-hot level target


def rps(logp: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.float()
    p = logp.exp() * m
    k = mask.sum(-1).clamp(min=2).float()
    return (((torch.cumsum(p, -1) - torch.cumsum(target * m, -1)) ** 2) * m).sum(-1) / (k - 1)


def soft_ordinal_target(level: torch.Tensor, mask: torch.Tensor, sigma: float) -> torch.Tensor:
    idx = torch.arange(mask.size(1), device=mask.device, dtype=torch.float32)[None, :]
    t = torch.exp(-0.5 * ((idx - level[:, None].float()) / sigma) ** 2) * mask.float()
    return t / t.sum(-1, keepdim=True).clamp_min(1e-9)


def decision_loss(q: dict[str, torch.Tensor], qtype: torch.Tensor, target: torch.Tensor, answerable: torch.Tensor,
                  cfg: LossConfig, sample_weight: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, float]]:
    """q: merge_chunks output (rank, apply?, mask, null_rank?, ans?); target: [Q, K] multi-hot (one-hot for
    choice/noul/score; all-zero for unanswerable choice / empty multi). Returns (mean loss, parts)."""
    mask = q["mask"]
    z = q["rank"].masked_fill(~mask, NEG)
    logp = torch.log_softmax(z, -1)
    Q = z.size(0)
    per = torch.zeros(Q, device=z.device)
    parts: dict[str, float] = {}
    is_choice = qtype == QTYPES_V1["choice"]
    is_multi = qtype == QTYPES_V1["multi"]
    is_score = qtype == QTYPES_V1["score"]
    is_noul = qtype == QTYPES_V1["noul"]
    ans_f = answerable.float()

    # --- single-winner distributions: noul, score, answerable choice ---
    tgt = target
    if cfg.score_sigma > 0 and is_score.any():
        lvl = target.argmax(-1)
        tgt = torch.where(is_score[:, None], soft_ordinal_target(lvl, mask, cfg.score_sigma), target)
    ce = -(tgt * logp).sum(-1)
    single = is_noul | is_score | (is_choice & answerable)
    if cfg.unanswerable_mode == "null" and "null_rank" in q:
        zn = torch.cat([z, q["null_rank"][:, None]], -1)
        logpn = torch.log_softmax(zn, -1)
        tn = torch.cat([target, (~answerable).float()[:, None]], -1)
        ce_null = -(tn * logpn).sum(-1)
        per = per + torch.where(is_choice, ce_null, torch.where(single, ce, torch.zeros_like(ce)))
    else:
        per = per + torch.where(single, ce, torch.zeros_like(ce))
        if cfg.unanswerable_mode == "oe":
            k = mask.sum(-1).clamp(min=1).float()
            uni = mask.float() / k[:, None]
            oe = -(uni * logp).sum(-1)
            per = per + torch.where(is_choice & ~answerable, cfg.w_oe * oe, torch.zeros_like(oe))
    if cfg.w_rps and is_score.any():
        per = per + torch.where(is_score, cfg.w_rps * rps(logp, target, mask), torch.zeros_like(ce))
    parts["rank"] = float(per.detach().mean())

    # --- applicability (per-candidate sigmoid): multi (main loss) and choice (auxiliary) ---
    if "apply" in q and (is_multi.any() or (cfg.w_apply and is_choice.any())):
        bce = F.binary_cross_entropy_with_logits(q["apply"], target, reduction="none") * mask.float()
        bce = bce.sum(-1) / mask.sum(-1).clamp(min=1).float()
        w = torch.where(is_multi, torch.full_like(bce, cfg.w_multi),
                        torch.where(is_choice, torch.full_like(bce, cfg.w_apply), torch.zeros_like(bce)))
        per = per + w * bce
        parts["apply"] = float((w * bce).detach().mean())

    # --- answerability (choice + multi) ---
    if "ans" in q and cfg.w_ans:
        has = is_choice | is_multi
        bce = F.binary_cross_entropy_with_logits(q["ans"], ans_f, reduction="none")
        per = per + torch.where(has, cfg.w_ans * bce, torch.zeros_like(bce))
        parts["ans"] = float((cfg.w_ans * bce * has.float()).detach().mean())

    loss = per.mean() if sample_weight is None else (per * sample_weight.to(per.dtype)).mean()
    parts["loss"] = float(loss.detach())
    return loss, parts


# ------------------------------------------------------------------------------------------------ decoding
def answerability_scores(q: dict[str, torch.Tensor], temperature: float = 1.0) -> dict[str, torch.Tensor]:
    """Every answerability signal the architecture offers, from one forward (higher = more answerable)."""
    mask = q["mask"]
    p = torch.softmax(q["rank"].masked_fill(~mask, NEG) / temperature, -1)
    top2 = p.topk(min(2, p.size(-1)), -1).values
    s = {"margin": top2[:, 0] - (top2[:, 1] if top2.size(-1) > 1 else 0), "top1": top2[:, 0]}
    if "null_rank" in q:
        zn = torch.cat([q["rank"].masked_fill(~mask, NEG), q["null_rank"][:, None]], -1)
        s["null_softmax"] = 1 - torch.softmax(zn / temperature, -1)[:, -1]
    if "ans" in q:
        s["ans_head"] = torch.sigmoid(q["ans"])
    if "apply" in q:
        s["max_apply"] = torch.sigmoid(q["apply"].masked_fill(~mask, NEG)).max(-1).values
    return s
