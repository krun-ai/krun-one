"""Export a self-contained Krun One inference artifact.

    artifacts/krun-one-v0/
      model.safetensors   DecisionModel state dict (fp32, exactly the evaluated weights; strict load)
      config.json         encoder architecture (ModernBERT-large config, from Laya)
      tokenizer/          tokenizer files (from Laya)
      krun_config.json    sequence budgets, per-family temperatures, abstention rule + thresholds
      metrics.json        headline validation / final metrics (copied from results/)
      provenance.json     base checkpoint, data manifest, training run, licenses / attributions
      README.md

Loading needs only torch + transformers + safetensors + this package's ``model.py``:
``KrunOne.from_pretrained("artifacts/krun-one-v0")`` — no ./laya, no training code.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping

from safetensors.torch import save_model

KRUN_CONFIG_VERSION = "krun_one.config.v0"


def build_krun_config(*, max_len: int, head_max_len: int, head_layers: int, n_act: int, temperatures: Mapping[str, float],
                      score: str, thresholds: Mapping[str, float], target: str, name: str = "krun-one-v0",
                      primitive_temperatures: Mapping[str, float] | None = None,
                      score_confidence: str | None = None) -> dict[str, Any]:
    cfg = _base_config(max_len=max_len, head_max_len=head_max_len, head_layers=head_layers, n_act=n_act,
                       temperatures=temperatures, score=score, thresholds=thresholds, target=target, name=name)
    if primitive_temperatures:  # KRUN-011: multi-primitive model
        cfg["config_version"] = "krun_one.config.v1"
        cfg["primitives"] = ["choice", "noul", "score"]
        cfg["temperature"].update({k: float(v) for k, v in primitive_temperatures.items()})
        cfg["score_confidence"] = score_confidence or "normalized_entropy"
        cfg["primitive_semantics"] = {
            "noul": "p_true = softmax(z / T_noul)[1] over options [false, true]",
            "score": "p = softmax(z / T_score) over levels in request order; score = sum(i * p_i); "
                     "confidence per score_confidence (ordinal_dispersion: 1 - Var_p(i) / ((K-1)/2)^2; "
                     "normalized_entropy: 1 - H(p)/ln K; top1_probability: max p)",
            "choice": "unchanged from krun-one-v0 (family temperature + top1_top2_margin abstention)"}
        cfg["serialization"] = ("Laya build_sequence: [CLS] '{type} question: {instructions}' [SEP] ([MASK] option)* [SEP] "
                                "context [SEP]; choice options = 'name: description' or humanized label; noul options = "
                                "['false: {criteria.false|default}', 'true: {criteria.true|default}']; score options = "
                                "'level i: {level_i}' in order; qtype selects type_emb (choice 0, score 1, noul 2)")
    return cfg


def _base_config(*, max_len, head_max_len, head_layers, n_act, temperatures, score, thresholds, target, name):
    return {
        "config_version": KRUN_CONFIG_VERSION,
        "name": name,
        "architecture": "laya.DecisionModel (ModernBERT-large encoder + 2-layer decision head + option-marker scorer)",
        "max_len": max_len,
        "head_max_len": head_max_len,
        "head_layers": head_layers,
        "n_act": n_act,
        "act_head": "present for state-dict compatibility; NOT used (upstream act_probability saturates at ~1.0)",
        "temperature": {**dict(temperatures), "default": temperatures.get("intent", 1.0)},
        "task_family_rule": "options with descriptions -> 'tool', otherwise 'intent'",
        "abstention": {"score": score, "threshold": dict(thresholds), "target_selective_accuracy": target,
                       "rule": "abstain (choice = null) when score(temperature-scaled probabilities) < threshold[family]"},
        "serialization": "Laya build_sequence: [CLS] 'choice question: {instructions}' [SEP] ([MASK] option)* [SEP] context [SEP];"
                         " option text = 'name: description' if described else humanized label; 48 tokens max per option",
    }


def export_artifact(model, tokenizer_dir: Path, encoder_config: Path, out: Path, krun_config: Mapping[str, Any],
                    metrics: Mapping[str, Any] | None = None, provenance: Mapping[str, Any] | None = None,
                    readme: str | None = None) -> Path:
    out = Path(out)
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    save_model(model.float().cpu(), str(tmp / "model.safetensors"))
    shutil.copy(encoder_config, tmp / "config.json")
    shutil.copytree(tokenizer_dir, tmp / "tokenizer")
    (tmp / "krun_config.json").write_text(json.dumps(krun_config, indent=2) + "\n")
    (tmp / "metrics.json").write_text(json.dumps(metrics or {}, indent=2) + "\n")
    (tmp / "provenance.json").write_text(json.dumps(provenance or {}, indent=2, default=str) + "\n")
    (tmp / "README.md").write_text(readme or "# Krun One\n")
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    return out
