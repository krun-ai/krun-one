"""FREEZE of the Krun One v1 text base: promote the Stage 2 selected checkpoint to artifacts/krun-one-v1-text-final.

    python scripts/v1/freeze_text_final.py build      # artifact dir (weights hard-linked, no copy), calibration, configs
    python scripts/v1/freeze_text_final.py reference  # reference cases + CPU fp32 reference outputs (inside the artifact)
    python scripts/v1/freeze_text_final.py sums       # SHA256SUMS (last step; the artifact is immutable afterwards)

Calibration is recomputed from the step_003299 V-fit signals with the registered protocols (kd1_evaluate: family
temperatures, 0.98 target margin; r2_protocol serving_point_rule: coverage grid on fit -> margin threshold). Nothing is
refitted on eval, nothing touches final_protected.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
SRC = ROOT / "runs/v1/ft/krun-one-v1-stage2/milestones/step_003299"
SPEC = ROOT / "runs/v1/ft/krun-one-v1-stage2/final/spec.json"
OUT = ROOT / "artifacts/krun-one-v1-text-final"
EXPECTED_SHA = "781e5c7b7006da630d6b77693de26bd886839b0f8f96673774f9ace136dc18b6"
MODEL_ID = "krun-one-v1-text-final"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(Path(path).parent))
    spec.loader.exec_module(m)
    return m


def calibration() -> tuple[dict, dict]:
    C = load_mod("C", ROOT / "scripts/v1/r2_compare.py")
    K = C.K
    rows = [r for r in K.load(SRC) if r["file"] != "val_tools_v2"]
    _, calib = K.evaluate(rows)
    serving = C.serving_point(rows, calib)            # registered rule, fit only
    tau = {}
    for fam in ("intent", "tool"):
        T = calib["temperature"][fam]
        fm = np.array([C.margin(r["rank"], T)[0] for r in rows if r["part"] == "fit" and r["family"] == fam
                       and r["type"] == "choice" and r["answerable"] and "permutation" not in r["slices"]])
        tau[fam] = float(np.quantile(fm, 1 - serving[fam]))
    out = {"temperature": calib["temperature"],
           "margin_threshold": tau,
           "margin_threshold_rule": "serving_point_rule of configs/v1/r2_protocol.yaml: per family the largest coverage "
                                    "in {50..90%} whose FIT selective accuracy >= 0.97 and OOD FA <= 0.05 (intent) / "
                                    "0.08 (tool); threshold = (1 - coverage) quantile of answerable fit margins",
           "serving_coverage": serving,
           "margin_threshold_target_0.98": calib["margin_threshold"],
           "target_selective_accuracy_reference": calib["target_selective_accuracy"],
           "multi_threshold": calib["multi_threshold"],
           "abstention_score": "top1_top2_margin (temperature-scaled)",
           "fitted_on": "V fit only (KrunBench v1 val_fit + val_ml fit part) of step_003299 signals",
           "signals_sha256": {p.name: sha(p) for p in sorted(SRC.glob("signals_*.jsonl.gz"))}}
    return out, serving


def build() -> None:
    if sha(SRC / "model.safetensors") != EXPECTED_SHA:
        raise SystemExit("step_003299 weights do not match the Stage 2 record")
    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / "model.safetensors"
    if not dst.exists():
        os.link(SRC / "model.safetensors", dst)        # same bytes, zero extra disk
    shutil.copy(SPEC, OUT / "spec.json")
    import glob
    snap = sorted(glob.glob(str(Path.home() / ".cache/huggingface/hub/models--jhu-clsp--mmBERT-small/snapshots/*/tokenizer.json")))
    tdir = OUT / "tokenizer"
    tdir.mkdir(exist_ok=True)
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        shutil.copy(Path(snap[0]).parent / f, tdir / f)
    calib, serving = calibration()
    (OUT / "calibration.json").write_text(json.dumps(calib, indent=2))
    import yaml
    s2 = yaml.safe_load((ROOT / "configs/v1/stage2.yaml").read_text())
    dcfg = {"config_version": "krun_one.v1.decision_config.1", "model_id": MODEL_ID,
            "architecture": "K1-W1536-L22 (ModernBERT/mmBERT encoder 22x1536 + 2-layer decision head + marker scorer)",
            "sequence": s2["sequence"], "dtype_serving": "bf16 autocast (scorer output fp32)",
            "primitives_public": ["choice", "noul", "score"], "primitives_internal": ["multi"],
            "family_rule": "task_type if given; else 'tool' when any option has a description; else 'intent'",
            "default_instructions": "krun_one.v1.serialization.DEFAULT_INSTRUCTIONS (by family / multi)",
            "abstention": {"public": "top1_top2_margin", "closed_signals": ["[NULL] slot", "answerability head",
                                                                             "applicability max", "fusion calibrator"]},
            "abstention_status": {"intent": "calibrated", "tool": "calibrated"},
            "score_confidence": "ordinal_dispersion",
            "answer_schema": "krun-worker v0.3 answer fields (choice/noul/score) — public API contract unchanged",
            "serving_coverage": serving,
            "runtime": "krun_one.v1.runtime.TextArtifact"}
    (OUT / "decision_config.json").write_text(json.dumps(dcfg, indent=2))
    cmp_ = json.loads((ROOT / "results/v1/stage2/stage2_comparison.json").read_text())
    m = cmp_["models"]["step_003299"]
    metrics = {"source": "results/v1/stage2/stage2_comparison.json (V eval; ToolACE v0 regression; gate harness bf16)",
               "selection": cmp_["selection"]["step_003299"] | {"selected": cmp_["selected"]},
               "gates": cmp_["gates"], "metrics_v_eval": m["metrics"], "coverage_curves": m["curves"],
               "val_tools_v2": m["val_tools_v2"]["metrics"], "toolace": m["toolace"],
               "paired_toolace": cmp_["paired_toolace"].get("step_003299"),
               "milestone_eval_validation": json.loads((SRC / "eval_validation.json").read_text())}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    prov = {"model": "Krun One v1 text — FROZEN (K1-W1536-L22, 879M params)",
            "freeze": {"FREEZE_TEXTUAL_BASE": True, "date": "2026-09-28",
                       "decision": "owner product/roadmap decision: freeze despite the known gaps (see README); "
                                   "the registered Stage 2 gate outcome (FREEZE_TEXTUAL_BASE=false) is NOT rewritten"},
            "weights": {"source": str(SRC.relative_to(ROOT)) + "/model.safetensors", "sha256": EXPECTED_SHA,
                        "selected_by": "configs/v1/stage2_protocol.yaml checkpoint_selection (ToolACE val_tools; holdout never used)",
                        "global_step": 3299, "total_steps": 4399},
            "lineage": ["mmBERT-base (Apache-2.0) -> HyperCloning x2 -> k1_0b (runs/v1/mlm1b_import, sha 061160e3...)",
                        "candidate 1: Krun Decisions v1 data/kd1/train.jsonl sha ddcf6d1b..., 21,757 steps",
                        "v1-lr1e5 -> v1-tool (data/kd1_r2, 1.33M) sha c52475ab...",
                        "Stage 2: data/kd1_s2/train.jsonl sha 9aad2018... -> step_003299 (this artifact)"],
            "dataset_recipe": {"stage2_train_sha256": "9aad201845d4683f6db3a0621602e578ac9ec79800878db3f5f9ce48c5d37e0f",
                               "stage2_build_manifest_sha256": sha(ROOT / "results/v1/stage2/stage2_build_manifest.json"),
                               "sources": "configs/v1/sources.yaml sha " + sha(ROOT / "configs/v1/sources.yaml")},
            "training_config": {"configs/v1/stage2.yaml": sha(ROOT / "configs/v1/stage2.yaml"),
                                "configs/v1/stage2_protocol.yaml": sha(ROOT / "configs/v1/stage2_protocol.yaml")},
            "tokenizer": "jhu-clsp/mmBERT-small tokenizer (Gemma-2 vocab 256k; Apache-2.0 google/gemma_pytorch "
                         "tokenizer.model; gate CLEARED docs/KRUN_ONE_V1_TOKENIZER_LICENSE.md)",
            "leakage": "Stage 2 LEAKAGE = 0 (build_krunbench_v1.py --check); final_protected never opened",
            "notice": "NOTICE (Laya Apache-2.0 decision head design; MASSIVE CC-BY-4.0; see docs/KRUN_DECISIONS_V1_SOURCES_AUDIT.md)",
            "code": {p: sha(ROOT / p) for p in ("src/krun_one/v1/model.py", "src/krun_one/v1/serialization.py",
                                                "src/krun_one/v1/arch.py", "src/krun_one/v1/runtime.py")}}
    (OUT / "provenance.json").write_text(json.dumps(prov, indent=2))
    print(json.dumps({"calibration": calib["margin_threshold"], "serving": serving, "T": calib["temperature"]}, indent=1))


# ------------------------------------------------------------------------------------------------ reference cases
def reference_cases() -> dict:
    """Fixed API-shaped requests covering every primitive, languages, families and k (intents, tools, noul, score,
    multi, cross-language, k=2..128) + a deterministic sample of V EVAL records (never fit-only, never protected)."""
    labels = ["book_flight", "cancel_subscription", "check_balance", "report_fraud", "play_music", "set_alarm",
              "weather_query", "order_status"]
    tools = [{"id": f"tool_{i}", "description": d} for i, d in enumerate([
        "Get the current weather forecast for a city", "Send an email to a contact", "Create a calendar event",
        "Search flights between two airports on a date", "Convert an amount between currencies",
        "Look up the status of a shipped order", "Cancel an active subscription", "Translate text to another language"])]
    many = [{"id": f"api_{i}", "description": f"Endpoint {i}: returns {w} records for the account"}
            for i, w in enumerate(["billing", "invoice", "shipping", "user", "audit", "session", "device", "payment"] * 16)]
    many[77] = {"id": "api_77", "description": "Cancel the user's active subscription and stop future charges"}
    reqs = {
        "intent_en": {"context": "I want to cancel my subscription, it's too expensive",
                      "questions": {"q": {"type": "choice", "task_type": "intent", "options": [{"id": x} for x in labels]}}},
        "intent_ptbr": {"context": "quero cancelar minha assinatura agora", "questions": {"q": {"type": "choice", "task_type": "intent", "options": [{"id": x} for x in labels]}}},
        "intent_es": {"context": "¿cuánto dinero tengo en mi cuenta?", "questions": {"q": {"type": "choice", "task_type": "intent", "options": [{"id": x} for x in labels]}}},
        "intent_fr": {"context": "réveille-moi à sept heures demain", "questions": {"q": {"type": "choice", "task_type": "intent", "options": [{"id": x} for x in labels]}}},
        "intent_k2": {"context": "play some jazz", "questions": {"q": {"type": "choice", "options": [{"id": "play_music"}, {"id": "set_alarm"}]}}},
        "intent_ood": {"context": "the mitochondria is the powerhouse of the cell", "questions": {"q": {"type": "choice", "task_type": "intent", "options": [{"id": x} for x in labels]}}},
        "tool_k8": {"context": "What's the weather like in Lisbon tomorrow?", "questions": {"q": {"type": "choice", "task_type": "tool", "options": tools}}},
        "tool_k8_pt": {"context": "Converte 100 dólares para reais, por favor", "questions": {"q": {"type": "choice", "task_type": "tool", "options": tools}}},
        "tool_k128": {"context": "Please cancel my subscription so I'm not charged again", "questions": {"q": {"type": "choice", "task_type": "tool", "options": many}}},
        "tool_irrelevant": {"context": "Tell me a joke about cats", "questions": {"q": {"type": "choice", "task_type": "tool", "options": tools}}},
        "noul_true": {"context": "The package arrived broken and I want my money back.", "questions": {"q": {"type": "noul", "instructions": "The customer is asking for a refund."}}},
        "noul_false_fr": {"context": "Merci beaucoup, tout fonctionne parfaitement !", "questions": {"q": {"type": "noul", "instructions": "Le client signale un problème."}}},
        "score_sentiment": {"context": "This is the worst service I have ever received.", "questions": {"q": {"type": "score", "instructions": "How positive is the message?", "levels": ["very negative", "negative", "neutral", "positive", "very positive"]}}},
        "score_urgency_pt": {"context": "Minha conta foi invadida e estão fazendo compras agora!", "questions": {"q": {"type": "score", "instructions": "Qual a urgência?", "levels": ["baixa", "média", "alta"]}}},
        "multi": {"context": "Book a flight to Paris and set an alarm for 6am", "questions": {"q": {"type": "multi", "options": [{"id": x} for x in labels]}}},
        "mixed_4q": {"context": "Quero cancelar minha assinatura, fui cobrado duas vezes este mês e estou muito irritado.",
                     "questions": {"a": {"type": "choice", "task_type": "intent", "options": [{"id": x} for x in labels]},
                                   "b": {"type": "choice", "task_type": "tool", "options": tools},
                                   "c": {"type": "noul", "instructions": "The customer mentions a double charge."},
                                   "d": {"type": "score", "instructions": "Customer anger", "levels": ["calm", "annoyed", "angry", "furious"]}}},
    }
    return reqs


def v_eval_sample(n: int = 384) -> list[dict]:
    """Deterministic sample of V EVAL records for the logit-level reload check (val_eval + val_ml eval part)."""
    rng = np.random.default_rng(20260928)
    recs = [json.loads(l) for l in open(ROOT / "data/krunbench_v1/val_eval.jsonl")]
    recs += [r for r in map(json.loads, open(ROOT / "data/kd1/val_ml.jsonl")) if r["metadata"].get("val_part", "eval") == "eval"]
    recs = [r for r in recs if r.get("split") not in ("test", "protected") and "final" not in r["id"]]
    by = {}
    for r in recs:
        by.setdefault(r["task"]["type"], []).append(r)
    out = []
    for t, rs in sorted(by.items()):
        idx = rng.choice(len(rs), size=min(len(rs), n // len(by)), replace=False)
        out += [rs[i] for i in sorted(idx)]
    return out


def reference() -> None:
    import torch

    from krun_one.v1.runtime import TextArtifact

    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    ref = OUT / "reference"
    ref.mkdir(exist_ok=True)
    reqs = reference_cases()
    (ref / "api_cases.json").write_text(json.dumps(reqs, indent=1, ensure_ascii=False))
    sample = v_eval_sample()
    with open(ref / "v_eval_sample.jsonl", "w") as f:
        for r in sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    a = TextArtifact.load(OUT, device="cpu", dtype="fp32", verify=False)
    outs = {k: a.decide(v, return_raw=True) for k, v in reqs.items()}
    (ref / "api_outputs_cpu_fp32.json").write_text(json.dumps(outs, indent=1, ensure_ascii=False))
    print({k: {q: (x.get("decision", x.get("noul", x.get("score", x.get("selected"))))) for q, x in o["answers"].items()}
           for k, o in outs.items()})


def sums() -> None:
    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    (OUT / "SHA256SUMS").write_text("".join(f"{sha(p)}  {p.relative_to(OUT)}\n" for p in files))
    print((OUT / "SHA256SUMS").read_text())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "reference", "sums"])
    {"build": build, "reference": reference, "sums": sums}[ap.parse_args().cmd]()
