"""Summarize the HyperCloning scaling experiment: MLM loss per language + downstream proxy metrics per milestone,
and apply the saturation rule of configs/v1/scaling.yaml.

    uv run python scripts/v1/scaling_summary.py   # -> results/v1/hyperclone_scaling.{json,md}
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load((ROOT / "configs/v1/scaling.yaml").read_text())
spec = importlib.util.spec_from_file_location("bk", ROOT / "scripts/v1/answerability_bakeoff.py")
bk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bk)


def downstream(run: Path, signal: str = "margin") -> dict | None:
    if not (run / "val_signals.jsonl.gz").exists():
        return None
    rows = [r for r in bk.load(run) if r["type"] == "choice"]
    fit = [r for r in rows if bk.is_fit(r["split_group_id"])]
    ev = [r for r in rows if not bk.is_fit(r["split_group_id"])]
    T = {f: bk.fit_temperature([r for r in fit if r["answerable"] and r["family"] == f]) for f in ("intent", "tool")}
    ff, fe = [bk.featurize(r, T[r["family"]]) for r in fit], [bk.featurize(r, T[r["family"]]) for r in ev]
    tau = {}
    for fam in ("intent", "tool"):
        idx = [i for i, r in enumerate(fit) if r["family"] == fam]
        tau[fam] = bk.threshold(bk.native_scores(ff, signal)[idx], np.array([ff[i]["correct"] for i in idx], dtype=float))
    sig = bk.evaluate_signal(ev, fe, bk.native_scores(fe, signal), tau)
    out = {"model": bk.model_metrics(ev, fe), "macro_fa": sig["macro_fa"], "coverage": sig["coverage"]}

    def acc(pred):
        m = [f["correct"] for r, f in zip(ev, fe) if r["answerable"] and pred(r)]
        return float(np.mean(m)) if m else None
    for lang in ("en", "pt", "es", "fr"):
        out[f"acc_{lang}"] = acc(lambda r, lang=lang: r["lang"].split("-")[0] == lang)
    out["acc_cross_language"] = acc(lambda r: "cross_language" in r["slices"])
    out["acc_hard"] = acc(lambda r: r.get("difficulty") == "hard")
    out["acc_unseen_tool"] = acc(lambda r: "unseen_tool" in r["slices"])
    e = json.loads((run / "eval_validation.json").read_text())
    out.update({k: e.get(k) for k in ("noul_auroc", "score_mae", "score_spearman", "multi_micro_f1")})
    out["mean_choice_acc"] = float(np.mean([out["model"]["accuracy_intent"], out["model"]["accuracy_tool"]]))
    return out


def main() -> None:
    mil = CFG["milestones_stage1"] + CFG.get("milestones_stage2", [])
    R = ROOT / "runs/v1/scaling" / CFG["name"]
    D = ROOT / "runs/v1/scaling_downstream"
    rows = []
    for m in mil:
        e = R / f"milestone-{m}" / "eval.json"
        if not e.exists():
            continue
        mlm = json.loads(e.read_text())["heldout_mlm_loss"]
        rows.append({"tokens": m, "mlm": mlm, "mlm_mean": float(np.mean([mlm[k] for k in ("en", "pt", "es", "fr") if k in mlm])),
                     "downstream": downstream(D / f"k1_t{m}")})
    ref = downstream(D / "ref_mmbert_base")
    # saturation rule between consecutive milestones from 1B on
    verdicts = []
    for a, b in zip(rows, rows[1:]):
        if a["tokens"] < 1_000_000_000:
            continue
        mlm_gain = (a["mlm_mean"] - b["mlm_mean"]) / a["mlm_mean"]
        da, db = a["downstream"] or {}, b["downstream"] or {}
        acc_gain = (db.get("mean_choice_acc", 0) - da.get("mean_choice_acc", 0)) if da and db else None
        fa_gain = (da.get("macro_fa", 0) - db.get("macro_fa", 0)) if da and db else None
        strong = mlm_gain >= 0.02 or (acc_gain or 0) >= 0.005 or (fa_gain or 0) >= 0.01
        verdicts.append({"from": a["tokens"], "to": b["tokens"], "mlm_rel_gain": mlm_gain, "acc_gain": acc_gain,
                         "fa_gain": fa_gain, "strong": strong})
    res = {"milestones": rows, "reference_mmbert_base": ref, "saturation": verdicts,
           "continue_to_stage2": bool(verdicts and verdicts[-1]["strong"])}
    out = ROOT / "results/v1"
    (out / "hyperclone_scaling.json").write_text(json.dumps(res, indent=2))
    md = ["| tokens | MLM EN | PT | ES | FR | choice intent | tool | PT | ES | FR | cross-lang | hard | unseen tool | "
          "noul AUROC | score MAE | macro FA |", "|" + "---|" * 16]
    for r in rows + ([{"tokens": "mmBERT-base ref", "mlm": {}, "downstream": ref}] if ref else []):
        d = r["downstream"] or {}
        f = lambda x: "—" if x is None else f"{100 * x:.1f}"  # noqa: E731
        md.append(f"| {r['tokens']} | " + " | ".join(f"{r['mlm'].get(k, float('nan')):.3f}" if r["mlm"] else "—"
                                                        for k in ("en", "pt", "es", "fr")) + " | "
                  + " | ".join(f(d.get("model", {}).get(k)) for k in ("accuracy_intent", "accuracy_tool")) + " | "
                  + " | ".join(f(d.get(k)) for k in ("acc_pt", "acc_es", "acc_fr", "acc_cross_language", "acc_hard",
                                                     "acc_unseen_tool", "noul_auroc")) + " | "
                  + ("—" if d.get("score_mae") is None else f"{d['score_mae']:.3f}") + f" | {f(d.get('macro_fa'))} |")
    md.append(f"\nContinue to stage 2: **{res['continue_to_stage2']}** (rule: configs/v1/scaling.yaml saturation_rule)")
    (out / "hyperclone_scaling.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
