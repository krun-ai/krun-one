"""Krun One v1.1 — multimodal M1.5c regression table: the registered fusion weights on top of each text core.

    uv run python scripts/v11/mm_report.py v1 r1_s1 r2l4_s1

Rows: the registered M1.5c result (results/v1/m15c/m15c_eval.json, production host), the v1 core re-dumped on this round's
host, and each candidate core. The registered criterion (configs/v11/promotion_gates.yaml, multimodal_m15c_suite) is that
every registered M1.5c gate still passes; the difference to the same-host v1 core is reported next to it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/v11/multimodal"


def flat(g: dict) -> dict:
    h = g["G4b_document_hybrid"]
    return {"G1 doc visual-only": g["G1_document_visual_only"]["visual_only_mean_C"], "G1 layout": g["G1_document_visual_only"]["layout_C"],
            "G2 screenshot": g["G2_screenshot_text"]["screenshot_choice_C"], "G2 text-dep doc": g["G2_screenshot_text"]["text_dep_doc_C"],
            "G3 real speech": g["G3_real_speech"]["real_C"], "G3 minds intent": g["G3_real_speech"]["minds_intent_C"],
            "G4 image": g["G4_image"]["C"], "G4b synthdoc": h["synthdoc_type"]["C"], "G4b doclaynet": h["doclaynet_category"]["C"],
            "G4b cord": h["cord_type"]["C"]}


def passes(g: dict) -> dict:
    return {k.split("_")[0]: bool(v["pass"]) for k, v in g.items()}


def main() -> None:
    tags = sys.argv[1:]
    rows = {"M1.5c registered": json.loads((ROOT / "results/v1/m15c/m15c_eval.json").read_text())}
    rows.update({t: json.loads((OUT / f"{t}.json").read_text()) for t in tags})
    f = {k: flat(v["gates_quality"]) for k, v in rows.items()}
    p = {k: passes(v["gates_quality"]) for k, v in rows.items()}
    cols = list(next(iter(f.values())))
    md = ["# Multimodal M1.5c regression — same fusion weights, different text core (val_eval; nothing multimodal retrained)", "",
          "| text core | " + " | ".join(cols) + " | registered gates | by modality / class |", "|---|" + "---:|" * len(cols) + "---|---|"]
    res = {}
    for k in rows:
        bm = rows[k]["by_modality_task_class"]
        md.append(f"| {k} | " + " | ".join(f"{100 * f[k][c]:.1f}" for c in cols) + " | " + " ".join(f"{g}{'✅' if ok else '❌'}" for g, ok in p[k].items())
                  + " | " + ", ".join(f"{m} {100 * v['C']:.1f}" for m, v in bm.items()) + " |")
        res[k] = {"values": f[k], "gates": p[k], "by_modality_task_class": {m: v["C"] for m, v in bm.items()}}
    base = tags[0]
    md += ["", f"## Candidate core − same-host {base} core (pp)", "", "| text core | " + " | ".join(cols) + " | worst | verdict |", "|---|" + "---:|" * (len(cols) + 1) + "---|"]
    for k in tags[1:]:
        d = {c: 100 * (f[k][c] - f[base][c]) for c in cols}
        worst = min(d.values())
        ok = all(p[k].values())
        still = all(p[k][g] or not p[base][g] for g in p[k])       # no gate that the same-host v1 core passes is lost
        res[k].update(delta_pp=d, worst_pp=worst, all_registered_gates=all(p[k].values()), no_gate_lost_vs_same_host_v1=still, PASS=bool(ok))
        md.append(f"| {k} | " + " | ".join(f"{d[c]:+.2f}" for c in cols) + f" | {worst:+.2f} | "
                  f"{'✅' if ok else '❌'} (gates lost vs same-host v1: {'none' if still else [g for g in p[k] if p[base][g] and not p[k][g]]}) |")
    (OUT / "multimodal_regression.json").write_text(json.dumps(res, indent=1))
    (OUT / "multimodal_regression.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
