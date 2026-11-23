"""Krun One v1.1 R3 — multimodal recovery table and gates (configs/v11/r3_protocol.yaml: multimodal).

    uv run python scripts/v11/mm_report_r3.py r3f_s1_m20260932_e1 [more tags ...]

Rows: the registered M1.5c result; the v1 core + M1.5c on this round's host (results/v11/r3/multimodal/v1_host.json); for each
tag "R3 core + old fusion" (system P of the dump) and "R3 core + re-aligned fusion" (system C).
Gates = the registered M1.5c thresholds; the two terms that were relative to system P use the v1-core values of this host.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/v11/r3/multimodal"
COLS = ["doc visual-only", "layout", "screenshot", "text-dep doc", "real speech", "MInDS intent", "image", "synthdoc", "DocLayNet cat.", "CORD"]


def vals(g: dict, s: str) -> dict:
    h = g["G4b_document_hybrid"]
    if s == "C":
        return dict(zip(COLS, [g["G1_document_visual_only"]["visual_only_mean_C"], g["G1_document_visual_only"]["layout_C"], g["G2_screenshot_text"]["screenshot_choice_C"],
                               g["G2_screenshot_text"]["text_dep_doc_C"], g["G3_real_speech"]["real_C"], g["G3_real_speech"]["minds_intent_C"], g["G4_image"]["C"],
                               h["synthdoc_type"]["C"], h["doclaynet_category"]["C"], h["cord_type"]["C"]]))
    return dict(zip(COLS, [g["G1_document_visual_only"]["P"], g["G1_document_visual_only"]["layout_P"], g["G2_screenshot_text"]["P"],
                           g["G2_screenshot_text"]["text_dep_doc_P"], g["G3_real_speech"]["real_P"], g["G3_real_speech"]["minds_intent_P"], g["G4_image"]["P"],
                           h["synthdoc_type"]["P"], h["doclaynet_category"]["P"], h["cord_type"]["P"]]))


def gates(v: dict, ref_p: dict) -> dict:
    return {"G1": v["doc visual-only"] >= 0.679 and v["layout"] >= ref_p["layout"] + 0.05,
            "G2": v["screenshot"] >= 0.979 and v["text-dep doc"] >= ref_p["text-dep doc"] - 0.01,
            "G3": v["real speech"] >= 0.841 and v["MInDS intent"] >= 0.890, "G4": v["image"] >= 0.633,
            "G4b": v["synthdoc"] >= 0.953 and v["DocLayNet cat."] >= 0.935 and v["CORD"] >= 0.95}


def main() -> None:
    reg = json.loads((ROOT / "results/v1/m15c/m15c_eval.json").read_text())["gates_quality"]
    host = json.loads((OUT / "v1_host.json").read_text())["gates_quality"]
    ref_p = vals(host, "P")                                  # M1.5 (system P) on the v1 core, this host: the registered reference of G1 / G2
    rows = {"M1.5c as registered (production host)": vals(reg, "C"), "v1 core + M1.5c fusion (this host)": vals(host, "C")}
    extra = {}
    for t in sys.argv[1:]:
        d = json.loads((OUT / f"{t}.json").read_text())
        rows[f"R3 core + OLD fusion [{t}]"] = vals(d["gates_quality"], "P")
        rows[f"R3 core + RE-ALIGNED fusion [{t}]"] = vals(d["gates_quality"], "C")
        tr = ROOT / "runs/v11/r3/multimodal" / t / "train_result.json"
        if tr.exists():
            r = json.loads(tr.read_text())
            extra[t] = {"selected": r["selected"], "eligible": r["selection_detail"][r["selected"]]["eligible"], "steps": r["total_steps"],
                        "train_seconds": r["train_seconds"], "core_unchanged": r["core_unchanged"], "r3": r.get("r3"),
                        "selection_detail": r["selection_detail"], "by_modality": d["by_modality_task_class"]}
    res = {"reference_P_v1_core_this_host": ref_p, "rows": {}, "runs": extra}
    md = ["# R3 multimodal recovery — val_eval (SigLIP2 / Whisper / OCR and the text core frozen; only the fusion is re-trained)", "",
          f"Gate references measured on the v1 core, this host: layout of the M1.5 reference {100 * ref_p['layout']:.1f} (G1 needs +5 pp), "
          f"text-dependent documents {100 * ref_p['text-dep doc']:.1f} (G2 needs -1 pp).", "",
          "| system | " + " | ".join(COLS) + " | G1 | G2 | G3 | G4 | G4b | passed |", "|---|" + "---:|" * len(COLS) + "---|---|---|---|---|---|"]
    base = rows["v1 core + M1.5c fusion (this host)"]
    for k, v in rows.items():
        g = gates(v, ref_p)
        res["rows"][k] = {"values": v, "gates": g, "passed": sum(g.values()), "delta_vs_v1_core_pp": {c: 100 * (v[c] - base[c]) for c in COLS}}
        md.append(f"| {k} | " + " | ".join(f"{100 * v[c]:.1f}" for c in COLS) + " | " + " | ".join("✅" if g[x] else "❌" for x in ("G1", "G2", "G3", "G4", "G4b"))
                  + f" | {sum(g.values())} / 5 |")
    md += ["", "## Difference to the v1 core + M1.5c on this host (pp)", "", "| system | " + " | ".join(COLS) + " | worst |", "|---|" + "---:|" * (len(COLS) + 1)]
    for k, v in res["rows"].items():
        if k.startswith("R3"):
            d = v["delta_vs_v1_core_pp"]
            md.append(f"| {k} | " + " | ".join(f"{d[c]:+.1f}" for c in COLS) + f" | {min(d.values()):+.1f} |")
    for t, e in extra.items():
        md += ["", f"Run `{t}`: {e['steps']} steps, {e['train_seconds'] / 60:.1f} min, selected `{e['selected']}` (eligible on val_fit: {e['eligible']}), "
                   f"text core unchanged: {e['core_unchanged']}."]
    (OUT / "multimodal_r3.json").write_text(json.dumps(res, indent=1))
    (OUT / "multimodal_r3.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
