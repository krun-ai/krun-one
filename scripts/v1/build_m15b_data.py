"""Krun One v1 — M1.5b dataset (configs/v1/m15b.yaml): the M1.5 dataset reused + layout replay / new layout decisions.

    python scripts/v1/build_m15b_data.py build          # data/m15b/{train,val_fit,val_eval}.jsonl (val = M1.5 val, unchanged)
    python scripts/v1/build_m15b_data.py ocr_missing    # OCR (frozen RapidOCR) for replay pages without M1.5 OCR text
    python scripts/v1/build_m15b_data.py finalize       # attach OCR text, loss weights, manifest, text views

Only TRAIN-split DocLayNet pages (M1 group split) produce new decisions; val_fit / val_eval are the M1.5 files verbatim
(so M1.5 and M1.5b are evaluated on identical decisions); protected is never touched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/v1"))
import build_m1_data as B1  # noqa: E402
import build_m15_data as B15  # noqa: E402

import os  # noqa: E402

CFG_PATH = ROOT / os.environ.get("KRUN_M15_CONFIG", "configs/v1/m15b.yaml")
CFG = yaml.safe_load(CFG_PATH.read_text())
VARIANT = CFG["name"].rsplit("-", 1)[-1]
D15 = ROOT / "data/m15"
D = ROOT / "data" / VARIANT
D1 = ROOT / "data/m1"
LAB = B1.LAB
NAMES = ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title']
LAY = list(LAB["doclaynet_layout"])
PAIRS = [("Caption", "Footnote"), ("Title", "Section-header"), ("Picture", "Table"), ("List-item", "Formula")]
CONTAINS = ["The page contains this element: {x}.", "A página contém este elemento: {x}.", "La página contiene este elemento: {x}.",
            "La page contient cet élément : {x}."]


def jl(p):
    for l in open(p, encoding="utf-8").read().split("\n"):
        if l:
            yield json.loads(l)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def label(c, L):
    return LAB["doclaynet_layout"][c][B1.LANGS.index(L)]


def new_layout_records(rng) -> list[dict]:
    import pyarrow.parquet as pq

    out = []
    for f in sorted((D1 / "raw/pierreguillou__DocLayNet-base").glob("*.parquet")):
        t = pq.read_table(f, columns=["page_hash", "texts", "categories", "bboxes_block", "original_filename", "coco_width", "coco_height"])
        for row in t.to_pylist():
            g = f"dln|{row['original_filename']}"
            if B1.split_of(g) != "train":
                continue
            p = D1 / "media/doclaynet" / f"{row['page_hash']}.jpg"
            if not p.exists():
                continue
            A = float(row["coco_width"] * row["coco_height"])
            area = defaultdict(float)
            seen_blocks = set()
            for c, bb in zip(row["categories"], row["bboxes_block"]):
                k = (c, tuple(bb))
                if k in seen_blocks:
                    continue
                seen_blocks.add(k)
                area[NAMES[c]] = max(area[NAMES[c]], bb[2] * bb[3] / A)
            pres = {n for n in area if n in LAY}
            media = {"path": f"media/doclaynet/{p.name}", "kind": "document"}
            text = "\n".join(row["texts"])[:1500] or None
            base = f"m15b/doclaynet/{row['page_hash'][:16]}"
            common = dict(teacher_filter="if_correct", speech=None, semantic_group_id=None, gt_text=text)
            L = B1.pick_lang(rng)
            out.append(B15.mk(f"{base}/layout_full", "doclaynet", "document", media, "multi", LAB["instructions"]["doc_layout"][B1.LANGS.index(L)],
                              L, [B1.cand(c, label(c, L), L) for c in LAY], [c for c in LAY if c in pres], g, "en-US", "visual_only",
                              slices=["layout_replay_new"], **common))
            small = [c for c in pres if area[c] < 0.015]
            if small:
                L = B1.pick_lang(rng)
                cl = rng.sample(LAY, rng.randint(5, 8))
                cl = list(dict.fromkeys(cl + small))
                rng.shuffle(cl)
                out.append(B15.mk(f"{base}/layout_small", "doclaynet", "document", media, "multi", LAB["instructions"]["doc_layout"][B1.LANGS.index(L)],
                                  L, [B1.cand(c, label(c, L), L) for c in cl], [c for c in cl if c in pres], g, "en-US", "visual_only",
                                  slices=["layout_replay_new", "small_region"], **common))
            for a, b in PAIRS:
                if (a in pres) != (b in pres):
                    truth = rng.random() < 0.5
                    x = (a if a in pres else b) if truth else (b if a in pres else a)
                    L = B1.pick_lang(rng)
                    out.append(B15.mk(f"{base}/contains_{x}", "doclaynet", "document", media, "noul", CONTAINS[B1.LANGS.index(L)].format(x=label(x, L)),
                                      L, [B1.cand("false", "false", L), B1.cand("true", "true", L)], ["true" if x in pres else "false"], g, "en-US",
                                      "visual_only", slices=["layout_replay_new", "confusable_pair"], **common))
                    break
    return out


def build() -> None:
    rng = random.Random(CFG["seed"])
    D.mkdir(parents=True, exist_ok=True)
    for sp in ("val_fit", "val_eval"):                       # identical evaluation decisions to M1.5
        shutil.copy(D15 / f"{sp}.jsonl", D / f"{sp}.jsonl")
    train15 = list(jl(D15 / "train.jsonl"))
    caps = CFG["data"]["preservation_replay_caps"]
    by = defaultdict(list)
    for r in train15:
        key = r["source"]
        if key == "doclaynet":
            key = "doclaynet_category" if r["id"].endswith("/category") else "doclaynet_layout"
        by[key].append(r)
    out = []
    for k, rs in sorted(by.items()):
        if k in caps:
            out += rng.sample(rs, min(caps[k], len(rs)))
    # layout replay: ALL M1 train layout decisions (M1 records -> m15 schema), x2
    layout_m1 = [B15.upgrade_m1(r) for r in jl(D1 / "train.jsonl") if r["source"] == "doclaynet" and r["id"].endswith("/layout")]
    new = new_layout_records(rng)
    share = CFG["data"].get("layout_share")
    if share is None:                                       # M1.5b: M1 layout x2 + every new layout decision
        for rep in (0, 1):
            out += [dict(r, id=f"{r['id']}#rep{rep}") for r in layout_m1]
        out += new
    else:                                                   # M1.5c: layout = share of the train decisions, each at most once
        target = int(round(share * len(out) / (1 - share)))
        rng.shuffle(new)
        pool = [dict(r, id=f"{r['id']}#rep0") for r in layout_m1] + new
        out += pool[:target]
        print("layout target", target, "pool", len(pool))
    with open(D / "train_pre.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"n": len(out), "layout_m1": len(layout_m1), "by": dict(Counter(r["source"] + ("/" + r["id"].rsplit("/", 1)[-1].split("#")[0]
                                                                                                    if r["source"] == "doclaynet" else "") for r in out))}, indent=1))


def ocr_missing() -> None:
    from krun_one.v1.multimodal.m15_ocr import build_ocr, chosen_ocr

    have = {x["path"] for f in sorted(D15.glob("ocr.*.jsonl")) for x in jl(f)}
    need = sorted({r["media"]["path"] for r in jl(D / "train_pre.jsonl") if r["modality"] == "document"} - have)
    ocr = build_ocr(chosen_ocr(ROOT))
    with open(D / "ocr_extra.jsonl", "w", encoding="utf-8") as f:
        for s in range(0, len(need), 16):
            for p, t in zip(need[s:s + 16], ocr.read([str(D1 / p) for p in need[s:s + 16]])):
                f.write(json.dumps({"path": p, "text": t}, ensure_ascii=False) + "\n")
    print("ocr extra", len(need))


def finalize() -> None:
    ocr = {x["path"]: x["text"] for f in list(sorted(D15.glob("ocr.*.jsonl"))) + [D / "ocr_extra.jsonl"] if f.exists() for x in jl(f)}
    asr = {x["path"]: x["text"] for x in jl(D15 / "asr.jsonl")}
    w = CFG["loss"]["layout_sample_weight"]
    out, miss = [], Counter()
    from krun_one.v1.leakage import normalize

    held = {normalize(r["gt_text"]) for sp in ("val_fit", "val_eval") for r in jl(D / f"{sp}.jsonl") if r.get("gt_text")}
    held_sha = {r["media"]["sha256"] for sp in ("val_fit", "val_eval") for r in jl(D / f"{sp}.jsonl")}
    held_sha |= {r["media"].get("sha256") for r in jl(D15 / "protected.jsonl")}
    dropped = 0
    for r in jl(D / "train_pre.jsonl"):
        p = r["media"]["path"]
        if r["modality"] == "document" and r["context"].get("source") is None:
            if p not in ocr:
                miss[r["source"]] += 1
                continue
            r["context"] = {"text": ocr[p], "language": r["context"]["language"], "source": "ocr:rapidocr:v5-latin"}
        elif r["modality"] == "audio" and r["context"].get("source") is None:
            r["context"] = {"text": asr.get(p, ""), "language": r["context"]["language"], "source": "asr:whisper-small"}
        if "sha256" not in r["media"]:
            r["media"]["sha256"] = sha(D1 / p)
        if r["media"]["sha256"] in held_sha or (r.get("gt_text") and r.get("teacher_filter") != "never" and normalize(r["gt_text"]) in held):
            dropped += 1
            continue
        r["loss_weight"] = w if (r["source"] == "doclaynet" and r["task_class"] == "visual_only") else 1.0
        out.append(r)
    assert all(B1.split_of(r["split_group_id"]) == "train" for r in out), "a replay decision is not in a train group"
    rng = random.Random(CFG["seed"] + 1)
    rng.shuffle(out)
    with open(D / "train.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    man = {"schema": "krun.m15.v1 (+loss_weight)", "config": str(CFG_PATH.relative_to(ROOT)), "config_sha256": sha(CFG_PATH),
           "files": {sp: {"n": sum(1 for _ in jl(D / f"{sp}.jsonl")), "sha256": sha(D / f"{sp}.jsonl")} for sp in ("train", "val_fit", "val_eval")},
           "val_identical_to_m15": all(sha(D / f"{sp}.jsonl") == sha(D15 / f"{sp}.jsonl") for sp in ("val_fit", "val_eval")),
           "missing_ocr": dict(miss), "dropped_for_leakage_hygiene": dropped,
           "by_source_task": dict(Counter(f"{r['source']}/{r['task_class']}/{r['task']['type']}" for r in out)),
           "weighted_decisions": sum(1 for r in out if r["loss_weight"] != 1.0)}
    (D / "manifest.json").write_text(json.dumps(man, indent=1))
    with open(D / "textview_train.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            for tag, txt in (("gt", r.get("gt_text")), ("ctx", r["context"]["text"])):
                if txt:
                    v = B1.text_view(dict(r, paired_text=txt))
                    v["id"] = f"{r['id']}#{tag}"
                    f.write(json.dumps(v, ensure_ascii=False) + "\n")
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "ocr_missing", "finalize"])
    {"build": build, "ocr_missing": ocr_missing, "finalize": finalize}[ap.parse_args().stage]()
