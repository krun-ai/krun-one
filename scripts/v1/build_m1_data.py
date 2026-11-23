"""Krun One v1 — M1 multimodal alignment dataset (Krun M1 v1): image / visual document / audio decisions.

Stages (run on the pod; idempotent):
    fetch      COCO 2017 annotations + license-filtered images (ids {4: CC BY 2.0, 7: no known copyright}), DocLayNet-base,
               CORD v2, MInDS-14 (en-US/en-GB/en-AU/fr-FR/es-ES/pt-PT)
    render     synthetic app screenshots (kd1 pool utterances) + synthetic business documents (12 types x 4 languages)
    tts_jobs   data/m1/tts_jobs.jsonl for scripts/v1/m1_tts.py (Kokoro-82M, separate venv)
    decisions  krun.m1.v1 records -> data/m1/{train,val_fit,val_eval,protected}.jsonl + text views + manifest

Sources and licenses: docs/KRUN_ONE_V1_M1_SOURCES_AUDIT.md. Splits: by group hash (image id / document / kd1
split_group_id / MInDS clip / DocLayNet original file) 80 / 5 / 7.5 / 7.5 (train / val_fit / val_eval / protected).
``protected`` is written and hashed but never read by training, calibration, retrieval, generation or mining.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import sys
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
D = ROOT / "data/m1"
MEDIA = D / "media"
RAW = D / "raw"
LAB = yaml.safe_load((ROOT / "configs/v1/m1_labels.yaml").read_text())
LANGS = ["en", "pt", "es", "fr"]
LANG_TAG = {"en": "en-US", "pt": "pt-BR", "es": "es-ES", "fr": "fr-FR"}
COCO_OK_LICENSES = {4, 7}          # CC BY 2.0, No known copyright restrictions (audit: id 5 BY-SA left out)
MINDS_LOCALES = ["en-US", "en-GB", "en-AU", "fr-FR", "es-ES", "pt-PT"]
SEED = 20260929


def split_of(group: str) -> str:
    b = int(hashlib.sha1(f"m1|{group}".encode()).hexdigest()[:8], 16) % 1000
    return "train" if b < 800 else "val_fit" if b < 850 else "val_eval" if b < 925 else "protected"


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def hf_get(url: str, dst: Path) -> Path:
    import urllib.request

    if dst.exists() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "krun-m1/1.0"})
    with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as f:
        while True:
            b = r.read(1 << 22)
            if not b:
                break
            f.write(b)
    tmp.rename(dst)
    return dst


# ================================================================================================ fetch
def fetch() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    ann = hf_get("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", RAW / "coco_ann.zip")
    with zipfile.ZipFile(ann) as z:
        for n in ("annotations/instances_train2017.json", "annotations/instances_val2017.json",
                  "annotations/captions_train2017.json", "annotations/captions_val2017.json"):
            if not (RAW / Path(n).name).exists():
                (RAW / Path(n).name).write_bytes(z.read(n))
    imgs = []
    for part in ("train2017", "val2017"):
        inst = json.loads((RAW / f"instances_{part}.json").read_text())
        imgs += [(part, im) for im in inst["images"] if im["license"] in COCO_OK_LICENSES]
    out = MEDIA / "coco"
    out.mkdir(parents=True, exist_ok=True)

    def get(x):
        part, im = x
        p = out / im["file_name"]
        if not p.exists():
            try:
                hf_get(im["coco_url"], p)
            except Exception as e:  # noqa: BLE001
                return f"fail {im['file_name']} {e}"
        return None

    with ThreadPoolExecutor(64) as ex:
        fails = [r for r in ex.map(get, imgs) if r]
    print(f"coco: {len(imgs)} license-ok images, {len(fails)} failures", flush=True)
    base = "https://huggingface.co/api/datasets"
    for ds, cfg, splits in [("pierreguillou/DocLayNet-base", "DocLayNet_2022.08_processed_on_2023.01", ["train", "validation", "test"]),
                            ("naver-clova-ix/cord-v2", "default", ["train", "validation", "test"])]:
        listing = json.loads(_http(f"{base}/{ds}/parquet"))
        for sp in splits:
            for i, u in enumerate(listing[cfg][sp]):
                hf_get(u, RAW / ds.replace("/", "__") / f"{sp}-{i}.parquet")
    listing = json.loads(_http(f"{base}/PolyAI/minds14/parquet"))
    for loc in MINDS_LOCALES:
        for i, u in enumerate(listing[loc]["train"]):
            hf_get(u, RAW / "minds14" / f"{loc}-{i}.parquet")
    print("fetch done", flush=True)


def _http(url: str) -> str:
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "krun-m1/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read().decode()


# ================================================================================================ render
def _render_screen(job):
    from krun_one.v1.multimodal.m1_render import render_screenshot

    rid, text, lang, seed = job
    p = MEDIA / "screens" / f"{rid}.jpg"
    if not p.exists():
        render_screenshot(text, lang, seed).save(p, quality=90)
    return rid


def _render_doc(job):
    from krun_one.v1.multimodal.m1_render import render_document

    did, dtype, lang, seed = job
    p = MEDIA / "docs" / f"{did}.jpg"
    meta = MEDIA / "docs" / f"{did}.json"
    if not meta.exists():
        title = LAB["doc_types"][dtype][LANGS.index(lang)]
        im, text, truth = render_document(dtype, title, lang, seed)
        im.save(p, quality=88)
        meta.write_text(json.dumps({"id": did, "doc_type": dtype, "lang": lang, "text": text, "truth": truth}, ensure_ascii=False))
    return did


def kd1_pool() -> list[dict]:
    out = []
    for l in open(D / "kd1_pool.jsonl", encoding="utf-8").read().split("\n"):
        if l:
            r = json.loads(l)
            r["_rid"] = hashlib.sha1(r["id"].encode()).hexdigest()[:16]
            out.append(r)
    lim = int(os.environ.get("M1_LIMIT", "0"))
    return out[:lim] if lim else out


def render(n_docs: int = 14000) -> None:
    (MEDIA / "screens").mkdir(parents=True, exist_ok=True)
    (MEDIA / "docs").mkdir(parents=True, exist_ok=True)
    pool = kd1_pool()
    jobs = [(r["_rid"], r["context"]["text"], r["context"]["language"].split("-")[0], SEED + i) for i, r in enumerate(pool)]
    workers = max(1, min(24, (os.cpu_count() or 2) - 2))   # 94 workers + TTS hit the container memory limit (OOM)
    with ProcessPoolExecutor(workers) as ex:
        n = sum(1 for _ in ex.map(_render_screen, jobs, chunksize=64))
    print("screens", n, flush=True)
    rng = random.Random(SEED)
    types = list(LAB["doc_types"])
    djobs = [(f"doc{i:06d}", types[i % len(types)], LANGS[(i // len(types)) % 4], SEED * 7 + i) for i in range(n_docs)]
    rng.shuffle(djobs)
    with ProcessPoolExecutor(workers) as ex:
        n = sum(1 for _ in ex.map(_render_doc, djobs, chunksize=32))
    print("docs", n, flush=True)


def tts_jobs() -> None:
    """Every kd1 pool utterance, in priority order: all evaluation-side records first, then train (the TTS script
    stops at its time budget, so evaluation-side coverage is complete)."""
    pool = kd1_pool()
    rng = random.Random(SEED + 1)
    rng.shuffle(pool)
    pool.sort(key=lambda r: 0 if split_of("kd1|" + r["split_group_id"]) != "train" else 1)
    with open(D / "tts_jobs.jsonl", "w", encoding="utf-8") as f:
        for r in pool:
            f.write(json.dumps({"id": r["_rid"], "text": r["context"]["text"], "lang": r["context"]["language"]},
                               ensure_ascii=False) + "\n")
    print("tts jobs", len(pool))


# ================================================================================================ decisions
def cand(cid: str, label: str, lang: str, desc: str | None = None) -> dict:
    return {"id": cid, "label": label, "description": desc, "language": lang}


def rec(rid, source, modality, media, qtype, instructions, lang_ins, cands, gold, group, lang_media, paired=None,
        teacher_filter="if_correct", slices=(), answerable=True, family=None, cross_group=None, criteria=None) -> dict:
    if qtype == "multi":
        answerable = bool(gold)
    return {"schema_version": "krun.m1.v1", "id": rid, "split": split_of(group), "source": source, "modality": modality,
            "media": media, "task": {"type": qtype, "instructions": instructions, "instructions_language": lang_ins,
                                     "criteria": criteria, "family": family or modality},
            "context": {"text": "", "language": lang_media}, "candidates": cands, "target": {"labels": list(gold)},
            "answerable": answerable, "paired_text": paired, "teacher_filter": teacher_filter,
            "slices": list(slices), "split_group_id": group, "cross_modal_group": cross_group}


def coco_label(cid: str, lang: str) -> str:
    return cid.replace("_", " ") if lang == "en" else LAB["coco"][cid][LANGS.index(lang) - 1]


def pick_lang(rng) -> str:
    return rng.choices(LANGS, weights=[0.4, 0.2, 0.2, 0.2])[0]


def coco_records(rng) -> list[dict]:
    out = []
    for part in ("train2017", "val2017"):
        inst = json.loads((RAW / f"instances_{part}.json").read_text())
        caps = defaultdict(list)
        for a in json.loads((RAW / f"captions_{part}.json").read_text())["annotations"]:
            caps[a["image_id"]].append(a["caption"].strip())
        cats = {c["id"]: c["name"].replace(" ", "_") for c in inst["categories"]}
        supercat = {c["name"].replace(" ", "_"): c["supercategory"] for c in inst["categories"]}
        names = sorted(set(cats.values()))
        by_img = defaultdict(list)
        for a in inst["annotations"]:
            by_img[a["image_id"]].append(a)
        imgs = [im for im in inst["images"] if im["license"] in COCO_OK_LICENSES and (MEDIA / "coco" / im["file_name"]).exists()]
        all_caps = [(im["id"], c) for im in imgs for c in caps[im["id"]]]
        for im in imgs:
            A = im["width"] * im["height"]
            anns = by_img[im["id"]]
            area = Counter()
            crowd = set()
            for a in anns:
                area[cats[a["category_id"]]] += a["area"]
                if a["iscrowd"]:
                    crowd.add(cats[a["category_id"]])
            present = {c for c, v in area.items() if v / A >= 0.02}
            anypres = set(area)
            absent = [c for c in names if c not in anypres]
            g = f"coco|{im['id']}"
            media = {"path": f"media/coco/{im['file_name']}", "kind": "image"}
            cap = rng.choice(caps[im["id"]]) if caps[im["id"]] else None
            base = f"m1/coco/{im['id']}"
            top = area.most_common(2)
            # main subject (choice)
            if top and top[0][1] / A >= 0.10 and (len(top) == 1 or top[0][1] >= 1.5 * top[1][1]):
                gold = top[0][0]
                k = rng.choice([2, 4, 4, 8, 8, 16])
                hard = [c for c in absent if supercat[c] == supercat[gold]]
                negs = rng.sample(hard, min(len(hard), (k - 1) // 2)) if rng.random() < .5 else []
                negs += rng.sample([c for c in absent if c not in negs], k - 1 - len(negs))
                unans = rng.random() < 0.08
                cl = negs + ([rng.choice([c for c in absent if c not in negs])] if unans else [gold])
                rng.shuffle(cl)
                L = pick_lang(rng)
                out.append(rec(f"{base}/main", "coco", "image", media, "choice", LAB["instructions"]["image_main"][LANGS.index(L)], L,
                               [cand(c, coco_label(c, L), L) for c in cl], [] if unans else [gold], g, "en", cap,
                               slices=["hard_negative"] * bool(len(negs) and any(supercat[n] == supercat[gold] for n in negs))
                               + ["unanswerable"] * unans + ([] if L == "en" else ["cross_language"]),
                               answerable=not unans))
            # multi: which objects appear
            if present and rng.random() < 0.6:
                pos = rng.sample(sorted(present), min(len(present), rng.randint(1, 4)))
                k = rng.randint(6, 10)
                cl = pos + rng.sample(absent, k - len(pos))
                rng.shuffle(cl)
                L = pick_lang(rng)
                out.append(rec(f"{base}/multi", "coco", "image", media, "multi", LAB["instructions"]["image_multi"][LANGS.index(L)], L,
                               [cand(c, coco_label(c, L), L) for c in cl], pos, g, "en", cap))
            # noul: contains X
            if rng.random() < 0.8:
                truth = rng.random() < 0.5 and present
                x = rng.choice(sorted(present)) if truth else rng.choice(absent)
                L = pick_lang(rng)
                ins = LAB["instructions"]["image_contains"][LANGS.index(L)].format(x=coco_label(x, L))
                out.append(rec(f"{base}/noul", "coco", "image", media, "noul", ins, L,
                               [cand("false", "false", L), cand("true", "true", L)], ["true" if truth else "false"], g, "en", cap))
            # score: people count
            if "person" not in crowd and rng.random() < 0.5:
                n = sum(1 for a in anns if cats[a["category_id"]] == "person" and a["area"] / A >= 0.005)
                lvl = 0 if n == 0 else 1 if n == 1 else 2 if n == 2 else 3 if n <= 5 else 4
                L = pick_lang(rng)
                lv = LAB["people_levels"][LANGS.index(L)]
                out.append(rec(f"{base}/people", "coco", "image", media, "score", LAB["instructions"]["image_people"][LANGS.index(L)], L,
                               [cand(str(i), x, L) for i, x in enumerate(lv)], [str(lvl)], g, "en", cap))
            # caption match (k = 4 captions, descriptions)
            if len(caps[im["id"]]) >= 2 and rng.random() < 0.5:
                gold_cap = rng.choice([c for c in caps[im["id"]] if c != cap] or caps[im["id"]])
                others = []
                while len(others) < 3:
                    oid, oc = rng.choice(all_caps)
                    if oid != im["id"]:
                        others.append(oc)
                cl = [("gold", gold_cap)] + [(f"n{i}", c) for i, c in enumerate(others)]
                rng.shuffle(cl)
                ids = "abcd"
                cands = [cand(ids[i], ids[i], "en", c) for i, (_, c) in enumerate(cl)]
                goldid = ids[[i for i, (t, _) in enumerate(cl) if t == "gold"][0]]
                out.append(rec(f"{base}/caption", "coco", "image", media, "choice", LAB["instructions"]["image_caption"][0], "en",
                               cands, [goldid], g, "en", cap, family="image_caption"))
    return out


def amount_level(v: float) -> int:
    return 0 if v < 100 else 1 if v < 500 else 2 if v < 2000 else 3


def doc_records(rng) -> list[dict]:
    out = []
    types = list(LAB["doc_types"])
    fields = list(LAB["doc_fields"])
    for p in sorted((MEDIA / "docs").glob("*.json")):
        m = json.loads(p.read_text())
        did, dt, lang, tr = m["id"], m["doc_type"], m["lang"], m["truth"]
        g = f"doc|{did}"
        media = {"path": f"media/docs/{did}.jpg", "kind": "document"}
        base = f"m1/doc/{did}"
        tasks = []
        L = pick_lang(rng)
        k = rng.choice([4, 6, 8, 12])
        cl = [dt] + rng.sample([t for t in types if t != dt], k - 1)
        rng.shuffle(cl)
        tasks.append(rec(f"{base}/type", "synthdoc", "document", media, "choice", LAB["instructions"]["doc_type"][LANGS.index(L)], L,
                         [cand(t, LAB["doc_types"][t][LANGS.index(L)], L) for t in cl], [dt], g, LANG_TAG[lang], m["text"],
                         slices=[] if L == lang else ["cross_language"]))
        L = pick_lang(rng)
        cl = rng.sample(fields, rng.randint(6, 8))
        gold = [f for f in cl if tr["fields"][f]]
        tasks.append(rec(f"{base}/fields", "synthdoc", "document", media, "multi", LAB["instructions"]["doc_fields"][LANGS.index(L)], L,
                         [cand(f, LAB["doc_fields"][f][LANGS.index(L)], L) for f in cl], gold, g, LANG_TAG[lang], m["text"]))
        noul = [("paid", tr["paid"], LAB["instructions"]["doc_paid"]), ("signed", tr["signed"], LAB["instructions"]["doc_signed"])]
        if tr["total"] is not None:
            x = round(tr["total"] * rng.choice([0.4, 0.6, 0.8, 1.25, 1.6, 2.5]))
            noul.append(("total_gt", tr["total"] > x, [s.format(x=x) for s in LAB["instructions"]["doc_total_gt"]]))
            L = pick_lang(rng)
            lv = LAB["amount_levels"][LANGS.index(L)]
            tasks.append(rec(f"{base}/amount", "synthdoc", "document", media, "score", LAB["instructions"]["doc_amount"][LANGS.index(L)], L,
                             [cand(str(i), x_, L) for i, x_ in enumerate(lv)], [str(amount_level(tr["total"]))], g, LANG_TAG[lang], m["text"]))
        name, val, ins = rng.choice(noul)
        L = pick_lang(rng)
        tasks.append(rec(f"{base}/{name}", "synthdoc", "document", media, "noul", ins[LANGS.index(L)], L,
                         [cand("false", "false", L), cand("true", "true", L)], ["true" if val else "false"], g, LANG_TAG[lang], m["text"]))
        out += rng.sample(tasks, min(3, len(tasks)))
    return out


def doclaynet_cord_records(rng) -> list[dict]:
    import pyarrow.parquet as pq

    out = []
    cats = list(LAB["doclaynet"])
    names = ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title']
    lay = list(LAB["doclaynet_layout"])
    (MEDIA / "doclaynet").mkdir(parents=True, exist_ok=True)
    for f in sorted((RAW / "pierreguillou__DocLayNet-base").glob("*.parquet")):
        t = pq.read_table(f, columns=["page_hash", "image", "texts", "categories", "original_filename", "doc_category"])
        for row in t.to_pylist():
            p = MEDIA / "doclaynet" / f"{row['page_hash']}.jpg"
            if not p.exists():
                p.write_bytes(row["image"]["bytes"])
            g = f"dln|{row['original_filename']}"
            media = {"path": f"media/doclaynet/{p.name}", "kind": "document"}
            text = "\n".join(row["texts"])[:1500] or None
            base = f"m1/doclaynet/{row['page_hash'][:16]}"
            L = pick_lang(rng)
            k = rng.choice([3, 4, 6])
            cl = [row["doc_category"]] + rng.sample([c for c in cats if c != row["doc_category"]], k - 1)
            rng.shuffle(cl)
            out.append(rec(f"{base}/category", "doclaynet", "document", media, "choice", LAB["instructions"]["doc_category"][LANGS.index(L)], L,
                           [cand(c, LAB["doclaynet"][c][LANGS.index(L)], L) for c in cl], [row["doc_category"]], g, "en-US", text))
            if rng.random() < 0.7:
                pres = {names[c] for c in row["categories"]}
                cl = rng.sample(lay, rng.randint(5, 8))
                L = pick_lang(rng)
                out.append(rec(f"{base}/layout", "doclaynet", "document", media, "multi", LAB["instructions"]["doc_layout"][LANGS.index(L)], L,
                               [cand(c, LAB["doclaynet_layout"][c][LANGS.index(L)], L) for c in cl], [c for c in cl if c in pres], g,
                               "en-US", text))
    (MEDIA / "cord").mkdir(parents=True, exist_ok=True)
    types = list(LAB["doc_types"])
    for f in sorted((RAW / "naver-clova-ix__cord-v2").glob("*.parquet")):
        t = pq.read_table(f, columns=["image"])
        for i, row in enumerate(t.to_pylist()):
            b = row["image"]["bytes"]
            h = sha_bytes(b)[:16]
            p = MEDIA / "cord" / f"{h}.jpg"
            if not p.exists():
                p.write_bytes(b)
            L = pick_lang(rng)
            k = rng.choice([4, 6, 8])
            cl = ["receipt"] + rng.sample([t_ for t_ in types if t_ != "receipt"], k - 1)
            rng.shuffle(cl)
            out.append(rec(f"m1/cord/{h}/type", "cord", "document", {"path": f"media/cord/{p.name}", "kind": "document"}, "choice",
                           LAB["instructions"]["doc_type"][LANGS.index(L)], L, [cand(t_, LAB["doc_types"][t_][LANGS.index(L)], L) for t_ in cl],
                           ["receipt"], f"cord|{h}", "id-ID", None, slices=["real_document"]))
    return out


def minds_records(rng) -> list[dict]:
    import pyarrow.parquet as pq
    import soundfile as sf

    from krun_one.v1.multimodal.m1_towers import load_audio  # noqa: F401  (resampling helper lives there)

    out = []
    intents = list(LAB["minds14"])
    (MEDIA / "minds14").mkdir(parents=True, exist_ok=True)
    for f in sorted((RAW / "minds14").glob("*.parquet")):
        loc = f.name.rsplit("-", 1)[0]
        t = pq.read_table(f)
        for row in t.to_pylist():
            b = row["audio"]["bytes"]
            h = sha_bytes(b)[:16]
            p = MEDIA / "minds14" / f"{h}.flac"
            if not p.exists():
                w, sr = sf.read(io.BytesIO(b), dtype="float32")
                if w.ndim > 1:
                    w = w.mean(1)
                n = int(round(len(w) * 16000 / sr))
                w = np.interp(np.linspace(0, len(w) - 1, n), np.arange(len(w)), w).astype(np.float32)
                sf.write(p, w, 16000)
            gold = intents[row["intent_class"]]
            g = f"minds|{h}"
            media = {"path": f"media/minds14/{p.name}", "kind": "audio"}
            ml = loc.split("-")[0]
            base = f"m1/minds14/{h}"
            L = ml if rng.random() < 0.6 else pick_lang(rng)
            k = rng.choice([4, 8, 14])
            cl = [gold] + rng.sample([c for c in intents if c != gold], k - 1)
            rng.shuffle(cl)
            out.append(rec(f"{base}/intent", "minds14", "audio", media, "choice", LAB["instructions"]["audio_intent"][LANGS.index(L)], L,
                           [cand(c, c, L, LAB["minds14"][c][LANGS.index(L)]) for c in cl], [gold], g, loc, row["transcription"],
                           teacher_filter="always", slices=["real_speech"] + ([] if L == ml else ["cross_language"]), family="audio_intent"))
            truth = rng.random() < 0.5
            x = gold if truth else rng.choice([c for c in intents if c != gold])
            L = ml if rng.random() < 0.6 else pick_lang(rng)
            out.append(rec(f"{base}/is", "minds14", "audio", media, "noul",
                           LAB["instructions"]["audio_is"][LANGS.index(L)].format(x=LAB["minds14"][x][LANGS.index(L)]), L,
                           [cand("false", "false", L), cand("true", "true", L)], ["true" if truth else "false"], g, loc,
                           row["transcription"], teacher_filter="always", slices=["real_speech"]))
    return out


def kd1_records() -> list[dict]:
    """Every kd1 pool decision with its context moved into a screenshot (document modality) and, when synthesized,
    into speech (audio modality). paired_text = the original text context (teacher = frozen text model)."""
    tts = {}
    for mf in sorted(D.glob("tts_manifest.*.jsonl")):
        for l in open(mf):
            x = json.loads(l)
            if (MEDIA / "tts" / f"{x['id']}.flac").exists():
                tts[x["id"]] = x
    out = []
    for r in kd1_pool():
        g = "kd1|" + r["split_group_id"]
        for mod, media in (("document", {"path": f"media/screens/{r['_rid']}.jpg", "kind": "screenshot"}),
                           ("audio", {"path": f"media/tts/{r['_rid']}.flac", "kind": "tts", "voice": tts.get(r["_rid"], {}).get("voice")})):
            if mod == "audio" and r["_rid"] not in tts:
                continue
            if mod == "document" and not (ROOT / "data/m1" / media["path"]).exists():
                continue
            x = {k: v for k, v in r.items() if not k.startswith("_") and k not in ("metadata",)}
            x.update({"schema_version": "krun.m1.v1", "id": f"m1/{'screen' if mod == 'document' else 'tts'}/{r['_rid']}",
                      "split": split_of(g), "source": "kd1_screen" if mod == "document" else "kd1_tts", "modality": mod,
                      "media": media, "paired_text": r["context"]["text"], "teacher_filter": "always",
                      "split_group_id": g, "cross_modal_group": r["_rid"], "kd1_id": r["id"]})
            x["context"] = {"text": "", "language": r["context"]["language"]}
            x["slices"] = list(r.get("slices") or []) + ["synthetic_speech" if mod == "audio" else "screenshot"]
            out.append(x)
    return out


def text_view(r: dict) -> dict:
    """The record as a text decision (context = paired text) for the KrunBench leakage checker."""
    return {"id": r["id"], "split": "train", "task": r["task"], "context": {"text": r.get("paired_text") or "", "language": r["context"]["language"]},
            "candidates": r["candidates"], "target": r["target"], "answerable": r["answerable"],
            "split_group_id": r["split_group_id"], "slices": r["slices"]}


def decisions() -> None:
    rng = random.Random(SEED + 2)
    parts = {}
    for name, fn in (("coco", lambda: coco_records(rng)), ("synthdoc", lambda: doc_records(rng)),
                     ("doclaynet+cord", lambda: doclaynet_cord_records(rng)), ("minds14", lambda: minds_records(rng)),
                     ("kd1", kd1_records)):
        try:
            parts[name] = fn()
        except FileNotFoundError as e:
            if not os.environ.get("M1_PARTIAL"):
                raise
            print("skip", name, e)
            parts[name] = []
    recs = [r for v in parts.values() for r in v]
    # media hashes (exact duplicate media across splits is leakage)
    msha, by_sha = {}, defaultdict(set)
    for r in recs:
        p = D / r["media"]["path"]
        if p.as_posix() not in msha:
            msha[p.as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
        r["media"]["sha256"] = msha[p.as_posix()]
        by_sha[r["media"]["sha256"]].add(r["split"])
    dup_media = {h for h, s in by_sha.items() if len(s) > 1}
    # exact normalized paired-text overlap: eval/protected text must not appear in train
    from krun_one.v1.leakage import normalize

    held = {normalize(r["paired_text"]) for r in recs if r["split"] != "train" and r.get("paired_text")}
    drop = [r["id"] for r in recs if r["split"] == "train" and (r["media"]["sha256"] in dup_media or
                                                                  (r.get("paired_text") and normalize(r["paired_text"]) in held))]
    dset = set(drop)
    recs = [r for r in recs if r["id"] not in dset]
    groups = defaultdict(set)
    for r in recs:
        groups[r["split_group_id"]].add(r["split"])
    assert all(len(s) == 1 for s in groups.values()), "group crosses splits"
    by_split = defaultdict(list)
    for r in recs:
        by_split[r["split"]].append(r)
    man = {"schema": "krun.m1.v1", "seed": SEED, "split_rule": "sha1('m1|'+group) % 1000: <800 train, <850 val_fit, <925 val_eval, else protected",
           "files": {}, "dropped_train_for_leakage": len(drop), "dup_media_across_splits": len(dup_media)}
    for sp, rs in by_split.items():
        rs.sort(key=lambda r: r["id"])
        p = D / f"{sp}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        man["files"][sp] = {"n": len(rs), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        if sp != "protected":
            with open(D / f"textview_{sp}.jsonl", "w", encoding="utf-8") as f:
                for r in rs:
                    f.write(json.dumps(text_view(r), ensure_ascii=False) + "\n")
    stats = defaultdict(Counter)
    for r in recs:
        stats["split"][r["split"]] += 1
        stats["source"][r["source"]] += 1
        stats["modality"][r["modality"]] += 1
        stats["type"][f"{r['modality']}/{r['task']['type']}"] += 1
        stats["media_lang"][r["context"]["language"]] += 1
        stats["schema_lang"][r["task"].get("instructions_language")] += 1
        stats["train_source"][r["source"]] += r["split"] == "train"
        stats["paired"][bool(r.get("paired_text"))] += 1
    man["stats"] = {k: dict(v) for k, v in stats.items()}
    man["n_media"] = len(msha)
    (D / "manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fetch", "render", "tts_jobs", "decisions"])
    ap.add_argument("--n-docs", type=int, default=14000)
    a = ap.parse_args()
    {"fetch": fetch, "render": lambda: render(a.n_docs), "tts_jobs": tts_jobs, "decisions": decisions}[a.stage]()
