"""Krun One v1 — M1.5 dataset (configs/v1/m15.yaml). Runs on the pod after the M1 media rebuild (build_m1_data.py
fetch / render / tts, deterministic).

    python scripts/v1/build_m15_data.py verify_media     # rebuilt M1 media vs the sha256 in the M1 records
    python scripts/v1/build_m15_data.py fetch_new        # FLEURS (en_us / pt_br / es_419 / fr_fr) + SIB-200, CREMA-D
    python scripts/v1/build_m15_data.py select           # the M1.5 mixture (records only) -> data/m15/selected.jsonl
    python scripts/v1/build_m15_data.py asr              # Whisper-small transcripts of every selected clip
    python scripts/v1/build_m15_data.py ocr --backend X  # OCR text of every selected document / screenshot page
    python scripts/v1/build_m15_data.py decisions        # data/m15/{train,val_fit,val_eval,protected}.jsonl + textviews

Record additions over krun.m1.v1: ``context.text`` = ASR transcript / OCR text (empty for natural images),
``gt_text`` (ground-truth transcript / document text; the teacher's input and the oracle), ``task_class``
(visual_only | text_dependent | hybrid | acoustic_only), ``speech`` (real | synthetic | null), ``semantic_group_id``.
Splits: the M1 split function (group hash) for every group, so M1 and M1.5 splits agree; protected never read.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/v1"))
import build_m1_data as B1  # noqa: E402

D1 = ROOT / "data/m1"
D = ROOT / "data/m15"
RAW = D1 / "raw"
MEDIA = D1 / "media"
CFG = yaml.safe_load((ROOT / "configs/v1/m15.yaml").read_text())
LAB = B1.LAB
SEED = CFG["seed"]
SHARD, NSHARDS = 0, 1
OCR_NAME = "?"
DROP_IDS: set = set()
FLEURS = {"en_us": "en", "pt_br": "pt", "es_419": "es", "fr_fr": "fr"}
SIB = {"en": "eng_Latn", "pt": "por_Latn", "es": "spa_Latn", "fr": "fra_Latn"}
SIB_TOPICS = {"science/technology": ["science / technology", "ciência / tecnologia", "ciencia / tecnología", "science / technologie"],
              "travel": ["travel", "viagem", "viajes", "voyage"], "politics": ["politics", "política", "política", "politique"],
              "sports": ["sports", "esportes", "deportes", "sport"], "health": ["health", "saúde", "salud", "santé"],
              "entertainment": ["entertainment", "entretenimento", "entretenimiento", "divertissement"],
              "geography": ["geography", "geografia", "geografía", "géographie"]}
TOPIC_INS = ["What is the topic of the recording?", "Qual é o tema da gravação?", "¿Cuál es el tema de la grabación?",
             "Quel est le sujet de l'enregistrement ?"]
TOPIC_IS = ["The recording is about {x}.", "A gravação é sobre {x}.", "La grabación trata sobre {x}.", "L'enregistrement porte sur {x}."]
EMO = {"ANG": ["anger", "raiva", "enojo", "colère"], "DIS": ["disgust", "nojo", "asco", "dégoût"], "FEA": ["fear", "medo", "miedo", "peur"],
       "HAP": ["happiness", "alegria", "alegría", "joie"], "NEU": ["neutral", "neutro", "neutral", "neutre"], "SAD": ["sadness", "tristeza", "tristeza", "tristesse"]}
EMO_INS = ["Which emotion does the speaker express?", "Que emoção a pessoa expressa?", "¿Qué emoción expresa la persona?",
           "Quelle émotion le locuteur exprime-t-il ?"]
URG_INS = ["How intense is the speaker's emotion?", "Qual a intensidade da emoção de quem fala?", "¿Qué tan intensa es la emoción de quien habla?",
           "Quelle est l'intensité de l'émotion du locuteur ?"]
URG_LV = [["low", "medium", "high"], ["baixa", "média", "alta"], ["baja", "media", "alta"], ["faible", "moyenne", "forte"]]
CREMA_SENT = {"IEO": "It's eleven o'clock", "TIE": "That is exactly what happened", "IOM": "I'm on my way to the meeting",
              "IWW": "I wonder what this is about", "TAI": "The airplane is almost full", "MTI": "Maybe tomorrow it will be cold",
              "IWL": "I would like a new alarm clock", "ITH": "I think I have a doctor's appointment",
              "DFA": "Don't forget a jacket", "ITS": "I think I've seen this before", "TSI": "The surface is slick",
              "WSI": "We'll stop in a couple of minutes"}
# task_class of the M1 task kinds (id suffix)
TASK_CLASS = {("coco", "main"): "visual_only", ("coco", "multi"): "visual_only", ("coco", "noul"): "visual_only",
              ("coco", "people"): "visual_only", ("coco", "caption"): "visual_only",
              ("synthdoc", "type"): "hybrid", ("synthdoc", "fields"): "hybrid", ("synthdoc", "paid"): "hybrid",
              ("synthdoc", "signed"): "visual_only", ("synthdoc", "total_gt"): "text_dependent", ("synthdoc", "amount"): "text_dependent",
              ("doclaynet", "category"): "hybrid", ("doclaynet", "layout"): "visual_only", ("cord", "type"): "hybrid",
              ("minds14", "intent"): "text_dependent", ("minds14", "is"): "text_dependent"}
PLAN = {"coco": 22000, "synthdoc": 22000, "doclaynet": 7000, "cord": 1000, "kd1_screen": 16000, "kd1_tts": 12000}   # train caps


def split_of(g: str) -> str:
    return B1.split_of(g)


def jl(p: Path):
    for l in open(p, encoding="utf-8").read().split("\n"):
        if l:
            yield json.loads(l)


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ================================================================================================ media
def verify_media() -> None:
    res, upd = Counter(), {}
    for sp in ("train", "val_fit", "val_eval", "protected"):
        for r in jl(D1 / f"{sp}.jsonl"):
            p = D1 / r["media"]["path"]
            if p.as_posix() in upd:
                continue
            if not p.exists():
                upd[p.as_posix()] = None
                res[f"{r['source']}/missing"] += 1
                continue
            s = sha_file(p)
            upd[p.as_posix()] = s
            res[f"{r['source']}/{'same' if s == r['media']['sha256'] else 'changed'}"] += 1
    D.mkdir(parents=True, exist_ok=True)
    (D / "media_rebuild_check.json").write_text(json.dumps(dict(sorted(res.items())), indent=1))
    (D / "media_sha_rebuilt.json").write_text(json.dumps(upd))
    print(json.dumps(dict(sorted(res.items())), indent=1))


def extract_media() -> None:
    """The M1 builder writes DocLayNet / CORD images and MInDS-14 audio while building its decisions; M1.5 only needs
    the media (records come from the M1 split files), so run those extractors and discard their records."""
    rng = random.Random(0)
    n1 = len(B1.doclaynet_cord_records(rng))
    n2 = len(B1.minds_records(rng))
    print("extracted media for", n1, "doclaynet/cord and", n2, "minds14 records")


def audio_media() -> None:
    """Write the FLEURS / CREMA-D media only (records discarded; used when M1.5 records are reused, e.g. M1.5b)."""
    print("audio media for", len(new_audio_records(random.Random(0))), "records")


def fetch_new() -> None:
    base = "https://huggingface.co/api/datasets"
    lst = json.loads(B1._http(f"{base}/google/fleurs/parquet"))
    for cfg in FLEURS:
        for sp, urls in lst[cfg].items():
            for i, u in enumerate(urls):
                B1.hf_get(u, RAW / "fleurs" / f"{cfg}-{sp}-{i}.parquet")
    lst = json.loads(B1._http(f"{base}/Davlan/sib200/parquet"))
    for l, cfg in SIB.items():
        for sp, urls in lst[cfg].items():
            for i, u in enumerate(urls):
                B1.hf_get(u, RAW / "sib200" / f"{cfg}-{sp}-{i}.parquet")
    # CREMA-D from the primary source (GitHub, git-LFS media URLs; ODbL + DbCL): filenames carry actor / sentence / emotion / level
    names = [l.split(",")[1].strip().strip('"') for l in B1._http(
        "https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D/master/SentenceFilenames.csv").split("\n")[1:] if "," in l]
    from concurrent.futures import ThreadPoolExecutor

    def get(n):
        try:
            B1.hf_get(f"https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/master/AudioWAV/{n}.wav", RAW / "crema" / f"{n}.wav")
            return None
        except Exception as e:  # noqa: BLE001
            return f"{n}: {e}"

    with ThreadPoolExecutor(32) as ex:
        fails = [x for x in ex.map(get, names) if x]
    print("crema", len(names), "failures", len(fails))
    print("fetch_new done")


def _norm(s: str) -> str:
    import re
    import unicodedata

    s = unicodedata.normalize("NFKC", s).lower()
    return " ".join(re.sub(r"[^\w\s]", " ", s).split())


def _save_wav(b: bytes, p: Path) -> float:
    import soundfile as sf

    if p.exists():
        return sf.info(str(p)).duration
    w, sr = sf.read(io.BytesIO(b), dtype="float32")
    if w.ndim > 1:
        w = w.mean(1)
    if sr != 16000:
        n = int(round(len(w) * 16000 / sr))
        w = np.interp(np.linspace(0, len(w) - 1, n), np.arange(len(w)), w).astype(np.float32)
    p.parent.mkdir(parents=True, exist_ok=True)
    sf.write(p, w, 16000)
    return len(w) / 16000


def _fleurs_records(rng, lang, r, top, topics) -> list[dict]:
    out = []
    b = r["audio"]["bytes"]
    h = hashlib.sha256(b).hexdigest()[:16]
    p = MEDIA / "fleurs" / f"{h}.flac"
    secs = _save_wav(b, p)
    g = f"flores|{r['id']}"                           # the sentence: same split in every language
    media = {"path": f"media/fleurs/{p.name}", "kind": "audio", "seconds": round(secs, 2)}
    base = f"m15/fleurs/{lang}/{h}"
    L = lang if rng.random() < 0.6 else B1.pick_lang(rng)
    k = rng.choice([4, 5, 7])
    cl = [top] + rng.sample([x for x in topics if x != top], k - 1)
    rng.shuffle(cl)
    tag = {"en": "en-US", "pt": "pt-BR", "es": "es-419", "fr": "fr-FR"}[lang]
    common = dict(teacher_filter="if_correct", speech="real", semantic_group_id=g, gt_text=r["raw_transcription"])
    out.append(mk(f"{base}/topic", "fleurs", "audio", media, "choice", TOPIC_INS[B1.LANGS.index(L)], L,
                  [B1.cand(x.replace("/", "_"), SIB_TOPICS[x][B1.LANGS.index(L)], L) for x in cl], [top.replace("/", "_")],
                  g, tag, "text_dependent", slices=["real_speech"] + ([] if L == lang else ["cross_language"]), **common))
    truth = rng.random() < 0.5
    x = top if truth else rng.choice([y for y in topics if y != top])
    L = lang if rng.random() < 0.6 else B1.pick_lang(rng)
    out.append(mk(f"{base}/topic_is", "fleurs", "audio", media, "noul", TOPIC_IS[B1.LANGS.index(L)].format(x=SIB_TOPICS[x][B1.LANGS.index(L)]),
                  L, [B1.cand("false", "false", L), B1.cand("true", "true", L)], ["true" if truth else "false"], g, tag,
                  "text_dependent", slices=["real_speech"], **common))
    return out


def new_audio_records(rng) -> list[dict]:
    import pyarrow.parquet as pq

    out = []
    # ---- FLEURS + SIB-200 topics (real speech, 4 languages, same sentence id across languages)
    sib = {}
    for f in sorted((RAW / "sib200").glob("*.parquet")):
        lang = {v: k for k, v in SIB.items()}[f.name.split("-")[0]]
        for r in pq.read_table(f).to_pylist():
            sib[(lang, _norm(r["text"]))] = r["category"]
    topics = list(SIB_TOPICS)
    by_id_topic = {}
    files = sorted((RAW / "fleurs").glob("*.parquet"))
    for f in files:                                     # pass 1: text only (cheap) -> topic per FLEURS sentence id
        lang = FLEURS[f.name.split("-")[0]]
        t = pq.read_table(f, columns=["id", "raw_transcription"])
        for i, tr in zip(t.column("id").to_pylist(), t.column("raw_transcription").to_pylist()):
            top = sib.get((lang, _norm(tr)))
            if top:
                by_id_topic[i] = top
    for f in files:                                     # pass 2: stream audio in batches (11 GB of parquet; 31 GB RAM cap)
        lang = FLEURS[f.name.split("-")[0]]
        for batch in pq.ParquetFile(f).iter_batches(batch_size=256, columns=["id", "audio", "raw_transcription"]):
            for r in batch.to_pylist():
                top = by_id_topic.get(r["id"])
                if not top:
                    continue
                out += _fleurs_records(rng, lang, r, top, topics)
    # ---- CREMA-D (real speech, acted emotion; the words carry no emotion -> acoustic_only)
    for f in sorted((RAW / "crema").glob("*.wav")):
        stem = f.stem                                       # e.g. 1001_DFA_ANG_XX
        parts = stem.split("_")
        if len(parts) < 4 or parts[2] not in EMO:
            continue
        actor, sent, emo, lvl = parts[:4]
        b = f.read_bytes()
        h = hashlib.sha256(b).hexdigest()[:16]
        p = MEDIA / "crema" / f"{h}.flac"
        try:
            secs = _save_wav(b, p)
        except Exception:  # noqa: BLE001  (a few CREMA-D files are known to be empty / corrupt)
            continue
        g = f"crema|{actor}"                               # speaker-disjoint splits
        media = {"path": f"media/crema/{p.name}", "kind": "audio", "seconds": round(secs, 2)}
        base = f"m15/crema/{stem}"
        L = B1.pick_lang(rng)
        k = rng.choice([3, 4, 6])
        cl = [emo] + rng.sample([e for e in EMO if e != emo], k - 1)
        rng.shuffle(cl)
        common = dict(teacher_filter="never", speech="real", semantic_group_id=None, gt_text=CREMA_SENT.get(sent))
        out.append(mk(f"{base}/emotion", "crema", "audio", media, "choice", EMO_INS[B1.LANGS.index(L)], L,
                      [B1.cand(e.lower(), EMO[e][B1.LANGS.index(L)], L) for e in cl], [emo.lower()], g, "en-US", "acoustic_only",
                      slices=["real_speech", "paralinguistic"], **common))
        if lvl in ("LO", "MD", "HI") and emo != "NEU":
            L = B1.pick_lang(rng)
            lv = URG_LV[B1.LANGS.index(L)]
            out.append(mk(f"{base}/intensity", "crema", "audio", media, "score", URG_INS[B1.LANGS.index(L)], L,
                          [B1.cand(str(i), x, L) for i, x in enumerate(lv)], [str({"LO": 0, "MD": 1, "HI": 2}[lvl])], g, "en-US",
                          "acoustic_only", slices=["real_speech", "paralinguistic"], **common))
    return out


def cord_text_records(rng) -> list[dict]:
    """Text-dependent decisions on REAL receipts (CORD v2 ground truth): item present, total above X."""
    import pyarrow.parquet as pq

    out, items_all, parsed = [], [], []
    for f in sorted((RAW / "naver-clova-ix__cord-v2").glob("*.parquet")):
        for r in pq.read_table(f, columns=["image", "ground_truth"]).to_pylist():
            gt = json.loads(r["ground_truth"])
            h = hashlib.sha256(r["image"]["bytes"]).hexdigest()[:16]
            menu = gt.get("gt_parse", {}).get("menu", [])
            menu = menu if isinstance(menu, list) else [menu]
            names = [m.get("nm") for m in menu if isinstance(m, dict) and isinstance(m.get("nm"), str) and len(m["nm"]) >= 3]
            tot = gt.get("gt_parse", {}).get("total", {})
            tp = tot.get("total_price") if isinstance(tot, dict) else None
            try:
                total = float(str(tp).replace(",", "").replace(".", "")) if tp else None
            except ValueError:
                total = None
            lines = [w["text"] for ln in gt.get("valid_line", []) for w in ln.get("words", [])]
            parsed.append((h, names, total, " ".join(lines)))
            items_all += names
    for h, names, total, text in parsed:
        if not (MEDIA / "cord" / f"{h}.jpg").exists():
            continue
        g = f"cord|{h}"
        media = {"path": f"media/cord/{h}.jpg", "kind": "document"}
        common = dict(teacher_filter="if_correct", speech=None, semantic_group_id=None, gt_text=text)
        if names:
            truth = rng.random() < 0.5
            x = rng.choice(names) if truth else rng.choice([i for i in items_all if i not in names])
            L = B1.pick_lang(rng)
            ins = ["The receipt lists the item \"{x}\".", "O recibo contém o item \"{x}\".", "El recibo incluye el artículo \"{x}\".",
                   "Le reçu contient l'article « {x} »."][B1.LANGS.index(L)].format(x=x)
            out.append(mk(f"m1/cord/{h}/item", "cord", "document", media, "noul", ins, L, [B1.cand("false", "false", L), B1.cand("true", "true", L)],
                          ["true" if truth else "false"], g, "id-ID", "text_dependent", slices=["real_document"], **common))
        if total:
            x = round(total * rng.choice([0.5, 0.8, 1.25, 2.0]))
            L = B1.pick_lang(rng)
            ins = LAB["instructions"]["doc_total_gt"][B1.LANGS.index(L)].format(x=f"{x:,}")
            out.append(mk(f"m1/cord/{h}/total_gt", "cord", "document", media, "noul", ins, L, [B1.cand("false", "false", L), B1.cand("true", "true", L)],
                          ["true" if total > x else "false"], g, "id-ID", "text_dependent", slices=["real_document"], **common))
    return out


def mk(rid, source, modality, media, qtype, ins, lang_ins, cands, gold, group, lang_media, task_class, slices=(), teacher_filter="if_correct",
       speech=None, semantic_group_id=None, gt_text=None) -> dict:
    r = B1.rec(rid, source, modality, media, qtype, ins, lang_ins, cands, gold, group, lang_media, gt_text,
               teacher_filter=teacher_filter, slices=slices, family=f"{modality}_{task_class}")
    r.update({"schema_version": "krun.m15.v1", "task_class": task_class, "speech": speech, "semantic_group_id": semantic_group_id,
              "gt_text": gt_text})
    return r


def upgrade_m1(r: dict) -> dict:
    """An M1 record -> krun.m15.v1 (task_class, speech, semantic group, gt_text = the M1 paired text)."""
    src = r["source"]
    kind = r["id"].rsplit("/", 1)[-1]
    if src == "kd1_screen":
        tc = "text_dependent"
    elif src == "kd1_tts":
        tc = "text_dependent"
    else:
        tc = TASK_CLASS.get((src, kind), "hybrid")
    r = dict(r, schema_version="krun.m15.v1", task_class=tc, gt_text=r.get("paired_text"),
             speech="synthetic" if src == "kd1_tts" else ("real" if src == "minds14" else None),
             semantic_group_id=r.get("cross_modal_group"))
    return r


def select() -> None:
    rng = random.Random(SEED)
    out = []
    per = defaultdict(list)
    for sp in ("train", "val_fit", "val_eval", "protected"):
        for r in jl(D1 / f"{sp}.jsonl"):
            per[(sp, r["source"])].append(upgrade_m1(r))
    for (sp, src), rs in sorted(per.items()):
        if sp == "train" and src in PLAN and len(rs) > PLAN[src]:
            rs = rng.sample(rs, PLAN[src])
        out += rs
    out += new_audio_records(rng)
    out += cord_text_records(rng)
    D.mkdir(parents=True, exist_ok=True)
    prot = [r for r in out if r["split"] == "protected"]          # never processed (no ASR / OCR / teacher), only hashed
    out = [r for r in out if r["split"] != "protected"]
    with open(D / "protected.jsonl", "w", encoding="utf-8") as f:
        for r in sorted(prot, key=lambda r: r["id"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(D / "selected.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    c = Counter((r["split"], r["modality"], r["source"]) for r in out)
    print(json.dumps({"/".join(k): v for k, v in sorted(c.items())}, indent=1))


# ================================================================================================ ASR / OCR passes
def media_list(kind: str) -> list[str]:
    seen, out = set(), []
    for r in jl(D / "selected.jsonl"):
        if (kind == "audio" and r["modality"] == "audio") or (kind == "ocr" and r["modality"] == "document"):
            p = r["media"]["path"]
            if p not in seen:
                seen.add(p)
                out.append(p)
    return out


def asr_pass() -> None:

    from krun_one.v1.multimodal.m1_towers import load_audio
    from krun_one.v1.multimodal.m15_asr import WhisperASR

    paths = media_list("audio")
    outp = D / "asr.jsonl"
    done = {json.loads(l)["path"] for l in open(outp)} if outp.exists() else set()
    todo = [p for p in paths if p not in done]
    asr = WhisperASR()
    from concurrent.futures import ThreadPoolExecutor

    with open(outp, "a", encoding="utf-8") as f, ThreadPoolExecutor(8) as ex:
        for s in range(0, len(todo), 48):
            chunk = todo[s:s + 48]
            wavs = list(ex.map(lambda p: load_audio(str(D1 / p)), chunk))
            feats, n = asr.logmel(wavs)
            enc = asr.encode(feats)
            texts = asr.transcribe(enc)
            for p, t, w in zip(chunk, texts, wavs):
                f.write(json.dumps({"path": p, "text": t, "seconds": round(len(w) / 16000, 2)}, ensure_ascii=False) + "\n")
            if s % 4800 == 0:
                print("asr", s, len(todo), flush=True)
    print("asr done", len(paths))


def ocr_pass(backend: str | None) -> None:
    from krun_one.v1.multimodal.m15_ocr import build_ocr, chosen_ocr

    backend = backend or chosen_ocr(ROOT)
    paths = [p for i, p in enumerate(media_list("ocr")) if i % NSHARDS == SHARD]
    outp = D / f"ocr.{SHARD}.jsonl"
    done = {json.loads(l)["path"] for l in open(outp)} if outp.exists() else set()
    todo = [p for p in paths if p not in done]
    ocr = build_ocr(backend)
    with open(outp, "a", encoding="utf-8") as f:
        for s in range(0, len(todo), 32):
            chunk = todo[s:s + 32]
            texts = ocr.read([str(D1 / p) for p in chunk])
            for p, t in zip(chunk, texts):
                f.write(json.dumps({"path": p, "text": t}, ensure_ascii=False) + "\n")
            if s % 3200 == 0:
                print("ocr", s, len(todo), flush=True)
    print("ocr done", len(paths))


# ================================================================================================ decisions
def decisions() -> None:
    global OCR_NAME
    from krun_one.v1.multimodal.m15_ocr import chosen_ocr

    OCR_NAME = chosen_ocr(ROOT)
    asr = {x["path"]: x["text"] for x in jl(D / "asr.jsonl")}
    ocr = {x["path"]: x["text"] for f in sorted(D.glob("ocr.*.jsonl")) for x in jl(f)}
    rebuilt = json.loads((D / "media_sha_rebuilt.json").read_text()) if (D / "media_sha_rebuilt.json").exists() else {}
    by_split = defaultdict(list)
    missing = Counter()
    for r in jl(D / "selected.jsonl"):
        p = r["media"]["path"]
        full = (D1 / p).as_posix()
        if not (D1 / p).exists():
            missing[r["source"]] += 1
            continue
        if full in rebuilt and rebuilt[full]:
            r["media"]["sha256"] = rebuilt[full]
        elif "sha256" not in r["media"]:
            r["media"]["sha256"] = sha_file(D1 / p)
        if r["modality"] == "audio":
            r["context"]["text"] = asr.get(p, "")
            r["context"]["source"] = "asr:whisper-small"
        elif r["modality"] == "document":
            r["context"]["text"] = ocr.get(p, "")
            r["context"]["source"] = "ocr:" + OCR_NAME
        else:
            r["context"]["source"] = "none"
        by_split[r["split"]].append(r)
    # leakage hygiene: no media sha across splits; no train gt/ASR/OCR text equal to an eval/protected gt text
    from krun_one.v1.leakage import normalize

    sh = defaultdict(set)
    for sp, rs in by_split.items():
        for r in rs:
            sh[r["media"]["sha256"]].add(sp)
    dup = {h for h, s in sh.items() if len(s) > 1}
    held = {normalize(r["gt_text"]) for sp, rs in by_split.items() if sp != "train" for r in rs if r.get("gt_text")}
    before = len(by_split["train"])
    # CREMA-D actors all read the same 12 sentences (the label is the emotion, splits are speaker-disjoint): exempt
    by_split["train"] = [r for r in by_split["train"] if r["media"]["sha256"] not in dup and not (
        r.get("gt_text") and r.get("teacher_filter") != "never" and normalize(r["gt_text"]) in held)]
    # KrunBench v1 --check hits (run on the laptop; final_protected never leaves it) + duplicate ids (identical CORD images)
    seen, dropped_ids = set(), 0
    for sp in list(by_split):
        keep = []
        for r in by_split[sp]:
            if r["id"] in DROP_IDS or r["id"] in seen:
                dropped_ids += 1
                continue
            seen.add(r["id"])
            keep.append(r)
        by_split[sp] = keep
    man = {"schema": "krun.m15.v1", "config_sha256": sha_file(ROOT / "configs/v1/m15.yaml"), "files": {},
           "dropped_train_for_leakage": before - len(by_split["train"]), "dup_media_across_splits": len(dup),
           "dropped_ids_check_or_duplicate": dropped_ids, "drop_ids": sorted(DROP_IDS),
           "missing_media": dict(missing)}
    man["files"]["protected"] = {"n": sum(1 for _ in jl(D / "protected.jsonl")), "sha256": sha_file(D / "protected.jsonl"),
                                 "note": "written by select; never processed (no ASR / OCR / teacher), never read"}
    for sp, rs in by_split.items():
        rs.sort(key=lambda r: r["id"])
        p = D / f"{sp}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        man["files"][sp] = {"n": len(rs), "sha256": sha_file(p)}
        if sp != "protected":
            with open(D / f"textview_{sp}.jsonl", "w", encoding="utf-8") as f:
                for r in rs:
                    for tag, txt in (("gt", r.get("gt_text")), ("ctx", r["context"]["text"])):
                        if txt:
                            v = B1.text_view(dict(r, paired_text=txt))
                            v["id"] = f"{r['id']}#{tag}"
                            f.write(json.dumps(v, ensure_ascii=False) + "\n")
    st = defaultdict(Counter)
    for sp, rs in by_split.items():
        for r in rs:
            st["split"][sp] += 1
            st["source"][f"{sp}/{r['source']}"] += 1
            st["task_class"][f"{r['modality']}/{r['task_class']}"] += 1
            st["speech"][f"{sp}/{r.get('speech')}"] += r["modality"] == "audio"
            st["type"][f"{r['modality']}/{r['task']['type']}"] += 1
    man["stats"] = {k: dict(sorted(v.items())) for k, v in st.items()}
    (D / "manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["audio_media", "extract_media", "verify_media", "fetch_new", "select", "asr", "ocr", "decisions"])
    ap.add_argument("--backend")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--drop-ids", default="")
    a = ap.parse_args()
    DROP_IDS = {x for x in a.drop_ids.split(",") if x}
    SHARD, NSHARDS = a.shard, a.nshards
    {"audio_media": audio_media, "extract_media": extract_media, "verify_media": verify_media, "fetch_new": fetch_new, "select": select, "asr": asr_pass,
     "ocr": lambda: ocr_pass(a.backend), "decisions": decisions}[a.stage]()
