# Krun M1 v1 — multimodal alignment dataset (2026-09-29)

The builder is `scripts/v1/build_m1_data.py`, which runs the stages fetch → render → tts_jobs → decisions. Speech comes
from `scripts/v1/m1_tts.py`. The leakage fix-up is `scripts/v1/m1_leakage_filter.py`.

**Where things are:**
* the manifest (hashes, stats and the leakage filter record) is `data/m1/manifest.json`;
* the protocol is `configs/v1/m1.yaml`, registered before the build;
* licenses are in `docs/KRUN_ONE_V1_M1_SOURCES_AUDIT.md`.

## Size and splits

The dataset has **200,963 decisions** built on **124,248 media files** (~13.7 GB on the pod).

| split | decisions | sha256 |
|---|---:|---|
| train | 160,589 | `01edd22f…` |
| val_fit | 9,897 | `8e7a3412…` |
| val_eval | 15,269 | `acb70731…` |
| protected | 15,208 | `9f4974b2…` (never read) |

The split is by group hash, 80 / 5 / 7.5 / 7.5. A group is one of:
* a COCO image;
* a synthetic document;
* a kd1 `split_group_id` (the text, its speech and its screenshot always share a split);
* a MInDS-14 clip;
* a DocLayNet original PDF;
* a CORD image.

## Sources

| source | modality | decisions | media | license decision |
|---|---|---:|---|---|
| COCO 2017 train + val, images with license id ∈ {4 CC BY 2.0, 7 no known copyright} (19,935 images; ids 1/2/3/6 NC/ND and id 5 BY-SA excluded) | image | 60,605 | 3.3 GB | approved_with_notice (annotations CC-BY-4.0, per-image Flickr license) |
| Synthetic business documents (12 types × EN/PT/ES/FR, Faker + DejaVu/Noto, scan/photo degradations) | document | 42,000 | 14,000 pages | own generation (Faker MIT, fonts Bitstream-Vera / OFL) |
| DocLayNet-base (pierreguillou) | document | 13,698 | 8,057 pages | approved_with_notice (CDLA-Permissive-1.0) |
| CORD v2 receipts | document | 1,000 | 1,000 | approved_with_notice (CC-BY-4.0) |
| kd1 pool → **app screenshots** (chat / ticket / search screens rendering the user message) | document | 38,481 | 38,601 | derived from Krun Decisions v1 train (sources already audited; LEAKAGE = 0) |
| kd1 pool → **synthetic speech** (Kokoro-82M, 17 voices, speed 0.85–1.15; 38,601 clips, 43.8 h) | audio | 38,481 | 2.5 GB | Kokoro Apache-2.0 (espeak-ng GPL used as a tool only) |
| MInDS-14 (en-US / en-GB / en-AU / fr-FR / es-ES / pt-PT), **real speech** | audio | 6,702 | 3,351 clips | approved_with_notice (CC-BY-4.0) |

**Excluded after the audit:**

| source | reason |
|---|---|
| SLURP audio | CC-BY-NC |
| Speech-MASSIVE | NC-SA |
| Fluent Speech Commands | NC-ND |
| RVL-CDIP | no grant |
| FUNSD | NC |
| SROIE, DocVQA | unverified |

## Composition

The dataset covers all four primitives.

| | choice | noul | score | multi |
|---|---:|---:|---:|---:|
| image | 22,685 | 16,556 | 9,899 | 11,465 |
| document | 47,725 | 17,883 | 7,540 | 22,031 |
| audio | 30,062 | 9,323 | 1,400 | 4,398 |

The tasks, by source:

* **image:**
  * main subject (k = 2–16, half with same-supercategory hard negatives, 8% unanswerable);
  * which objects appear (multi);
  * "contains X" (noul);
  * people count (score, 5 levels);
  * caption matching (k = 4 descriptions).
* **document:**
  * document type (k = 4–12);
  * fields present (multi: signature, due date, tax id, bank details, item table, barcode, logo, paid stamp);
  * paid / signed / total > X (noul);
  * amount bucket (score);
  * DocLayNet category (k = 3–6) and layout elements (multi);
  * CORD receipt type.
* **audio and screenshots:** the original kd1 decision — intent / tool choice, noul, score, multi, including its
  unanswerable items. MInDS-14 adds intent (k = 4–14) and "the caller is asking about X" (noul).

**Cross-language schemas.** Instructions and candidate labels are in EN 40% / PT 20% / ES 20% / FR 20%, independent
of the media language. The labels are hand-written in `configs/v1/m1_labels.yaml`.

Media languages:

| language | decisions |
|---|---:|
| EN | 110k (incl. COCO / DocLayNet) |
| PT-BR | 25k |
| PT-PT | 5k |
| ES | 30k |
| FR | 30k |
| ID | 1k (CORD) |

**Cross-modal pairs.**
* **199,909 decisions carry a paired text:**
  * caption;
  * the rendered document text (the text fast-path equivalent);
  * DocLayNet page text;
  * transcript;
  * the original kd1 message.
* **Teacher.** The frozen text model answered every paired text. That produced 184,790 teacher outputs, 140,496 of
  them used for the consistency loss. The rules:
  * `teacher_filter=always` for speech, screenshots and MInDS;
  * `if_correct` for captions and documents.
* **Triples.** Every kd1 decision exists as text, speech **and** screenshot. That is 38.5k triples, used for the
  cross-modal consistency metric.

## Leakage (G5)

* Groups are disjoint across splits (asserted).
* 0 media sha256 values are shared across splits.
* 590 train decisions were dropped because their paired text equaled an eval / protected paired text.
* **KrunBench v1 `--check`** was run on the laptop, so final_protected never leaves the laptop. It covered the text
  views of train + val_fit + val_eval (185,755 records).
  * **First run: LEAKAGE = 4.**
    * 2 duplicate ids came from identical images inside CORD.
    * 1 MInDS-14 clip had a transcript that is a near-duplicate (Jaccard 0.944) of a CLINC `final_protected`
      utterance, which counts twice (intent + noul).
  * Removed with `m1_leakage_filter.py`: 1 group plus the duplicate ids.
  * **Re-check: LEAKAGE = 0.** Files: `results/v1/m1/krunbench_v1_leakage_check_m1.json` and the first-run log.
* Training was gated on a flag holding the sha256 of the filtered `train.jsonl`.
