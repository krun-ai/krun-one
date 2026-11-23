# Krun One v1 M1.5: OCR, ASR and real-speech source audit (2026-09-29)

Scope: the OCR and ASR models that Krun One ships or serves locally, and the real-speech datasets used to train its small
fusion layers. Krun One is a **commercial** product, so both the code license and the weights/data license must allow
commercial use. Every license below was fetched on 2026-09-29 from a primary source: the HF API or card, a GitHub
`LICENSE`/README (via the GitHub API or raw), PyPI JSON, Zenodo metadata, or the official site. Anything not confirmed
that way is marked **UNVERIFIED**. This is an engineering assessment, not legal advice.

## Summary A: OCR candidates

| name | license code | license weights | commercial | decision | notes (accent coverage / size) |
|---|---|---|---|---|---|
| docTR 1.1.0 (`python-doctr`), detectors db_resnet50 / db_mobilenet_v3_large / fast_* | Apache-2.0 | Apache-2.0 (project-wide; no weight-specific license text) | yes | approved_with_notice | Detectors do not depend on language. db_mobilenet_v3_large 4.2M, db_resnet50 25.4M, fast_small 14.7M (9.7M reparam), fast_base 16.3M (10.6M) |
| docTR default recognizers (crnn_vgg16_bn, crnn_mobilenet_v3_*, parseq, vitstr_*, master, viptr_tiny) | Apache-2.0 | Apache-2.0 (project-wide) | yes | approved_with_notice, **but not fit for PT/ES** | All use `VOCABS["french"]` (126 chars). It has à â é è ê ë î ï ô ù û ü ç but **no ã õ ñ á í ó ú**. crnn_mobilenet_v3_small 2.1M, crnn_mobilenet_v3_large 4.5M, crnn_vgg16_bn 15.8M, parseq 23.8M, vitstr_small 21.4M, master 58.7M |
| docTR parseq multilingual (`Felix92/doctr-torch-parseq-multilingual-v1`; ONNX `Felix92/onnxtr-parseq-multilingual-v1`) | Apache-2.0 | Apache-2.0 (HF card) | yes | approved_with_notice | Vocab covers **every** required accent (ã õ ç ñ á é í ó ú à è ù ê ô, uppercase too, plus ¡ ¿ €). Languages: en da fr it es de pt cs pl nl no fi. About 23.8M params (ONNX file 96.7 MB). Personal account of the docTR maintainer. Training data not disclosed |
| RapidOCR 3.9.2 (`rapidocr`) + PP-OCRv5 `latin` rec ONNX | Apache-2.0 | Apache-2.0 (PaddlePaddle HF card; RapidOCR `MODEL_LICENSES.md`) | yes | approved_with_notice | Latin dictionary has 772 chars and covers every required accent (no ¡). ONNX 8.0 MB (≈2M params, estimated from file size) |
| PP-OCRv6 (tiny/small/medium det+rec, ONNX on HF and in RapidOCR) | Apache-2.0 | Apache-2.0 (HF cards) | yes | approved_with_notice | One model covers 50 languages (46 Latin-script languages plus zh, zh-Hant, en, ja; tiny has 49). The dictionary covers every required accent. Rec params: small 5.2M, tiny 1.1M (cards). Medium rec ONNX is 76.6 MB |
| PaddleOCR 3.7.0 (`paddleocr`) | Apache-2.0 | Apache-2.0 | yes | approved_with_notice | Same weights as above. Only needed for the Paddle runtime. We prefer the ONNX route |
| Tesseract 5.5.3 + tessdata(_fast/_best) eng/por/spa/fra | Apache-2.0 | Apache-2.0 (tessdata, tessdata_best and tessdata_fast repos) | yes | approved_with_notice | CPU-only LSTM. por/spa/fra traineddata files exist in _fast (1.1–4.1 MB) and _best (4–15 MB). Weak on photos and scene text. Use as a baseline only |
| EasyOCR 1.7.2 (CRAFT + `latin` CRNN) | Apache-2.0 | EasyOCR repo is Apache-2.0. CRAFT code+weights come from clovaai/CRAFT-pytorch, MIT (NAVER) | yes | approved_with_notice | The latin model covers the pt/es/fr language codes. Weight training data is **UNVERIFIED**. Weights download from GitHub releases. Params not measured. Older, slower stack. Not shortlisted |
| Surya OCR 0.22.1 | Apache-2.0 | modified AI Pubs OpenRAIL-M: free only for research, personal use and startups under $5M funding/revenue | **conditional** | **excluded** | The revenue/funding cap is not an open commercial grant |
| Florence-2 base/large (microsoft) | MIT | MIT (HF) | yes | excluded (fit) | 231.6M / 776.7M params. A general VLM. Accent OCR quality UNVERIFIED. Too heavy for the OCR stage |
| GOT-OCR2.0 (`stepfun-ai/GOT-OCR2_0`, `-hf`) | GitHub: repo has no LICENSE file; README badges say "Code Apache-2.0 / Data CC-BY-NC-4.0" | HF card says apache-2.0 | UNVERIFIED | **excluded** | 716M / 560.5M params. The sources contradict each other. Too large anyway |
| TrOCR (`microsoft/trocr-*`) | MIT (microsoft/unilm) | `trocr-base-handwritten` card is MIT. `trocr-base/small-printed` cards have **no license tag** (UNVERIFIED) | partly | excluded (fit) | English only, line-level only. 61.4M (small) / 333.3M (base) |
| PaddleOCR-VL / 1.5 / 1.6 | Apache-2.0 | Apache-2.0 (HF) | yes | excluded (fit) | 0.96B params (1.6). Too heavy for a per-request OCR stage |

## Summary B: ASR

| name | license code | license weights | commercial | decision | notes |
|---|---|---|---|---|---|
| `openai/whisper-small` (full encoder+decoder) | MIT (openai/whisper `LICENSE`, © 2022 OpenAI) | README: "Whisper's code and model weights are released under the MIT License". HF card tag: apache-2.0 | yes | approved_with_notice | 241.73M params total: encoder 88.15M, decoder 153.58M (includes the 39.8M tied token embedding). 12+12 layers, d_model 768, vocab 51,865. 99 languages, including en/pt/es/fr. The model card advises against "high-risk domains like decision-making contexts" and against "subjective classification". That is guidance, not a license restriction, but record it in our model card |

## Summary C: speech datasets

| name | license code | license data | commercial | decision | notes |
|---|---|---|---|---|---|
| FLEURS (`google/fleurs`) | n/a | CC-BY-4.0 (HF card) | yes | approved_with_notice | en_us 2602/394/647, pt_br 2793/386/919, es_419 2796/408/908, fr_fr 3193/289/676 (train/val/test). Native parquet in `main` (`parquet-data/<lang>/`). No loading script. A `refs/convert/parquet` branch also exists. The text comes from FLoRes (CC-BY-SA-4.0) |
| SIB-200 (`Davlan/sib200`) | Apache-2.0 (dadelani/sib-200) | HF tag CC-BY-SA-4.0. Card text: "CC 4.0 Commercial". Built on FLoRes-200 (CC-BY-SA-4.0) | yes, share-alike | approved_with_notice | 7 topics. 701/99/204 per language. `index_id` is the FLoRes-200 sentence index. Mapping to FLEURS is below |
| SIB-Fleurs (`WueNLP/sib-fleurs`) | fdschmidt93/fleurs-slu: no license (UNVERIFIED) | CC-BY-SA-4.0 (HF) | yes, share-alike | approved_with_notice | SIB labels already joined to FLEURS audio. Keeps the **FLEURS** splits. eng 738/71/177, por 728/70/177, spa 676/71/177, fra 753/65/164 (rows = sentences, about 2.3 utterances each). **Drop the `seamlessm4t_*` columns**: they are outputs of SeamlessM4T-v2, which is CC-BY-NC-4.0 |
| CREMA-D (official GitHub) | R scripts in the repo | ODbL-1.0 (database) + DbCL-1.0 (contents), from `LICENSE.txt` | yes | approved_with_notice | 7,442 clips, 91 actors, 12 English sentences, 6 emotions (anger, disgust, fear, happy, neutral, sad), 4 levels. About 7.55 GB repo. ODbL: attribution. Share-alike covers derived *databases*, not Produced Works. Mirrors: `myleslinder/crema-d` (odbl tag, loading script) and `confit/cremad-parquet` (no license tag) |
| Mozilla Common Voice | n/a | CC0 historically. **Now distributed only through Mozilla Data Collective**, under new MDC terms | UNVERIFIED | UNVERIFIED | Since Oct 2025 the HF repos (`mozilla-foundation/common_voice_17_0`) contain only a README pointing to MDC. Mozilla says it will not host through third parties and wants consent revocations honored. Do not use HF mirrors (e.g. `fsicoli/common_voice_19_0`). The MDC terms need a separate review |
| Multilingual LibriSpeech (`facebook/multilingual_librispeech`) | n/a | CC-BY-4.0 (card: "Public Domain, CC-BY-4.0") | yes | approved_with_notice | Parquet, not gated. Configs: dutch, french, german, italian, polish, portuguese, spanish (English is not in this repo). Read audiobooks, so no intent labels |
| Google Speech Commands (`google/speech_commands`) | loading script | CC-BY-4.0 (card) | yes | approved_with_notice | English only, keyword spotting. The HF repo holds only `speech_commands.py`. No parquet conversion |
| MInDS-14 (`PolyAI/minds14`) | n/a | CC-BY-4.0 | yes | approved_with_notice (already in use) | Configs include es-ES, fr-FR, pt-PT, en-US/GB/AU. 8 kHz |
| Picovoice speech-to-intent-benchmark | Apache-2.0 | Apache-2.0 (HF + GitHub) | yes | approved_with_notice (optional) | English only. 619 commands from more than 50 speakers. Small |
| Speech-MASSIVE | n/a | CC-BY-NC-SA-4.0 | no | **excluded** | Confirmed again |
| `OpenVoiceOS/ovos-intent-bench-speech-massive-*` | n/a | tagged apache-2.0 but derived from Speech-MASSIVE | no | **excluded** | The tag does not override the NC upstream |
| SLURP audio | n/a | CC-BY-NC-4.0 | no | **excluded** | See the M1 audit |
| RAVDESS (Zenodo 1188976) | n/a | CC-BY-NC-SA-4.0 | no | **excluded** | |
| EmoV-DB | n/a | `LICENSE.md`: "Non-commercial Purposes" only | no | **excluded** | |
| ESD (HLTSingapore) | MIT (repo) | A signed license is required: "can only be used for research purpose" | no | **excluded** | |
| Expresso (Meta) | MIT (textlesslib) | CC-BY-NC-4.0 | no | **excluded** | |

## Details A: OCR

**docTR** (github.com/mindee/doctr): `LICENSE` is Apache-2.0 (© 2022 Mindee). README: "Distributed under the Apache 2.0
License". PyPI `python-doctr` is 1.1.0, needs Python ≥3.11, and is **PyTorch-only** (torch/torchvision/onnx are core
dependencies, and the TF backend is gone). Install with `pip install python-doctr` (extras: `[viz,html,contrib]`). The
pretrained weights are served from `doctr-static.mindee.com`. No separate weights license exists, so they fall under
the project's Apache-2.0. The pretraining datasets are not documented (UNVERIFIED provenance). The docs say: "most of our
recognition models were trained on our french vocab". The `default_cfgs` in `doctr/models/recognition/*/pytorch.py` use
`VOCABS["french"]` for crnn_*, parseq, vitstr_*, master and viptr_tiny. Coverage from `doctr/datasets/vocabs.py`:
french lacks ã õ ñ á í ó ú. portuguese lacks ñ è ù. spanish lacks ã õ ç à è ù ê ô. `multilingual` (601 chars) covers
everything. ONNX: `doctr.models.utils.export_model_to_onnx`, or OnnxTR (github.com/felixdittrich92/OnnxTR,
Apache-2.0, PyPI `onnxtr` 0.9.0). Param table: `docs/source/using_doctr/using_models.rst`.
The multilingual checkpoint is https://huggingface.co/Felix92/doctr-torch-parseq-multilingual-v1 (apache-2.0, card
languages en/da/fr/it/es/de/pt/cs/pl/nl/no/fi). The vocab string in its `config.json` contains all required accents.
Its weights are the v0.7.0 parseq URL on doctr-static. `mindee` has no HF models (API returned an empty list).

**RapidOCR** (github.com/RapidAI/RapidOCR): code is Apache-2.0. `python/MODEL_LICENSES.md` says the model artifacts are
Baidu/PaddleOCR works under Apache-2.0, that converted ONNX files keep the upstream attribution, and that redistribution
needs the Apache-2.0 text, a modification notice ("Converted from an official PaddleOCR model…") and no trademark or
endorsement claims. In `python/rapidocr/default_models.yaml` (v3.9.2), the ONNX Runtime models include
`latin_PP-OCRv3_rec_mobile`, `latin_PP-OCRv5_rec_mobile` and `multi_PP-OCRv6_{det,rec}_{tiny,small,medium}`, all served
from ModelScope `RapidAI/RapidOCR`. PyPI: `rapidocr` 3.9.2 (current, Apache-2.0). `rapidocr_onnxruntime` 1.4.4 is the
legacy package. Add `onnxruntime-gpu` 1.30.0 (MIT) for GPU.
**PaddleOCR** (github.com/PaddlePaddle/PaddleOCR): Apache-2.0. The HF cards `PaddlePaddle/latin_PP-OCRv5_mobile_rec_onnx`
and `PaddlePaddle/PP-OCRv6_{tiny,small,medium}_{det,rec}_onnx` are all license apache-2.0. PP-OCRv5 multilingual doc:
"106 languages, including Korean, Spanish, French, Portuguese…". The PP-OCRv6 doc: medium/small "support 50 languages
with a single unified model, including … 46 Latin-script languages", and "Dictionary extended with ~200 diacritical
characters". Accent coverage was checked programmatically against `character_dict` in each `inference.yml`. The
latin_v5 (772), v6_small (18,708) and v6_tiny (6,904) dictionaries all contain ã õ ç ñ á é í ó ú à è ù ê ô and the
matching uppercase forms. ¡ is missing from all three. Training data is not disclosed (UNVERIFIED provenance).

**Tesseract**: github.com/tesseract-ocr/tesseract, tessdata, tessdata_best and tessdata_fast are all Apache-2.0 (GitHub
API). Latest release: 5.5.3.
**EasyOCR**: github.com/JaidedAI/EasyOCR is Apache-2.0. Its README says detection uses the CRAFT "pretrained model" from
github.com/clovaai/CRAFT-pytorch (MIT, © NAVER Corp.). `easyocr/config.py` lists `latin_lang_list` (includes es, fr,
pt) and a `latin.pth` download from GitHub releases.
**Surya**: README "Commercial usage": "The model weights use a modified AI Pubs Open Rail-M license (free for research,
personal use, and startups under $5M funding/revenue)". The code is Apache-2.0.
**GOT-OCR2.0**: github.com/Ucas-HaoranWei/GOT-OCR2.0 has no LICENSE (API: null). Its README badges link to the Alpaca
code/data licenses (Data CC-BY-NC-4.0). The HF card says apache-2.0. Treat as UNVERIFIED.
**Florence-2**: HF license:mit (231,567,705 / 776,721,497 params, safetensors).
**TrOCR**: microsoft/unilm is MIT. The HF printed cards carry no license tag.
**PaddleOCR-VL-1.6**: HF apache-2.0, 958.6M params.

## Details B: Whisper

`openai/whisper` `LICENSE`: MIT. README § License: "Whisper's code and model weights are released under the MIT
License." HF `openai/whisper-small`: `license: apache-2.0`, 241,734,912 F32 params. The split was computed from the
safetensors header: encoder 88.154M, decoder 153.581M. `model-card.md` recommends against "use in high-risk domains
like decision-making contexts" and warns about transcribing people "without their consent". Both statements are
advisory. Krun One's model card should say that Whisper supplies transcripts and features, and that the product's
decisions carry their own evaluation. Keep both the MIT and Apache-2.0 notices.

## Details C: speech datasets

**FLEURS**: the HF card declares `license: cc-by-4.0` and the dataset is not gated. `main` holds
`data/<lang>/{train,dev,test}.tsv` with audio tarballs, and `parquet-data/<lang>/*.parquet`. Neither `main` nor any
other branch has a `fleurs.py` loading script. The refs are `main`, `streaming` and `refs/convert/parquet`. Features:
`id` (int32), `num_samples`, `path`, `audio` (16 kHz), `transcription`, `raw_transcription`, `gender`, `lang_id`,
`language`, `lang_group_id`. The FLEURS `id` is shared across languages and identifies a FLoRes sentence (range 1–2009).
One id has several utterances, so there are several rows per id. Counts come from the datasets-server `/size`.

**FLEURS `id` ↔ FLoRes/SIB `index_id` (verified empirically):** I joined SIB-200 to FLEURS on normalized text for
por_Latn↔pt_br and eng_Latn↔en_us. For every SIB `index_id` from 0 to 996 (the FLoRes **dev** part), `fleurs_id =
index_id + 1` (pt 497/497, en 498/498). For `index_id` ≥ 997 (FLoRes **devtest**) there is **no constant offset**
(pt 478 mismatches, en 489). The SIB-Fleurs card example also shows `fleurs_id 1895` ↔ `index_id 1592`. Unmatched SIB
rows: pt 29, en 17 (sentences FLEURS never recorded). So, to join: either use `WueNLP/sib-fleurs` directly, or build an
id map by normalized-text join on en_us↔eng_Latn and apply it to every language. Do **not** assume `id-1`. The SIB and
FLEURS splits disagree (SIB-train sentences appear in FLEURS-test, and the reverse). Pick one split scheme (FLEURS's, as
SIB-Fleurs does) to avoid leakage.

**SIB-200**: the HF tag is `cc-by-sa-4.0`. The card's "Licensing Information" says "CC 4.0 Commercial". The
github.com/dadelani/sib-200 repo is Apache-2.0 (code). The upstream facebookresearch/flores README gives "FLORES-200:
CC-BY-SA 4.0". Fields: `index_id`, `category`, `text`. Labels: science/technology, travel, politics, sports, health,
entertainment, geography. Treat the share-alike terms as in the M1 audit: attribute, keep derived label files under
CC-BY-SA, and note that whether SA reaches model weights is unsettled.
**CREMA-D**: `LICENSE.txt`: "made available under the Open Database License … Any rights in individual contents of the
database are licensed under the Database Contents License". ODbL allows commercial use. A trained model is a "Produced
Work", which needs a notice naming the database. Emotion/prosody only: English, 12 scripted sentences, no intent.
**Common Voice**: the `mozilla-foundation/common_voice_17_0` README says: "Effective October 2025, Mozilla Common Voice
datasets are now exclusively available through Mozilla Data Collective". The MDC FAQ (19 Aug 2025) says: "We have no
plans to host Mozilla community datasets through third parties … Our new terms reflect this." Whether CC0 still applies
under the MDC terms was not reviewed, so it stays **UNVERIFIED**.
**MLS**: card § Licensing: "Public Domain, Creative Commons Attribution 4.0". Parquet on HF, not gated.
**Speech Commands**: card § Licensing: "Creative Commons BY 4.0". HF holds only a loading script, so download from the
official TF tarball if needed.
**PT/ES/FR real-speech intent/SLU with a commercial license**: after searching HF (sib-fleurs, slu, intent speech,
spoken intent) and checking the M1 audit, only MInDS-14 (pt-PT, es-ES, fr-FR) and SIB-Fleurs/FLEURS+SIB-200 (topic, not
intent) qualify. Speech-MASSIVE, SLURP audio and Fluent Speech Commands are NC. Nothing else was verified; French MEDIA,
PortMEDIA and DIHANA are ELRA-licensed and were not checked (UNVERIFIED).
**RAVDESS**: Zenodo 1188976 is `cc-by-nc-sa-4.0`. **EmoV-DB**: `LICENSE.md` is "Non-commercial Purposes" only.
**ESD**: README says "This database can only be used for research purpose". **Expresso**: textlesslib
`examples/expresso/dataset/README.md` says "CC BY-NC 4.0". The HF `ylacombe/expresso` is cc-by-nc-4.0.

## Required notices (NOTICE file)

docTR (Apache-2.0, Mindee), parseq-multilingual (Apache-2.0, Felix92/docTR), PaddleOCR/PP-OCR weights (Apache-2.0,
Baidu, with a conversion notice when we re-export), RapidOCR (Apache-2.0), onnxruntime (MIT), Whisper (MIT + HF Apache-2.0
tag), FLEURS (CC-BY-4.0, Google), SIB-200/FLoRes-200 (CC-BY-SA-4.0), CREMA-D (ODbL/DbCL, Cao et al. 2014), MInDS-14 and
MLS (CC-BY-4.0).
