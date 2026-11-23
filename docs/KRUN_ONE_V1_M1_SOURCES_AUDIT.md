# Krun One v1 M1 — multimodal source audit (2026-09-28)

Scope: the data and models used to train Krun One's small multimodal projectors (vision and audio) for a
**commercial** product. Every license below comes from a primary source that was fetched on 2026-09-28: the HF API
or card, the official GitHub `LICENSE`/README, Zenodo record metadata, or the dataset's own website. Anything not
confirmed that way is marked **UNVERIFIED**. This is an engineering assessment, not legal advice.

## Summary

| source | modality | license (primary) | commercial | decision | notes |
|---|---|---|---|---|---|
| COCO 2017 annotations (captions/instances) | text/boxes | CC-BY-4.0 (COCO Consortium) | yes | approved_with_notice | Attribute COCO Consortium |
| COCO 2017 images, license id 4 (CC-BY-2.0) / 7 (no known copyright) / 8 (US Gov) | image | per-image Flickr license | yes | approved_with_notice | Filter by `images[].license`. Keep `flickr_url` and the owner attribution list |
| COCO 2017 images, license id 5 (CC-BY-SA-2.0) | image | CC-BY-SA-2.0 | yes, share-alike | approved_with_notice (optional) | It is unsettled whether share-alike reaches model weights. Leave it out if the owner wants zero SA exposure |
| COCO 2017 images, ids 1/2/3 (NC) and 6 (ND) | image | CC-BY-NC(-SA/-ND)-2.0, CC-BY-ND-2.0 | no / doubtful | **excluded** | 87,451 of 118,287 train images |
| DocLayNet (`docling-project/DocLayNet-v1.2`, `pierreguillou/DocLayNet-small`/`-base`) | image (doc pages) | CDLA-Permissive-1.0 | yes | approved_with_notice | HF tag says `other`. The card text and GitHub LICENSE say CDLA-Permissive-1.0 |
| CORD v2 (`naver-clova-ix/cord-v2`) | image (receipts) | CC-BY-4.0 | yes | approved_with_notice | 800/100/100 |
| MInDS-14 (`PolyAI/minds14`) | audio 8 kHz | CC-BY-4.0 (metadata); card text says "CC-BY" with no version | yes | approved_with_notice | Parquet, no loading script |
| SLURP text (`pswietojanski/slurp` `dataset/slurp/`) | text | CC-BY-4.0 | yes | approved_with_notice | Fine to re-synthesize with Kokoro |
| SLURP audio (Zenodo 4274930) and every HF audio mirror | audio | CC-BY-NC-4.0 (Zenodo `other-nc`, LICENSE.txt) | **no** | **excluded** | Mirrors inherit NC |
| Speech-MASSIVE (`FBK-MT/Speech-MASSIVE`) | audio | CC-BY-NC-SA-4.0 | **no** | **excluded** | Confirmed |
| Fluent Speech Commands | audio | CC-BY-NC-ND-4.0, "academic research only" | **no** | **excluded** | Confirmed on fluent.ai |
| Kokoro-82M (`hexgrad/Kokoro-82M`) | TTS model | Apache-2.0 (weights + voices) | yes | approved_with_notice | Used to generate synthetic audio. espeak-ng/phonemizer are GPL (tooling only) |
| SigLIP2 base NaFlex (`google/siglip2-base-patch16-naflex`) | vision encoder | Apache-2.0 | yes | approved_with_notice | Vision tower 92.93M params |
| Whisper small (`openai/whisper-small`) | audio encoder | Apache-2.0 (HF card) | yes | approved_with_notice | Encoder 88.15M params |
| RVL-CDIP | image | `other` (UCSF Legacy Tobacco Library terms) | unclear | **excluded** | No commercial grant |
| FUNSD | image | EPFL terms: "non-commercial, research purposes only" | **no** | **excluded** | |
| SROIE (ICDAR 2019) | image | UNVERIFIED (no license grant found. Mirror repos cover code only) | unclear | **excluded** | |
| DocVQA | image | UNVERIFIED upstream (UCSF Industry Docs pages. RRC terms). `lmms-lab/DocVQA` tag says apache-2.0, which is not an upstream grant | unclear | **excluded** | |
| DejaVu fonts | font | Bitstream Vera license + Arev (DejaVu changes public domain) | yes | approved | Installed: `/usr/share/fonts/truetype/dejavu` |
| Noto fonts | font | OFL-1.1 | yes | approved | Installed: `/usr/share/fonts/truetype/noto` (272 files) |

## 1. COCO 2017

* **Terms** (github.com/cocodataset/cocodataset.github.io `dataset/termsofuse.htm`): "The annotations ... belong to
  the COCO Consortium and are licensed under a Creative Commons Attribution 4.0 License." "The COCO Consortium does
  not own the copyright of the images. Use of the images must abide by the Flickr Terms of Use."
  → Annotations are **CC-BY-4.0**. Image rights come **per image**, from the `license` field.
* **`licenses[]` in `captions_train2017.json`** (downloaded and parsed; the same table is in every 2017 annotation JSON):

| id | name | URL | commercial-OK |
|---|---|---|---|
| 1 | Attribution-NonCommercial-ShareAlike | CC BY-NC-SA 2.0 | no |
| 2 | Attribution-NonCommercial | CC BY-NC 2.0 | no |
| 3 | Attribution-NonCommercial-NoDerivs | CC BY-NC-ND 2.0 | no |
| 4 | Attribution | CC BY 2.0 | **yes** |
| 5 | Attribution-ShareAlike | CC BY-SA 2.0 | **yes (SA caveat)** |
| 6 | Attribution-NoDerivs | CC BY-ND 2.0 | excluded (ND, doubtful for training) |
| 7 | No known copyright restrictions | flickr.com/commons/usage | **yes** |
| 8 | United States Government Work | usa.gov/copyright | **yes** |

* **Measured image counts** per license id:

| split | total | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| train2017 | 118,287 | 32,878 | 16,397 | 32,184 | 19,470 | 10,901 | 5,992 | 465 | 0 |
| val2017 | 5,000 | 1,431 | 630 | 1,414 | 857 | 417 | 246 | 5 | 0 |

  Commercial-OK pools: train ids {4,7} = **19,935**; train ids {4,5,7} = **30,836**. Val ids {4,7} = 862;
  val ids {4,5,7} = 1,279. Captions: 591,753 in train (about 5 per image), 25,014 in val.
* **Fields per image**: `id, license, file_name, coco_url, flickr_url, height, width, date_captured`. Captions:
  `annotations[] {image_id, id, caption}`. Instances: 80 categories, with bbox and segmentation.
* **Download**: `http://images.cocodataset.org/annotations/annotations_trainval2017.zip` (252.9 MB). It contains
  instances, captions, and person_keypoints for train and val.
* **Fetching single images works.** `coco_url` (for example `http://images.cocodataset.org/train2017/000000000009.jpg`)
  returned HTTP 200 `image/jpeg`. So only the license-filtered images need to be downloaded; the full 18 GB zip is not needed.
* **Obligations**: credit the COCO Consortium (CC-BY-4.0) for the annotations. For CC-BY images, keep a manifest of
  `image_id, license, flickr_url` for attribution. CC-BY-SA-2.0 images carry share-alike; this is the owner's call.
  The images' EXIF and Flickr owner names are not in the JSON; `flickr_url` is the attribution pointer.

## 2. DocLayNet

* **License**: **CDLA-Permissive-1.0**. Sources: the `Licensing Information` section of the `docling-project/DocLayNet`
  and `DocLayNet-v1.2` cards, and the full `LICENSE` text in github.com/DS4SD/DocLayNet (GitHub API shows NOASSERTION
  because it does not recognise CDLA). The HF metadata tag `license: other` is just the unrecognised-SPDX fallback.
  CDLA-P allows use and redistribution. **Results** (for example trained weights) carry no obligations. Anyone who
  redistributes the *data* must include the agreement text. `ds4sd/*` now redirects to `docling-project/*`.
* **Content**: 80,863 pages from 6 `doc_category` values: `financial_reports, scientific_articles,
  laws_and_regulations, government_tenders, manuals, patents`. There are 11 layout classes: Caption, Footnote,
  Formula, List-item, Page-footer, Page-header, Picture, Section-header, Table, Text, Title. Pages are 1025×1025 PNG.
  About 95% of pages are English; the rest include German 2.5%, French 1% and Japanese 1%.
* **Download options**:
  * `docling-project/DocLayNet-v1.2`: parquet, 39.8 GB. train 69,375 / validation 6,489 / test 4,999. Fields:
    `image, bboxes, category_id, segmentation, area, pdf_cells(text+font), metadata{doc_category, collection,
    page_no, ...}, pdf`.
  * `docling-project/DocLayNet` (v1.0): script-only repo (`DocLayNet.py`), not loadable by datasets-server.
    Raw zips: DocLayNet_core.zip (28 GiB) and DocLayNet_extra.zip (7.5 GiB).
  * `pierreguillou/DocLayNet-small`: 804 rows (691/64/49), about 315 MB parquet. `pierreguillou/DocLayNet-base`: 8,057 rows
    (6,910/648/499), about 3.2 GB parquet. Both repos use a loading script, but `refs/convert/parquet` exists
    (`/api/datasets/<id>/parquet`). Fields: `texts, bboxes_block, bboxes_line, categories, image, doc_category,
    collection, page_no, ...`. The card says the data stays under CDLA-Permissive-1.0 from IBM.

## 3. CORD v2

* **License**: **CC-BY-4.0**. Sources: HF card metadata, `dataset_infos.json` ("Creative Commons Attribution 4.0
  International License"), and github.com/clovaai/cord (README badge; GitHub API spdx `CC-BY-4.0`).
* **HF**: `naver-clova-ix/cord-v2`, parquet, 2.31 GB. train 800 / validation 100 / test 100. Fields: `image` and
  `ground_truth` (a JSON string with `gt_parse{menu[{nm,cnt,price}], sub_total, total}`, `meta`, and `valid_line`
  word quads). Indonesian receipts.
* **Obligation**: attribution to NAVER Clova (cite the CORD paper).

## 4. MInDS-14

* **License**: HF metadata `cc-by-4.0`. The card text reads "All datasets are licensed under the Creative Commons
  license (CC-BY)" without a version. Treat it as CC-BY-4.0 and attribute PolyAI (arXiv 2104.08524).
* **Format**: `PolyAI/minds14` now serves **plain parquet** (one config per locale plus `all`, `train` split only).
  There is no loading script; `load_dataset("PolyAI/minds14", "en-US")` works directly.
  Audio is **8 kHz** (telephony, e-banking), so resample to 16 kHz for Whisper.
* **Fields**: `path, audio, transcription, english_transcription, intent_class, lang_id`.
  The 14 `intent_class` values are: abroad, address, app_error, atm_limit, balance, business_loan, card_issues, cash_deposit, direct_debit, freeze,
  high_value_payment, joint_account, latest_transactions, pay_bill.
* **Row counts** (parquet size): en-US 563 (34 MB), en-GB 592 (35 MB), en-AU 654 (37 MB), fr-FR 539 (33 MB),
  es-ES 486 (39 MB), pt-PT 604 (52 MB). `all` = 8,168 rows (567 MB) across 14 locales. **No pt-BR.** The
  Portuguese audio is European Portuguese.
* Every locale has only a `train` split, so evaluation splits have to be made by grouping by speaker or file
  (`path`).

## 5. SLURP

* **License** (github.com/pswietojanski/slurp README, "License"): "Textual data (whatever is distributed via this
  repository) is released on CC BY 4.0". "Audio data (whatever is stored on Zenodo.org) is released on CC BY-NC
  4.0 (one may obtain a less-strict license by contacting info@emotech.co)". The Zenodo record 4274930 has
  `license: other-nc`, and its LICENSE.txt says CC BY-NC 4.0.
* **Audio tarballs** (Zenodo API): `slurp_real.tar.gz` 3,918,185,662 B (about 3.9 GB) and `slurp_synth.tar.gz`
  2,839,670,545 B (about 2.8 GB), both FLAC.
* **HF parquet mirrors** exist (`qmeeus/slurp`, `marcel-gohsen/slurp`, `DynamicSuperb/SuperbIC_SLURP`,
  `yhfang/slurp_dataset_audio`, ...). None has a license that could override the upstream NC grant, so **all
  audio mirrors are excluded**.
* **Decision**: audio **excluded**. The text annotations (CC-BY-4.0: intents, scenarios, entities) are
  **approved_with_notice** as a source for Kokoro re-synthesis. Note that MASSIVE, which is already approved, is
  derived from SLURP text.

## 6. Speech-MASSIVE

The `FBK-MT/Speech-MASSIVE` metadata is `cc-by-nc-sa-4.0`. The card text says "All datasets are licensed under the
CC-BY-NC-SA-4.0 license". **Excluded (non-commercial). Confirmed.**

## 7. Fluent Speech Commands

The fluent.ai dataset page says "We are releasing this dataset for academic research only. It is licensed under
Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International license." It also says "released strictly
for academic research only". Splits are 23,132 / 3,118 / 3,793 utterances from 77/10/10 speakers.
**Excluded. Confirmed.**

## 8. Kokoro-82M

* **License**: `apache-2.0` (HF metadata). The README says the weights are Apache-licensed and "deployed in numerous
  projects and commercial APIs". The voices (`voices/*.pt`) are in the same Apache repo. Generated audio can be used
  commercially.
* **Provenance note**: the card says Kokoro was trained on "permissive/non-copyrighted audio", including *synthetic
  audio from closed TTS models from large providers*. Some voices use CC-BY data: `ff_siwis` (SIWIS, CC BY 4.0) and
  the `jf_*`/`jm_kumo` voices (Koniwa, CC BY 3.0). Residual ToS-provenance risk is low; record it in the notice.
* **Voices** (from VOICES.md; grade in parentheses):
  * **en-US** (`lang_code='a'`, espeak `en-us` fallback): af_heart (A), af_bella (A-), af_nicole (B-), af_aoede,
    af_kore, af_sarah, am_fenrir, am_michael, am_puck (C+), af_alloy, af_nova (C), af_sky (C-); weaker voices:
    af_jessica, af_river, am_echo, am_eric, am_liam, am_onyx (D), am_santa (D-), am_adam (F+).
  * **en-GB** (`'b'`, espeak `en-gb`): bf_emma (B-), bf_isabella, bm_fable, bm_george (C), bm_lewis (D+),
    bf_alice, bf_lily, bm_daniel (D).
  * **Spanish** (`'e'`, espeak `es`): ef_dora, em_alex, em_santa (ungraded).
  * **French** (`'f'`, espeak `fr-fr`): ff_siwis (B-). This is the only French voice.
  * **pt-BR** (`'p'`, espeak `pt-br`): pf_dora, pm_alex, pm_santa (ungraded).
* **Packages**: `pip install kokoro>=0.9.2 soundfile` (PyPI `kokoro` 0.9.4, Apache-2.0). It requires
  `huggingface-hub, loguru, misaki[en]>=0.9.4, numpy, torch, transformers`. `misaki` 0.9.4 (Apache-2.0) with extra
  `[en]` pulls in `espeakng-loader, num2words, phonemizer-fork, spacy, spacy-curated-transformers`. The system
  package is `espeak-ng` (the card uses `apt-get install espeak-ng`); it is needed for non-English and OOV fallback.
  espeak-ng and phonemizer are GPL-3.0. They run only as data-generation tooling, are not distributed with Krun One,
  and do not affect the output audio or the weights.

## 9. google/siglip2-base-patch16-naflex

* **License**: `apache-2.0` (HF metadata/card). Pretrained on WebLI. Commercial use of weights and derivatives is allowed
  with an Apache NOTICE.
* **Params** (counted from the safetensors header): `vision_model.*` **92,930,304**; text_model 282,303,744;
  total 375.2M. Only the vision tower is needed.
* **NaFlex processor** (`Siglip2ImageProcessorFast`, preprocessor_config `max_num_patches: 256`, `patch_size: 16`,
  mean/std 0.5): `processor(images=imgs, max_num_patches=N, return_tensors="pt")` resizes each image, keeping the
  aspect ratio, to at most N 16×16 patches, then pads to N. It returns `pixel_values` (B, N, 3·16·16),
  `pixel_attention_mask` (B, N), and `spatial_shapes` (B, 2) = (patch rows, patch cols).
  Call it as `model.vision_model(pixel_values=..., pixel_attention_mask=..., spatial_shapes=...)` or
  `model.get_image_features(**inputs)`. Positional embeddings are interpolated per `spatial_shapes`.

## 10. openai/whisper-small

`apache-2.0` (HF metadata; the openai/whisper code on GitHub is MIT). Safetensors count: encoder **88,154,112** params
(including the 1500×768 position table); decoder 153,580,800; total 241.7M. Commercial use is allowed with an Apache notice.

## 11. Document datasets expected to be excluded

* **RVL-CDIP** (`aharley/rvl_cdip`, tag `license:other`). The card points to the UCSF Legacy Tobacco Document Library
  copyright page; there is no permissive grant. **Excluded.**
* **FUNSD** (guillaumejaume.github.io/FUNSD/work/). EPFL terms: "to be used for non-commercial, research purposes
  only". **Excluded.**
* **SROIE** (ICDAR 2019 RRC). **UNVERIFIED**: no data license grant found. Mirrors (for example
  zzzDavid/ICDAR-2019-SROIE) license only their code (MIT). **Excluded.**
* **DocVQA**. Images come from the UCSF Industry Documents Library and the challenge is distributed through RRC under
  registration terms. **UNVERIFIED** (no license text found). The `lmms-lab/DocVQA` HF tag says apache-2.0, but that
  is the mirror's label, not an upstream grant. **Excluded.**

## 12. Fonts for synthetic documents

* **DejaVu**: the Bitstream Vera Fonts license, with DejaVu changes in the public domain and Arev glyphs under their
  own copyright (github.com/dejavu-fonts `LICENSE`). SPDX `Bitstream-Vera`. Permissive, and rendered output is
  unrestricted.
* **Noto**: SIL Open Font License 1.1 (`OFL-1.1`). Sources: notofonts `OFL.txt` and Debian `fonts-noto-core` copyright.
  Rendered pages are not bound by OFL; only redistributing the font files is.

## Notice obligations (for `THIRD_PARTY_NOTICES`)

COCO Consortium (CC-BY-4.0 annotations) plus a per-image attribution manifest for CC-BY and CC-BY-SA Flickr images.
IBM DocLayNet (CDLA-P-1.0, which asks for no notice for Results). NAVER CORD (CC-BY-4.0). PolyAI MInDS-14 (CC-BY-4.0).
SLURP text (CC-BY-4.0, Emotech/Heriot-Watt). Apache-2.0 NOTICEs for Kokoro-82M, SigLIP2 and Whisper-small.
