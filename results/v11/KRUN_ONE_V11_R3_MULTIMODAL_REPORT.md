# Krun One v1.1 — R3 multimodal report (recovery of the M1.5c path on the R3 text core)

Date: 2026-10-02 · Config: `configs/v11/r3_m15c_realign.yaml` · Script: `scripts/v11/mm_realign.py` (the registered M1.5c
trainer, unchanged, with the text core swapped) · Tables: `results/v11/r3/multimodal/multimodal_r3.md` · Protocol:
`configs/v11/r3_protocol.yaml` (multimodal).

## 1. Verdict

**Largely recovered, not fully: 2 of 5 registered gates pass.** Re-fitting only the fusion (26.7M parameters) on the frozen
R3 core takes the multimodal path from 0 / 5 gates and −17 to −27 pp (old fusion) to within **3.0 pp of the v1 stack on
every gated cell**, and above it on visual-only documents and layout. Three gates still fail: **image** (61.8 vs 63.3
required; −3.0 pp against v1), **MInDS-14 intent** (88.6 vs 89.0) and **DocLayNet category** (93.1 vs 93.5). The
promotion gate "5 / 5" is not met → registered **case B: no promotion; investigate fusion re-alignment / training mix /
latent shift only.**

## 2. What was done

| | |
|---|---|
| text core | R3 candidate seed `r3r_s1` (sha c151c7b0d71f7aca…), **frozen** — core digest identical before and after training |
| frozen | SigLIP2 (vision), Whisper-small (audio encoder + ASR), OCR |
| trained | M1.5c fusion adapters / projectors, modality embeddings, head LoRA — 26,681,856 parameters |
| initialisation | the registered M1.5c fusion weights (not from scratch) |
| data | the M1.5c training set, unchanged: 45,895 decisions, layout share 22%, same replay caps (image, visual documents, screenshots, speech, MInDS-14, document type, DocLayNet, CORD) |
| teacher | the R3 core's own answers on the ground-truth text (regenerated: 12 min) |
| schedule | registered M1.5c schedule, 1 epoch = 1,559 steps (15.3 min); registered fallback after a failed gate: 3 epochs = 4,677 steps (40.0 min) |
| checkpoint selection | `val_fit` only, against the M1.5c checkpoint on the v1 core; **no checkpoint of either run was eligible** (image visual-only stayed 3.8–5.0 pp under the reference), so the least-regressing one was taken, as registered |
| seeds | one fusion seed (20260932). The second seed was registered "if the first passes"; it did not, so it was not run |
| same-host reference | v1 core + registered M1.5c fusion re-dumped on the same pod: 5 / 5 gates, numbers identical to the R1 / R2 host |

## 3. Results (val_eval; gates = registered M1.5c thresholds, G1 / G2 references measured on the v1 core on this host)

| system | doc visual-only | layout | screenshot | text-dep doc | real speech | MInDS intent | image | synthdoc | DocLayNet cat. | CORD | G1 | G2 | G3 | G4 | G4b | passed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| M1.5c as registered (production host) | 68.0 | 38.7 | 99.2 | 67.2 | 85.1 | 89.7 | 64.8 | 96.1 | 94.5 | 100.0 | ✅ | ✅ | ✅ | ✅ | ✅ | 5 / 5 |
| v1 core + M1.5c fusion (this host) | 68.7 | 40.1 | 99.2 | 67.1 | 85.1 | 89.7 | 64.8 | 96.2 | 94.5 | 100.0 | ✅ | ✅ | ✅ | ✅ | ✅ | 5 / 5 |
| R3 core + OLD fusion (no training) | 46.9 | 15.2 | 97.3 | 64.8 | 79.0 | 86.8 | 46.9 | 69.0 | 76.8 | 93.1 | ❌ | ❌ | ❌ | ❌ | ❌ | 0 / 5 |
| R3 core + RE-ALIGNED fusion — 1 epoch | 67.4 | 36.2 | 98.3 | 67.3 | 84.8 | 87.5 | 60.1 | 92.8 | 91.4 | 100.0 | ❌ | ✅ | ❌ | ❌ | ❌ | 1 / 5 |
| R3 core + RE-ALIGNED fusion — 3 epochs | 70.4 | 42.8 | 98.8 | 67.3 | 85.2 | 88.6 | 61.8 | 95.9 | 93.1 | 100.0 | ✅ | ✅ | ❌ | ❌ | ❌ | 2 / 5 |

Candidate − (v1 core + M1.5c), same host, pp:

| system | doc visual-only | layout | screenshot | text-dep doc | real speech | MInDS intent | image | synthdoc | DocLayNet cat. | CORD | worst |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R3 core + old fusion | −21.8 | −24.9 | −1.9 | −2.3 | −6.2 | −2.8 | −17.9 | −27.2 | −17.7 | −6.9 | −27.2 |
| R3 core + re-aligned fusion, 1 epoch | −1.3 | −3.9 | −0.8 | +0.1 | −0.4 | −2.1 | −4.8 | −3.4 | −3.1 | 0.0 | −4.8 |
| R3 core + re-aligned fusion, 3 epochs | **+1.7** | **+2.6** | −0.4 | +0.2 | +0.1 | −1.1 | **−3.0** | −0.3 | −1.3 | 0.0 | −3.0 |

By modality and class (accuracy; v1 stack → re-aligned 3 epochs): image visual-only 65.8 → 62.7 · document visual-only
63.4 → 65.2 · document hybrid 80.9 → 80.1 · document text-dependent 81.9 → 80.9 · audio text-dependent 90.1 → 90.1 ·
audio acoustic-only 37.3 → 42.8.

## 4. Did image / document / audio regress?

* **Documents:** no regression that matters — visual-only +1.7 pp, layout +2.6 pp, screenshots −0.4 pp, text-dependent
  +0.2 pp, CORD equal, synthetic type −0.3 pp; DocLayNet category −1.3 pp (misses its threshold by 0.4 pp).
* **Audio:** real speech +0.1 pp, MInDS-14 intent −1.1 pp (misses its threshold by 0.4 pp), acoustic-only +5.5 pp.
* **Image: yes, −3.0 pp**, and it is the one that does not close: −5.0 → −4.4 → −3.8 → −4.0 pp over the four checkpoints
  of the 3-epoch run on `val_fit`. Natural images are the only modality with no text channel at all (no OCR, no
  transcript); everything has to reach the decision through media tokens that the re-trained last four blocks and head
  now read differently. The adapters recover most of it and then plateau.

## 5. Latency (same pod; total p50 / p95 ms)

| request | v1 core + M1.5c | R3 core + re-aligned fusion |
|---|---|---|
| image | 13.5 / 13.9 | 13.5 / 14.2 |
| scanned document (with OCR) | 78.7 / 97.4 | 80.3 / 95.2 |
| screenshot | 67.3 / 87.4 | 67.0 / 85.9 |
| document with text layer | 7.9 / 8.4 | 7.9 / 8.8 |
| audio 5 s | 77.3 / 81.1 | 77.6 / 79.0 |
| audio 15 s | 110.3 / 115.0 | 109.6 / 113.1 |
| text-only through the stack: typical / k = 128 | 7.1 / 43.7 | 7.2 / 43.7 |

Unchanged (same modules, same sizes). The registered latency checks of M1.5c still pass (image and scanned page ≤ 150 ms).

## 6. Diagnosis and what would close the gap (not started)

1. **The misalignment is real and large, and it is repairable from the fusion side** for every modality that carries text
   (documents, screenshots, speech): one epoch brings them within ~3 pp, three epochs within ~1 pp.
2. **Image is the residual.** Longer training of the same 26.7M parameters plateaus at −3 to −4 pp. Options, in order of
   cost: (a) more fusion capacity or an image-heavier replay mix for the re-alignment; (b) keep the fused path alive
   *while* the text core is trained — replay of fused media inputs or distillation of the v1 stack's outputs on them —
   so that the last blocks never drift away from what the media tokens need; (c) serve multimodal requests on the v1 core.
3. Two of the three failing gates miss by 0.4 pp on a single fusion seed. A second seed could land on either side; the
   image gate would still fail, so the verdict would not change.
4. The multimodal gate should be part of the selection of the text recipe from the start (R1 / R2 conclusion, confirmed):
   every text run in R3 was selected without looking at it, as the brief ordered, and the fusion had to absorb the whole
   shift afterwards.

## 7. Incidents on the pod (no result affected)

* Speech synthesis shared the GPU with text training: one shard ran out of GPU memory and others skipped clips; the media
  check added after R1 / R2 caught it (4,496 records without media), the shards were re-run one at a time and the check
  passed (74,956 records, 0 missing) before any multimodal step. One CREMA-D clip that failed to download was re-fetched.
* The first dump failed because the registered dump code builds its comparison system with a gate that the M1.5c fusion
  does not have; the wrapper now builds it without gates. Re-run, same checkpoint.

Kept locally (run outputs, not artifacts): `runs/v11/r3/text/r3r_s1/model_v1.safetensors` (sha c151c7b0…) and
`runs/v11/r3/multimodal/r3r_s1_m20260932_e3/step_003507/m15.safetensors` (sha db8c607a…).
