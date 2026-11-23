# R3 multimodal recovery — val_eval (SigLIP2 / Whisper / OCR and the text core frozen; only the fusion is re-trained)

Gate references measured on the v1 core, this host: layout of the M1.5 reference 32.2 (G1 needs +5 pp), text-dependent documents 66.9 (G2 needs -1 pp).

| system | doc visual-only | layout | screenshot | text-dep doc | real speech | MInDS intent | image | synthdoc | DocLayNet cat. | CORD | G1 | G2 | G3 | G4 | G4b | passed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| M1.5c as registered (production host) | 68.0 | 38.7 | 99.2 | 67.2 | 85.1 | 89.7 | 64.8 | 96.1 | 94.5 | 100.0 | ✅ | ✅ | ✅ | ✅ | ✅ | 5 / 5 |
| v1 core + M1.5c fusion (this host) | 68.7 | 40.1 | 99.2 | 67.1 | 85.1 | 89.7 | 64.8 | 96.2 | 94.5 | 100.0 | ✅ | ✅ | ✅ | ✅ | ✅ | 5 / 5 |
| R3 core + OLD fusion [r3r_s1_m20260932_e1] | 46.9 | 15.2 | 97.3 | 64.8 | 79.0 | 86.8 | 46.9 | 69.0 | 76.8 | 93.1 | ❌ | ❌ | ❌ | ❌ | ❌ | 0 / 5 |
| R3 core + RE-ALIGNED fusion [r3r_s1_m20260932_e1] | 67.4 | 36.2 | 98.3 | 67.3 | 84.8 | 87.5 | 60.1 | 92.8 | 91.4 | 100.0 | ❌ | ✅ | ❌ | ❌ | ❌ | 1 / 5 |
| R3 core + OLD fusion [r3r_s1_m20260932_e3] | 46.9 | 15.2 | 97.3 | 64.8 | 79.0 | 86.8 | 46.9 | 69.0 | 76.8 | 93.1 | ❌ | ❌ | ❌ | ❌ | ❌ | 0 / 5 |
| R3 core + RE-ALIGNED fusion [r3r_s1_m20260932_e3] | 70.4 | 42.8 | 98.8 | 67.3 | 85.2 | 88.6 | 61.8 | 95.9 | 93.1 | 100.0 | ✅ | ✅ | ❌ | ❌ | ❌ | 2 / 5 |

## Difference to the v1 core + M1.5c on this host (pp)

| system | doc visual-only | layout | screenshot | text-dep doc | real speech | MInDS intent | image | synthdoc | DocLayNet cat. | CORD | worst |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R3 core + OLD fusion [r3r_s1_m20260932_e1] | -21.8 | -24.9 | -1.9 | -2.3 | -6.2 | -2.8 | -17.9 | -27.2 | -17.7 | -6.9 | -27.2 |
| R3 core + RE-ALIGNED fusion [r3r_s1_m20260932_e1] | -1.3 | -3.9 | -0.8 | +0.1 | -0.4 | -2.1 | -4.8 | -3.4 | -3.1 | +0.0 | -4.8 |
| R3 core + OLD fusion [r3r_s1_m20260932_e3] | -21.8 | -24.9 | -1.9 | -2.3 | -6.2 | -2.8 | -17.9 | -27.2 | -17.7 | -6.9 | -27.2 |
| R3 core + RE-ALIGNED fusion [r3r_s1_m20260932_e3] | +1.7 | +2.6 | -0.4 | +0.2 | +0.1 | -1.1 | -3.0 | -0.3 | -1.3 | +0.0 | -3.0 |

Run `r3r_s1_m20260932_e1`: 1559 steps, 15.3 min, selected `step_001169` (eligible on val_fit: False), text core unchanged: True.

Run `r3r_s1_m20260932_e3`: 4677 steps, 40.0 min, selected `step_003507` (eligible on val_fit: False), text core unchanged: True.
