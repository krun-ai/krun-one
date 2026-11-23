# Multimodal M1.5c regression — same fusion weights, different text core (val_eval; nothing multimodal retrained)

| text core | G1 doc visual-only | G1 layout | G2 screenshot | G2 text-dep doc | G3 real speech | G3 minds intent | G4 image | G4b synthdoc | G4b doclaynet | G4b cord | registered gates | by modality / class |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| M1.5c registered | 68.0 | 38.7 | 99.2 | 67.2 | 85.1 | 89.7 | 64.8 | 96.1 | 94.5 | 100.0 | G1✅ G2✅ G3✅ G4✅ G4b✅ | audio/acoustic_only 37.3, audio/text_dependent 90.1, document/hybrid 81.0, document/text_dependent 81.9, document/visual_only 62.5, image/visual_only 65.8 |
| v1 | 68.7 | 40.1 | 99.2 | 67.1 | 85.1 | 89.7 | 64.8 | 96.2 | 94.5 | 100.0 | G1✅ G2✅ G3✅ G4✅ G4b✅ | audio/acoustic_only 37.3, audio/text_dependent 90.1, document/hybrid 80.9, document/text_dependent 81.9, document/visual_only 63.4, image/visual_only 65.8 |
| v1_wrapped | 68.7 | 40.1 | 99.2 | 67.1 | 85.1 | 89.7 | 64.8 | 96.2 | 94.5 | 100.0 | G1✅ G2✅ G3✅ G4✅ G4b✅ | audio/acoustic_only 37.3, audio/text_dependent 90.1, document/hybrid 80.9, document/text_dependent 81.9, document/visual_only 63.4, image/visual_only 65.8 |
| r1_s1 | 50.8 | 6.5 | 96.1 | 66.7 | 81.0 | 87.2 | 56.3 | 80.6 | 81.5 | 81.9 | G1❌ G2❌ G3❌ G4❌ G4b❌ | audio/acoustic_only 29.9, audio/text_dependent 87.5, document/hybrid 61.5, document/text_dependent 78.7, document/visual_only 42.5, image/visual_only 57.0 |
| r2l4_s1 | 56.0 | 18.6 | 97.6 | 67.3 | 82.4 | 89.3 | 47.3 | 80.3 | 88.8 | 98.6 | G1❌ G2❌ G3❌ G4❌ G4b❌ | audio/acoustic_only 30.2, audio/text_dependent 88.2, document/hybrid 66.8, document/text_dependent 79.6, document/visual_only 49.0, image/visual_only 49.6 |
| r2l4na_s1 | 57.5 | 17.1 | 97.4 | 66.6 | 81.5 | 87.9 | 46.5 | 78.7 | 88.0 | 93.1 | G1❌ G2❌ G3❌ G4❌ G4b❌ | audio/acoustic_only 33.7, audio/text_dependent 87.8, document/hybrid 63.3, document/text_dependent 79.3, document/visual_only 49.9, image/visual_only 48.0 |

## Candidate core − same-host v1 core (pp)

| text core | G1 doc visual-only | G1 layout | G2 screenshot | G2 text-dep doc | G3 real speech | G3 minds intent | G4 image | G4b synthdoc | G4b doclaynet | G4b cord | worst | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| v1_wrapped | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | ✅ (gates lost vs same-host v1: none) |
| r1_s1 | -17.91 | -33.64 | -3.00 | -0.41 | -4.13 | -2.49 | -8.56 | -15.63 | -12.96 | -18.06 | -33.64 | ❌ (gates lost vs same-host v1: ['G1', 'G2', 'G3', 'G4', 'G4b']) |
| r2l4_s1 | -12.68 | -21.56 | -1.50 | +0.17 | -2.71 | -0.36 | -17.49 | -15.86 | -5.67 | -1.39 | -21.56 | ❌ (gates lost vs same-host v1: ['G1', 'G2', 'G3', 'G4', 'G4b']) |
| r2l4na_s1 | -11.25 | -23.05 | -1.76 | -0.57 | -3.59 | -1.78 | -18.28 | -17.47 | -6.48 | -6.94 | -23.05 | ❌ (gates lost vs same-host v1: ['G1', 'G2', 'G3', 'G4', 'G4b']) |
