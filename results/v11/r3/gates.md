# R3 promotion gates (registered in configs/v11/r3_protocol.yaml before training)

Seeds: r3r_s1, r3r_s2 · candidate seed: r3r_s1 · multimodal run: r3r_s1_m20260932_e3

| gate | required | measured | result |
|---|---|---|---|
| KrunTechBench v1 macro top-1 (raw) | mean ≥ 61.7 | 62.9 / 61.6 (mean 62.2) | ✅ |
| unseen requirement combinations | mean ≥ 71.2 (R2 61.2 + 10), each ≥ 66.2 | 58.3 / 59.8 (mean 59.0) | ❌ |
| constraint flips, both correct | mean ≥ 53.3 (R2 48.3 + 5) | 49.7 / 44.4 (mean 47.0) | ❌ |
| stuck rate | mean ≤ 13.7 (R2 18.7 − 5) | 20.1 / 21.7 (mean 20.9) | ❌ |
| hard-negative margin (logit) | mean ≥ 3.99 | +4.08 / +4.19 | ✅ |
| text preservation gates | 19 / 19 on every seed | 19/19 · 19/19 (k = 128: -0.80 / -1.07 pp) | ✅ |
| multimodal gates, re-aligned fusion | 5 / 5 | 2 / 5 (old fusion: 0 / 5) | ❌ |
| JevBench, all 231 (candidate − v1) | CI excludes 0, other seed above v1 | +0.4 pp [-4.3, +5.6]; other seed +3.0 pp | ❌ |
| latency / VRAM vs same-host v1 | ±1 ms typical, ±2 ms k128, +50 MiB | r3r_s1: -1.08 / -0.32 ms, +0 MiB · r3r_s2: -0.16 / -0.21 ms, +0 MiB | ✅ |
| **all gates** | | | ❌ |
