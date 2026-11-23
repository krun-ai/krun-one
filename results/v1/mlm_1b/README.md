# Continued MLM 1B: artifacts (run 2026-09-26/27, RTX 4090 Secure, pod pxoxp8ye2xtrer, terminated)

Gate result: `results/v1/mlm_gate_1b.md` → **STOP_AT_1B**; G2 failed, so decision training starts from **0B** (registered rule).

| file (runs/v1/mlm1b_import/) | content | size | sha256 |
|---|---|---:|---|
| k1_0b.tar | `milestone-0` = HyperClone(mmBERT-base) + 1e-3 noise (0B init) | 3.3 GB | 061160e3…7565 |
| k1_1b.tar | `milestone-1000000000` (after 1B tokens, WSD decayed) | 3.3 GB | d36438f0…6378 |
| k1_800m_ckpt3052.tar | full resume checkpoint at 800M (pre-decay; branch point for +2B) | 10.0 GB | 226c1e4c…165b |
| mlm1b_small.tgz | milestone evals, probes, gate, latency, manifests, logs | 3.4 MB | 693e6a18…76a9 |

Full hashes in `runs/v1/mlm1b_import/SHA256SUMS` (all verified locally). Extracted small files: this directory.

Held-out MLM loss (0B → 250M → 500M → 800M → 1B):
en 1.391 → 1.441 → 1.437 → 1.427 → 1.365 · es 1.085 → 1.101 → 1.090 → 1.085 → 1.027 ·
fr 0.983 → 0.999 → 0.988 → 0.979 → 0.924 · pt 1.106 → 1.119 → 1.099 → 1.089 → 1.029.

Incidents fixed during the run (all with regression tests):
* sparse MLM head not applied after `from_pretrained` (OOM);
* non-fused AdamW (OOM on 22 GiB);
* micro-batch 4 allocator thrash (→ 2);
* HF `blobs/` not deleted (disk quota at the 300M checkpoint);
* probe checkpoints tripping the volume quota (→ `--save-every` huge).
