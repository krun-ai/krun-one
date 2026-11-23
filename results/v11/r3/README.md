# R3 — result index

Reports (in `results/v11/`): `KRUN_ONE_V11_R3_FINAL_REPORT.md` (start here), `…_TEXT_REPORT.md`, `…_GENERALIZATION_REPORT.md`,
`…_JEVBENCH_REPORT.md`, `…_MULTIMODAL_REPORT.md`.

| what | where |
|---|---|
| registered gates, measured | `gates.md`, `gates.json` |
| batching selection (validation) | `batching_selection.json` |
| R2 re-scored with the R3 definitions (before R3 results) | `r2_reference.json` |
| KrunTechBench v1 + R3 extension, protected / validation / diagnostics | `techbench/protected.md`, `techbench/val.md`, `techbench/val_diag.json` |
| v1 regression (19 gates), text latency / VRAM | `regression/regression.md` |
| JevBench by tier / length / family / type | `jevbench/jevbench_r3.md` (per-item lengths: `../jevbench/lengths.json`) |
| multimodal recovery, gates, latency | `multimodal/multimodal_r3.md`, `multimodal/latency_rtx4090_*.json`, `multimodal/v1_host.json` |
| Tetris views | `tetris/tetris_views.md` |
| Ollama quality review of the generators | `quality_review/README.md` |
| run records (arguments, steps, hashes) | `runs_meta/` |
| pod logs and result checksums | `pod_logs/` |
| raw dumps | `../eval/<run>/` (R1 / R2-host dumps of v1: `../eval_r12_host/v1/`) |
| kept weights (run outputs, not artifacts) | `runs/v11/r3/text/r3r_s1/model_v1.safetensors`, `runs/v11/r3/multimodal/r3r_s1_m20260932_e3/step_003507/m15.safetensors` |
