# KrunBench v1 — especificação

Status: **proposta**. Os sets `test` do KrunBench v1 são materializados **uma vez**, com hash registrado, e ficam
fechados até o candidato final do Krun One v1 estar congelado. Durante tuning só se usa `validation`
(val_fit / val_eval por grupo, como no v0.3).

## 1. Camadas

| camada | conteúdo | papel |
|---|---|---|
| **R — regressão** (decisão do dono, 2026-09-26: o KrunBench v0 **não** serve como evidência de zero-shot/unseen, só de regressão) | KrunBench v0 (kb_val, kb_test incl. BFCL), val_hard, val_tools, holdout_tools, primitives v2 test (noul/score/mixed), BoolQ, SST-5 | comparar com v0.3 no mesmo código; **já abertos 4×**, não são test cego |
| **V — validação v1** | novas fatias construídas do mesmo jeito que o test v1, a partir de grupos de validação | tuning, calibração, thresholds, seleção de variantes |
| **T — test v1 (protegido)** | fatias novas de §3, grupos exclusivos | uma avaliação final por candidato congelado |
| **X — externos protegidos** | BFCL v3 (live + non-live, irrelevância), outros benchmarks de function-calling multilíngue a registrar | generalização a tools nunca vistas |

Tudo que está em T e X é "protegido": nunca entra em treino, mining de hard negatives, prompts de teacher, tradução,
índices de similaridade. Verificação automática: `src/krun_one/v1/leakage.py` (grupo, exato, near-dup, strings de
tool) sobre **todo** artefato de lado-de-treino, com o relatório anexado ao run.

`check_leakage` precisa retornar **zero** violações sobre o corpus de treino do candidato (e sobre pools de mining e inputs de teacher) antes de qualquer avaliação final — o `evaluate --final` exige o relatório de leakage com `ok: true` e sha256 do corpus. O lock precisa ser de código, não convenção: `evaluate --final` exige (1) um arquivo de decisões congelado com
sha256 do checkpoint, (2) que o checkpoint não tenha sido avaliado em T antes (registro append-only
`results/v1/final_openings.jsonl`). O lock do v0.3 só bloqueava tags `v02*` — corrigir.

## 2. Correções herdadas do audit

* Remover de R/X qualquer tool que aparece no treino (372 decisões de treino do v0.3 usam tools idênticas às do
  BFCL): BFCL passa a ter fatia **`bfcl_unseen_tools`** (tools ausentes do corpus de treino por string normalizada) e
  **`bfcl_seen_tools`**, reportadas separadamente.
* Deduplicar Civil Comments / BANKING77 entre splits por near-dup (Jaccard ≥ 0.9).
* Corrigir a fatia unseen-tool do noul (`metadata.unseen_tool` vs `tool_seen`).
* Anotar idioma nos itens BFCL live (hoje `language=null`).

## 3. Fatias

Cada item de T/V é `krun.decision.v1`; fatias são `slices` + campos (`context.language`, candidato `language`, k).

| eixo | fatias |
|---|---|
| idioma do contexto | EN, PT-BR, PT-PT, ES (es-ES, es-419), FR (fr-FR, fr-CA) |
| relação de idioma | same-language schema; cross-language schema (contexto X, candidatos EN); EN→X; code-switching |
| generalização | unseen labels, unseen tools, unseen domains, unseen instructions (paráfrases de instrução não vistas) |
| tamanho do conjunto | k = 2, 4, 8, 16, 32, 64, 128 |
| dificuldade | near-duplicate candidates, minimal pairs (métrica de conjunto), hard negatives v2 |
| answerability | OOD near (gold removido), OOD far, native irrelevance, cross-task, multi com conjunto vazio |
| robustez | permutação (4 rotações latin-square), typos, gírias, regional |
| primitivas | choice, noul, score (2–10 níveis, não só 5), multi |

Tamanho-alvo de T: ≥ 1,000 decisões por célula principal (idioma × tipo) e ≥ 300 por fatia secundária, para IC 95% de
acurácia de ±2–3 pp. Itens de EN→PT/ES/FR vêm de itens nativos traduzidos por teacher + verificação humana
(test é pequeno o bastante para revisão humana completa das fatias sintéticas).

## 4. Métricas (todas obrigatórias no relatório)

| métrica | definição / implementação |
|---|---|
| accuracy | choice/score/noul@0.5 (`krunbench.metrics.accuracy`) |
| macro-F1 | choice por rótulo; multi micro/macro-F1 @ threshold calibrado |
| OOD AUROC | answerable vs unanswerable, **por sinal**: margem (baseline v0.3), `ans_head`, `null_softmax`, `max_apply` |
| AURC | área sob risco-cobertura (`krunbench.metrics.aurc`) |
| ECE | 15 bins, após temperatura; por família e por idioma |
| NLL, Brier | por primitiva |
| false acceptance | fração de unanswerable aceitos no threshold de operação (selective acc 0.98 em val_fit) |
| coverage @ target selective accuracy | 0.95 / 0.98 / 0.99 |
| permutation consistency | mesma decisão nas 4 rotações |
| cross-language consistency | mesma decisão para o par (EN, tradução) do mesmo `semantic_group_id`; e |Δp| médio |
| unseen-tool / unseen-label performance | acurácia + AUROC nas fatias unseen |
| score | MAE, Spearman, Kendall τb, RPS, distância ponderada |
| multi | micro/macro-F1, exact-set match, taxa de conjunto vazio correto |
| minimal pairs | acurácia por membro + "todos os membros corretos" |
| latência | p50/p95 de tokenização, H2D, forward, pós-processamento, worker total, end-to-end (§5) |

Todas as métricas com IC bootstrap por grupo (1,000 reamostragens).

## 5. Latência

Harness: `scripts/v1/latency_proxy.py` (modelo) + worker LB (worker total, end-to-end). Estágios medidos
separadamente: tokenização, collate, H2D, forward, pós-processamento, worker total, end-to-end.

Workloads: comprimentos 128/256/512/1024; k = 2..128 com descrições reais de tools; 1/4/16 perguntas por request;
stress (16 perguntas × 32 tools × 512); throughput (32 requests pequenos em batch); contexto PT e EN (efeito do
tokenizer). GPUs: RTX 4090 (produção) e RTX 5090 quando houver estoque; L40S/H100 apenas se houver argumento de custo.

**Workload típico de produção** (definição para a meta de ~50 ms): 1–4 perguntas, ≤ 256 tokens por pergunta,
≤ 16 candidatos — no KRUN-014 as requests em produção tinham ~43 tokens por pergunta.

Qualidade e calibração são re-medidas após cada otimização de serving (bf16 puro, FP8, INT8, TensorRT): gate
= |Δ accuracy| ≤ 0.2 pp, |Δ ECE| ≤ 0.003, argmax flips ≤ 0.5% vs bf16 no set de fidelidade.

## 6. Protocolo de uso

1. Construção de V e T no mesmo script, por grupo, com sementes registradas; hash de cada arquivo em
   `benchmarks/krunbench/v1/manifest.json`.
2. Tuning (temperaturas, thresholds, variantes) só em V: val_fit ajusta, val_eval reporta.
3. Regras de decisão registradas **antes** de cada treino (como v0.3: `decision_rules` no config).
4. T aberto uma vez por candidato congelado; resultado vai ao relatório mesmo se pior.
5. R roda sempre (regressão vs v0.3 no mesmo código).

---

## 7. Materialized (2026-09-26) — `scripts/v1/build_krunbench_v1.py` → `data/krunbench_v1/`

**KrunBench v0 = REGRESSION ONLY.** v0/v0.3 already opened all of the following, so none of it can be a final test:
* MASSIVE, BANKING77, HWU64, BFCL;
* v0 holdout_tools;
* primitives v2 test (BoolQ and SST-5 included).

MASSIVE fr-FR is excluded from final for the same reason: it translates groups that were already opened.

Splits (`final_protected` is **not used in this phase**):

| split | total | choice | multi | noul | score | role |
|---|---:|---:|---:|---:|---:|---|
| dev | 695 | 544 | 32 | 77 | 42 | debugging |
| val_fit | 2,844 | 2,304 | 114 | 244 | 182 | temperatures, thresholds, variant selection |
| val_eval | 2,905 | 2,325 | 112 | 281 | 187 | reported validation numbers |
| **final_protected** | **11,715** | 9,717 | 511 | 898 | 589 | opened once per frozen candidate |

Never-opened sources (EN):

| source | license | goes to | provides |
|---|---|---|---|
| CLINC150 | CC-BY-3.0 | validation → dev/val; test → final | 150 intents, out-of-scope (far-OOD / native irrelevance) |
| SNIPS test | Apache-2.0 | final | |
| Bitext, 3 held-out intents | CDLA-Sharing-1.0 | val / final | unseen labels |
| ToolACE items v0 never used | Apache-2.0 | 40% → training pool (`train_pool_toolace_unused.jsonl`, **not protected**); 20% → val; 40% → final | native irrelevance (first turn without a call), multi-call → multi-label |
| Civil Comments test rows never used by v0 | CC0 | val / final | noul (6 propositions, balanced) and score (5 toxicity levels) |

Slices present in val and final:
* **Candidate count:** k2 through k128 (final: 2,186 / 2,119 / 1,441 / 1,305 / 902 / 897 / 852 at k = 2 / 4 / 8 / 16 / 32 / 64 / 128);
* **Difficulty / seen-ness:** near_duplicate (hard negatives by description similarity), unseen_label, unseen_tool (1,695 protected tools);
* **Answerability:** gold_removed / ood_near, ood_far, native_irrelevance;
* **Noise:** typo (rule).

**PT-BR / PT-PT / ES-ES / ES-LATAM / FR-FR / FR-CA, cross-language, slang and code-switching:** emitted as **12,000
protected translation jobs** (`pending_translation.jsonl`: source item, target locale, schema kept in EN). They enter
`final_protected` only after translation **and human review**. Until then these slices do not exist; no automatic
result may be reported for them. Translation is a protected-data operation: its output stays protected and never
becomes training data.

Protection (`protected_manifest.json` + `--check`):
* hashes of the normalized texts;
* split / semantic group ids;
* unseen tools (id, name, description).

`--check <training-side files>` validates, and aborts on any violation:
* exact / normalized text;
* groups;
* near-duplicates (Jaccard 0.9);
* unseen-tool **id / name / description / equivalent description**.

Current result against the training side (`data/v1_probe/train.jsonl`): **LEAKAGE = 0**. The build itself drops
benchmark items that overlap the training side (30 dropped). The MLM corpus is decontaminated against the same texts
(`results/v1/mlm_protected_texts.json`, 55,848 texts).
