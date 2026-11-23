# Krun One v1 — Rodada 2: answerability, leakage, licença do tokenizer, scaling do HyperCloning

Data: 2026-09-26. Arquitetura congelada provisoriamente: **K1-W1536-L22** (879.4M total / 486.1M não-embedding,
vocabulário Gemma-2 completo). O modelo de 879M não foi treinado.

## 1. Answerability Bakeoff — resultado

Proxy: mmBERT-small + pilha v1 (144.5M), corpus limpo `data/v1_proxy` (110,650 treino / 12,897 validação), 5
variantes de treino × 3 sementes, na RTX 4090. Calibrador E (regressão logística) ajustado só em val_fit e avaliado
em val_eval. Nenhum set protegido foi lido. A regra e os gates foram registrados antes do treino
(`configs/v1/proxy.yaml`). Tabela completa: `results/v1/answerability_bakeoff.md`.

| mecanismo | macro FA @ sel. 0.98 | FA irrelevância tool | FA gold removido tool | FA near-OOD intent | cobertura | FA com cobertura = A (post-hoc) | AUROC | AURC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **A margem (baseline)** | 2.8 ±0.2 | 1.8 | 1.6 | 5.0 | **64.0** | 2.8 | 0.959 | **0.052** |
| B `[NULL]` no softmax | 0.1 ±0.1 | 0.0 | 0.2 | 0.0 | 6.5 | 5.9 | 0.937 | 0.098 |
| C answer head | 0.9 ±0.4 | 0.0 | 0.0 | 2.6 | 28.9 | 2.3 | 0.979 | 0.076 |
| D máx. aplicabilidade | 1.7 ±0.8 | 1.6 | 0.8 | 2.8 | 49.2 | 3.4 | 0.962 | 0.055 |
| E fusão (sobre J) | 1.9 ±0.4 | 1.6 | 0.6 | 3.5 | 58.2 | 2.3 | 0.973 | 0.055 |
| J answer head | 0.5 ±0.2 | 0.0 | 0.2 | 1.3 | 32.6 | **1.9** | 0.979 | 0.070 |

Risco seletivo (aceito e errado / aceito) no ponto de operação: tools não vistas A 1.0% · C 5.6% · J 3.2% · E 2.1%;
negativos difíceis A 0.9% · C 2.5% · E 1.6%.

### Regressões / gates vs A (média de 3 sementes)

| variante | acurácia intent | acurácia tool | ECE | NLL | consistência de permutação | noul AUROC | score MAE | multi F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 90.0 | 93.8 | 0.014 | 0.273 | 96.6 ±1.3 | 0.929 | 0.666 | 0.814 |
| B | 89.9 | 93.0 | 0.011 | 0.263 | 94.6 ±2.2 | 0.927 | 0.667 | 0.809 |
| C | 90.0 | 93.2 | 0.009 | 0.267 | 97.2 ±1.3 | 0.928 | 0.672 | 0.812 |
| D | 90.0 | 93.3 | 0.011 | 0.267 | 95.1 ±1.0 | 0.927 | 0.674 | 0.805 |
| J | 90.1 | 92.8 | 0.011 | 0.270 | 95.3 ±0.6 | 0.928 | 0.674 | 0.818 |

### Decisão

**Nenhum vencedor → a margem top1−top2 continua como mecanismo público de abstenção.**

Nenhum candidato passa nos gates:
* todos reprovam em cobertura, porque abstêm-se mais;
* os que treinam objetivos de OOD perdem 0.5–1.0 pp de acurácia de tool;
* com cobertura igual, o melhor (J answer head) reduz a FA macro em só 0.9 pp, abaixo do critério registrado de
  ≥ 3 pp e ≥ 25%.

A AUROC maior dos answer heads (0.979 vs 0.959) não bastou para selecionar, como pedido.

**Por quê.** O answer head mede "existe candidato aplicável?", mas não mede "o top-1 está certo?". O ponto de operação
de seletividade 0.98 conta os dois erros, e a margem cobre os dois ao mesmo tempo. `[NULL]` no softmax é o pior
sinal: instável entre sementes e com AUROC 0.937. Não adotar.

**O que fica na arquitetura:** `[NULL]` + answer head + aplicabilidade continuam como saídas internas. Custo de
latência ≤ 0.25 ms (§3), e a aplicabilidade é o mecanismo de multi-label.

**Limitação importante.** O OOD de validação do proxy é relativamente fácil (AUROC da margem 0.96). O caso que motivou
este trabalho — irrelevância de tools do mesmo domínio, como o BFCL live-irrelevance, onde o v0.3 teve AUROC 0.795 —
está protegido e não foi usado. Próximo passo registrável: um set V de irrelevância difícil no KrunBench v1, mais a
hipótese de dois limiares (answer head para OOD, margem para ranking), avaliada contra a margem com a mesma regra.

## 2. Latência (impacto de answerability)

RTX 4090, host EU-RO-1, pesos bf16, contexto pt-PT, total p50 em ms:

| workload | v0.3 | K1 sem answerability | K1 com answerability | Δ |
|---|---:|---:|---:|---:|
| 1 pergunta, k=4 | 7.9 | 6.4 | 6.6 | +0.18 |
| 1 pergunta, 8 tools | 8.3 | 6.7 | 6.7 | −0.07 |
| 4 perguntas mistas | 12.3 | 12.9 | 13.0 | +0.10 |
| 32 tools | 9.8 | 10.2 | 10.3 | +0.02 |
| 64 tools | 20.1 | 23.5 | 23.6 | +0.09 |
| 128 tools | 39.3 | 36.0 | 36.1 | +0.10 |
| 128 / 256 / 512 / 1024 tokens | 7.9 / 7.8 / 8.2 / 11.2 | 6.7 / 6.6 / 7.8 / 13.4 | 6.6 / 6.7 / 7.8 / 13.2 | ~0 |
| stress 16 × 32 tools × 1024 | 181.0 | 217.3 | 217.5 | +0.23 |

* O fast path típico fica em 6.6–13 ms, bem abaixo da meta de ~50 ms.
* O batch 1 depende da CPU do host (lançamento de kernels): este host foi cerca de 2× mais rápido que o da rodada 1
  (US-NC-1, típico de 13.9 ms). Aqui o K1 fica 0.8× o v0.3 com 1 pergunta e 1.05× com 4 perguntas.
* RTX 5090 continua sem estoque na RunPod (Secure e Community). Nenhum pod foi criado para esperar estoque.

## 3. Dataset / leakage

* **Bug da fatia unseen-tool do noul corrigido** (`scripts/prepare_primitives_data.py`), com a definição nova em
  `src/krun_one/v1/tools.py`. "Vista" = gold **ou** distrator no treino, com a mesma chave ou com descrição
  equivalente (mesma descrição sob outro nome, ou Jaccard ≥ 0.8).
  * Invariantes de build: a fatia não pode ficar vazia em choice nem em noul, e nenhuma tool "unseen" pode aparecer
    no treino.
  * Testes de regressão sobre os dados reais: o noul agora tem itens unseen em val e test e zero em train.
* Achado extra: 7 golds que o v0 marcava como "unseen" já estavam no treino (5 em val_tools e 2 em holdout_tools; 1
  exato e 6 por descrição equivalente).
* Proxy reconstruído com os metadados novos do `krun.decision.v1`:
  * campos novos: `schema_language`, `cross_language`, `difficulty` e `candidate_set.strategy` (obrigatório);
  * mantidos: `semantic_group_id`, `split_group_id`, `provenance`, transforms e `provenance.synthetic`.
* **Leakage = 0** após remover os 454 registros com vazamento (25 exatos, 131 near-duplicates, 372 com tools
  idênticas às do BFCL, 1 exato do BFCL).
  * Fatia unseen-tool: 1,453 choice e 137 noul.
  * Nenhum treino desta rodada usou registros removidos.
* O KrunBench v0 passa a valer só como regressão. O spec do KrunBench v1 exige relatório de leakage com `ok: true`
  antes de qualquer avaliação final.

## 4. Licença do tokenizer (`docs/KRUN_ONE_V1_TOKENIZER_LICENSE.md`)

* O vocabulário do mmBERT é o do Gemma-2. **Verificado localmente:** é idêntico (255,999 de 256,000 peças; só muda a
  grafia do byte TAB) ao `tokenizer/tokenizer.model` publicado pelo Google em `google/gemma_pytorch` sob
  **Apache-2.0** (sha256 `61a7b147…`).
* Os pesos do mmBERT são **MIT**.
* Os Gemma Terms definem "Gemma" como modelos, pesos e parâmetros e não mencionam tokenizers. O Krun não usaria
  pesos nem outputs do Gemma.
* **Veredito técnico: sem bloqueio material.** Treinar, servir via API e publicar pesos é permitido, com os avisos
  MIT (mmBERT), Apache-2.0 (tokenizer e Laya) e a nota de modificação. Opcionalmente, por cautela: incorporar a Gemma
  Prohibited Use Policy aos termos da API.
* Não é parecer jurídico. **O gate está `PENDING-OWNER`:** o pré-treino se recusa a rodar até o dono do projeto trocar
  a linha para `GATE: CLEARED`.

## 5. Experimento de scaling do HyperCloning — pronto para executar

| peça | estado |
|---|---|
| `scripts/v1/grow_encoder.py` (mmBERT-base → 1536, verificação de função) | pronto; carregamento bit-exato no modelo v1 testado |
| `scripts/v1/prepare_mlm_corpus.py` (FineWeb-Edu EN + FineWeb-2 PT/ES/FR, held-out por idioma, descontaminação) | pronto, não executado |
| `scripts/v1/continued_mlm.py` + `src/krun_one/v1/mlm.py` (marcos 0/100M/500M/1B/5B; estágio 2 10B/25B/50B/100B) | pronto; smoke com marcos e **resume bit-idêntico** testado; bloqueado pelo gate de licença (testado) |
| downstream por marco (`train_proxy.py --spec k1 --encoder-init`) + referência mmBERT-base | pronto |
| `scripts/v1/scaling_summary.py` + regra de saturação (`configs/v1/scaling.yaml`) | pronto |
| `scripts/v1/runpod_scaling.sh` | pronto |

**Custo estimado do estágio 1:** ~16 H100-h (≈ US$ 40–50). Numa 4090 seria ~6× mais lento.

**Pré-requisitos:**
1. gate de licença CLEARED;
2. confirmação da licença ODC-By do FineWeb (+ termos do Common Crawl) e construção do corpus;
3. `DOWNSTREAM_VARIANT=A_margin`, conforme o resultado do bakeoff.

## 5.1 Testes

A suíte completa passa: todos os testes do v0.3, mais os testes v1 de serialização, modelo, losses, schema com os
novos metadados, leakage, tools unseen (unitário, invariantes do builder e regressão em dados reais), growth,
carregamento de checkpoint crescido, resume bit-idêntico dos trainers de decisão e de MLM, e o bloqueio do gate de
licença.

## 6. RunPod

| pod | GPU | região | US$/h | workload | runtime | artefatos | custo | status final |
|---|---|---|---:|---|---|---|---:|---|
| `qqtofic3qn1vt0` (rodada 1) | RTX 4090 Secure | US-NC-1 | 0.74 | proxy de latência, 6 arquiteturas | ~20 min | `results/v1/latency_rtx4090_sdpa.*` | ≈ 0.25 | **TERMINATED** |
| `6jxb91hm8j0dp2` | RTX 4090 Secure | EU-RO-1 | 0.74 | testes, resume smoke na GPU, bakeoff (15 treinos), latência de answerability | 04:08:42 → 05:03 UTC (~55 min) | `runs/v1/*/{val_signals.jsonl.gz,eval_validation.json,train_result.json}`, `results/v1/bakeoff_logs/`, `results/v1/latency_answerability_rtx4090.json` | ≈ 0.68 | **TERMINATED** |

Confirmado via MCP: `list-pods` retorna vazio. Total GPU até aqui: ≈ US$ 0.93.
