# Krun One v1 — Rodada 3: orçamento, custo por token, plano de dados

Data: 2026-09-26. Arquitetura fixa: **K1-W1536-L22** (879.4M). Abstenção pública: **margem**. Gate do tokenizer:
**CLEARED** (aprovação técnica do dono; não é parecer jurídico). Nenhum treino foi executado nesta rodada, só
benchmarks curtos de vazão.

## 1. Custo por token — medido (não estimado)

Benchmark: `scripts/v1/bench_training.py`. Mede o modelo K1 real (pesos aleatórios; a velocidade não depende deles)
com bf16, AdamW fused, SDPA, torch 2.14 e transformers 5.17, em 1 GPU:

* **MLM:** seq 1024, 15% de máscara, decoder esparso;
* **decisão:** fine-tuning completo com 16k tokens por passo.

Para cada GPU está a melhor configuração que coube na memória. Arquivos: `results/v1/bench_training/*.json` e
`results/v1/decisions_plan.md`.

| GPU | MLM tok/s | VRAM (melhor config) | decisão tok/s reais | US$/h community / secure | **US$ por 1B tokens de MLM** (community) | h por 1B | US$ por 1B tokens de decisão |
|---|---:|---|---:|---|---:|---:|---:|
| **RTX 4090 24GB** | 19,912 | 17.6 GiB (mb 4, sem checkpointing) | 13,381 (com checkpointing) | **0.34** / 0.74 | **4.74** (secure: 10.3) | 14.0 | **7.06** |
| L40S 48GB | 20,552 | 24.2 GiB (mb 8) | 15,830 | 0.79 / 1.09 | 10.68 | 13.5 | 13.86 |
| A100 SXM 80GB | 30,607 | mb 16 | 25,102 | 1.39 / 1.59 | 12.61 | 9.1 | 15.38 |
| RTX PRO 6000 (Blackwell) 96GB | 36,926 | 24.3 GiB (mb 8) | 32,377 | 1.69 / 2.19 | 12.71 | 7.5 | 14.50 |
| H100 NVL 94GB | 43,137 | 63.7 GiB (mb 32) | 35,513 | 2.59 / 3.19 | 16.68 | 6.4 | 20.26 |
| RTX 5090 | — | — | — | 0.69 / 0.99 | **sem estoque** (Secure e Community) | — | — |

**Escolha: RTX 4090 Community.** Custa 2.3× menos por token que a segunda opção (L40S) e 3.5× menos que a H100.
Até a 4090 Secure (US$ 10.3/1B) empata com a L40S Community.

Congelar os embeddings (393M parâmetros) acelera a 4090 em 17% (23.3k tok/s → US$ 4.05/1B), mas não há evidência de
qualidade. O padrão continua sendo treinar tudo; essa opção é uma alavanca de custo para depois.

**Parâmetros de uma execução de MLM na 4090:**
* 17.6 GiB de VRAM com micro-batch 4, sem gradient checkpointing (com checkpointing: 14.1 GiB e −16% de vazão);
* checkpoint de resume: ~10 GB (pesos fp32 3.3 GB + AdamW 6.7 GB), com `keep_last: 1`;
* marcos com pesos bf16: 1.7 GB cada;
* corpus uint32: 4 GB por 1B tokens;
* disco do pod: ~40 GB para 1B tokens e ~60 GB para 5B;
* volume de rede: opcional (~US$ 0.07/GB/mês), só se for preciso retomar em outro host.

Riscos operacionais da Community:
* **host sem TCP direto:** aconteceu em 1 de 2 criações da 4090; basta recriar o pod;
* **host menos confiável:** os checkpoints a cada 100M tokens (~50 min) limitam a perda a menos de US$ 0.30.

## 2. Quanto continued pretraining dá para pagar

| tokens de MLM | tempo (1× 4090) | custo (community) | com 2× 4090 em DDP |
|---:|---:|---:|---|
| 100M | 1.4 h | US$ 0.47 | — |
| **1B** | **14 h** | **US$ 4.74** | ~7 h, mesmo custo (+ overhead de comunicação **[não medido]**) |
| 2B | 28 h | US$ 9.5 | ~14 h |
| 3B | 42 h | US$ 14.2 | ~21 h |
| 5B | 70 h | US$ 23.7 | ~35 h |

Config pronta para uma execução única: `configs/v1/continued_mlm_budget.yaml` (1B, marcos só de pesos em
250M/500M/1B, sem estudo de scaling).

## 3. Dataset de decisões: 1M / 3M / 12M

Plano executável: `configs/v1/decisions_plan.yaml` + `scripts/v1/plan_decisions.py` (→ `results/v1/decisions_plan.md`).
Tokens e bytes por decisão foram medidos no proxy (148.6 tokens em média; intent 66, tool 270, noul 117, score 309;
2.3 KB por registro).

| escala | reais | derivados (grátis) | MT local | regra (typos etc.) | LLM local | OOD | cross-language | armazenamento jsonl / gz | tokens de treino | passos (16k tok) | treino na 4090 | geração LLM | MT |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---|---|
| **1M** (2 épocas) | 97k | 390k | 274k | 80k | 160k | 130k | 280k | 2.3 / 0.4 GB | 0.46B | 27.9k | 7.6 h / **US$ 2.6** | 5.3 h / US$ 1.8 | US$ 0.4 |
| **3M** (1.5 épocas) | 291k | 1.17M | 821k | 239k | 480k | 390k | 840k | 7.0 / 1.2 GB | 1.03B | 62.8k | 17 h / **US$ 5.8** | 16 h / US$ 5.4 | US$ 1.2 |
| **12M** (1 época) | 1.16M | 4.68M | 3.28M | 954k | 1.92M | 1.56M | 3.36M | 28 / 4.7 GB | 2.74B | 167.5k | 46 h / **US$ 15.5** | 63 h / US$ 21.5 | US$ 4.9 |

* "Reais" = textos originais de fontes licenciadas. Os cerca de 150k contextos reais únicos disponíveis limitam a
  escala: em 12M cada contexto real seria reusado muitas vezes.
* **1M–3M é o ponto de melhor qualidade por dólar.**
* Os custos de LLM e de MT são **estimativas** (premissas de vazão no YAML). O piloto de 2k decisões mede o valor real
  antes de qualquer geração em escala.

## 4. Orçamento ponta a ponta

Premissas:
* treino em RTX 4090 Community;
* avaliação e latência em RTX 4090 Secure (host estável);
* contingência para re-execuções, setup de pods e falhas.

| item | MÍNIMO | EQUILIBRADO | IDEAL |
|---|---|---|---|
| continued MLM | 1B tokens · 14 h · US$ 4.7 | 3B · 42 h · US$ 14.2 | 5B · 70 h · US$ 23.7 |
| dataset de decisões | 1M | 3M | 3M + 12M quando houver fonte real suficiente |
| geração local (LLM + MT) | US$ 2.2 | US$ 6.6 | US$ 26.4 |
| fine-tuning de decisão | 1M × 2 épocas · US$ 2.6 | 3M × 1.5 · US$ 5.8 | 12M × 1 · US$ 15.5 + 2ª semente US$ 15.5 |
| verificação barata do valor do MLM (fine-tune de 1M a partir de HyperClone@0 e @1B) | — | 2 × US$ 2.6 | 3 × US$ 2.6 (0/1B/5B) |
| avaliação (KrunBench v1, latência, export, fidelidade) | US$ 1.5 | US$ 2.5 | US$ 4 |
| subtotal | US$ 11.0 | US$ 34.3 | US$ 93 |
| contingência | +50% | +30% | +20% |
| **total estimado** | **≈ US$ 16** | **≈ US$ 45** | **≈ US$ 112** |
| tempo de parede (1 GPU, sequencial) | ~1.5 dia | ~4 dias | ~9 dias (ou ~4–5 com 2 GPUs) |

**Recomendação: EQUILIBRADO.** O custo marginal sobre o mínimo (~US$ 30) compra:
* 3× mais continued MLM;
* 3× mais dados de decisão;
* a única evidência barata de que o MLM ajuda: o mesmo fine-tune de 1M a partir de HyperClone@0 e @1B. Não é um
  estudo de scaling, são 2 execuções de US$ 2.6.

## 5. Plano mínimo viável até o Krun One

1. Montar o corpus de MLM, 1B tokens (§2 do doc de proveniência). A tokenização roda localmente, sem custo de GPU.
2. HyperClone + 1e-3 de ruído (crescimento verificado) → continued MLM de 1B numa 4090 Community (~US$ 5).
3. Krun Decisions v1 com 1M decisões:
   * reais + derivadas + MT local + regras;
   * LLM local só para minimal pairs, near-OOD, gírias/regional e code-switching;
   * leakage = 0 contra o KrunBench v1.
4. Fine-tuning de decisão, 2 épocas (~US$ 3).
5. Calibração (temperaturas por família) e abstenção pela margem, como hoje.
6. KrunBench v1 (T protegido, aberto uma vez) + regressão v0 + latência/fidelidade no worker.
7. Se passar nos gates: export drop-in (`krun_config` v2) → staging.

## 6. Datasets

Tabela completa: `docs/KRUN_ONE_V1_CORPUS_PROVENANCE.md`.

* **Entram:**
  * MLM: FineWeb-Edu, FineWeb2-HQ, FineWeb-2 (fatia de fóruns), FinePDFs-edu, FineWiki*, Bitext, Taskmaster,
    MultiWOZ, Multi3WOZ, oasst e ToolACE;
  * decisões: BANKING77, MASSIVE (+fr-FR), ToolACE, Civil Comments, HelpSteer2, MHS, Bitext (novo), Taskmaster,
    MultiWOZ e Multi3WOZ.
* **Revisar (decisão do dono):**
  * **xLAM e xlam-irrelevance**: o card do xLAM diz também "for research purposes only". É a maior fonte de tools
    (~40k de ~52k) e **já está no treino do v0.3 em produção**;
  * hermes e glaive (modelo gerador não declarado);
  * The Stack e specs OpenAPI.
* **Ficam fora:**
  * HWU64 (licença não confirmada);
  * BFCL, BoolQ e SST-5 (só avaliação/protegidos);
  * OpenSubtitles.

\* FineWiki é CC-BY-SA; para zero exposição a share-alike, retirar (~7% do corpus).

## 7. Dados sintéticos de baixo custo

| técnica | ferramenta | custo | uso |
|---|---|---|---|
| derivação (outros candidatos, k, hard negatives, gold removido, proposições noul, permutações) | código | US$ 0 | maior parte do volume |
| ruído por regra (typos por teclado, acentos, abreviações como "vc", "pq", "q", "xq", "stp", "slt", caixa) | código | US$ 0 | robustez / informal |
| tradução PT/ES/FR ↔ EN dos contextos (nomes de tools nunca traduzidos) | Helsinki-NLP opus-mt (CC-BY-4.0), local + filtro de ida e volta | ~US$ 0.15 por 100k (4090), ou 0 na máquina local | cross-language, pt-BR / es-419 / fr-CA |
| minimal pairs, contrafactuais, near-OOD, ambiguidade, gírias e variantes regionais, code-switching | Qwen3-8B (Apache-2.0) via vLLM numa 4090 Community | **~US$ 1.1 por 100k aceitas (estimativa)** | só essas categorias |
| verificador independente | outra família Apache-2.0 (p.ex. Mistral-7B-Instruct), às cegas | incluído acima | aceite só com concordância |

**Evitar como gerador:**
* Gemma: a cláusula de "Model Derivatives" cobre modelos treinados com dados sintéticos gerados por Gemma;
* Llama: restrições da licença;
* APIs fechadas: custo, e termos que costumam proibir treinar modelos concorrentes.

**Antes de gerar em escala:** piloto de 2k decisões para medir custo real, taxa de aceite e qualidade. Amostra
humana de 200 por transformação.

## 8. Uso de GPU nesta rodada

| pod | GPU | US$/h | workload | uso aprox. | custo aprox. | status |
|---|---|---:|---|---|---:|---|
| `xr1tyxe70fdfni` | RTX 4090 Community | 0.34 | nenhum (host sem TCP direto) | ~2 min | ~0.01 | **TERMINATED** |
| `hz8md0mruzd003` | RTX 4090 Secure (EUR-IS-1) | 0.74 | benchmark de vazão | ~27 min | ~0.33 | **TERMINATED** |
| `qxveltsp8jhn31` | L40S Secure (US-MO-1) | 1.09 | benchmark de vazão | ~33 min | ~0.60 | **TERMINATED** |
| `jvrywoprne2ewd` | A100 SXM 80GB Community | 1.39 | benchmark de vazão | ~43 min | ~1.00 | **TERMINATED** |
| `d4bcu822rup2wt` | RTX PRO 6000 Community | 1.69 | benchmark de vazão | ~31 min | ~0.87 | **TERMINATED** |
| `ydlq1edh4aiuku` | H100 NVL Community | 2.59 | benchmark de vazão | ~36 min | ~1.55 | **TERMINATED** |

* **Total da rodada: ≈ US$ 4.4.** Inclui ~5 min por pod perdidos na 1ª tentativa, que falhou porque não existe wheel
  torch 2.14 para cu128 (só cu126/cu130); corrigido.
* **Total do projeto v1: ≈ US$ 5.3.** As duas rodadas anteriores estão confirmadas no billing: US$ 0.25 + US$ 0.67.
* `list-pods` retorna vazio.
