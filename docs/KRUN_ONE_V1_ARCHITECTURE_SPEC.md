# Krun One v1 — especificação de arquitetura

Status: **arquitetura congelada provisoriamente (K1-W1536-L22, 879.4M, vocabulário Gemma-2 completo — decisão do dono em 2026-09-26)**; nada de 879M foi treinado. O recorte de vocabulário fica apenas como ablação futura. Contagens de parâmetros são
calculadas pelo código e checadas contra os módulos reais instanciados (`scripts/v1/param_budget.py`,
`tests/v1/test_v1_core.py`; a mesma conta reproduz exatamente os 421,293,827 parâmetros do artefato v0.3).

Auditoria de base: `docs/KRUN_ONE_V1_AUDIT.md`. Dados: `docs/KRUN_DECISIONS_V1_SPEC.md`. Avaliação:
`docs/KRUNBENCH_V1_SPEC.md`.

## 1. Resumo da decisão

**K1-W1536-L22**: encoder na arquitetura mmBERT/ModernBERT, 22 camadas, largura 1536, 24 cabeças (dim 64), FFN GLU
2304, tokenizer Gemma-2 de 256k do mmBERT, inicializado por **crescimento em largura (HyperCloning) do mmBERT-base**
seguido de pré-treino MLM continuado em EN/PT/ES/FR; em cima, a pilha de decisão do Laya/v0.3 (type embedding,
head transformer, scorer por marcador) com duas saídas novas: **aplicabilidade por candidato** (multi-label) e
**answerability** (slot `[NULL]`).

Por quê, em uma linha cada:
* **Latência:** medido na RTX 4090, o workload típico fica **mais rápido que o v0.3** (§9): batch 1 na 4090 é
  limitado por lançamento de kernels, e 22 camadas < 28; e o tokenizer Gemma usa ~1/3 menos tokens em PT/ES.
* **Multilíngue:** vocabulário e pesos pré-treinados em ~1800 línguas em vez de um BPE inglês.
* **Custo de pré-treino:** partir de pesos mmBERT-base crescidos custa ~250 H100-h por 100B tokens, contra um
  pré-treino do zero de trilhões de tokens.
* **Risco:** a pilha de decisão, a serialização e a semântica da API são as do v0.3 (validadas em produção).

## 2. Orçamento de parâmetros (calculado)

`results/v1/param_budget.md`:

| spec | L | d | heads (hd) | FFN | vocab | total | embeddings | não-embedding | pilha de decisão | GFLOP/token @256 | × v0.3 |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| v0.3 (referência) | 28 | 1024 | 16 (64) | 2624 | 50,368 | 421,029,889¹ | 51,577,856 | 369,452,033 | 26,248,193 | 0.759 | 1.00 |
| **K1-W1536-L22** | 22 | 1536 | 24 (64) | 2304 | 256,000 | **879,356,931** | 393,217,536 | **486,139,395** | 44,883,459 | 0.985 | **1.30** |
| K1-W1536-L22-H1 (head 1 camada) | 22 | 1536 | 24 (64) | 2304 | 256,000 | 869,905,923 | 393,217,536 | 476,688,387 | 35,432,451 | 0.964 | 1.27 |
| K1-W1280-L28 | 28 | 1280 | 20 (64) | 2560 | 256,000 | 817,681,923 | 327,681,280 | 490,000,643 | 31,176,963 | 0.998 | 1.31 |
| K1-W1792-L18 | 18 | 1792 | 28 (64) | 2688 | 256,000 | 1,011,222,019 | 458,753,792 | 552,468,227 | 61,080,323 | 1.111 | 1.47 |
| K1-W1536-L22-hd128 | 22 | 1536 | 12 (128) | 2304 | 256,000 | 879,356,931 | 393,217,536 | 486,139,395 | 44,883,459 | 0.985 | 1.30 |
| proxy (mmBERT-small) | 22 | 384 | 6 (64) | 1152 | 256,000 | 144,490,755 | 98,304,384 | 46,186,371 | 3,997,059 | 0.098 | 0.13 |

¹ + 263,938 do act head não usado = 421,293,827 no artefato real.

Decomposição do K1-W1536-L22: embeddings 393.2M · corpo do encoder 441.3M (20.06M/camada) · head de decisão
37.8M (2 camadas, FFN 2d) · type_emb + scorer + answerability 7.1M.

**Faixa 750–850M.** Com o vocabulário completo o total é 879M (3.5% acima). Os 486M que custam latência não mudam com
o vocabulário. Duas saídas, a decidir com dados (Fase G):

| vocab | total |
|---|---:|
| 256,000 (Gemma-2 completo) | 879.4M |
| 200,000 (recorte) | 793.3M |
| 160,000 (recorte) | 731.9M |

O recorte remove linhas de embedding de tokens que nunca aparecem num corpus amostrado de EN/PT/ES/FR + código
(principalmente CJK/índicos) **sem retreinar o tokenizer**: um token removido é re-emitido como seus bytes UTF-8
(`<0xNN>`, o Gemma tem byte-fallback) — sem perda, só mais tokens em scripts raros. Critério para aceitar:
≥ 99.99% das ocorrências de tokens do corpus amostrado preservadas nos 4 idiomas-alvo. Se o critério não for
atingido, mantém-se 256k e 879M.

## 3. Encoder

| | valor | origem |
|---|---|---|
| arquitetura | ModernBERT (pre-norm, GLU GELU, sem bias, RoPE, embedding LayerNorm) — `transformers.ModernBertModel` | mmBERT = ModernBERT + Gemma vocab |
| camadas | 22 | = mmBERT-base (crescimento só em largura) |
| hidden | 1536 | 2× mmBERT-base |
| cabeças | 24 × 64 | cabeças duplicadas pelo HyperCloning |
| FFN | 2304 (GLU; `Wi` 1536→4608) | 2× mmBERT-base. Pode crescer depois com colunas zero-init (preserva função) se o proxy mostrar ganho |
| atenção | global a cada 3 camadas (0, 3, …, 21 → 8 globais), local janela 128 nas outras 14 | mmBERT |
| RoPE | θ = 160k global e local | mmBERT |
| contexto | 8192 posições no pré-treino; serving 1024 tokens por sequência (como v0.3) | |
| dtype | treino bf16 autocast + master fp32; serving bf16 | |

hd128 (12 cabeças de 128) foi medido como alternativa de kernel; não é compatível com a clonagem exata de cabeças.

### Tokenizer

Gemma-2 BPE 256k com byte-fallback, idêntico ao do mmBERT-small/base e do Laya Multilingual. Tokens especiais
existentes (nenhum token novo, a matriz de embedding do mmBERT serve sem redimensionar):

| papel | token | id |
|---|---|---|
| CLS / SEP / PAD | `<bos>` / `<eos>` / `<pad>` | 2 / 1 / 0 |
| marcador de candidato | `<mask>` | 4 |
| slot de answerability `[NULL]` | `<unused0>` | reservado do Gemma |

Fertilidade medida (`results/v1/tokenizer_fertility.json`, 2,974 frases paralelas MASSIVE test; 3,000 strings de
tools): tokens Gemma / tokens ModernBERT = **EN 0.96 · pt-PT 0.66 · es-ES 0.65 · fr-FR 0.71 · descrições de tools
0.90**. O prefixo `" " + opção` do Laya não gera `▁` duplicado com o pré-tokenizador Metaspace (verificado).

## 4. Serialização (`src/krun_one/v1/serialization.py`)

```
<bos> "{type} question: {instructions}" <eos> <mask> cand_0 <mask> cand_1 … [NULL] <eos> contexto <eos>
```

* Idêntica ao v0.3 em tudo que o v0.3 já faz (renderização `nome: descrição`, rótulos humanizados, `level i: …`,
  noul `[false, true]`, cap de 48 tokens por candidato, `max_len` 1024 / `head_max_len` 896, candidatos antes do
  contexto).
* `[NULL]` só em `choice` e `multi`.
* **Chunking**: se os candidatos não cabem em `head_max_len`, são distribuídos (ordem preservada) em várias
  sequências, cada uma com cabeçalho, contexto e `[NULL]` próprios; os logits são reunidos por pergunta
  (`merge_chunks`: rank/aplicabilidade por índice do candidato; `null` = média; answerability = máximo). Noul e score
  nunca são divididos. Substitui o encolhimento uniforme do Laya.
* Continua uma sequência por pergunta e um forward por request (invariância entre perguntas, requisito de
  fidelidade do worker).

## 5. Pilha de decisão (`src/krun_one/v1/model.py`)

```
h = encoder(ids)                                    [B, L, 1536]
h = h + type_emb[qtype]                             choice 0 · score 1 · noul 2 · multi 3 (0–2 = ordem do Laya)
h = head(h)                                         2 × TransformerEncoderLayer(1536, 24 cabeças, FFN 3072, pre-norm)
s = scorer(h[markers]) -> [rank, apply]             LN → Linear(d,d) → GELU → Linear(d,2)
z0 = scorer(h[NULL])[rank]                          logit do slot "nenhum"
g = answer_head([h[NULL]; h[CLS]])                   LN(2d) → Linear(2d,d) → GELU → Linear(d,1)
```

Mudanças vs v0.3: FFN do head 4d → 2d (economiza 18.9M e ~2% de FLOPs; ablação no proxy), scorer com 2 saídas,
`type_emb` com 4 linhas, act head removido, `answer_head` novo.

## 6. Primitivas

| primitiva | serialização | saída pública (compatível com v0.3) | treino |
|---|---|---|---|
| **choice** | candidatos em ordem, `[NULL]` | `softmax(rank / T_fam,K)`; `decision`, `probabilities`, `confidence`, `abstain` | CE no gold (answerable) |
| **noul** | `[false: …, true: …]` | `noul = p_true = σ((z_t − z_f)/T)` | CE 2-way (= BCE em z_t − z_f) |
| **score** | `level i: …` ordenado, nunca permutado | `score = Σ i·p_i`, `confidence` = dispersão ordinal, `probabilities` | CE + (ablação) alvo soft-ordinal gaussiano σ=0.75 e/ou RPS. v0.3: RPS não melhorou ranking, só ECE — manter CE puro salvo evidência do proxy. Treinar 2–10 níveis (v0.3 só viu 5) |
| **multi** (interno) | como choice, `type=multi` | `σ(apply / T)` por candidato; conjunto = {p ≥ τ}; **conjunto vazio permitido** | BCE multi-hot |
| **answerability** | `[NULL]` + `answer_head` | `abstain` e `abstention_status` como hoje | ver §7 |

## 7. Answerability

Pergunta separada de "qual é o melhor candidato": "existe algum candidato aplicável?". Todas as variantes saem do
mesmo forward, então podem ser comparadas no mesmo checkpoint:

| sinal | cálculo | treino que o sustenta |
|---|---|---|
| A. margem (baseline v0.3) | `p1 − p2` do softmax calibrado | OE: CE contra uniforme ×0.5 nos unanswerable |
| B. null-softmax | `1 − softmax([z; z0])[NULL]` | CE K+1, alvo `[NULL]` quando unanswerable |
| C. answer head | `σ(g)` | BCE(g, answerable); rank sem loss nos unanswerable |
| D. max aplicabilidade | `max_c σ(apply_c)` | BCE one-hot da aplicabilidade (compartilhada com multi) |

Hipótese: C (+D auxiliar) separa melhor, porque o softmax é invariante a deslocamento — nenhuma função de `p`
consegue distinguir "todos ruins" de "todos bons", só a forma da distribuição (é o que falha no BFCL
live-irrelevance, onde tools do mesmo domínio produzem distribuições confiantes). B não é uma opção "none" textual;
é um slot sem texto, testado explicitamente contra A porque o pedido era não adotar "none" sem evidência.

### 7.1 Resultado do Answerability Bakeoff (2026-09-26, proxy, 3 sementes; `results/v1/answerability_bakeoff.md`)

**Decisão: nenhum vencedor pela regra registrada → a margem top1−top2 continua como mecanismo público de
abstenção.** As saídas novas (`[NULL]`, aplicabilidade, answer head) permanecem na arquitetura como sinais
**internos/experimentais**. O custo de serving delas é ≤ 0.25 ms (ruído) e elas servem para multi-label.

| mecanismo | macro FA @0.98 | cobertura | FA com cobertura igual à de A (post-hoc) | AUROC | acurácia tool |
|---|---:|---:|---:|---:|---:|
| A margem (baseline) | 2.8% | 64.0% | 2.8% | 0.959 | 93.8% |
| B `[NULL]` no softmax | 0.1% | **6.5%** | 5.9% | 0.937 | 93.0% |
| C answer head | 0.9% | 28.9% | 2.3% | 0.979 | 93.2% |
| D máx. aplicabilidade | 1.7% | 49.2% | 3.4% | 0.962 | 93.3% |
| J answer head (treino conjunto) | 0.5% | 32.6% | **1.9%** | 0.979 | 92.8% |
| E fusão (LR sobre J) | 1.9% | 58.2% | 2.3% | 0.973 | 92.8% |

Leitura:
1. No ponto de operação (seletividade 0.98), todo sinal novo reduz a falsa aceitação **abstendo-se mais**. A
   cobertura cai de 64% para 7–58%, e todos reprovam no gate de cobertura (−2 pp).
2. Com cobertura igual, o melhor sinal (answer head do modelo J) reduz a FA macro em 0.9 pp (2.8% → 1.9%). Isso fica
   abaixo do critério de ganho claro (≥ 3 pp e ≥ 25%). A AUROC maior (0.979 vs 0.959) não basta.
3. O answer head responde "existe candidato aplicável?", mas não detecta erro de ranking. O risco nas tools não vistas
   sobe de 1.0% para 3–6%. A seletividade 0.98 conta os dois tipos de erro, e a margem cobre os dois.
4. Treinar objetivos de OOD custa 0.5–1.0 pp de acurácia de tool (gate). Noul, score e multi não mudam.
5. `[NULL]` dentro do softmax é o pior: instável entre sementes e com baixa AUROC. **Não adotar.**
6. Limitação: o OOD de validação do proxy é relativamente fácil (AUROC da margem 0.96). O caso em que a margem
   falhou no v0.3 (BFCL live-irrelevance, AUROC 0.795) está protegido e não foi usado. Antes de reabrir a questão, é
   preciso um set de validação de irrelevância difícil (KrunBench v1 V) e testar a hipótese de dois limiares:
   answer head para OOD + margem para ranking.

Regra registrada (`configs/v1/proxy.yaml`): uma variante substitui A só se ganhar ≥ 0.01 de AUROC OOD na validação,
sem perder > 0.5 pp de acurácia de choice e sem ECE > A + 0.005. Senão v1 mantém a margem e as saídas novas ficam
internas. A decisão final de abstenção pública combina: `abstain = (sinal < τ_fam)`, τ calibrado em val_fit para
seletividade 0.98, como hoje.

## 8. Losses e estágios de treino

Loss por pergunta (`src/krun_one/v1/losses.py`): `CE_rank + w_ans·BCE_ans + w_apply·BCE_apply (choice) +
w_multi·BCE_apply (multi) [+ w_rps·RPS (score)]`. Padrões: w_ans 0.5, w_apply 0.25, w_multi 1.0, w_rps 0.

| estágio | o quê | dados | custo estimado |
|---|---|---|---|
| G0 crescimento | mmBERT-base → 1536 por HyperCloning (+ ruído 1e-3·std para quebrar simetria) | — | minutos (teste de preservação de função existe) |
| G1 pré-treino continuado | MLM 15%, decodificador esparso, seq 1024, **quantidade decidida por experimento de scaling** (0 → 100M → 500M → 1B → 5B; 10B/25B/50B/100B só se ainda houver ganho forte — `configs/v1/scaling.yaml`, §8.1) | FineWeb-Edu (EN) + FineWeb-2 (PT/ES/FR), ODC-By, descontaminado contra sets protegidos | ~3.5e18 FLOPs por 1B tokens ≈ 2.5 H100-h/1B (40% MFU); estágio 1 (5B) ≈ 13 H100-h |
| G2 decisão | todas as primitivas, KD v1 (12M) | `KRUN_DECISIONS_V1_SPEC.md` | ~12M decisões × ~250 tokens ≈ 3B tokens ≈ 15–25 H100-h |
| G3 hard/OOD | hard negatives v2 + mineração de confusão online + minimal pairs, LR menor | subconjunto difícil | ~5 H100-h |
| G4 calibração | T por (tipo, faixa de K) e thresholds em val_fit | val_fit | CPU |
| G5 avaliação final | KrunBench v1 T (uma vez) + regressão R | | |

### 8.1 Experimento de scaling do HyperCloning (pronto, não executado)

Objetivo: achar o joelho da curva "tokens de MLM continuado → qualidade de decisão" em vez de assumir 100–300B.

| peça | arquivo |
|---|---|
| crescimento G0 + verificação de função (máx |Δ| logits/hidden, concordância top-1) | `scripts/v1/grow_encoder.py` |
| corpus tokenizado + held-out por idioma + descontaminação (8-gramas e substrings de contextos protegidos) | `scripts/v1/prepare_mlm_corpus.py` |
| MLM continuado com marcos (0, 100M, 500M, 1B, 5B; estágio 2: 10B, 25B, 50B, 100B), WSD, checkpoint/resume atômico (bit-idêntico em CPU, testado) e **bloqueio automático enquanto o gate de licença não estiver CLEARED** | `scripts/v1/continued_mlm.py`, `src/krun_one/v1/mlm.py`, `configs/v1/scaling.yaml` |
| downstream por marco: mesmo fine-tune de decisão (proxy data, mesma receita e semente) + referência mmBERT-base sem crescimento | `scripts/v1/train_proxy.py --spec k1 --encoder-init <marco>` |
| resumo + regra de saturação (MLM −2% relativo, ou +0.5 pp de acurácia, ou −1 pp de FA entre marcos consecutivos) | `scripts/v1/scaling_summary.py` |
| runner de pod (termina com DONE; o chamador copia os artefatos e **termina o pod**) | `scripts/v1/runpod_scaling.sh` |

Métricas por marco: MLM loss held-out EN/PT/ES/FR; choice (intent/tool, por idioma), roteamento cross-language,
hard negatives, unseen tools, noul AUROC, score MAE/Spearman e false acceptance OOD @0.98.

Custo estimado do estágio 1 (1× H100): ~13 h de MLM + 7 fine-tunes de ~20 min ≈ 16 H100-h (≈ US$ 40–50 a
US$ 2.5–3/h). Pré-requisitos: gate de licença CLEARED, corpus construído com as licenças confirmadas, e a variante
de loss escolhida no bakeoff.

Otimização: AdamW, LR separado encoder / pilha de decisão (v0.3: 1e-5/5e-5 no fine-tuning), warmup 5% + decaimento
linear, grad clip 1.0, batches por orçamento de tokens, checkpoint atômico com estado completo e hash de config e
dados (`src/krun_one/v1/training.py`; resume bit-idêntico testado em CPU). Distribuído (G1): DDP/FSDP com o mesmo
contrato de checkpoint (um diretório por passo, `trainer_state.json` escrito por último, RNG por rank).

Memória de treino (AdamW misto): 879M × 16 B ≈ 14 GB de estado + ativações; cabe em 1 H100 80 GB com seq 1024 sem
gradient checkpointing; numa 4090 só com checkpointing e batches pequenos.

## 9. Latência (medida — `results/v1/latency_rtx4090_sdpa.md`)

RTX 4090 (Secure Cloud, US-NC-1), torch 2.14.0+cu126, transformers 5.17.0, SDPA — a mesma pilha do worker de
produção. Pesos aleatórios (latência não depende dos valores), serializador e tokenizer reais, contextos MASSIVE pt-PT,
candidatos = strings reais de tools. `total` = tokenização + collate + H2D + forward + pós-processamento (a parte do
worker que depende do modelo). Pesos em bf16; p50 (p95) em ms; razão vs v0.3 na mesma harness. EN dá o mesmo quadro
(`results/v1/latency_rtx4090_sdpa.md`).

| workload | v0.3 (421M) | **K1-W1536-L22** | L22 head 1 camada | W1280-L28 | W1792-L18 | L22 hd128 |
|---|---:|---:|---:|---:|---:|---:|
| **típico** 1 pergunta, k=4, ~90 tok | 17.8 (18.8) | **13.9 (14.5) 0.78×** | 13.4 0.75× | 16.2 0.91× | 11.6 0.66× | 13.9 0.78× |
| **típico** 1 pergunta, 8 tools, ~240 tok | 17.5 (19.4) | **13.7 (14.1) 0.78×** | 13.4 0.76× | 16.3 0.93× | 11.8 0.67× | 13.9 0.80× |
| **típico** 4 perguntas mistas (tools8, noul, score5, k4) | 19.0 (19.5) | **15.4 (15.8) 0.81×** | 15.0 0.79× | 17.8 0.94× | 14.8 0.78× | 15.2 0.80× |
| 1 pergunta, 1024 tok | 17.5 | 14.8 0.85× | 14.1 | 17.2 | 15.2 | 15.2 |
| 32 tools (680 tok) | 17.8 | 14.9 0.84× | 15.1 | 17.0 | 12.9 | 15.1 |
| 64 tools (2 chunks × 972) | 22.7 | 26.2 1.15× | 25.2 | 27.6 | 27.4 | 26.4 |
| 128 tools (3–4 chunks) | 44.4 | 40.2 0.90× | 38.4 | 41.0 | 42.5 | 40.8 |
| 4 perguntas × 448 tok | 20.1 | 21.3 1.06× | 20.8 | 22.7 | 22.6 | 21.4 |
| 16 perguntas × 472 tok | 78.7 | 95.0 1.21× | 92.4 | 96.0 | 99.9 | 96.1 |
| stress 16 perguntas × 32 tools × 1024 | 187.6 | 223.1 1.19× | 215.8 | 230.6 | 231.6 | 227.1 |
| throughput 32 requests × ~96 tok | 32.4 | 35.7 1.10× | 34.7 | 36.4 | 37.9 | 35.7 |
| VRAM pico (bf16) | 2.0 GB | 3.5 GB | 3.5 | 3.1 | 4.0 | 3.5 |

Decomposição no típico de 4 perguntas (K1-W1536-L22): tokenização 1.2 · collate 0.2 · H2D 0.1 · **forward 13.3** ·
pós 0.5 ms.

Leitura:
1. **Workload típico: ~14–15 ms, 20–25% mais rápido que o v0.3 em produção hoje**, com folga de >3× para a meta de
   ~50 ms. Na 4090 o batch 1 é limitado por lançamento de kernels: o forward do v0.3 fica em ~15–17 ms de 128 a 1024
   tokens. Por isso profundidade manda: 22 camadas < 28.
2. No regime limitado por compute (muitas perguntas/tokens por request) o custo segue os FLOPs: **1.15–1.21×** o
   v0.3. O stress de 16×1024 tokens já passa de 50 ms no v0.3 (188 ms) e continua fora da meta (223 ms): não é
   workload típico; mitigação por FlashAttention-2 com unpadding e CUDA Graphs (§10).
3. W1792-L18 é o mais rápido em batch 1 (0.66×), mas tem 1.01B parâmetros (fora da faixa), é o mais lento sob carga
   e não tem fonte pré-treinada — descartado. W1280-L28 não tem ganho em nada e exigiria pré-treino do zero —
   descartado. hd128 não ajuda com SDPA — descartado.
4. Head de 1 camada: −3 a −4% de latência.
   *Nota (2026-09-26, 2ª medição num host EU-RO-1 com CPU mais rápida):* em batch 1 a latência depende da CPU do
   host, porque o tempo é dominado pelo lançamento de kernels. Nesse host o típico caiu para 6.6 ms (K1) e 7.9 ms
   (v0.3), e com 4 perguntas mistas o K1 ficou 1.05× o v0.3 (13.0 vs 12.3 ms). A meta de ~50 ms continua com folga
   de mais de 3× nos dois hosts (`results/v1/latency_answerability_rtx4090.json`). Fica como ablação de qualidade no proxy; o padrão continua 2 camadas
   (Laya) até haver evidência.
5. Pesos bf16 em vez de fp32 + autocast: ganho pequeno em batch 1, ~9% no stress (v0.3: 207 → 188 ms). Adotar.
6. RTX 5090 estava sem estoque na RunPod durante a medição (Secure e Community); a comparação 4090 vs 5090 fica
   pendente. A imagem atual (torch cu126) também não tem kernels sm_120.

## 10. Serving (drop-in no worker)

Contrato edge ↔ worker **inalterado** (payload v1/v2 do queue e `/infer` do LB, mesmos campos de resposta).
Mudanças só no que é específico do modelo:

| arquivo do worker | mudança |
|---|---|
| `krun_one/model.py` vendorizado | + `krun_one/v1/{arch,model,serialization}.py`; `Engine.load` escolhe pela `config_version` |
| artefato | `model.safetensors` (bf16), `config.json` (ModernBERT 1536/22), `tokenizer/` (Gemma), `krun_config.json` `krun_one.config.v2`: `spec`, `serialization` (budget, `null_token`), temperaturas por (tipo, faixa de K), `abstention.signal` + thresholds, `primitives`, `multi_label.enabled=false` |
| carga | pesos já em bf16 (hoje: fp32 + autocast a cada forward) |
| imagem | torch/transformers atuais servem; FP8/INT8/TensorRT só após gate de qualidade (KrunBench §5) |
| VRAM | pesos 1.76 GB bf16 (v0.3: 0.84 GB bf16 / 1.7 GB fp32 como carregado hoje) |
| cold start | artefato ~1.8 GB bf16 vs 1.6 GB hoje (fp32) → sem piora material |
| fidelidade | novo baseline de `fidelity_compare.py`; testes de vendorização / batch-vs-single / LB==queue continuam |

Otimizações candidatas, em ordem de retorno esperado (medir cada uma): (1) CUDA Graphs por bucket de forma — o batch 1
na 4090 é dominado por lançamento de kernels (§9); (2) pesos bf16; (3) FlashAttention-2 com unpadding para requests
multi-pergunta; (4) FP8 na 4090/5090 (Ada/Blackwell) com verificação de calibração.

## 11. Proxy (Fase E) — `configs/v1/proxy.yaml`, `scripts/v1/train_proxy.py`

* Modelo: mmBERT-small (22L/384, pesos pré-treinados) + a mesma pilha v1 = 144.5M total, **46.2M não-embedding**
  (a faixa 50–100M do pedido em compute; o total é inflado pelo vocabulário 256k — o proxy também valida o
  tokenizer do modelo final).
* Dados: corpus v0.3 convertido para `krun.decision.v1` (+ fr-FR paralelo do MASSIVE, + multi-label de mecânica)
  com leakage zerado (`scripts/v1/build_proxy_data.py`).
* Valida: forward/backward, serialização com chunking, todas as losses, multi-label, answerability (variantes A–D
  com regra registrada), francês/cross-language, checkpoint/resume, export, latência.
* Custo previsto: ~1–2 h de RTX 4090 por variante ($0.74/h).

## 12. Portões antes de gastar com o 800M

| portão | estado |
|---|---|
| testes unitários (serialização, modelo, losses, schema, leakage, growth, resume) | ✅ `tests/v1` |
| contagem de parâmetros calculada e verificada | ✅ |
| proxy de latência no GPU de produção | ✅ (§9) |
| validação de dataset (schema + leakage) | ✅ no corpus do proxy |
| smoke de forward/backward/overfit | ✅ tiny; ⏳ proxy real |
| proxy treinado + regra de answerability decidida | ✅ margem mantida (§7.1) |
| recorte de vocabulário decidido (879M vs ~793M) | ✅ 879M, vocabulário completo (decisão do dono); recorte só como ablação |
| licença do tokenizer | ✅ análise sem bloqueio material; **gate aguardando o dono** (`KRUN_ONE_V1_TOKENIZER_LICENSE.md`) |
| corpus de pré-treino com licenças revisadas | ⏳ FineWeb-Edu/FineWeb-2 (ODC-By) a confirmar |
| experimento de scaling do HyperCloning | ✅ pronto (§8.1), não executado |
| estimativa de VRAM/throughput medida no H100 com o modelo crescido | ⏳ |
| KrunBench v1 T materializado e trancado | ⏳ |

### 8.2 Continued-MLM checkpoint policy (1B run, RTX 4090 Community, or a 3090 that passed the benchmark)

| item | value |
|---|---|
| GPU | 4090 Community ($4.74/1B measured). Fallback: 3090 Community, measured **10,512 tok/s = $5.81/1B** (≤ $6 rule, `results/v1/bench_training/bench_training_3090.json`); 1B ≈ 26.4 h on a 3090 vs ~14 h on a 4090 |
| resume checkpoint interval | every **100M tokens**, atomic (`.tmp` + rename, `trainer_state.json` last) |
| maximum compute lost per interruption | ≤ 100M tokens: ~1.4 h ≈ **US$0.47** on a 4090 ($0.34/h), ~2.6 h ≈ **US$0.58** on a 3090 ($0.22/h) |
| checkpoint size | ~10 GB (fp32 weights 3.3 GB + AdamW 6.7 GB + RNG/sampler state) |
| retention | `keep_last: 3`, so the run ends with the **800M pre-decay** checkpoint (branch point for a possible +2B), 900M and the final 1B; milestone weights (0 / 250M / 500M / 800M / 1B, 3.3 GB each fp32) are written for evaluation |
| storage on the pod | corpus ~4 GB + raw downloads ~30 GB (**deleted by the runner once the corpus is verified**) + checkpoints ≤ 30 GB + milestones ~17 GB + venv/models ~11 GB → **90 GB persistent pod volume** (survives stop/restart; lost only on host failure). A pod with only 70 GB of container disk does not qualify. |
| off-pod persistence | before terminating: copy the 1B milestone weights (3.3 GB), the 800M resume checkpoint (10 GB, only needed if the gate says CONTINUE), probe signals, MLM eval JSONs and logs to local disk; checksums are verified before `terminate` |
| health | `scripts/v1/gpu_preflight.py` (nvidia-smi, `cuda.is_available`, capability ≥ 8.0, bf16 op, tiny model forward) runs before any work; any failure → terminate immediately |
