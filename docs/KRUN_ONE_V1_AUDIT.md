# Krun One v1 — Fase A: auditoria

Data: 2026-09-26. Baseline oficial: **`artifacts/krun-one-v0.3`** (em produção desde KRUN-011; `provenance.json`
diz "promoted to production"; o `README.md` do artefato ainda diz "NOT promoted" — texto desatualizado, o
provenance e o estado do endpoint prevalecem).

Tudo aqui foi lido dos arquivos. Onde algo vem de conhecimento prévio e não foi verificado localmente, está marcado
**[não verificado]**.

## 1. O que existe de fato

| caminho | conteúdo real |
|---|---|
| `artifacts/krun-one-v0.3/` | ModernBERT-large 28L/1024/16h/FFN 2624, vocab 50,368 + head Laya; 421,293,827 parâmetros; `krun_config.json` com T(intent 1.405, tool 1.535, noul 1.222, score 1.169), abstenção `top1_top2_margin` (intent 0.8195, tool 0.7049 @ 0.98) |
| `artifacts/krun-one-v0.3-candidate/` | mesmo run `v03_B` (sha `7432c2c0…`), antes da promoção |
| `artifacts/krun-one-v0/` | KRUN-005, só choice |
| `krun-research/models/GLiNER2/` | repo GLiNER2 (Apache-2.0) completo |
| `krun-research/models/krun-research/models/laya-multilingual/` | pesos reais (644 MB, F16) + configs; encoder **mmBERT-base** |
| `…/gliner2.5-multi-v1/` | pesos (1.15 GB), encoder **mDeBERTa-v3-base** (não é o "multi-Decide") |
| `…/gliner2.5-decide/` | pesos (1.95 GB), encoder **DeBERTa-v3-large, só inglês** |
| HF cache | `jhu-clsp/mmBERT-small` (pesos + tokenizer), ModernBERT-base, e5-small; **mmBERT-base não está no cache** |

Não existe "mmBERT-large" público. A família mmBERT publicada tem small (140M) e base (307M) **[não verificado
além dos configs locais]**.

## 2. Comparação estruturada

| | Krun One v0.3 | Laya | Laya Multilingual | mmBERT-base | GLiNER2.5 Multi v1 | GLiNER2.5 Decide |
|---|---|---|---|---|---|---|
| encoder | ModernBERT-large | ModernBERT-large | mmBERT-base | — | mDeBERTa-v3-base | DeBERTa-v3-large |
| L / d / heads / FFN | 28 / 1024 / 16 / 2624 GLU | igual | 22 / 768 / 12 / 1152 GLU | 22 / 768 / 12 / 1152 | 12 / 768 / 12 | 24 / 1024 / 16 |
| vocab | 50,368 (BPE, inglês) | igual | 256,000 (Gemma-2, BPE byte-fallback) | igual | 250,112 (SPM) | 128,011 |
| atenção | global a cada 3 + local 128; RoPE 160k/10k | igual | global a cada 3 + local 128; RoPE 160k/160k | igual | relativa (DeBERTa), 512 pos. | relativa |
| params | 421.3M | 421.3M | 321.9M (encoder 306.9M, 64% embeddings) | 306.9M | 287.4M | 486.4M (repo diz 340M) |
| serialização | `[CLS] "{type} question: {ins}" [SEP] ([MASK] opção)* [SEP] contexto [SEP]` | igual | igual | — | `( [P] tarefa [DESCRIPTION] l: d … ( [L] l1 [L] l2 ) ) [SEP_STRUCT] … [SEP_TEXT] texto` | idem (span) |
| slot por candidato | `[MASK]` antes do texto do candidato | igual | igual | — | `[L]` antes do rótulo | `[L]` |
| descrição | inline `"nome: descrição"` junto ao marcador | igual | igual | — | num bloco `[DESCRIPTION]` separado do `[L]` | idem |
| head | type_emb(3) + 2 camadas transformer (d, 4d) + scorer MLP → 1 logit | igual + act head | igual | — | MLP H→2H→1 por `[L]`, sem head transformer | idem |
| loss | CE soft-target; OOD = CE uniforme ×0.5 | RLCD (log+esférico−RPS) | RLCD | MLM | BCE multi-hot (inclusive single-label) | idem |
| inferência | softmax + T por família + margem | softmax + T por (tipo, K) | sem calibração | — | softmax (single) / sigmoid≥0.5 (multi, com fallback argmax) | idem |
| abstenção | margem top1−top2 calibrada | act head (satura ≈1.0) | idem | — | nenhuma em classificação (o `null_projection` é só de NER) | nenhuma |
| multi-label | não | não | não | — | sim (sigmoid) | sim |
| ordinal | "level i: …" + E[score] | + RPS | idem | — | `ordered=True`, sem loss ordinal | — |
| perguntas/req. | 1 sequência por pergunta, 1 forward | igual | igual | — | várias tarefas numa sequência | idem |
| idiomas | EN + pt-PT/es-ES (só contexto) | EN | multilíngue (README: pior em inglês que Laya) | 1800+ **[não verificado]** | multilíngue | EN |
| licença | NOTICE credita Laya Apache-2.0 | Apache-2.0 | Apache-2.0 | MIT **[não verificado]** | Apache-2.0 | Apache-2.0 |

Latência medida (repo): v0.3 bf16 na RTX 4090, 1 pergunta ≈ 9.3 ms (bench isolado); forward em produção 19–28 ms
(4090, `LATENCY_INVESTIGATION.md`); a inferência é ~5–12% da latência end-to-end.

## 3. Achados que mudam o desenho

1. **Tokenizer é alavanca de latência.** Medido (`results/v1/tokenizer_fertility.json`, MASSIVE test paralelo,
   2,974 frases): o tokenizer Gemma-2/mmBERT usa **0.96× tokens em EN, 0.66× em pt-PT, 0.65× em es-ES, 0.71× em
   fr-FR** e 0.90× em descrições de tools, versus o ModernBERT do v0.3. Um encoder ~1.3× mais caro por token custa
   *menos* por decisão em PT/ES.
2. **Embeddings dominam a contagem com vocab 256k.** Em mmBERT-base 64% dos parâmetros são embeddings. Para um
   modelo "800M" o que custa latência é o corpo (não-embedding); a contagem total precisa ser lida com esse split.
3. **Não há checkpoint multilíngue grande para partir.** mmBERT-base (768) é o maior mmBERT. As opções são
   pré-treinar do zero ou **crescer** mmBERT-base. Crescimento em largura 2× preservando a função (HyperCloning) está
   implementado e testado (`src/krun_one/v1/growth.py`, teste de preservação exata).
4. **Laya encolhe todos os candidatos igualmente** quando não cabem (`(head_max_len−16)//K` tokens): em k=64 cada
   descrição vira ~13 tokens, em k=128 ~6. Laya Multilingual documenta colapso acima de ~20 opções. O v1 precisa de
   *chunking* com merge de logits, não encolhimento.
5. **Não há sinal de answerability treinado em nenhuma referência.** Laya: act head saturado. GLiNER: só opção
   "none" explícita ou `default`. v0.3: margem calibrada (funciona in-domain, AUROC 0.95–0.99; **falha em BFCL
   live-irrelevance: AUROC 0.795, falsa aceitação 39.4% @0.98**).
6. **GLiNER treina BCE e infere softmax** para single-label — incoerente; e nunca devolve conjunto vazio em
   multi-label (fallback para argmax). Não copiar.
7. **Lacunas de idioma do corpus atual:** francês ausente, pt-BR ausente (MASSIVE só tem pt-PT), es-LatAm ausente;
   tools, noul e score 100% em inglês; instruções/critérios sempre em inglês.
8. **Bug latente** no pipeline v0.3: a fatia "unseen tool" do noul está sempre vazia (o builder lê
   `metadata.unseen_tool`, o choice grava `tool_seen`) — `prepare_primitives_data.py:201`.
9. **Lock de final-test fraco:** `evaluate.py:84-87` só bloqueia tags `v02*`. KrunBench v0 test, holdout_tools e
   primitives test já foram abertos 4× — **não servem mais como test cego para o v1** (ficam como regressão).
10. Serving: flash-attn não é usado; o worker carrega pesos fp32 e usa autocast bf16; o worker vendoriza
    `krun_one/model.py` (teste exige cópia idêntica). Fidelidade queue↔LB é bit-idêntica; drift entre tipos de GPU
    ~0.05 em p (aceito).

## 4. Manter / remover / adaptar / novo

| | item | motivo |
|---|---|---|
| **manter** | layout `build_sequence`, marcador `[MASK]` por candidato, header `"{type} question: …"`, renderizações (nome: descrição, humanize, `level i:`, noul `[false,true]`), cap 48 tokens | validado em produção; mantém semântica da API |
| manter | type embedding + head transformer + scorer compartilhado para todos os tipos | um só forward heterogêneo, sem cabeças por tipo |
| manter | CE para choice/noul/score; T por família; margem calibrada como baseline de abstenção | referência do v0.3 |
| manter | 1 sequência por pergunta, 1 forward por request | invariância entre perguntas (teste de fidelidade) |
| manter | checkpoint atômico, RNG, config hash, resume validado; batches por orçamento de tokens | infra provada |
| manter | hard-negative mining v1 (RRF label/e5/confusão; filtro de intercambiáveis) | base do v2 |
| **remover** | act head | satura, não usado (263k params carregados à toa) |
| remover | encolhimento uniforme de candidatos | destrói descrições em k≥32 |
| **adaptar** | encoder → arquitetura mmBERT (tokenizer Gemma-2 256k), largura 1536 | multilíngue + menos tokens |
| adaptar | scorer com 2 saídas (rank, aplicabilidade) | multi-label sem nova cabeça |
| adaptar | aumentos do GLiNER: shuffle de candidatos, nomes sintéticos "label i" mantendo descrição, drop de candidatos | forçar leitura da descrição (unseen tools) |
| adaptar | temperatura por (tipo, faixa de K) como Laya | v0.3 usa só por família |
| **novo** | marcador `[NULL]` (token reservado `<unused0>` do Gemma) + cabeça de answerability | "existe candidato aplicável?" separado de "qual é o melhor?" |
| novo | chunking de candidatos + merge de logits (k até 128) | k=64/128 sem truncar descrições |
| novo | tipo `multi` (type_emb linha 3) | multi-label interno |
| novo | crescimento HyperCloning mmBERT-base → 1536 | inicialização multilíngue pré-treinada |
| novo | schema `krun.decision.v1`, leakage automatizado (grupo, exato, near-dup MinHash, strings de tool protegidas) | dados em escala sem vazamento |
| **ignorar** (GLiNER) | spans, count heads, relations, records, JSON-structure, `[E]/[C]/[R]`, decodificadores com restrição, few-shot `[EXAMPLE]` | específicos de extração |
