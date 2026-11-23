# Krun Decisions v1 — especificação do dataset

Status: **proposta**. Nenhum dado em escala deve ser gerado antes de: (1) este schema aprovado, (2) as regras de
leakage automatizadas passando (já implementadas: `src/krun_one/v1/leakage.py`), (3) KrunBench v1 congelado
(`docs/KRUNBENCH_V1_SPEC.md`), porque os sets protegidos precisam existir antes de minerar negativos ou gerar dados.

Implementação executável do schema: `src/krun_one/v1/schema.py` (`validate`, conversores do formato v0).

## 1. Schema `krun.decision.v1`

Um único formato para choice, noul, score, multi e OOD. A diferença entre tipos está em `task.type` e em regras de
consistência, nunca em formatos paralelos.

```json
{
  "schema_version": "krun.decision.v1",
  "id": "kd1/tool/xlam/48671/pt-BR/cl-03",
  "split": "train",
  "task": {
    "type": "choice",
    "instructions": "Choose the function that should be called to fulfil the user request.",
    "instructions_language": "en",
    "criteria": null,
    "family": "tool",
    "domain": "billing"
  },
  "context": {"text": "quero cancelar minha assinatura", "language": "pt-BR",
              "code_switching": false, "register": "informal", "noise": []},
  "candidates": [
    {"id": "cancel_subscription", "label": "cancel_subscription",
     "description": "Cancel an active customer subscription.", "language": "en"},
    {"id": "pause_subscription", "label": "pause_subscription",
     "description": "Temporarily pause billing for a subscription.", "language": "en"}
  ],
  "target": {"labels": ["cancel_subscription"], "distribution": null},
  "answerable": true,
  "semantic_group_id": "sg:xlam:48671",
  "split_group_id": "spg:xlam:48671",
  "candidate_set": {"strategy": "hard_v2", "k": 2, "hard_negative_ids": ["pause_subscription"], "removed_gold": null},
  "slices": ["cross_language"],
  "minimal_pair": null,
  "provenance": {
    "source_dataset": "xlam", "hf_id": "argilla/apigen-function-calling", "revision": "170a8a7e…",
    "license": "CC-BY-4.0", "original_split": "train", "original_id": "xlam-48671",
    "transforms": ["translate:context:pt-BR"],
    "synthetic": {"teacher": "<model id>", "prompt_id": "translate-v1", "verifier": "<model id or human>",
                  "verdict": "accepted", "source_example_id": "kd1/tool/xlam/48671/en-US",
                  "transformation": "translation", "generated_utc": "2026-10-01T12:00:00Z"}
  },
  "metadata": {}
}
```

| campo | regra |
|---|---|
| `task.type` | `choice` (exatamente 1 vencedor ou nenhum), `multi` (0..K), `noul` (candidatos fixos `false`,`true`), `score` (candidatos = níveis ordenados, 2–16) |
| `target.labels` | ids de candidatos; choice: 1 ou 0; multi: 0..K; noul/score: exatamente 1 |
| `target.distribution` | opcional (rótulos soft: anotadores múltiplos, p.ex. MHS/HelpSteer2) |
| `answerable` | choice/multi ⇔ labels não vazio; noul/score sempre `true` |
| `context.language` | BCP-47. First-class: `en-US`, `pt-BR`, `pt-PT`, `es-ES`, `es-419` (+`es-MX` etc.), `fr-FR`, `fr-CA`; `mul` para code-switching |
| `candidates[].language` | idioma do texto do candidato (cross-language = difere do contexto) |
| `semantic_group_id` | mesmo significado: traduções, paráfrases, variantes com typo do mesmo item fonte |
| `split_group_id` | unidade que nunca cruza splits: ⊇ grupo semântico; conjuntos de minimal pairs; item fonte; em MASSIVE, o `id` compartilhado entre locales |
| `schema_language` | idioma do lado do schema (instruções + candidatos), BCP-47 primário ou `mul` — derivado (`schema.derive_fields`) |
| `cross_language` | `context.language` ≠ `schema_language` (subtag primária; `mul` conta como cross) — derivado e validado |
| `difficulty` | `easy` / `medium` / `hard` — derivado da estratégia de negativos, tipo de OOD, near-duplicates e minimal pairs |
| `candidate_set.strategy` | estratégia de negativos (obrigatória): `native`, `pool`, `hard_v1`, `hard_v2`, `random`, `mixed`, `fixed` |
| `slices` | vocabulário fechado (`schema.SLICES`) |
| `provenance.synthetic` | obrigatório para qualquer dado gerado; exige `teacher`, `prompt_id`, `verifier`, `verdict`, `source_example_id`, `transformation`; só `verdict = accepted` entra em treino |

OOD é representado como `answerable = false` com `slices` indicando o tipo (`ood_near`, `ood_far`, `gold_removed`,
`native_irrelevance`, `cross_task`). **Nenhuma opção artificial "none/other" é injetada** — a abstenção é um sinal
do modelo (`docs/KRUN_ONE_V1_ARCHITECTURE_SPEC.md` §7), não um candidato.

## 2. Fontes

Estado de licenças como registrado nos pipelines atuais; itens "a verificar" **não entram no treino** até revisão.

| fonte | id @ revisão | licença | uso em v1 |
|---|---|---|---|
| BANKING77 | `legacy-datasets/banking77` @ `f5412156` | CC-BY-4.0 | treino / val; test = regressão |
| MASSIVE | `AmazonScience/massive` @ `ed58ac42` | CC-BY-4.0 (GitHub: NOASSERTION — confirmar) | treino (en-US, pt-PT, es-ES, **fr-FR novo**; outros locales para robustez opcional) |
| xLAM 60k | `argilla/apigen-function-calling` @ `170a8a7e` (rows `origin == xLAM`) | CC-BY-4.0 no card **+ "research purposes only"** | **EXCLUÍDO do v1** (`excluded_pending_license_clarification`, decisão do dono 2026-09-26); proveniência histórica do v0.3 |
| ToolACE | `Team-ACE/ToolACE` @ `6bda777c` | Apache-2.0 | treino tools |
| xlam-irrelevance | `MadeAgents/xlam-irrelevance-7.5k` @ `34323bf0` | CC-BY-4.0 | **EXCLUÍDO do v1** (derivado do xLAM) |
| Civil Comments | `google/civil_comments` @ `f2970eb3` | CC0-1.0 | noul; **deduplicar entre splits** (§6) |
| HelpSteer2 | `nvidia/HelpSteer2` @ `990b2711` | CC-BY-4.0 | score/noul |
| Measuring Hate Speech | `ucberkeley-dlab/measuring-hate-speech` @ `5468f6e1` | CC-BY-4.0 | score/noul (+ `distribution` dos anotadores) |
| HWU64 | `FastFit/hwu_64` @ `9fb5bff1` | **não confirmada** no mirror | só avaliação (regressão) |
| BFCL v3 | `gorilla-llm/Berkeley-Function-Calling-Leaderboard` @ `61fc0608` | Apache-2.0 | **protegido**: nunca treino, mining, teacher ou tradução |
| BoolQ | `google/boolq` @ `35b264d0` | CC-BY-SA-3.0 | só avaliação externa |
| SST-5 | `SetFit/sst5` @ `e51bdcd8` | **não confirmada** | só avaliação externa |
| e5-small (mining) | `intfloat/multilingual-e5-small` @ `614241f6` | MIT **[confirmar]** | ferramenta de mining, não dado |

Candidatas a adicionar (cada uma precisa de revisão de licença **antes** do download para treino): fontes nativas de
pt-BR / es-419 / fr-CA de intents e atendimento; datasets de function-calling multilíngues; ASR/typo corpora. Nenhuma
está aprovada nesta proposta.

Registro por fonte (arquivo `configs/v1/sources.yaml`, **criado**; builders ignoram fontes `excluded_*`): `dataset, hf_id, revision, license,
license_url, usage (train|eval|protected), provenance notes, reviewer, reviewed_utc`.

## 3. Splits e grupos

* Splits: `train`, `validation` (tuning; dividido em val_fit / val_eval por grupo, como v0.3), `test` (KrunBench v1
  final, protegido), `protected` (benchmarks externos).
* **A unidade de split é `split_group_id`.** Traduções, paráfrases, variantes com ruído e membros de um conjunto de
  minimal pairs herdam o grupo do item fonte. Um dado sintético derivado de um item herda o split dele; nunca se
  gera dado de treino a partir de um item de validação/test.
* MASSIVE: o `id` é paralelo entre locales → um grupo por `id` (já é assim no v0).
* Tools: grupo por query normalizada (v0) **e** held-out por tool (6% das tools nunca aparecem em treino, nem como
  negativo) + held-out por **domínio** (novo: domínios inteiros de tools reservados para `unseen_domain`).

### 3.1 Tools não vistas (unseen) — definição e invariantes

Uma tool é **vista** se aparece em qualquer lugar do lado de treino, como gold **ou** como distrator/negativo, seja
com a mesma chave normalizada (`nome: descrição`), seja de forma **equivalente**: mesma descrição com outro nome,
ou descrição quase duplicada (Jaccard de 5-gramas de caractere ≥ 0.8). Implementação: `src/krun_one/v1/tools.py`.

Invariantes verificados automaticamente (o build aborta se falharem): a fatia `unseen_tool` não pode estar vazia
nem em choice nem em noul, e nenhuma tool dessa fatia pode ser vista no treino. Testes de regressão:
`tests/v1/test_v1_core.py::test_noul_builder_unseen_tool_slice_regression` e `::test_proxy_unseen_tool_marking_and_invariants`.

Bug corrigido (v0.3): o builder de noul lia `metadata.unseen_tool`, mas o choice grava `tool_seen`. Por isso a fatia
de noul estava sempre vazia. Além disso, a marcação usava o gold da decisão e não a tool da proposição. Ao
reaplicar a definição acima, 7 golds que o v0 marcava como "unseen" (5 em val_tools e 2 em holdout_tools)
**já apareciam no treino**, 1 de forma exata e 6 por descrição equivalente.

## 4. Hard negatives v2

Preserva o v1 (`src/krun_one/mining.py`: RRF k=10 de similaridade de rótulo, e5 contexto↔centroide, confusão
leave-one-group-out; tools: query↔tool, gold↔tool, char-TF-IDF do nome; filtro de intercambiáveis cos ≥ 0.95).

Acrescenta, todos combinados por RRF e amostrados do top-2(k−1) com semente por decisão:

| sinal | como |
|---|---|
| query↔candidato cross-lingual | e5 multilíngue (ou o próprio encoder do proxy) com o contexto no idioma original e o candidato em EN — negativos difíceis *para o caso PT→EN* |
| gold↔candidato por descrição | embeddings de `nome: descrição`; hoje só nome+descrição juntos |
| near-duplicate de candidato | pares com cos 0.85–0.95 marcados `near_duplicate` (fatia própria), ≥ 0.95 continuam proibidos como negativo |
| confusão online | a cada época do proxy / checkpoint do modelo: top-k errados com alta confiança em val_fit viram prioridade de mining **de treino** (nunca a partir de itens de val/test) |
| histórico de confusão | tabela `confusion_pairs(label_a, label_b, count)` acumulada por versão |

Salvaguardas: (1) o pool de mining é construído **só** com registros `train` que passaram `check_leakage` contra os
protegidos; (2) um negativo que seja semanticamente correto (falso negativo) é o maior risco: todo par com
similaridade > 0.9 ao gold vai para verificação (verificador §6) antes de ser usado; (3) negativos fáceis (random)
limitados a ≤ 25% das decisões; OOD fácil (`ood_far`) ≤ 20% do total de OOD.

## 5. Minimal pairs

Objetivo: forçar o modelo a ler o predicado, não palavras-chave.

```
"I want to cancel my transfer."   -> cancel_transfer
"My transfer was cancelled."      -> transfer_cancelled
"Can a transfer be cancelled?"    -> transfer_information
```

Metodologia:
1. **Seleção de âncoras:** pares de rótulos com alta confusão (mining v2) no mesmo domínio.
2. **Geração controlada:** o teacher recebe o item âncora + o rótulo-alvo contrastante + a *transformação* permitida
   (`negation`, `tense/aspect`, `question_vs_request`, `agent_vs_patient`, `modality`, `quantity`, `entity_swap`)
   e deve produzir a menor edição que troca o rótulo. Nunca volume livre.
3. **Verificação independente** (§6): outro modelo (ou humano no set de avaliação) rotula a frase **sem ver** o rótulo
   pretendido, escolhendo entre os candidatos do conjunto; aceita se concordar. Distância de edição ≤ limiar
   registrado (caracteres) para garantir que é mínimo.
4. **Armazenamento:** cada membro é um registro normal com `minimal_pair = {set_id, member, contrast}`; todos os
   membros compartilham `split_group_id` (o conjunto inteiro fica num split) mas têm `semantic_group_id` distintos.
5. **Avaliação:** a métrica de conjunto é "todos os membros corretos" (além da acurácia por membro).

## 6. Dados sintéticos

Permitido apenas para: minimal pairs, contrafactuais, hard negatives (textos de candidatos), ambiguidades, near-OOD,
expressões regionais, gírias, typos, code-switching e pares cross-language. **Não** para aumentar volume de classes
já cobertas.

Pipeline obrigatório:

```
item fonte (train) -> teacher (prompt versionado) -> filtros determinísticos (idioma, tamanho, duplicata,
leakage vs protegidos) -> verificador independente (modelo diferente do teacher, sem ver o alvo)
-> verdict accepted|rejected -> registro com provenance.synthetic completo
```

* O verificador responde à *mesma pergunta* do registro (escolhe candidato / verdadeiro-falso / nível); aceita se
  bater o alvo. Taxa de rejeição é métrica de qualidade do prompt; prompts com rejeição > 30% são revisados.
* Auditoria humana: amostra estratificada de 200 por (transformação × idioma) antes de liberar o lote.
* Teacher/verificador precisam de termos de uso que permitam treinar outro modelo com a saída — registrar em
  `configs/v1/sources.yaml` junto com as fontes. **A escolha de provedores é decisão do dono do projeto.**
* Itens de validação/test/protected nunca entram em prompt de teacher, tradução ou busca por similaridade
  (`check_leakage` roda sobre os *inputs* do teacher também).

## 7. Política cross-language

Casos (todos first-class):

| contexto | candidatos | exemplo |
|---|---|---|
| X | X | pt-BR → rótulos pt-BR |
| X ≠ EN | EN | "quero cancelar minha assinatura" → `cancel_subscription: Cancel an active…` (**caso central**) |
| EN | X | contexto EN, catálogo de tools descrito em PT |
| X | Y (ambos ≠ EN) | es → fr (menor peso) |
| code-switching | EN | "preciso dar um refund pro cliente" |

Regras: (1) nomes de tools **nunca** são traduzidos (são ids); descrições podem ser; (2) a tradução de contextos é
feita por teacher + verificador e herda o grupo; (3) para cada item traduzido guarda-se a versão EN no mesmo grupo,
permitindo medir consistência cross-language (mesma decisão nas duas línguas); (4) instruções: 70% EN, 30% no idioma
do contexto.

Variedades: pt-BR como variante principal do português (≥ 65% do PT), pt-PT ≥ 25%; espanhol: es-ES ~40% /
es-419 ~60% (MX, AR, CO no mínimo); francês: fr-FR ~80% / fr-CA ~20%. Marcas regionais (vocabulário, gírias, você/tu,
vosotros/ustedes) viram fatia `regional`.

## 8. OOD

| tipo | construção | alvo no mix de OOD |
|---|---|---|
| gold removido (near-OOD) | remover o gold; manter hard negatives v2 | 35% |
| irrelevância nativa | xlam-irrelevance; catálogos reais sem a tool | 20% |
| near-OOD sintético | teacher gera pedido plausível no domínio que nenhum candidato cobre; verificador confirma que nenhum se aplica | 20% |
| cross-task | contexto de intent vs catálogo de tools e vice-versa | 10% |
| far-OOD | outro dataset/domínio | ≤ 15% |

Proporção de `answerable = false`: 15–20% das decisões choice/multi (v0.3: 10.6%). Multi com conjunto vazio: 10–15%
das decisões multi.

## 9. Regras de leakage (automatizadas — `src/krun_one/v1/leakage.py`)

1. `split_group_id` / `semantic_group_id` em dois splits → erro (`check_split_groups`).
2. Texto normalizado idêntico entre lado-de-treino e protegido → erro.
3. Near-duplicate (Jaccard de 5-gramas de caractere ≥ 0.9, MinHash/LSH) → erro (limiar registrado).
4. Strings de tool de benchmarks cujo objetivo é medir tools não vistas (BFCL, holdouts) usadas como candidato em
   treino → erro.
5. "Lado-de-treino" inclui: arquivos de treino, pools de mining, inputs de teacher/tradução, índices de similaridade.
6. O build **remove** violações e reroda até zero; a lista removida vai para o relatório.

**Achados ao aplicar essas regras ao corpus do v0.3** (`data/v1_proxy/leakage_report.json`): 25 contextos idênticos
a sets protegidos (Civil Comments repete comentários entre splits oficiais; um coincide com uma query BFCL
live-irrelevance), 131 near-duplicates (ex.: BANKING77 train vs test, Jaccard 0.93) e **372 decisões de treino que
usam como candidato uma tool idêntica a uma tool do BFCL** (xLAM e BFCL compartilham funções). Consequência: o
número "BFCL 92.3%" do v0.3 não mede tools totalmente não vistas. O KrunBench v1 corrige isso.

## 10. Mistura-alvo (~12M decisões)

| categoria | % | ~decisões |
|---|---:|---:|
| choice / intent | 20 | 2.4M |
| tool/function routing | 25 | 3.0M |
| noul (proposições) | 14 | 1.7M |
| score / ordinal (2–10 níveis) | 9 | 1.1M |
| multi-label | 8 | 1.0M |
| OOD / answerability (choice+multi, `answerable=false`) | 12 | 1.4M |
| minimal pairs / contrafactuais | 5 | 0.6M |
| ruído (typos, gírias, code-switching, regional) — sobreposto às categorias acima | (15 do total) | — |

Por idioma do contexto: EN 34%, PT 24% (pt-BR ≥ 16%), ES 22%, FR 17%, outros/`mul` 3%. Cross-language (contexto ≠
candidatos) ≥ 35% das decisões não-EN.

Por k (choice/multi): 2: 10%, 4: 20%, 8: 25%, 16: 22%, 32: 13%, 64: 7%, 128: 3%.

A mistura é hipótese: o proxy decide ajustes (p.ex. se a proporção de OOD degrada choice), e a versão é congelada
num `configs/v1/mixture.yaml` com hash antes do treino de 800M.

## 11. Portões antes de gerar em escala

1. `docs/KRUNBENCH_V1_SPEC.md` congelado e sets protegidos materializados (hash).
2. `configs/v1/sources.yaml` com todas as licenças revisadas.
3. Piloto de 50k decisões: 100% `validate()`, `check_leakage` zero, auditoria humana das fatias sintéticas.
4. Proxy treinado no piloto sem regressão vs proxy no corpus v0.3 convertido.
