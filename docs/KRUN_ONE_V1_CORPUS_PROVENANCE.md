# Krun One v1 — corpus de continued MLM e fontes de decisão: proveniência e licenças

Data: 2026-09-26 (atualizado com as decisões do dono na mesma data). Relatório técnico para a decisão do dono do projeto. **Não é parecer jurídico.** Licenças e SHAs
foram lidos da API do Hugging Face e dos cards dos datasets nesta data. O que não foi confirmado está marcado
**[não verificado]**.

## 1. Corpus do continued MLM (curto: 1–5B tokens, qualidade > quantidade)

| fonte (HF id @ sha[:10]) | licença | termos extras | uso previsto | veredito |
|---|---|---|---|---|
| `HuggingFaceFW/fineweb-edu` @ 87f09149ef (config `sample-10BT`) | ODC-By 1.0 | Termos de Uso do Common Crawl | EN educacional/técnico | **USAR com aviso** |
| `epfml/FineWeb2-HQ` @ c0c06e94fd | ODC-By 1.0 | Common Crawl; opt-out do FineWeb; o card recomenda filtro extra de PII | PT/ES/FR: os 10% melhores do FineWeb-2 (classificador de "texto estruturado e rico em conhecimento") | **USAR com aviso** — base principal para PT/ES/FR |
| `HuggingFaceFW/fineweb-2` @ af9c13333e | ODC-By 1.0 | Common Crawl; e-mails/IPs anonimizados; opt-out | fatia de fóruns (texto informal) em PT/ES/FR, selecionada por padrão de URL; pt-BR aproximado por TLD `.br` | **USAR com aviso** |
| `HuggingFaceFW/finepdfs-edu` @ 9cfabe2127 | ODC-By 1.0 | Common Crawl (PDFs) | manuais, especificações e documentação técnica, em 4 idiomas | **USAR com aviso** |
| `HuggingFaceFW/finewiki` @ 8bd13e72e6 | CC-BY-SA 4.0 + GFDL | share-alike | EN/PT/ES/FR enciclopédico | **EXCLUÍDO** — `excluded_for_v1_initial_training` (simplicidade de licença / revisão de share-alike; decisão do dono) |
| `bitext/Bitext-customer-support-llm-chatbot-training-dataset` @ 430d1a89bd | CDLA-Sharing-1.0 | "no obligations on Your Use or Publication of Results"; o share-alike vale só se os dados forem republicados | conversas de suporte (EN), com marcas de coloquialidade e typos | **USAR com aviso** |
| `google-research-datasets/taskmaster1/2/3` | CC-BY-4.0 | atribuição | diálogos orientados a tarefa (EN) | **USAR com aviso** |
| MultiWOZ 2.2 (`pfb30/multi_woz_v22`) | Apache-2.0 no card (o repositório upstream é MIT) | — | diálogos de serviço (EN) | **USAR com aviso** |
| `cambridgeltl/multi3woz` | MIT | — | diálogos em FR | **USAR com aviso** |
| `OpenAssistant/oasst1`, `oasst2` | Apache-2.0 | — | chat com registro informal; EN/ES grandes, PT/FR menores | **USAR com aviso** |
| `Team-ACE/ToolACE` | Apache-2.0 | sintético; modelo gerador não declarado **[não verificado]** | descrições e schemas de tools (JSON) | **USAR com aviso** |
| `HuggingFaceTB/smollm-corpus` (`cosmopedia-v2`) | ODC-By | sintético (Mixtral) | texto técnico EN, se necessário | **USAR com aviso** (baixa prioridade) |
| `bigcode/the-stack-v2`, `starcoderdata`, `stack-edu`, `codeparrot/github-code-clean` | outro / gated / por arquivo | acordo com a Software Heritage; licenças por arquivo (inclusive GPL) | código / Markdown | **REVISAR** — fora nesta fase |
| APIs-guru `openapi-directory` | CC0 no repositório; cada spec tem licença própria | uso justo alegado para parte das specs | specs OpenAPI | **REVISAR** — só specs com origem aberta |
| `google-research-datasets/schema_guided_dstc8` | CC-BY-SA-4.0 | share-alike | diálogos | **REVISAR** — pequeno, descartável |
| OpenSubtitles / OPUS-100 | desconhecida | diálogo de filmes com copyright | — | **EXCLUIR** |

`HuggingFaceFW/fineweb-2-edu`, `fineweb-edu-2` e `fineweb-2-hq` **não existem**. O subset HQ oficial é o
`epfml/FineWeb2-HQ`.

### Termos de Uso do Common Crawl (atualização de 2024-03-07) — trechos relevantes

* §2: o conteúdo rastreado "may be subject to separate terms of use"; é proibido usá-lo para invadir privacidade ou
  violar direitos de terceiros.
* §3: "YOU AGREE TO RESPECT THE COPYRIGHTS AND OTHER APPLICABLE RIGHTS OF THIRD PARTIES".
* §9: indenização ao Common Crawl por reclamações de terceiros, inclusive pelo uso "for the purpose of developing,
  training, or deploying AI Systems".

Na prática: **treinar não é proibido e não há exigência de atribuição**. O risco de copyright de terceiros não fica
coberto e passa para quem usa. Mitigação: registrar os SHAs usados, respeitar o opt-out do FineWeb antes de cada
release e não redistribuir o corpus montado.

## 2. Mistura do primeiro corpus (1B tokens) — implementada em `scripts/v1/prepare_mlm_corpus.py`

Decisões do dono (2026-09-26):
* **FineWiki fica fora.** Os ~7% dele não foram substituídos token por token: o corpus fica com ~0.93B tokens únicos,
  e o sampler faz ~1.08 passagens para chegar a 1B tokens de treino.
* Taskmaster, MultiWOZ e Multi3WOZ ficam fora do primeiro corpus. São distribuídos por script de carga e não foram
  implementados; o volume deles é pequeno.
* Entram como diálogo/suporte: Bitext (EN) e oasst1/oasst2 (EN/PT-BR/ES/FR, pelo campo `lang`).
* Entram como tools: schemas JSON e turnos do ToolACE.

| idioma | web | técnico (FinePDFs-edu) | fóruns (FineWeb-2) | suporte | tools |
|---|---:|---:|---:|---:|---:|
| EN | 250M (FineWeb-Edu) | 40M | — | 25M (Bitext + oasst) | 10M (ToolACE) |
| PT | 190M (FineWeb2-HQ) | 25M | 12M | 3M (oasst) | — |
| ES | 150M (FineWeb2-HQ) | 20M | 10M | 5M (oasst) | — |
| FR | 150M (FineWeb2-HQ) | 20M | 10M | 5M (oasst) | — |

## 2.1 Mistura proposta originalmente (histórico)

| fatia | fontes (tokens aprox.) |
|---|---|
| **EN 35%** (350M) | FineWeb-Edu 250M · FinePDFs-edu 40M · FineWiki 25M · suporte/diálogo (Bitext, Taskmaster, MultiWOZ, oasst) 25M (≤ 3 épocas) · ToolACE, descrições/schemas de tools 10M |
| **PT 25%** (250M) | FineWeb2-HQ por 190M (priorizar `.br`) · FinePDFs-edu 25M · FineWiki 20M · fatia de fóruns 12M · MASSIVE/oasst 3M |
| **ES 20%** (200M) | FineWeb2-HQ spa 150M · FinePDFs-edu 20M · FineWiki 15M · fóruns 10M · oasst/MASSIVE 5M |
| **FR 20%** (200M) | FineWeb2-HQ fra 150M · FinePDFs-edu 20M · FineWiki 15M · fóruns 10M · Multi3WOZ/MASSIVE/oasst 5M |

Lacunas: não se confirmou nenhum corpus de suporte ao cliente em PT/ES/FR com licença permissiva. O caminho limpo é
tradução local (opus-mt, CC-BY-4.0) de Bitext, Taskmaster e ToolACE.

Descontaminação obrigatória (implementada em `scripts/v1/prepare_mlm_corpus.py`): descartar qualquer documento que
contenha um 8-grama, ou a string inteira, de um contexto protegido (KrunBench v1 T/X, BFCL, holdouts).

## 3. Fontes do dataset de decisões (Krun Decisions v1)

| fonte | licença | status |
|---|---|---|
| BANKING77, MASSIVE (en-US/pt-PT/es-ES/fr-FR) | CC-BY-4.0 | **ENTRA** (MASSIVE: o GitHub diz NOASSERTION — confirmar) |
| ToolACE | Apache-2.0 | **ENTRA** |
| Civil Comments | CC0-1.0 | **ENTRA** (deduplicar entre os splits oficiais) |
| HelpSteer2, Measuring Hate Speech | CC-BY-4.0 | **ENTRA** |
| Bitext customer support (27 intents de suporte) | CDLA-Sharing-1.0 | **ENTRA** (novo; roteamento de suporte, com typos e coloquialismo) |
| Taskmaster 1/2/3, MultiWOZ, Multi3WOZ (FR) | CC-BY-4.0 / MIT / Apache | **ENTRA** (novo; intents e domínios de diálogo, com proveniência) |
| **xLAM function-calling-60k** (`Salesforce/xlam-function-calling-60k`, via o re-host `argilla/apigen-function-calling`) | card: CC-BY-4.0, gated, **e também "for research purposes only in support of an academic paper"** | **EXCLUÍDO do v1** — `excluded_pending_license_clarification` (decisão do dono, 2026-09-26). Continua como proveniência histórica do v0/v0.3. Os builders removem também as tools do xLAM usadas como distratoras em decisões de outras fontes |
| xlam-irrelevance-7.5k (MadeAgents) | CC-BY-4.0 | **EXCLUÍDO do v1** — derivado do xLAM, segue a mesma decisão |
| hermes-function-calling-v1, glaive-function-calling-v2 | Apache-2.0 | **REVISAR** — o modelo gerador não é declarado (pode ter termos proibindo treinar modelos concorrentes) |
| HWU64 (mirror) | não confirmada | **FORA** (só regressão) |
| BFCL | Apache-2.0 | **FORA do treino** (protegido) |
| BoolQ (CC-BY-SA-3.0), SST-5 (não confirmada) | — | **FORA do treino** (avaliação externa) |
| CLINC150 (tem a classe out-of-scope) | CC-BY-3.0 **[não verificado]** | candidato; verificar antes de usar |

## 4. Share-alike (Wikipedia / FineWiki) — interpretação

Leitura corrente na comunidade de modelos abertos: pesos treinados não são "Adapted Material" do texto. XLM-R,
ModernBERT e muitos outros treinaram em Wikipedia e publicaram pesos sem aplicar share-alike. Para um encoder MLM
o risco de reproduzir texto literal é baixo. Não há decisão judicial sobre isso. Se o dono quiser zero exposição a
share-alike, basta tirar o FineWiki (cerca de 7% do corpus) sem mudar o resto.

## 5. Avisos para o model card / NOTICE

> Contains information from FineWeb-Edu, FineWeb-2, FineWeb2-HQ, FinePDFs-Edu (and SmolLM-Corpus, if used), which are made available under the ODC Attribution License (https://opendatacommons.org/licenses/by/1-0/).

> Portions of the training data derive from Common Crawl (https://commoncrawl.org); use is subject to the Common Crawl Terms of Use (https://commoncrawl.org/terms-of-use).

> Trained in part on Taskmaster-1/2/3 (Google, CC BY 4.0), MASSIVE (Amazon, CC BY 4.0), BANKING77 (CC BY 4.0), HelpSteer2 (NVIDIA, CC BY 4.0), Measuring Hate Speech (UC Berkeley D-Lab, CC BY 4.0) and Wikipedia text (Wikimedia contributors, CC BY-SA 4.0 / GFDL).

E também: MultiWOZ (MIT), Multi3WOZ (MIT), OpenAssistant (Apache-2.0), ToolACE (Apache-2.0), Bitext
(CDLA-Sharing-1.0: o modelo conta como "Results"), Civil Comments (CC0), mmBERT (MIT), tokenizer Gemma via
`google/gemma_pytorch` (Apache-2.0) e Laya (Apache-2.0).

## 6. Bloqueios (atualizado)

* xLAM e xlam-irrelevance: excluídos do v1 (ver §3). As tools reais passam a ser ToolACE mais as fontes novas
  levantadas no `KRUN_V1_TOOL_DATA_REPORT.md`.
* FineWiki: excluído do corpus inicial.

### Texto original desta seção

* **Nenhum bloqueio material** para o corpus de MLM recomendado (§2).
* **Decisão pendente do dono:** manter ou retirar xLAM e xlam-irrelevance do corpus de decisões (§3).
  * Sem eles, as tools vêm de ToolACE + sintético local e o volume de tools reais cai de ~52k para ~11k.
  * O v0.3 em produção foi treinado com xLAM.
