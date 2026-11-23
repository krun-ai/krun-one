# Krun One v1 — licença do tokenizer / vocabulário (Gemma-2 via mmBERT)

Data da verificação: 2026-09-26. Isto é uma análise técnica e documental, **não é parecer jurídico**. Recomenda-se
revisão por advogado antes de publicar pesos.

**GATE: CLEARED** — 2026-09-26, decisão do dono do projeto:

> Owner technical approval.
> Not legal advice.
> Preserve MIT + Apache-2.0 notices and provenance.

O tokenizer completo de ~256k peças é mantido; não se cria tokenizer próprio nesta fase. Todo artefato publicado
precisa levar os avisos da §5 e o registro de proveniência do tokenizer (fonte Apache-2.0 `google/gemma_pytorch`,
sha256 e teste de equivalência).

## 1. Perguntas

1. Podemos treinar, redistribuir e publicar o Krun One comercialmente usando esse tokenizer/vocabulário?
2. Quais notices/termos precisam acompanhar o modelo?

## 2. Fatos verificados

| # | fato | evidência |
|---|---|---|
| F1 | O tokenizer do mmBERT é o vocabulário Gemma-2: BPE de 256.000 entradas, com `<start_of_turn>`, `<end_of_turn>`, `<unused0..99>`, `<2mass>`, `[@BOS@]` e os primeiros ids na ordem do Gemma (`<pad>` 0, `<eos>` 1, `<bos>` 2, `<unk>` 3, `<mask>` 4) | arquivos locais em `~/.cache/huggingface/hub/models--jhu-clsp--mmBERT-small/snapshots/abc32620…/`; os arquivos não têm nenhum texto de licença ou copyright |
| F2 | O Google publica o mesmo vocabulário sob **Apache-2.0**: `github.com/google/gemma_pytorch`, `tokenizer/tokenizer.model` (4.241.003 bytes, sha256 `61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2`). O LICENSE do repositório é Apache-2.0 | baixado e verificado localmente com sentencepiece: **255.999 de 256.000 peças idênticas** às do tokenizer do mmBERT. A única diferença é o id 226: o arquivo do Google escreve o byte TAB como `<0x09>` e o do mmBERT como `\t` |
| F3 | mmBERT-small e mmBERT-base são **MIT** nos model cards ("License: mit"; "Tokenizer: Gemma 2") | huggingface.co/jhu-clsp/mmBERT-base, /mmBERT-small. Os repositórios no HF não têm arquivo LICENSE. No GitHub (JHU-CLSP/mmBERT) há um badge MIT, mas a API do GitHub retorna `license: null` |
| F4 | O paper do mmBERT (arXiv 2509.06888) diz que o modelo usa o tokenizer do Gemma 2 e não trata dos termos desse tokenizer | arXiv 2509.06888v1 |
| F5 | Gemma Terms of Use (versão de 2026-04-01): "Gemma" = "the set of machine learning language models, trained model weights and parameters identified in the Appendix". "Model Derivatives" = modificações do Gemma, "works based on Gemma", ou modelos criados por transferência de pesos, parâmetros, operações ou Outputs do Gemma. Os termos **não mencionam tokenizers**. O Gemma 4 foi publicado sob Apache-2.0; Gemma 2 e 3 continuam sob os Gemma Terms | ai.google.dev/gemma/terms · ai.google.dev/gemma/apache_2 |
| F6 | O Laya Multilingual (também sobre mmBERT-base e o mesmo vocabulário) é publicado como Apache-2.0 | huggingface.co/convaiinnovations/laya-multilingual |
| F7 | Precedentes: fine-tunes públicos do mmBERT-base são publicados como MIT, Apache-2.0, CC-BY-NC e EUPL, e nenhum inclui os Gemma Terms. Não há discussão pública sobre a licença do tokenizer do mmBERT (issues no HF e no GitHub revisadas) | HF API / discussions |

## 3. Interpretação

* O Krun One v1 usaria **pesos do mmBERT** (MIT, treinados do zero pelos autores em dados abertos) e o **vocabulário
  Gemma-2**. Não usaria pesos do Gemma nem outputs gerados pelo Gemma. Pela definição, um vocabulário não é "models,
  trained model weights and parameters". Os itens (i) e (iii) de "Model Derivatives" não se aplicam.
* O único ponto aberto é a expressão ampla "works based on Gemma". O risco é pequeno, porque o próprio Google
  distribui o arquivo idêntico sob Apache-2.0 (F2). Usando essa cópia como fonte declarada, o vocabulário chega a nós
  sob Apache-2.0.
* **Treinar: sim. Servir comercialmente via API: sim. Publicar pesos: sim**, sob a licença que escolhermos, desde que
  os avisos do MIT e do Apache-2.0 acompanhem.

## 4. Veredito

| pergunta | resposta | risco |
|---|---|---|
| Pode treinar | Sim | baixo |
| Pode servir comercialmente (API) | Sim | baixo |
| Pode redistribuir/publicar pesos | Sim, com os notices da §5 | baixo |
| Bloqueio material? | **Não identificado** → não é necessário tokenizer próprio | — |

## 5. O que precisa acompanhar o modelo

1. **Declarar a fonte do tokenizer** como `google/gemma_pytorch` `tokenizer/tokenizer.model` (Apache-2.0),
   registrando o sha256 e o teste de equivalência. Salvar esse teste como script
   `scripts/v1/verify_tokenizer_source.py` quando o artefato for montado. Na publicação, incluir o texto da
   Apache-2.0 e um aviso de origem (Apache §4).
2. **MIT do mmBERT**: incluir o texto da licença MIT e o crédito aos autores ("mmBERT authors, JHU CLSP"), com um aviso
   de que o modelo foi modificado (alargado por HyperCloning e re-treinado).
3. **Apache-2.0 do Laya**: já está no `NOTICE`, pelo código derivado.
4. **Opcional, por cautela e de baixo custo**: incorporar a Gemma Prohibited Use Policy aos termos de uso da API.
   Isso cobre o pior caso de interpretação sem afetar o produto.
5. Não usar as marcas "Gemma" ou "Google" no nome ou no marketing do modelo.

## 6. Pior caso (se os Gemma Terms fossem considerados aplicáveis)

* Incluir as restrições da §3.2 (Prohibited Use Policy) de forma exigível nos termos do produto e na licença dos pesos.
* Entregar cópia dos Gemma Terms a quem receber os pesos, junto com o arquivo de Notice "Gemma is provided under and
  subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms". A API fica isenta desse arquivo, mas não das
  restrições.
* Marcar os arquivos modificados.
* A licença própria não poderia contrariar os Gemma Terms, ou seja, não poderia ser totalmente permissiva.

Mesmo nesse cenário, o uso comercial via API continua permitido.

## 7. Alternativa conservadora (só se o dono ou o advogado decidir)

Treinar um tokenizer próprio (BPE com byte-fallback sobre corpus EN/PT/ES/FR + código). Custo: descartar a matriz
de embeddings do mmBERT (393M parâmetros no K1) e fazer um re-warmup mais longo, o que invalida a vantagem de partir
de pesos pré-treinados. Pela evidência acima, isso não parece justificado.

## 8. Decisão

Tomada em 2026-09-26: aprovação técnica do dono (ver GATE no topo). O caminho da §7 não é seguido nesta fase.
