# Plano de Otimização do Recall — "contexto curado, não despejo"

> Problema medido (1 turno, user=thiago): **~5.581 tokens** de memória no `system`,
> sendo **93%** (20.815 chars) um despejo de **20 memórias cruas** (8 episódica +
> 8 conversa + 4 aurora), **sem corte de relevância**. A camada "identidade"
> (semântica) vem **vazia** porque nada popula `semantic_memory`.
>
> **North star:** trocar o despejo por um **contexto curado e enxuto** —
> um perfil semântico pequeno + poucas memórias realmente relevantes + Aurora só
> quando couber. Meta: cortar ~70-80% do bloco mantendo (ou melhorando) qualidade.
>
> Tudo vive em `app/chat_pipeline.py` (já extraído), então as mudanças são locais
> e testáveis. Cada fase = um commit, com testes meus (automatizados) e seus
> (manual), antes de seguir.

---

## Fase A — Cortes rápidos (relevância + top_k + dedup + budget) 🟢 maior corte, menor risco

**Objetivo:** reduzir o tamanho do bloco já, sem mudar a natureza do recall.

**Mudanças (em `chat_pipeline.py` + `config.py`)**
- **Corte de relevância:** filtrar a busca de conversa por `similarity >= recall_min_similarity`
  (a episódica já corta em 0.65; a de conversa hoje não corta nada). Se nada passar,
  o bloco de memória simplesmente não é enviado (o modelo diz "não tenho registrado").
- **top_k menor, configurável:** `recall_episodic_k`, `recall_convo_k`, `recall_aurora_k`
  (ex.: 8→4). Buscar menos.
- **Dedup episódica↔conversa:** as duas cobrem os mesmos turnos em formatos diferentes.
  Mesclar e remover quase-duplicatas (conteúdo de uma contido na outra / alta sobreposição),
  mantendo a de maior similaridade.
- **Teto de tokens:** `recall_char_budget` — truncar o bloco no orçamento, mantendo
  primeiro as de maior similaridade.

**Novos knobs de config** (com defaults seguros): `recall_min_similarity=0.7`,
`recall_episodic_k=4`, `recall_convo_k=4`, `recall_aurora_k=3`, `recall_char_budget=6000`.

**Meus testes (unit):** extrair `merge_and_dedup(mems)` e `apply_budget(text|mems, budget)`
como funções **puras** e testá-las (dedup remove o contido; budget corta na ordem certa;
gating descarta abaixo do limiar). A suíte da rede de segurança segue verde.
**Seus testes (manual):** no `/inspect`, o mesmo prompt de antes deve mostrar o bloco
**bem menor** (medir no inspetor) e o chat continuar lembrando dos fatos principais.
**Gate:** commit `perf(recall): relevância + top_k + dedup + budget`.

---

## Fase B — Popular a memória semântica (extração de fatos) 🟡 conserta a camada morta

**Objetivo:** a camada "identidade" deixar de ser vazia — virar um **perfil compacto e
sempre útil**, em vez de depender de turnos crus.

**Mudanças**
- Novo `app/semantic_extractor.py` (espelho do `aurora_extractor`): dado um turno, o LLM
  extrai **fatos duráveis** `{key, content, importance}` (nome, profissão, preferências,
  fatos estáveis). Usa o `llm_client.complete_text` (Fase 1).
- Persistir via `store_semantic_memory` (upsert por `key` → idempotente) no passo
  fire-and-forget do chat (junto de aurora/episódica).
- **Recall de identidade melhor:** em vez de `search_semantic_hybrid("user name assistant name")`,
  buscar os **top-N fatos por importância** do usuário (já há índice por importância).
- Opcional: um endpoint/aba no inspetor pra ver/editar o perfil semântico.

**Meus testes:** unit do parser de fatos (enums/validação, descarta malformado) — igual
ao padrão do aurora_extractor; integração store→recall do perfil.
**Seus testes (manual):** conversar dando um fato novo ("trabalho com X"), e no `/inspect`
ver o fato aparecer na aba **Semântica** e na camada **identidade** do preview de recall.
**Gate:** commit `feat(memory): extração e recall de perfil semântico`.

---

## Fase C — Re-ranking (qualidade do que entra) 🟡 precisão

**Objetivo:** garantir que os poucos trechos enviados sejam **os mais relevantes**, não só
os primeiros do vetor.

**Mudanças**
- Buscar **mais candidatos** (ex.: 15-20 mesclados) e **re-rankear** antes de cortar p/ top-5.
- Estratégia (escolher na hora): (a) **fusão de score** leve — combinar similaridade +
  importância + recência (sem dep nova); ou (b) **cross-encoder** (`sentence-transformers
  CrossEncoder`) — mais preciso, porém adiciona um modelo/custo. Começo por (a); (b) fica
  opcional.
- Aplicar o re-rank dentro do `merge_and_dedup` (vira `merge_dedup_rerank`).

**Meus testes:** unit da função de ranking (dado candidatos com scores, ordena/seleciona
como esperado; pura, determinística).
**Seus testes:** no `/inspect`, comparar o preview antes/depois — os trechos do topo devem
ser visivelmente mais "na mosca" para a pergunta.
**Gate:** commit `perf(recall): re-ranking dos candidatos`.

---

## Como as fases se combinam

| Fase | O que ataca | Efeito no prompt | Risco |
|---|---|---|---|
| A | despejo sem filtro | **−70%** no tamanho | baixo |
| B | camada identidade vazia | +perfil compacto (alto valor/baixo custo) | médio |
| C | "primeiros ≠ melhores" | mesma qtde, mais relevante | médio |

- **A** é o ganho imediato pro seu incômodo (tamanho). Independente.
- **B** conserta o gap real (semântica vazia) e dá a peça mais valiosa do "contexto curado".
- **C** é polimento; rende mais **depois** de A (com menos itens, re-rankear importa mais).

**Ordem recomendada:** A → B → C. Mas você decide — as três são independentes o
suficiente pra reordenar.

## Definição de pronto (cada fase)
1. Meus testes unit verdes (rede de segurança + novos).
2. Seu checklist manual no `/inspect` (medir o antes/depois).
3. Commit na branch + seguir.

---

### Aparte operacional (fora do recall, mas atrapalha os testes)
Todo rebuild do `memory-api` **re-baixa o modelo de embedding (~2GB)** porque não há
volume pro cache do HuggingFace. Um volume (`~/.cache/huggingface`) elimina isso e deixa
os testes ao vivo instantâneos. Posso encaixar como um commit rápido `chore(docker)`
quando você quiser — não faz parte deste plano, mas economiza muito tempo.
