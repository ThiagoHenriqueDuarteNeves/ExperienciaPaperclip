# Plano de Refatoração — SOLID, modularização e extensibilidade

> Objetivo: elevar o `memory-api` (e limpar o `frontend`) para SOLID de verdade —
> interfaces nos pontos de extensão reais, zero duplicação divergente, superfície
> honesta — **sem mudar o comportamento observável** (a não ser onde explícito).
>
> Regra de ouro: **cada fase é um commit isolado**, com testes automatizados (meus)
> e um checklist manual (seus) ANTES de seguir. Rollback = `git revert` da fase.

## Princípios do processo

- **Branch dedicada** por fase: `refactor/<n>-<slug>` → revisão → merge no `master`.
- **Refatores são "no-behavior-change"**: a rede de testes da Fase 0 é o que garante isso.
- **Lembrete operacional:** editar `memory-api/` exige `docker compose up --build`
  (não `restart`). Teste mobile é via HTTP no IP da LAN — evitar APIs de contexto seguro.
- **Decisão LangGraph:** manter isolado p/ futuro (Fase 8), não ligar no chat agora.

---

## Fase 0 — Rede de segurança (testes primeiro) 🔴 pré-requisito

**Por quê primeiro:** não dá pra refatorar com segurança sem um guard. Caracterizamos
o comportamento ATUAL antes de mexer.

**Mudanças**
- Validar que `pytest` roda no container (`pytest.ini` já existe).
- Testes de caracterização (capturam o comportamento de hoje, não o ideal):
  - `auth.py`: hash/verify PIN, issue/verify token, expiração, assinatura inválida.
  - Formatação do contexto de memória do chat (extrair a montagem do prompt p/ testar).
  - Híbrido RRF: dado 2 rankings, a fusão ordena como esperado.
- Marcar testes que precisam de DB/Chroma com `@pytest.mark.integration`.

**Meus testes:** `pytest -m "not integration"` verde; integração verde com `compose up`.
**Seus testes:** abrir o app, logar, mandar 2-3 mensagens, confirmar que lembra de um
fato dito antes (baseline — é o que vamos preservar em todas as fases).
**Gate:** commit `test: rede de segurança de caracterização`.

---

## Fase 1 — Extrair `llm_client.py` (DRY) 🟠 alto valor, baixo risco

**Problema:** a chamada ao `/messages` da Anthropic está duplicada 4× (main, entity,
aurora, langgraph) com headers e o truque de "concatenar blocos text" repetidos.

**Mudanças**
- Novo `app/llm_client.py` com:
  - `stream_messages(...)` (async, SSE) — usado pelo chat.
  - `complete(...)` (sync/async) — usado pelos extractores.
  - Centraliza provider/base/headers/retries e o parsing de blocos `text` (modelos de raciocínio).
- `main.py`, `entity_extraction.py`, `aurora_extractor.py` passam a chamar o client.
- Corrige a inconsistência: nada mais hardcoda URL.

**Meus testes:** unit do client com `httpx` mockado (provider→base correto, retry,
concat de blocos, erro `max_tokens`). Suíte da Fase 0 segue verde.
**Seus testes:** mandar mensagem normal (streaming OK), e uma troca emocional p/
confirmar que a extração Aurora ainda grava (`docker logs` mostra `aurora record stored`).
**Gate:** commit `refactor(llm): unifica chamadas em llm_client`.

---

## Fase 2 — Quebrar o `main.py` (SRP) 🔴 maior impacto estrutural

**Problema:** `main.py` (1054 linhas) acumula models + rotas + pipeline + util.

**Mudanças (puro refactor, sem mudar rotas/contratos)**
- `app/schemas.py` — todos os modelos Pydantic.
- `app/routers/` — `health.py`, `episodic.py`, `graph.py`, `conversation.py`,
  `semantic.py`, `aurora.py`, `letta.py`, `auth.py`, `chat.py`.
- `app/services/chat_pipeline.py` — recall + montagem de prompt + persistência.
- `main.py` vira só: cria `app`, middleware, `include_router(...)`, lifespan.

**Meus testes:** suíte verde sem alteração (os contratos não mudaram); checagem de
import; `GET /health` e cada rota responde igual. Adiciono testes de rota com `TestClient`.
**Seus testes:** smoke completo — login, chat com streaming, nova conversa, trocar
perfil, logout. Tudo idêntico a antes.
**Gate:** commit `refactor(api): divide main em routers e services`.

---

## Fase 3 — Protocolo `MemoryRetriever` + registry (OCP) 🔴 o coração da extensibilidade

**Problema:** o recall fixa as 4 fontes no `asyncio.gather` e a formatação à mão.
Adicionar uma 5ª camada hoje = editar o pipeline.

**Mudanças**
- `app/retrievers/base.py`:
  ```python
  class MemoryRetriever(Protocol):
      name: str
      async def recall(self, query: str, user_id: str, top_k: int) -> list[Memory]: ...
      def render(self, memories: list[Memory]) -> str: ...
  ```
- Adaptar cada fonte (episódica, conversacional, semântica, aurora) como um retriever.
- `RetrieverRegistry` — o pipeline itera sobre os registrados (gather automático).
- Adicionar uma camada nova passa a ser: criar a classe + registrar. **Zero edição do pipeline.**

**Meus testes:** um `FakeRetriever` registrado aparece no contexto montado, provando OCP;
testes de cada retriever real (recall + render). Suíte da Fase 0 verde.
**Seus testes:** chat lembra de fatos (episódica/conversacional), do nome (semântica) e
revive o tom (aurora) — as 4 camadas ainda contribuem.
**Gate:** commit `feat(memory): retrievers plugáveis via protocolo + registry`.

---

## Fase 4 — Feature flags no lugar dos stubs por `ImportError` 🟠

**Problema:** `try/import … except ImportError: def stub()` (~5×) esconde dependência
quebrada como "feature desligada".

**Mudanças**
- Flags explícitas em `config` (`enable_neo4j`, `enable_letta`, `enable_aurora`...).
- Import real falha alto (bug de dependência aparece no deploy).
- Feature desligada por flag = caminho controlado, logado uma vez no startup.

**Meus testes:** com flag off, rota responde o "desabilitado" esperado; com on e dep
ausente, **estoura** (comportamento desejado). 
**Seus testes:** `GET /health` reflete cada serviço; desligar Aurora por env e ver o
chat seguir sem ela.
**Gate:** commit `refactor(config): feature flags substituem stubs de import`.

---

## Fase 5 — `lifespan` + erros de migration visíveis 🟡

**Problema:** `@app.on_event` depreciado e `except: pass` engole falha de migration.

**Mudanças**
- Migrar startup/shutdown para `lifespan`.
- Migrations: logar e, em falha real, falhar o startup (ou health degradado claro) —
  nada de silêncio.

**Meus testes:** startup com DB ok sobe; com migration forçada a falhar, o erro aparece.
**Seus testes:** `docker compose up --build` sobe limpo; `docker logs` sem warnings novos.
**Gate:** commit `refactor(startup): lifespan e migrations sem swallow`.

---

## Fase 6 — DRY das 3 CTEs híbridas 🟠

**Problema:** `search_semantic_hybrid`, `search_similar_hybrid`, `search_aurora_hybrid`
são ~50 linhas quase iguais (tabela, coluna FTS, SELECT).

**Mudanças**
- Um builder parametrizado de SQL híbrido (tabela, coluna fts, colunas de retorno,
  filtro de user). As 3 funções viram chamadas finas.

**Meus testes:** as 3 buscas retornam os mesmos resultados de antes (golden tests sobre
dados de fixture). Suíte da Fase 0 verde.
**Seus testes:** buscar por um nome próprio dito antes (BM25) e por paráfrase (vetor) —
ambos recuperam.
**Gate:** commit `refactor(db): builder único p/ busca híbrida`.

---

## Fase 7 — Remover código morto do frontend 🟡

**Problema:** `lib/claude/client.ts`, `lib/memory/client.ts`, `lib/claude/tools.ts`
não são importados (confirmado) e contradizem a arquitetura ("só backend chama LLM").

**Mudanças**
- Remover os 3 arquivos e a dep `@anthropic-ai/sdk` do `frontend/package.json` (se nada mais usar).
- Conferir que `next build` segue verde.

**Meus testes:** `next build` ok; grep confirma nenhum import órfão.
**Seus testes:** app no Vercel/local carrega, login e chat funcionam (não dependiam desse código).
**Gate:** commit `chore(frontend): remove cliente LLM/memory legado`.

---

## Fase 8 — LangGraph: isolar com honestidade (decisão tomada) 🟡

**Decisão:** manter para o futuro, fora da superfície ativa.

**Mudanças**
- Trocar os endpoints 501 por algo honesto: removê-los das rotas **ou** colocá-los atrás
  de `enable_langgraph=false` (Fase 4) com resposta clara de "experimental, desligado".
- Mover `langgraph_agent.py` para `app/experimental/` com um README curto: o que é, por que
  está parado, o que falta p/ ligar (streaming, integrar retrievers da Fase 3, usar `llm_client`).
- Manter as deps no `requirements.txt` (anotadas como experimentais) — ou movê-las p/ um
  extra opcional, à sua escolha.

**Meus testes:** suíte verde; nenhuma rota ativa promete o que não cumpre.
**Seus testes:** `GET /health` e o chat inalterados; `/agent/chat` não finge mais 501 enganoso.
**Gate:** commit `chore(langgraph): isola como experimental, remove 501 enganoso`.

---

## Fase 9 — Endurecimento (opcional) 🟡

- CORS: restringir `allow_origins` aos domínios reais (Vercel + zrok) por env.
- Auth secret/Letta pass: garantir que produção exige segredo forte (já há warning).
- Cobertura: completar testes de auth, aurora, híbrido e pipeline de chat.

**Gate:** commit `chore(security): restringe CORS e reforça segredos`.

---

## Ordem e racional

| Fase | Entrega | Risco | Depende de |
|---|---|---|---|
| 0 | Rede de testes | — | — |
| 1 | `llm_client` | baixo | 0 |
| 2 | Routers/services | baixo* | 0 |
| 3 | Retrievers plugáveis | médio | 0,2 |
| 4 | Feature flags | baixo | 2 |
| 5 | lifespan/migrations | baixo | 2 |
| 6 | Builder híbrido | baixo | 0 |
| 7 | Limpeza frontend | baixo | — |
| 8 | LangGraph isolado | baixo | 4 |
| 9 | Hardening | baixo | — |

\* baixo *porque* a Fase 0 existe. Sem ela, seria alto.

## Definição de pronto (cada fase)
1. Meus testes automatizados verdes.
2. Seu checklist manual aprovado.
3. Commit na branch da fase + merge.
4. Só então a próxima fase começa.
