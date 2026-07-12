# Assistente com Memória Persistente

Um chatbot que **lembra**. Em vez de esquecer tudo a cada conversa, o assistente
mantém memória de longo prazo em camadas — o que aconteceu, fatos sobre você, as
relações entre as coisas e até a *carga emocional* dos momentos.

---

## Visão geral

Dois blocos desacoplados:

- **`frontend/`** — interface de chat em Next.js (React 19). Hospedada no Vercel.
  **Não fala com nenhum LLM** — apenas com a `memory-api` (via uma rota-proxy de
  borda, que resolve CORS e a tela-interstitial do zrok).
- **`memory-api/`** — o cérebro, em FastAPI (Python). Faz o *recall* de memória,
  chama o LLM e persiste tudo. Roda local via Docker e pode ser exposta à internet
  por um túnel (zrok).

```
┌──────────────┐     HTTPS      ┌───────────────────────────────────────────┐
│  Frontend    │  /api/chat     │                 memory-api (FastAPI)        │
│  (Next.js)   │ ─────────────► │  recall → monta prompt → LLM (stream) →     │
│  Vercel      │ ◄───────────── │  persiste (fire-and-forget)                 │
└──────────────┘   SSE stream   └───────┬─────────┬─────────┬────────┬────────┘
                                         │         │         │        │
                                    ┌────▼───┐ ┌───▼───┐ ┌───▼───┐ ┌──▼────┐
                                    │ pgvec  │ │Chroma │ │ Neo4j │ │ Letta │
                                    │(pg16)  │ │(episó-│ │(grafo)│ │(proce-│
                                    │conv +  │ │ dica) │ │       │ │dural) │
                                    │semânt+ │ └───────┘ └───────┘ └───────┘
                                    │aurora  │
                                    └────────┘
```

## Camadas de memória

| Camada | Onde mora | O que guarda |
|---|---|---|
| **Episódica** | ChromaDB | turnos de conversa brutos |
| **Conversacional** | PostgreSQL + pgvector | histórico por thread, busca híbrida (vetor + BM25 com RRF) |
| **Semântica (perfil)** | pgvector | fatos duráveis sobre o usuário (nome, profissão, preferências) |
| **Afetiva (Arquivo Aurora)** | pgvector | *como* um momento foi sentido (tom, reação, "saudade") |
| **Grafo de conhecimento** | Neo4j | entidades e relações extraídas das conversas |
| **Procedural** | Letta (MemGPT) | blocos de persona / memória arquival |

O recall é **curado, não despejado**: a cada turno, [memory-api/app/chat_pipeline.py](memory-api/app/chat_pipeline.py)
recupera as camadas relevantes, filtra por relevância, deduplica e monta um contexto
enxuto — a mesma função alimenta o chat e o inspetor, então o que você inspeciona é
exatamente o que vai ao LLM.

## Stack

- **Backend:** FastAPI, asyncpg, slowapi (rate limit), httpx.
- **LLM:** API compatível com Anthropic Messages (Anthropic, Deepseek ou um proxy
  local como LM Studio) — configurável por env var.
- **Embeddings:** `intfloat/multilingual-e5-large` (1024-dim, local) ou um provider
  via API (OpenAI-compat).
- **Bancos:** PostgreSQL 16 + pgvector, ChromaDB, Neo4j 5, Letta.
- **Frontend:** Next.js 16, React 19.

---

## Como subir (local)

Pré-requisitos: Docker Desktop.

```bash
# 1. configure o ambiente
cp .env.example .env
#    edite .env: ANTHROPIC_API_KEY (ou credenciais do seu provider), MEMORY_AUTH_SECRET, etc.

# 2. suba o stack (6 serviços)
docker compose up -d --build

# 3. (opcional) suba o frontend
cd frontend && npm install && npm run dev   # http://localhost:3001
```

A API fica em **http://localhost:8001** (health em `/health`).

> Editar código em `memory-api/` exige `docker compose up --build` (não `restart`).

### Variáveis de ambiente principais

| Var | Para quê |
|---|---|
| `ANTHROPIC_API_KEY` | chave do LLM |
| `LLM_PROVIDER` | `anthropic` (padrão) ou `deepseek` |
| `LLM_API_BASE` | sobrescreve o endpoint do LLM (ex.: proxy local `http://host.docker.internal:1111/v1`) |
| `CLAUDE_MODEL` | modelo a usar |
| `MEMORY_AUTH_SECRET` | segredo HMAC dos tokens de sessão (defina em produção) |
| `INSPECT_ENABLED` | `true` liga o inspetor de memória (default off) |
| `MEMORY_EMBEDDING_API_BASE` / `_KEY` | usar embeddings via API em vez do modelo local |

Portas: frontend `3001`, memory-api `8001`, ChromaDB `8000`, pgvector `5434→5432`,
Neo4j `7474`/`7687`, Letta `8283`.

---

## Inspetor de memória

Ferramenta de observabilidade local: dado um prompt, mostra **exatamente** o que
seria enviado ao LLM (cada camada recuperada + o `system` montado), e permite
navegar livremente o conteúdo de todos os bancos.

```bash
INSPECT_ENABLED=true docker compose up -d --build memory-api
# abra http://localhost:8001/inspect
```

Desligado por padrão (para não expor a memória via túnel público).

---

## Testes

```bash
# unit (rápido, sem serviços) — usa um venv leve em memory-api/.venv
cd memory-api
./.venv/Scripts/python.exe -m pytest -m unit          # Windows
# python -m pytest -m unit                            # Linux/Mac

# integração (precisa dos bancos no ar)
docker compose up -d memory-db chromadb neo4j
docker compose exec -T memory-api pytest -m integration
```

Detalhes em [memory-api/tests/README.md](memory-api/tests/README.md).

---

## Estrutura

```
frontend/                  Interface de chat (Next.js)
memory-api/app/
  main.py                  FastAPI app + rotas
  chat_pipeline.py         recall + montagem do prompt (chat e inspetor)
  llm_client.py            cliente unificado do LLM
  schemas.py               modelos Pydantic
  retrieval.py             memória episódica (ChromaDB)
  conversation_store.py    memória conversacional (pgvector)
  semantic_store.py        perfil semântico (pgvector)
  aurora_store.py          memória afetiva (pgvector)
  neo4j_client.py          grafo de conhecimento
  *_extractor.py           extração via LLM (entidades, aurora, fatos semânticos)
  inspect_api.py           inspetor (/inspect)
  scripts/                 reprocessamento de grafo / backfill semântico
migrations/                schema SQL (init + migrações versionadas)
```

## Documentação adicional

- [memory-architecture-report.md](memory-architecture-report.md) — relatório técnico que embasou a arquitetura.
- [REFACTOR_PLAN.md](REFACTOR_PLAN.md) / [RECALL_OPTIMIZATION_PLAN.md](RECALL_OPTIMIZATION_PLAN.md) — planos de evolução.
