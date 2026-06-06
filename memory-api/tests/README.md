# Testes do memory-api

Dois níveis, separados por marcador (ver `pytest.ini`):

- **`unit`** — rápidos, sem serviços externos. São o *guard de refatoração*: rodam
  a cada fase pra provar que o comportamento não mudou.
- **`integration`** — precisam de PostgreSQL/pgvector (e afins) no ar. Pulam
  automaticamente quando o banco não está acessível.

## Rodando localmente (host, venv leve)

O venv `.venv/` tem só as deps necessárias para coletar e rodar os testes `unit`
(sem `torch`/`chromadb`). Criado uma vez com:

```
python -m venv .venv
./.venv/Scripts/python.exe -m pip install pytest pytest-asyncio fastapi \
    pydantic-settings httpx asyncpg pgvector
```

Guard rápido (use a cada fase):

```
./.venv/Scripts/python.exe -m pytest -m unit
```

## Rodando a integração (precisa de Docker)

```
docker compose up -d --build memory-db
docker compose run --rm memory-api pytest -m integration
```

> Lembrete: editar `memory-api/` exige `docker compose up --build` (não `restart`).
