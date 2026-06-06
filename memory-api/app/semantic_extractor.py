"""Semantic fact extraction — builds the durable user profile.

Mirrors aurora_extractor, but instead of affective moments it distills STABLE,
long-lived facts about the user (name, profession, goals, strong preferences,
ongoing situations) into semantic_memory. Each fact carries a canonical snake_case
`key` so repeated writes UPSERT (update in place) instead of duplicating — keeping
the profile compact and current.

This is what the recall "identity" layer reads. Without it, semantic_memory stays
empty and identity recall returns nothing.
"""

from __future__ import annotations

import json
import logging

import httpx

from app.config import settings
from app.llm_client import LLMError, complete_text

logger = logging.getLogger(__name__)


SEMANTIC_EXTRACTION_PROMPT = """Você é um extrator de FATOS DURÁVEIS sobre o usuário, para uma memória de perfil.

Analise o último turno da conversa (mensagem do usuário + resposta da assistente) e extraia APENAS fatos estáveis e de longo prazo sobre o usuário: nome, profissão, objetivos, preferências fortes, relacionamentos, condições, situações em andamento.

NÃO extraia: conversa trivial, perguntas, estados momentâneos, opiniões passageiras, nada sobre a assistente. A maioria dos turnos NÃO tem fato durável — nesse caso retorne exatamente: []

Para cada fato use uma CHAVE canônica em snake_case que se repita entre turnos (ex.: "nome", "profissao", "objetivo_carreira", "cidade", "neurodivergencia"), para que o fato seja ATUALIZADO e não duplicado.

Retorne um array JSON (sem markdown, sem texto extra):
[{"key": "profissao", "content": "Trabalha como QA.", "importance": 0.8}]

Regras:
- content: frase objetiva, em terceira pessoa.
- importance: número de 0.0 a 1.0 (quão central/duradouro é o fato).
- Se não houver nenhum fato durável, retorne exatamente: []
"""


def extract_semantic_facts(user_message: str, assistant_reply: str) -> list[dict]:
    """Extract durable user-profile facts from a conversation turn.

    Returns a list of {key, content, importance} dicts (possibly empty). Never
    raises — on any failure it returns [] so the caller's persist path is safe.
    """
    api_key = settings.effective_claude_api_key
    if not api_key:
        logger.warning("semantic_extractor: no API key configured, skipping")
        return []

    turn = f"Usuário: {user_message}\nAssistente: {assistant_reply}"
    messages = [
        {"role": "user", "content": SEMANTIC_EXTRACTION_PROMPT + "\n\nTurno:\n" + turn}
    ]

    for attempt in range(settings.max_extraction_retries + 1):
        try:
            content_text = complete_text(messages, max_tokens=1024)
            return _parse_facts(content_text)
        except LLMError as exc:
            logger.warning(
                "semantic_extractor: attempt %d/%d — %s",
                attempt + 1, settings.max_extraction_retries + 1, exc,
            )
            if attempt >= settings.max_extraction_retries:
                return []
        except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.ConnectError) as exc:
            logger.warning(
                "semantic_extractor: attempt %d/%d — HTTP error: %s",
                attempt + 1, settings.max_extraction_retries + 1, exc,
            )
            if attempt >= settings.max_extraction_retries:
                return []
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("semantic_extractor: parse error: %s", exc)
            if attempt >= settings.max_extraction_retries:
                return []

    return []


def _parse_facts(text: str) -> list[dict]:
    """Parse the LLM response into a validated list of fact dicts."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    if text.startswith("json"):
        text = text[4:].strip()
    if not text:
        return []

    data = json.loads(text)
    if not isinstance(data, list):
        return []

    facts: list[dict] = []
    seen_keys: set[str] = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        key = _norm_key(item.get("key"))
        content = item.get("content")
        if not key or not content or not isinstance(content, str):
            continue
        if key in seen_keys:  # de-dupe within a single extraction
            continue
        seen_keys.add(key)
        facts.append({
            "key": key,
            "content": content.strip(),
            "importance": _clamp_importance(item.get("importance")),
        })
    return facts[:10]


# Canonical aliases: LLM sometimes generates these variants — collapse them to the
# canonical key so they upsert the same row instead of creating a parallel fact.
_KEY_ALIASES: dict[str, str] = {
    "profissao_atual":  "profissao",
    "trabalho_atual":   "profissao",
    "emprego_atual":    "profissao",
    "cargo_atual":      "profissao",
    "nome_completo":    "nome",
    "primeiro_nome":    "nome",
    "nome_usuario":     "nome",
}


def _norm_key(value) -> str | None:
    if not isinstance(value, str):
        return None
    key = value.strip().lower().replace(" ", "_")
    key = "".join(c for c in key if c.isalnum() or c == "_")
    key = key[:512] or None
    if key:
        key = _KEY_ALIASES.get(key, key)
    return key


def _clamp_importance(value) -> float:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, n))
