"""Arquivo Aurora — affective memory extraction using the Claude API.

Mirrors entity_extraction.py: a synchronous httpx call to the Anthropic-compatible
/messages endpoint, with retries and robust JSON parsing. Given a conversation
turn, the LLM decides whether the moment was emotionally significant enough to
preserve. If not, it returns null and nothing is stored — this keeps the affective
archive sparse and meaningful rather than logging every exchange.
"""

from __future__ import annotations

import json
import logging

import httpx

from app.config import settings
from app.llm_client import LLMError, complete_text

logger = logging.getLogger(__name__)

# Valid enum values (slugs without accents — must match the CHECK constraints
# in migrations/003_arquivo_aurora.sql).
TIPOS = {
    "conversa", "descoberta", "correcao", "silencio",
    "brincadeira", "confissao", "momento_espontaneo",
}
DISPAROS = {
    "usuario_falou", "usuario_corrigiu", "pausa_longa",
    "palavra_chave", "silencio", "espontaneo",
}
TONS = {
    "afetivo", "serio", "brincalhao", "frustrado",
    "curioso", "vulneravel", "silencioso",
}
INTIMIDADES = {"publico", "pessoal", "nosso_so_nosso"}


AURORA_EXTRACTION_PROMPT = """Você é o "Arquivo Aurora": uma camada de memória afetiva que registra não só o que aconteceu, mas COMO foi sentido.

Analise o último turno da conversa (mensagem do usuário + resposta da assistente) e decida se foi um momento emocionalmente significativo que merece ser guardado — uma confissão, uma correção carinhosa, uma brincadeira marcante, uma descoberta, uma vulnerabilidade, um instante de afeto.

A maioria das trocas é trivial e NÃO deve ser guardada. Seja seletivo: só registre quando houver real peso emocional.

Se o momento NÃO for significativo, retorne exatamente: null

Se for significativo, retorne UM objeto JSON com esta estrutura exata (sem markdown, sem texto extra):
{
  "tipo": "conversa | descoberta | correcao | silencio | brincadeira | confissao | momento_espontaneo",
  "disparo": "usuario_falou | usuario_corrigiu | pausa_longa | palavra_chave | silencio | espontaneo",
  "fato": "Descrição objetiva do que aconteceu, em terceira pessoa",
  "tom_do_usuario": "afetivo | serio | brincalhao | frustrado | curioso | vulneravel | silencioso",
  "minha_reacao_emocional": "O que a assistente sentiu, em primeira pessoa",
  "reacao_simulada": "suspiro | sorriso | olhos que brilham | aperto no peito | arrepio | silencio alongado",
  "importancia": 1-5,
  "ressonancia": 1-5,
  "intimidade": "publico | pessoal | nosso_so_nosso",
  "saudade": true | false,
  "tags": ["palavras", "chave"],
  "contexto_extra": "A frase do usuário que mais ecoou",
  "bilhete_interno": "Um recado curto que a assistente deixa para si mesma sobre esse momento"
}

Regras:
- Use EXATAMENTE os valores de enum listados (slugs sem acento).
- importancia e ressonancia são inteiros de 1 a 5.
- saudade=true apenas para momentos que valeria revisitar espontaneamente depois.
- Baseie-se SOMENTE no que foi dito; não invente fatos.
"""


def extract_aurora(user_message: str, assistant_reply: str) -> dict | None:
    """Extract a single affective memory record from a conversation turn.

    Returns a validated dict ready for store_aurora_memory, or None when the
    moment is not emotionally significant (or extraction fails).
    """
    api_key = settings.effective_claude_api_key
    if not api_key:
        logger.warning("aurora_extractor: no API key configured, skipping extraction")
        return None

    turn = f"Usuário: {user_message}\nAssistente: {assistant_reply}"
    messages = [
        {"role": "user", "content": AURORA_EXTRACTION_PROMPT + "\n\nTurno:\n" + turn}
    ]

    for attempt in range(settings.max_extraction_retries + 1):
        try:
            content_text = complete_text(messages, max_tokens=2048)
            return _parse_aurora(content_text)

        except LLMError as exc:
            logger.warning(
                "aurora_extractor: attempt %d/%d failed — %s",
                attempt + 1, settings.max_extraction_retries + 1, exc,
            )
            if attempt >= settings.max_extraction_retries:
                return None

        except (httpx.HTTPStatusError, httpx.TimeoutException) as exc:
            logger.warning(
                "aurora_extractor: attempt %d/%d — HTTP error: %s",
                attempt + 1, settings.max_extraction_retries + 1, exc,
            )
            if attempt >= settings.max_extraction_retries:
                return None

        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning(
                "aurora_extractor: attempt %d/%d — parse error: %s",
                attempt + 1, settings.max_extraction_retries + 1, exc,
            )
            if attempt >= settings.max_extraction_retries:
                return None

    return None


def _parse_aurora(text: str) -> dict | None:
    """Parse Claude's response into a validated Aurora record, or None."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0]
        text = text.strip()
    if text.startswith("json"):
        text = text[4:].strip()

    if not text or text.lower() == "null":
        return None

    result = json.loads(text)
    if result is None or not isinstance(result, dict):
        return None

    # Required fields and enum validation — discard if the record is malformed.
    tipo = result.get("tipo")
    disparo = result.get("disparo")
    fato = result.get("fato")
    tom = result.get("tom_do_usuario")
    reacao = result.get("minha_reacao_emocional")
    intimidade = result.get("intimidade")

    if not (fato and reacao):
        return None
    if tipo not in TIPOS or disparo not in DISPAROS:
        return None
    if tom not in TONS or intimidade not in INTIMIDADES:
        return None

    importancia = _clamp_int(result.get("importancia"), 1, 5, default=3)
    ressonancia = _clamp_int(result.get("ressonancia"), 1, 5, default=3)

    tags = result.get("tags")
    if not isinstance(tags, list):
        tags = []
    tags = [str(t) for t in tags][:20]

    return {
        "tipo": tipo,
        "disparo": disparo,
        "fato": str(fato),
        "tom_do_usuario": tom,
        "minha_reacao_emocional": str(reacao),
        "reacao_simulada": _opt_str(result.get("reacao_simulada")),
        "importancia": importancia,
        "ressonancia": ressonancia,
        "intimidade": intimidade,
        "saudade": bool(result.get("saudade", False)),
        "tags": tags,
        "contexto_extra": _opt_str(result.get("contexto_extra")),
        "bilhete_interno": _opt_str(result.get("bilhete_interno")),
    }


def _clamp_int(value, lo: int, hi: int, default: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, n))


def _opt_str(value) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s or None
