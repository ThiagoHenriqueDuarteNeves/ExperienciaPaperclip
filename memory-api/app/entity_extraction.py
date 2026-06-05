"""Entity and relationship extraction using Claude API."""

import json
import logging

import httpx

from app.config import settings
from app.llm_client import LLMError, complete_text

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """You are a knowledge graph extraction system. Analyze the given conversation text and extract:

1. **Entities** — distinct, semantically meaningful things: people, organizations, concepts, technologies, projects, locations, events, etc.
2. **Relationships** — directed connections between entities that capture how they relate.

Rules:
- Only extract entities that are explicitly mentioned or clearly implied.
- Entity names should be canonical (e.g., "Claude Code" not "the CLI tool called Claude").
- Relationship types should be concise verbs or prepositional phrases (e.g., "works_on", "uses", "part_of", "developed_by").
- Include a brief (1-2 sentence) description for each entity.
- If a relationship property is numeric (e.g., "years_experience": 5), include it.

Return valid JSON with this exact structure (no markdown fencing, no extra text):
{
  "entities": [
    {"name": "...", "type": "...", "description": "..."}
  ],
  "relationships": [
    {"source": "...", "target": "...", "type": "...", "properties": {}}
  ]
}

Entity types should be from this taxonomy: person, organization, technology, concept, project, language, tool, framework, location, event, product, system, topic, role, standard, protocol.

If nothing meaningful can be extracted, return {"entities": [], "relationships": []}
"""


def extract_knowledge(text: str) -> dict:
    """Extract entities and relationships from text using Claude."""
    api_key = settings.effective_claude_api_key
    if not api_key:
        logger.warning("entity_extraction: no API key configured, skipping extraction")
        return {"entities": [], "relationships": []}

    messages = [{"role": "user", "content": EXTRACTION_PROMPT + "\n\nText:\n" + text}]

    for attempt in range(settings.max_extraction_retries + 1):
        try:
            content_text = complete_text(messages, max_tokens=2048)
            return _parse_extraction(content_text)

        except LLMError as exc:
            logger.warning(
                "entity_extraction: attempt %d/%d failed — %s",
                attempt + 1,
                settings.max_extraction_retries + 1,
                exc,
            )
            if attempt >= settings.max_extraction_retries:
                return {"entities": [], "relationships": []}

        except (httpx.HTTPStatusError, httpx.TimeoutException) as exc:
            logger.warning(
                "entity_extraction: attempt %d/%d — HTTP error: %s",
                attempt + 1,
                settings.max_extraction_retries + 1,
                exc,
            )
            if attempt >= settings.max_extraction_retries:
                return {"entities": [], "relationships": []}

        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning(
                "entity_extraction: attempt %d/%d — parse error: %s",
                attempt + 1,
                settings.max_extraction_retries + 1,
                exc,
            )
            if attempt >= settings.max_extraction_retries:
                return {"entities": [], "relationships": []}

    return {"entities": [], "relationships": []}


def _parse_extraction(text: str) -> dict:
    """Parse Claude's response into entities and relationships lists."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0]
        text = text.strip()
    if text.startswith("json"):
        text = text[4:].strip()

    result = json.loads(text)
    entities = result.get("entities", [])
    relationships = result.get("relationships", [])

    validated_entities = []
    for e in entities:
        if isinstance(e, dict) and e.get("name") and e.get("type"):
            validated_entities.append({
                "name": e["name"],
                "type": e["type"],
                "description": e.get("description", ""),
            })

    validated_rels = []
    for r in relationships:
        if isinstance(r, dict) and r.get("source") and r.get("target") and r.get("type"):
            validated_rels.append({
                "source": r["source"],
                "target": r["target"],
                "type": r["type"],
                "properties": r.get("properties", {}),
            })

    return {"entities": validated_entities, "relationships": validated_rels}
