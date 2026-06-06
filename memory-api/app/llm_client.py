"""Unified LLM client for the Anthropic-Messages-compatible API.

Single source of truth for talking to the model. Centralizes:
  - provider/base-URL resolution (settings.effective_llm_api_base)
  - auth headers and api-version
  - the "concatenate every text block" handling that reasoning models
    (e.g. deepseek via the anthropic-compat endpoint) require — they emit a
    `thinking` block BEFORE the `text` block, so content[0] is not the answer
  - stop_reason validation (truncation / error)

Two entry points:
  - complete_text(...)  — sync, one-shot, returns the full text. Used by the
    extractors (entity / aurora), which wrap it in their own retry loops.
  - stream_text(...)    — async, yields text chunks as they arrive. Used by the
    chat pipeline, which re-wraps the chunks into its browser-facing SSE frames.

Retries are intentionally NOT handled here: each caller has its own retry/return
policy (extractors return a safe default; the chat endpoint surfaces an error
frame). This module owns the transport + parsing, not the control flow.
"""

from __future__ import annotations

import json
from typing import AsyncIterator, Iterable, Iterator

import httpx

from app.config import settings

_ANTHROPIC_VERSION = "2023-06-01"


class LLMError(Exception):
    """Raised when the LLM returns an unusable response (truncated or error)."""


# ---------------------------------------------------------------------------
# Pure helpers (no I/O — unit-testable in isolation)
# ---------------------------------------------------------------------------


def _text_from_content_blocks(content: list[dict]) -> str:
    """Concatenate every `text` block, skipping `thinking`/other block types."""
    return "".join(
        b.get("text", "")
        for b in content
        if isinstance(b, dict) and b.get("type") == "text"
    )


def _process_sse_line(line: str) -> tuple[str | None, bool]:
    """Parse one upstream SSE line.

    Returns (text_chunk_or_None, should_stop). Mirrors the Anthropic streaming
    format: `data: {...}` frames with `content_block_delta`/`text_delta` carrying
    text, and `message_stop` / `[DONE]` ending the stream.
    """
    if not line.startswith("data: "):
        return None, False
    raw = line[6:]
    if raw == "[DONE]":
        return None, True
    try:
        ev = json.loads(raw)
    except json.JSONDecodeError:
        return None, False
    etype = ev.get("type")
    if etype == "content_block_delta":
        delta = ev.get("delta", {})
        if delta.get("type") == "text_delta":
            return delta.get("text", ""), False
    elif etype == "message_stop":
        return None, True
    return None, False


def iter_sse_text(lines: Iterable[str]) -> Iterator[str]:
    """Yield text chunks from an iterable of upstream SSE lines (sync, for tests)."""
    for line in lines:
        chunk, stop = _process_sse_line(line)
        if chunk:
            yield chunk
        if stop:
            break


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


def _headers() -> dict[str, str]:
    return {
        "x-api-key": settings.effective_claude_api_key,
        "anthropic-version": _ANTHROPIC_VERSION,
        "content-type": "application/json",
    }


def _endpoint() -> str:
    return f"{settings.effective_llm_api_base.rstrip('/')}/messages"


def _validate_stop_reason(data: dict) -> None:
    stop_reason = data.get("stop_reason")
    if stop_reason == "max_tokens":
        raise LLMError("Response truncated (max_tokens reached)")
    if stop_reason == "error" or "error" in data:
        raise LLMError(f"LLM returned an error response: {data.get('error')}")


def complete_text(
    messages: list[dict],
    *,
    max_tokens: int = 2048,
    timeout: float = 30.0,
) -> str:
    """One-shot completion. Returns the concatenated text of the reply.

    Raises LLMError on truncation/error responses; lets httpx exceptions
    (HTTPStatusError, TimeoutException) propagate so callers can handle them.
    """
    resp = httpx.post(
        _endpoint(),
        headers=_headers(),
        json={
            "model": settings.claude_model,
            "max_tokens": max_tokens,
            "messages": messages,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    _validate_stop_reason(data)
    return _text_from_content_blocks(data.get("content", []))


async def stream_text(
    messages: list[dict],
    *,
    system: list[dict] | str | None = None,
    max_tokens: int = 4096,
    timeout: float = 120.0,
) -> AsyncIterator[str]:
    """Stream a completion, yielding text chunks as they arrive.

    The caller is responsible for any downstream framing (e.g. the chat endpoint
    re-wraps these chunks into its own SSE protocol for the browser).
    """
    payload: dict = {
        "model": settings.claude_model,
        "max_tokens": max_tokens,
        "messages": messages,
        "stream": True,
    }
    if system is not None:
        payload["system"] = system

    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "POST", _endpoint(), headers=_headers(), json=payload
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                chunk, stop = _process_sse_line(line)
                if chunk:
                    yield chunk
                if stop:
                    break
