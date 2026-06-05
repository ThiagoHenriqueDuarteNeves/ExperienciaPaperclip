"""Unit tests for app.llm_client — the unified Anthropic-compatible LLM client.

No network: httpx.post is monkeypatched and the SSE/text helpers are pure.

Run with:  pytest -m unit
"""

from __future__ import annotations

import httpx
import pytest

from app import llm_client
from app.config import settings

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# _text_from_content_blocks — reasoning models put a `thinking` block first
# ---------------------------------------------------------------------------


def test_text_blocks_concatenates_only_text():
    content = [
        {"type": "thinking", "text": "internal reasoning that must be ignored"},
        {"type": "text", "text": "Hello"},
        {"type": "text", "text": " world"},
    ]
    assert llm_client._text_from_content_blocks(content) == "Hello world"


def test_text_blocks_empty_when_no_text():
    assert llm_client._text_from_content_blocks([{"type": "thinking", "text": "x"}]) == ""
    assert llm_client._text_from_content_blocks([]) == ""


# ---------------------------------------------------------------------------
# _process_sse_line / iter_sse_text — streaming frame parsing
# ---------------------------------------------------------------------------


def test_process_sse_line_text_delta():
    line = 'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hi"}}'
    chunk, stop = llm_client._process_sse_line(line)
    assert chunk == "hi"
    assert stop is False


def test_process_sse_line_message_stop():
    chunk, stop = llm_client._process_sse_line('data: {"type": "message_stop"}')
    assert chunk is None
    assert stop is True


def test_process_sse_line_done_sentinel():
    assert llm_client._process_sse_line("data: [DONE]") == (None, True)


def test_process_sse_line_ignores_non_data_and_bad_json():
    assert llm_client._process_sse_line("event: ping") == (None, False)
    assert llm_client._process_sse_line("data: {not json") == (None, False)
    # a non-text delta (e.g. input_json_delta) yields nothing
    assert llm_client._process_sse_line(
        'data: {"type": "content_block_delta", "delta": {"type": "input_json_delta"}}'
    ) == (None, False)


def test_iter_sse_text_assembles_full_message_and_stops():
    lines = [
        'data: {"type": "message_start"}',
        'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "The "}}',
        'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "answer"}}',
        'data: {"type": "message_stop"}',
        'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "AFTER STOP"}}',
    ]
    assert "".join(llm_client.iter_sse_text(lines)) == "The answer"


# ---------------------------------------------------------------------------
# complete_text — provider/base/headers + stop_reason handling
# ---------------------------------------------------------------------------


class _FakePost:
    """Records the last httpx.post call and returns a canned httpx.Response."""

    def __init__(self, payload: dict, status: int = 200):
        self.payload = payload
        self.status = status
        self.url = None
        self.headers = None
        self.json_body = None

    def __call__(self, url, headers=None, json=None, timeout=None):
        self.url = url
        self.headers = headers
        self.json_body = json
        return httpx.Response(self.status, json=self.payload, request=httpx.Request("POST", url))


def test_complete_text_happy_path(monkeypatch):
    fake = _FakePost({
        "stop_reason": "end_turn",
        "content": [
            {"type": "thinking", "text": "ignore me"},
            {"type": "text", "text": '{"ok": true}'},
        ],
    })
    monkeypatch.setattr(llm_client.httpx, "post", fake)

    out = llm_client.complete_text([{"role": "user", "content": "hi"}], max_tokens=1234)

    assert out == '{"ok": true}'
    # Hits the configured endpoint with the right model + headers.
    assert fake.url.endswith("/messages")
    assert fake.headers["anthropic-version"] == "2023-06-01"
    assert fake.headers["x-api-key"] == settings.effective_claude_api_key
    assert fake.json_body["model"] == settings.claude_model
    assert fake.json_body["max_tokens"] == 1234
    assert "stream" not in fake.json_body  # complete_text is non-streaming


def test_complete_text_raises_on_max_tokens(monkeypatch):
    fake = _FakePost({"stop_reason": "max_tokens", "content": []})
    monkeypatch.setattr(llm_client.httpx, "post", fake)
    with pytest.raises(llm_client.LLMError, match="truncated"):
        llm_client.complete_text([{"role": "user", "content": "hi"}])


def test_complete_text_raises_on_error_stop_reason(monkeypatch):
    fake = _FakePost({"stop_reason": "error", "error": {"message": "boom"}})
    monkeypatch.setattr(llm_client.httpx, "post", fake)
    with pytest.raises(llm_client.LLMError):
        llm_client.complete_text([{"role": "user", "content": "hi"}])


def test_complete_text_raises_on_error_field_without_stop_reason(monkeypatch):
    fake = _FakePost({"error": {"type": "overloaded_error"}})
    monkeypatch.setattr(llm_client.httpx, "post", fake)
    with pytest.raises(llm_client.LLMError):
        llm_client.complete_text([{"role": "user", "content": "hi"}])


def test_complete_text_propagates_http_status_error(monkeypatch):
    def boom(url, headers=None, json=None, timeout=None):
        return httpx.Response(500, request=httpx.Request("POST", url))

    monkeypatch.setattr(llm_client.httpx, "post", boom)
    with pytest.raises(httpx.HTTPStatusError):
        llm_client.complete_text([{"role": "user", "content": "hi"}])
