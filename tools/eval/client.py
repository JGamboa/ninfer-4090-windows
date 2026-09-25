"""Non-streaming Chat Completions exchange normalized for checkers."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from tools.ninfer_serve.client import NInferServeClient, ProtocolRequest
from tools.streaming_http.client import HttpClientError

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: Any  # parsed JSON object, or the raw string when it is not valid JSON
    arguments_valid: bool


@dataclass
class ChatResult:
    content: str = ""
    reasoning: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, Any] = field(default_factory=dict)
    latency_s: float = 0.0
    error: str | None = None


def strip_think(content: str) -> str:
    """Remove reasoning that leaked into answer content."""
    content = _THINK_BLOCK.sub("", content)
    # A template-opened think block leaks only its closing marker.
    if "</think>" in content:
        content = content.rsplit("</think>", 1)[1]
    return content.replace("<think>", "").strip()


def split_base_url(base_url: str) -> str:
    """Accept both `http://host:port` and the OpenAI-style `http://host:port/v1`."""
    trimmed = base_url.rstrip("/")
    return trimmed[: -len("/v1")] if trimmed.endswith("/v1") else trimmed


def parse_completion(body: dict[str, Any]) -> ChatResult:
    result = ChatResult(usage=body.get("usage") or {}, timings=body.get("timings") or {})
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        result.error = "response has no choices"
        return result
    choice = choices[0]
    result.finish_reason = choice.get("finish_reason")
    message = choice.get("message") or {}
    content = message.get("content")
    result.content = strip_think(content) if isinstance(content, str) else ""
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    result.reasoning = reasoning if isinstance(reasoning, str) else ""
    for call in message.get("tool_calls") or []:
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict):
            continue
        raw = function.get("arguments")
        if isinstance(raw, dict):
            arguments, valid = raw, True
        else:
            try:
                arguments, valid = json.loads(raw or "{}"), True
            except (TypeError, json.JSONDecodeError):
                arguments, valid = raw, False
        result.tool_calls.append(ToolCall(str(function.get("name", "")), arguments, valid))
    return result


class ChatClient:
    def __init__(self, base_url: str, timeout_s: float, api_key: str | None = None) -> None:
        self._client = NInferServeClient(split_base_url(base_url), timeout_s, api_key)

    def discover_model(self) -> str:
        return self._client.discover_model()

    def complete(self, payload: dict[str, Any]) -> ChatResult:
        request = ProtocolRequest("openai_chat", "/v1/chat/completions", payload, stream=False)
        started = time.perf_counter()
        try:
            exchange = self._client.prepare(request).execute()
        except (OSError, HttpClientError) as error:
            return ChatResult(latency_s=time.perf_counter() - started,
                              error=f"{type(error).__name__}: {error}")
        latency = time.perf_counter() - started
        http = exchange.http
        if http.error is not None:
            return ChatResult(latency_s=latency, error=http.error)
        if http.status != 200:
            code, message = exchange.error_code, exchange.error_message
            return ChatResult(latency_s=latency,
                              error=f"HTTP {http.status} {code or ''}: {message or ''}".strip())
        try:
            body = json.loads(http.body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return ChatResult(latency_s=latency, error="response body is not JSON")
        if not isinstance(body, dict):
            return ChatResult(latency_s=latency, error="response body is not a JSON object")
        result = parse_completion(body)
        result.latency_s = latency
        return result
