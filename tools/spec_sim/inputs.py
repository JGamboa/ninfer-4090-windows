"""Load simulation records from JSONL token ids, text, chat transcripts, or Claude Code sessions.

JSONL record forms (one JSON value per line; blank lines are ignored):
  [1, 2, 3]                                    whole sequence is generated, empty prompt
  {"prompt_ids": [...], "completion_ids": [...]}
  {"prompt": "text", "completion": "text"}     needs a tokenizer
  {"text": "text"}                             whole text is generated
  {"messages": [{"role": ..., "content": ...}, ...]}
      one record per assistant message: its prompt is every earlier message rendered in the
      Qwen ChatML layout plus the assistant header; its completion is the assistant body and
      <|im_end|>. Messages are tokenized one at a time, so each turn's prompt extends the
      previous turn's full sequence exactly as a prefix-reusing server sees it.

Claude Code session files (`~/.claude/projects/<project>/<session>.jsonl`) are converted to one
`messages` transcript per file with `--claude-code`: text blocks stay text, tool_use becomes a
Qwen <tool_call> JSON object and tool_result a <tool_response> user turn; thinking is dropped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Protocol, Sequence

from .simulate import Record


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...


class HfTokenizer:
    """`tokenizers` wrapper; special tokens in text (<|im_start|>, ...) map to their ids."""

    def __init__(self, tokenizer_json: str) -> None:
        try:
            from tokenizers import Tokenizer as _Tokenizer
        except ImportError as error:  # pragma: no cover - depends on the environment
            raise SystemExit(
                "text input needs the `tokenizers` package (pip install tokenizers); "
                "token-id input does not") from error
        self._tokenizer = _Tokenizer.from_str(tokenizer_json)

    def encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=False).ids)


def tokenizer_json_from_artifact(path: Path) -> str:
    """Read the tokenizer.json resource embedded in a v3 `.ninfer` artifact (no GPU, no weights
    are read)."""
    from tools.artifact.reader import Artifact

    with Artifact(path) as artifact:
        text = artifact.directory.components.get("text", {})
        object_id = text.get("resources", {}).get("tokenizer.json")
        if object_id is None:
            raise SystemExit(f"{path}: artifact has no text/tokenizer.json resource")
        return artifact.read_object(object_id).decode("utf-8")


def load_tokenizer(tokenizer_json: Path | None, artifact: Path | None) -> Tokenizer | None:
    if tokenizer_json is not None:
        return HfTokenizer(tokenizer_json.read_text(encoding="utf-8"))
    if artifact is not None:
        return HfTokenizer(tokenizer_json_from_artifact(artifact))
    return None


def _ids(value: Any, where: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not all(type(t) is int and t >= 0 for t in value):
        raise ValueError(f"{where}: expected a list of non-negative token ids")
    return tuple(value)


def _need(tokenizer: Tokenizer | None, where: str) -> Tokenizer:
    if tokenizer is None:
        raise ValueError(f"{where}: text input needs --tokenizer-json or --artifact")
    return tokenizer


def content_text(content: Any) -> str:
    """OpenAI/Anthropic content: a string or a list of parts; only text parts are kept."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "".join(parts)
    return json.dumps(content, ensure_ascii=False)


def _tool_call_text(name: str, arguments: Any) -> str:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            pass
    call = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
    return f"<tool_call>\n{call}\n</tool_call>"


def message_body(message: dict) -> tuple[str, str]:
    """Return the ChatML role and rendered body of one OpenAI-style message."""
    role = message.get("role", "user")
    body = content_text(message.get("content"))
    if role == "assistant":
        calls = [_tool_call_text(call.get("function", {}).get("name", ""),
                                 call.get("function", {}).get("arguments", {}))
                 for call in message.get("tool_calls") or []]
        body = "\n".join([body] + calls if body else calls)
        return "assistant", body
    if role == "tool":
        return "user", f"<tool_response>\n{body}\n</tool_response>"
    return role, body


def records_from_messages(messages: Sequence[dict], tokenizer: Tokenizer,
                          label: str = "") -> list[Record]:
    history: list[int] = []
    records: list[Record] = []
    end = tokenizer.encode("<|im_end|>")
    newline = tokenizer.encode("\n")
    for index, message in enumerate(messages):
        role, body = message_body(message)
        header = tokenizer.encode(f"<|im_start|>{role}\n")
        body_ids = tokenizer.encode(body)
        if role == "assistant" and body_ids:
            prompt = tuple(history + header)
            completion = tuple(body_ids + end)
            records.append(Record(prompt, completion, f"{label}#{index}"))
        history.extend(header + body_ids + end + newline)
    return records


def records_from_value(value: Any, tokenizer: Tokenizer | None, where: str) -> list[Record]:
    if isinstance(value, list):
        return [Record((), _ids(value, where), where)]
    if not isinstance(value, dict):
        raise ValueError(f"{where}: expected a token list or an object")
    if "completion_ids" in value:
        return [Record(_ids(value.get("prompt_ids", []), where),
                       _ids(value["completion_ids"], where), where)]
    if "completion" in value:
        encoder = _need(tokenizer, where)
        prompt = value.get("prompt", "")
        return [Record(tuple(encoder.encode(prompt)) if prompt else (),
                       tuple(encoder.encode(value["completion"])), where)]
    if "text" in value:
        return [Record((), tuple(_need(tokenizer, where).encode(value["text"])), where)]
    if "messages" in value:
        return records_from_messages(value["messages"], _need(tokenizer, where), where)
    raise ValueError(f"{where}: unrecognized record keys {sorted(value)}")


def read_jsonl(path: Path, tokenizer: Tokenizer | None) -> Iterator[Record]:
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                yield from records_from_value(json.loads(line), tokenizer, f"{path}:{number}")


def claude_code_messages(path: Path) -> list[dict]:
    """Convert one Claude Code session transcript into OpenAI-style messages."""
    messages: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("type") not in ("user", "assistant") or event.get("isSidechain"):
                continue
            message = event.get("message") or {}
            role = message.get("role", event["type"])
            content = message.get("content")
            if isinstance(content, str):
                messages.append({"role": role, "content": content})
                continue
            texts: list[str] = []
            calls: list[dict] = []
            for block in content or []:
                kind = block.get("type") if isinstance(block, dict) else None
                if kind == "text":
                    texts.append(block.get("text", ""))
                elif kind == "tool_use":
                    calls.append({"function": {"name": block.get("name", ""),
                                               "arguments": block.get("input", {})}})
                elif kind == "tool_result":
                    messages.append({"role": "tool",
                                     "content": content_text(block.get("content"))})
            if texts or calls:
                entry: dict = {"role": role, "content": "".join(texts)}
                if calls:
                    entry["tool_calls"] = calls
                messages.append(entry)
    # One API turn is often split across several transcript events; merge adjacent assistant
    # events so each simulated completion is one model response.
    merged: list[dict] = []
    for message in messages:
        if merged and message["role"] == "assistant" == merged[-1]["role"]:
            previous = merged[-1]
            previous["content"] = "\n".join(p for p in (previous["content"],
                                                         message["content"]) if p)
            previous["tool_calls"] = (previous.get("tool_calls") or []) + \
                (message.get("tool_calls") or [])
        else:
            merged.append(dict(message))
    return merged


def load_records(paths: Sequence[Path], tokenizer: Tokenizer | None,
                 claude_code: bool = False) -> list[Record]:
    records: list[Record] = []
    for path in paths:
        if claude_code:
            records.extend(records_from_messages(claude_code_messages(path),
                                                 _need(tokenizer, str(path)), str(path)))
        else:
            records.extend(read_jsonl(path, tokenizer))
    return records
