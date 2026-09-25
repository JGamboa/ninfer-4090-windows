"""Execute tasks against one server and record per-attempt results."""

from __future__ import annotations

import datetime as dt
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from tools.eval.checkers import run_check
from tools.eval.client import ChatClient, ChatResult
from tools.eval.tasks import Task

RESULTS_SCHEMA = "ninfer_eval_results"
RESULTS_VERSION = 1
CONTENT_LIMIT = 4000
REASONING_LIMIT = 2000


@dataclass(frozen=True)
class RunConfig:
    model: str
    label: str
    base_url: str
    max_tokens: int = 4096
    thinking: str | None = None  # "on", "off", or None for the server default
    temperature: float = 0.0
    seed: int = 1234
    repeats: int = 1
    timeout_s: float = 600.0


def build_payload(task: Task, config: RunConfig, attempt: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": config.model,
        "messages": task.messages,
        "max_completion_tokens": task.max_tokens or config.max_tokens,
        "temperature": config.temperature,
        "seed": config.seed + attempt,
        "stream": False,
    }
    if task.tools:
        payload["tools"] = task.tools
        payload["tool_choice"] = "auto"
    if task.response_format is not None:
        payload["response_format"] = task.response_format
    if config.thinking is not None:
        payload["chat_template_kwargs"] = {"enable_thinking": config.thinking == "on"}
    return payload


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + f"...[+{len(text) - limit} chars]"


def evaluate(task: Task, result: ChatResult) -> tuple[bool, str]:
    if result.error is not None:
        return False, f"request failed: {result.error}"
    try:
        passed, detail = run_check(task.check, result, task)
    except Exception as error:  # a checker bug must not abort the remaining tasks
        return False, f"checker error: {type(error).__name__}: {error}"
    if not passed and result.finish_reason == "length":
        detail += " [output truncated at the token limit]"
    return passed, detail


def record(task: Task, attempt: int, result: ChatResult, passed: bool, detail: str) -> dict[str, Any]:
    return {
        "id": task.id,
        "category": task.category,
        "attempt": attempt,
        "passed": passed,
        "detail": detail,
        "error": result.error,
        "finish_reason": result.finish_reason,
        "content": _truncate(result.content, CONTENT_LIMIT),
        "reasoning": _truncate(result.reasoning, REASONING_LIMIT),
        "reasoning_chars": len(result.reasoning),
        "tool_calls": [
            {"name": call.name, "arguments": call.arguments} for call in result.tool_calls
        ],
        "latency_s": round(result.latency_s, 4),
        "usage": result.usage,
        "timings": result.timings,
    }


def run_suite(
    client: ChatClient,
    tasks: list[Task],
    config: RunConfig,
    *,
    on_record: Callable[[Task, dict[str, Any]], None] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    started = dt.datetime.now(dt.timezone.utc)
    records: list[dict[str, Any]] = []
    total = len(tasks) * config.repeats
    for attempt in range(config.repeats):
        for task in tasks:
            result = client.complete(build_payload(task, config, attempt))
            passed, detail = evaluate(task, result)
            entry = record(task, attempt, result, passed, detail)
            records.append(entry)
            if on_record is not None:
                on_record(task, entry)
            tps = result.timings.get("predicted_per_second")
            speed = f" {tps:.1f} tok/s" if isinstance(tps, (int, float)) else ""
            log(
                f"[{len(records)}/{total}] {'PASS' if passed else 'FAIL'} {task.id}"
                f" {result.latency_s:.1f}s{speed}" + ("" if passed else f" - {detail}")
            )
    return {
        "schema": RESULTS_SCHEMA,
        "version": RESULTS_VERSION,
        "meta": {
            "label": config.label,
            "model": config.model,
            "base_url": config.base_url,
            "thinking": config.thinking or "server-default",
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "seed": config.seed,
            "repeats": config.repeats,
            "task_count": len(tasks),
            "started_utc": started.isoformat(timespec="seconds"),
            "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "client": f"python {sys.version.split()[0]} on {platform.system()}",
        },
        "results": records,
    }


def write_results(results: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def load_results(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != RESULTS_SCHEMA or value.get("version") != RESULTS_VERSION:
        raise ValueError(f"{path} is not a {RESULTS_SCHEMA} v{RESULTS_VERSION} file")
    return value
