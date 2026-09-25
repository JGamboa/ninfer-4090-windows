"""Optional Opik trace logging; the suite never depends on it."""

from __future__ import annotations

from typing import Any, Callable

from tools.eval.tasks import Task


def _shorten(message: dict[str, Any], limit: int = 2000) -> dict[str, Any]:
    content = message.get("content")
    if isinstance(content, str) and len(content) > limit:  # long-context haystacks
        return {**message, "content": content[:limit] + f"...[+{len(content) - limit} chars]"}
    return message


def opik_recorder(project: str, label: str) -> tuple[Callable[[Task, dict[str, Any]], None],
                                                     Callable[[], None]]:
    """Return `(on_record, flush)`; raises RuntimeError when the SDK is unavailable."""
    try:
        import opik  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("--opik-project requires the `opik` package") from error
    client = opik.Opik(project_name=project)

    def on_record(task: Task, entry: dict[str, Any]) -> None:
        try:
            client.trace(
                name=task.id,
                input={"messages": [_shorten(m) for m in task.messages]},
                output={"content": entry["content"], "tool_calls": entry["tool_calls"]},
                metadata={key: entry[key] for key in
                          ("category", "attempt", "detail", "latency_s", "usage", "timings")},
                tags=[label, task.category],
                feedback_scores=[{"name": "passed", "value": float(entry["passed"]),
                                  "reason": str(entry["detail"])[:500]}],
            )
        except Exception as error:  # tracing is advisory; keep evaluating
            print(f"opik: failed to log {task.id}: {error}")

    return on_record, client.flush
