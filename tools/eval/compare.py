"""Side-by-side Markdown comparison of result files."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Aggregate:
    passed: int = 0
    attempts: int = 0
    latency: list[float] = field(default_factory=list)
    tokens: list[float] = field(default_factory=list)
    speed: list[float] = field(default_factory=list)

    def add(self, entry: dict[str, Any]) -> None:
        self.attempts += 1
        self.passed += bool(entry["passed"])
        if entry.get("error") is None:
            self.latency.append(float(entry["latency_s"]))
        tokens = (entry.get("usage") or {}).get("completion_tokens")
        if isinstance(tokens, (int, float)):
            self.tokens.append(float(tokens))
        speed = (entry.get("timings") or {}).get("predicted_per_second")
        if isinstance(speed, (int, float)) and speed > 0:
            self.speed.append(float(speed))

    @property
    def rate(self) -> float:
        return self.passed / self.attempts if self.attempts else 0.0


def _mean(values: list[float], digits: int) -> str:
    return f"{sum(values) / len(values):.{digits}f}" if values else "-"


def _aggregate(results: dict[str, Any], key: str | None) -> dict[str, Aggregate]:
    groups: dict[str, Aggregate] = {}
    for entry in results["results"]:
        groups.setdefault(entry[key] if key else "all", Aggregate()).add(entry)
    return groups


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


def _labels(runs: list[dict[str, Any]]) -> list[str]:
    labels = [run["meta"]["label"] for run in runs]
    if len(set(labels)) != len(labels):
        labels = [f"{label}#{index + 1}" for index, label in enumerate(labels)]
    return labels


def compare(runs: list[dict[str, Any]]) -> str:
    labels = _labels(runs)
    lines = ["# NInfer eval comparison", "", "## Overall", ""]
    rows = []
    for label, run in zip(labels, runs):
        meta, total = run["meta"], _aggregate(run, None).get("all", Aggregate())
        rows.append([
            label, meta["model"], meta["thinking"], f"{total.passed}/{total.attempts}",
            f"{100 * total.rate:.1f}%", _mean(total.latency, 1), _mean(total.tokens, 0),
            _mean(total.speed, 1),
        ])
    lines += _table(["run", "model", "thinking", "passed", "pass rate", "mean latency s",
                     "mean out tokens", "mean tok/s"], rows)

    by_category = [_aggregate(run, "category") for run in runs]
    categories = sorted({c for groups in by_category for c in groups})
    header = ["category"]
    for metric in ("pass", "latency s", "out tok", "tok/s"):
        header += [f"{label} {metric}" for label in labels]
    rows = []
    for category in categories:
        cells = [groups.get(category, Aggregate()) for groups in by_category]
        row = [category]
        row += [f"{a.passed}/{a.attempts}" for a in cells]
        row += [_mean(a.latency, 1) for a in cells]
        row += [_mean(a.tokens, 0) for a in cells]
        row += [_mean(a.speed, 1) for a in cells]
        rows.append(row)
    lines += ["", "## By category", ""] + _table(header, rows)

    by_task = [_aggregate(run, "id") for run in runs]
    category_of = {e["id"]: e["category"] for run in runs for e in run["results"]}
    header = ["task", "category"]
    header += [f"{label} pass" for label in labels] + [f"{label} s" for label in labels]
    header += [f"{label} tok" for label in labels] + ["differs"]
    rows = []
    for task_id in sorted(category_of, key=lambda t: (category_of[t], t)):
        cells = [groups.get(task_id, Aggregate()) for groups in by_task]
        row = [task_id, category_of[task_id]]
        row += [f"{a.passed}/{a.attempts}" if a.attempts else "-" for a in cells]
        row += [_mean(a.latency, 1) for a in cells]
        row += [_mean(a.tokens, 0) for a in cells]
        row.append("yes" if len({round(a.rate, 6) for a in cells}) > 1 else "")
        rows.append(row)
    lines += ["", "## By task", ""] + _table(header, rows)

    failures = []
    for label, run in zip(labels, runs):
        for entry in run["results"]:
            if not entry["passed"]:
                detail = " ".join(str(entry["detail"]).split()).replace("|", "\\|")
                failures.append(f"- **{label}** `{entry['id']}` #{entry['attempt']}: {detail[:240]}")
    if failures:
        lines += ["", "## Failure details", ""] + failures
    return "\n".join(lines) + "\n"
