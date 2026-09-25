"""Task files and deterministic long-context haystacks."""

from __future__ import annotations

import copy
import fnmatch
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.eval.checkers import CHECKERS

TASK_DIR = Path(__file__).with_name("tasks")
CHARS_PER_TOKEN = 4  # English prose with the Qwen tokenizer; sizes are approximate by design.

_VAULTS = [
    "ORION", "LYRA", "CARINA", "VELA", "PYXIS", "DORADO", "FORNAX", "TUCANA", "PAVO", "GRUS",
    "INDUS", "MENSA", "OCTANS", "NORMA", "CIRCINUS", "HOROLOGIUM",
]
_ADJ = ["northern", "old", "blue", "quiet", "eastern", "narrow", "busy", "upper", "lower", "red"]
_NOUN = ["warehouse", "depot", "mill", "workshop", "terminal", "granary", "foundry", "yard"]
_PLACE = ["Harbor District", "Millbrook", "Stonegate", "the river quarter", "Ashford", "Kelso"]
_MATERIAL = ["copper wire", "oak planks", "wool bales", "salt", "glass panes", "iron nails"]
_VERB = ["logged", "shipped", "received", "inspected", "counted", "stored"]


@dataclass(frozen=True)
class Needle:
    key: str
    secret: str
    distractors: tuple[str, ...]  # codes of the other vaults; answering one of them fails
    haystack: str


@dataclass(frozen=True)
class Task:
    id: str
    category: str
    messages: list[dict[str, Any]]
    check: dict[str, Any]
    tools: list[dict[str, Any]] | None = None
    response_format: dict[str, Any] | None = None
    max_tokens: int | None = None
    needle: Needle | None = None


def _code(rng: random.Random) -> str:
    letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    return "".join(rng.choice(letters) for _ in range(3)) + "-" + str(rng.randrange(1000, 10000))


def make_needle(approx_tokens: int, depth: float, seed: int) -> Needle:
    """Filler ledger prose with one planted vault code and several same-shaped distractors."""
    if not 0.0 <= depth <= 1.0:
        raise ValueError("needle depth must be in [0, 1]")
    rng = random.Random(seed)
    names = rng.sample(_VAULTS, 7)
    key, distractors = names[0], names[1:]
    secret = _code(rng)
    target_chars = approx_tokens * CHARS_PER_TOKEN
    sentences: list[str] = []
    length = 0
    while length < target_chars:
        sentence = (
            f"Entry {len(sentences) + 1}: the {rng.choice(_ADJ)} {rng.choice(_NOUN)} in "
            f"{rng.choice(_PLACE)} {rng.choice(_VERB)} {rng.randrange(10, 999)} crates of "
            f"{rng.choice(_MATERIAL)} on day {rng.randrange(1, 366)}."
        )
        sentences.append(sentence)
        length += len(sentence) + 1
    distractor_codes = []
    for name in distractors:
        distractor_codes.append(_code(rng))
        sentences.insert(rng.randrange(len(sentences)),
                         f"The access code for vault {name} is {distractor_codes[-1]}.")
    sentences.insert(round(depth * len(sentences)), f"The access code for vault {key} is {secret}.")
    return Needle(key, secret, tuple(distractor_codes), " ".join(sentences))


def _substitute(value: Any, fields: dict[str, str]) -> Any:
    if isinstance(value, str):
        for name, text in fields.items():
            value = value.replace("{" + name + "}", text)
        return value
    if isinstance(value, list):
        return [_substitute(item, fields) for item in value]
    if isinstance(value, dict):
        return {key: _substitute(item, fields) for key, item in value.items()}
    return value


def _validate_check(spec: dict[str, Any], task_id: str) -> None:
    kind = spec.get("kind")
    if kind not in CHECKERS:
        raise ValueError(f"task {task_id}: unknown checker kind {kind!r}")
    for sub in spec.get("checks", []):
        _validate_check(sub, task_id)
    if "content" in spec:
        _validate_check(spec["content"], task_id)


def parse_task(raw: dict[str, Any]) -> Task:
    raw = copy.deepcopy(raw)
    task_id = raw["id"]
    needle = None
    if "needle" in raw:
        spec = raw["needle"]
        needle = make_needle(int(spec["approx_tokens"]), float(spec["depth"]), int(spec["seed"]))
        raw["messages"] = _substitute(
            raw["messages"], {"haystack": needle.haystack, "needle_key": needle.key}
        )
    _validate_check(raw["check"], task_id)
    return Task(
        id=task_id,
        category=raw["category"],
        messages=raw["messages"],
        check=raw["check"],
        tools=raw.get("tools"),
        response_format=raw.get("response_format"),
        max_tokens=raw.get("max_tokens"),
        needle=needle,
    )


def load_tasks(pattern: str = "*", task_dir: Path = TASK_DIR) -> list[Task]:
    """Load every task file; `pattern` is a glob over task ids or categories."""
    tasks: list[Task] = []
    seen: set[str] = set()
    for path in sorted(task_dir.glob("*.json")):
        for raw in json.loads(path.read_text(encoding="utf-8")):
            if raw["id"] in seen:
                raise ValueError(f"duplicate task id {raw['id']!r} in {path.name}")
            seen.add(raw["id"])
            if fnmatch.fnmatchcase(raw["id"], pattern) or fnmatch.fnmatchcase(
                raw["category"], pattern
            ):
                tasks.append(parse_task(raw))
    return tasks
