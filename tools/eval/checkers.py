"""Deterministic answer checkers.

Every checker receives its JSON spec, the normalized chat result and the task, and returns
`(passed, detail)`. The detail explains a failure well enough to judge whether the model or the
task is at fault without rerunning it.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import tempfile
import unicodedata
from typing import TYPE_CHECKING, Any, Callable

from tools.eval.client import ChatResult

if TYPE_CHECKING:
    from tools.eval.tasks import Task

Outcome = tuple[bool, str]

_FENCE = re.compile(r"```([A-Za-z0-9_+-]*)[ \t]*\n(.*?)```", re.DOTALL)
_WORD = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*")
_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:[.,]\d+)*(?:[eE][-+]?\d+)?")
_ANSWER_MARKER = re.compile(r"(?:answer|respuesta|resultado|result)\s*[:=]", re.IGNORECASE)
_BULLET = re.compile(r"^\s*(?:[-*•+]|\d+[.)])\s+\S")


def normalize(text: str, options: list[str]) -> str:
    if "accents" in options:
        text = "".join(
            ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)
        )
    if "case" in options:
        text = text.casefold()
    if "punct" in options:
        text = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in text)
    if "whitespace" in options:
        text = " ".join(text.split())
    return text


def _norm_options(spec: dict[str, Any]) -> list[str]:
    return list(spec.get("normalize", ["case", "whitespace"]))


def _snippet(text: str, limit: int = 160) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "..."


def _find(needle: str, haystack: str, words: bool) -> bool:
    if not words:
        return needle in haystack
    return re.search(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", haystack) is not None


# ---- text ---------------------------------------------------------------------------------------


def check_exact(spec: dict[str, Any], result: ChatResult, task: Task) -> Outcome:
    options = _norm_options(spec)
    got = normalize(result.content, options)
    expected = [normalize(v, options) for v in _as_list(spec["value"])]
    if got in expected:
        return True, "exact match"
    return False, f"expected {expected!r}, got {_snippet(got)!r}"


def check_contains(spec: dict[str, Any], result: ChatResult, task: Task) -> Outcome:
    options = _norm_options(spec)
    words = bool(spec.get("words", False))
    text = normalize(result.content, options)
    # Each entry is a required term; a list entry is a set of accepted alternatives.
    groups = [[normalize(v, options) for v in _as_list(entry)] for entry in spec["values"]]
    hits = [any(_find(v, text, words) for v in group) for group in groups]
    mode = spec.get("mode", "all")
    passed = all(hits) if mode == "all" else any(hits)
    if passed:
        return True, f"found {sum(hits)}/{len(groups)} terms"
    missing = [group[0] if len(group) == 1 else group for group, hit in zip(groups, hits) if not hit]
    return False, f"missing {missing!r} in {_snippet(text)!r}"


def check_not_contains(spec: dict[str, Any], result: ChatResult, task: Task) -> Outcome:
    options = _norm_options(spec)
    words = bool(spec.get("words", True))
    text = normalize(result.content, options)
    found = [v for v in spec["values"] if _find(normalize(v, options), text, words)]
    if found:
        return False, f"forbidden terms present: {found!r}"
    return True, "no forbidden terms"


def check_regex(spec: dict[str, Any], result: ChatResult, task: Task) -> Outcome:
    options = list(spec.get("normalize", []))
    text = normalize(result.content, options)
    flags = 0
    for letter in spec.get("flags", ""):
        flags |= {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL}[letter]
    matcher = re.fullmatch if spec.get("fullmatch", False) else re.search
    if matcher(spec["pattern"], text, flags):
        return True, "regex matched"
    return False, f"/{spec['pattern']}/ did not match {_snippet(text)!r}"


def check_word_count(spec: dict[str, Any], result: ChatResult, task: Task) -> Outcome:
    count = len(_WORD.findall(result.content))
    low, high = spec.get("min", 0), spec.get("max", math.inf)
    if low <= count <= high:
        return True, f"{count} words"
    return False, f"{count} words, expected [{low}, {high}]"


def check_bullets(spec: dict[str, Any], result: ChatResult, task: Task) -> Outcome:
    lines = [line for line in result.content.splitlines() if line.strip()]
    bullets = [line for line in lines if _BULLET.match(line)]
    low, high = spec.get("min", 0), spec.get("max", math.inf)
    if not low <= len(bullets) <= high:
        return False, f"{len(bullets)} bullet lines, expected [{low}, {high}]"
    if spec.get("only", False) and len(bullets) != len(lines):
        return False, f"{len(lines) - len(bullets)} non-bullet lines present"
    return True, f"{len(bullets)} bullet lines"


_ES_WORDS = frozenset(
    "el la los las de del que y en un una unos unas es son por para con no se su sus al lo como "
    "más pero le ya o este esta estos estas porque entre cuando muy sin sobre también me hasta hay "
    "donde desde todo todos nos durante ni otro otra ese eso esto está están ser tiene puede "
    "cada mi tu usted nuestro".split()
)
_EN_WORDS = frozenset(
    "the and of to is in that it for was on are with as be this by at from have an or not but "
    "you which they their we can will has would there what when your its were been these".split()
)


def check_language(spec: dict[str, Any], result: ChatResult, task: Task) -> Outcome:
    if spec.get("lang", "es") != "es":
        raise ValueError("language checker supports only 'es'")
    words = [w.casefold() for w in _WORD.findall(result.content)]
    es = sum(w in _ES_WORDS for w in words)
    en = sum(w in _EN_WORDS for w in words)
    min_hits = spec.get("min_hits", 3)
    if es >= min_hits and en <= max(1, es // 10):
        return True, f"spanish function words {es}, english {en}"
    return False, f"not Spanish-only: spanish function words {es}, english {en}"


# ---- numbers ------------------------------------------------------------------------------------


def parse_number(token: str) -> float:
    """Parse one numeric token written with either English or Spanish separators."""
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", token):
        token = token.replace(",", "")
    elif re.fullmatch(r"-?\d{1,3}(\.\d{3})+(,\d+)?", token):
        token = token.replace(".", "").replace(",", ".")
    elif token.count(",") == 1 and "." not in token:
        token = token.replace(",", ".")
    return float(token)


def final_number(text: str) -> float | None:
    """The number after the last `Answer:` marker (or `\\boxed{}`), else the last number."""
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        text = boxed[-1]
    else:
        markers = list(_ANSWER_MARKER.finditer(text))
        if markers:
            tail = text[markers[-1].end():]
            first = _NUMBER.search(tail.replace("$", "").replace("€", ""))
            if first:
                return parse_number(first.group(0))
    tokens = _NUMBER.findall(text.replace("$", "").replace("€", ""))
    return parse_number(tokens[-1]) if tokens else None


def check_number(spec: dict[str, Any], result: ChatResult, task: Task) -> Outcome:
    got = final_number(result.content)
    if got is None:
        return False, f"no number in {_snippet(result.content)!r}"
    expected = float(spec["value"])
    tolerance = max(spec.get("tol", 1e-6), abs(expected) * spec.get("rel_tol", 0.0))
    if abs(got - expected) <= tolerance:
        return True, f"{got:g} == {expected:g}"
    return False, f"got {got:g}, expected {expected:g} (tol {tolerance:g})"


# ---- JSON ---------------------------------------------------------------------------------------


def extract_json(text: str) -> Any:
    """Decode a fenced JSON block, the whole text, or the first decodable object/array."""
    fenced = [body for lang, body in _FENCE.findall(text) if lang.lower() in {"", "json"}]
    for candidate in fenced + [text]:
        try:
            return json.loads(candidate.strip())
        except json.JSONDecodeError:
            pass
    decoder = json.JSONDecoder()
    for index, ch in enumerate(text):
        if ch in "{[":
            try:
                return decoder.raw_decode(text, index)[0]
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON value found")


_TYPES: dict[str, Callable[[Any], bool]] = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool)
    or isinstance(v, float) and v.is_integer(),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


def validate_schema(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Validate the supported subset: type, properties, required, additionalProperties(false),
    enum, const, items, minItems/maxItems, minimum/maximum, minLength/maxLength, pattern."""
    errors: list[str] = []
    expected_type = schema.get("type")
    if expected_type is not None:
        types = _as_list(expected_type)
        if not any(_TYPES[t](value) for t in types):
            return [f"{path}: expected {'|'.join(types)}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} not in {schema['enum']!r}")
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: {value!r} != {schema['const']!r}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: {value} < minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: {value} > maximum {schema['maximum']}")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path}: shorter than {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: longer than {schema['maxLength']}")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errors.append(f"{path}: {value!r} does not match /{schema['pattern']}/")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing required {key!r}")
        for key, item in value.items():
            if key in properties:
                errors.extend(validate_schema(item, properties[key], f"{path}.{key}"))
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}: unexpected property {key!r}")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: more than {schema['maxItems']} items")
        if "items" in schema:
            for index, item in enumerate(value):
                errors.extend(validate_schema(item, schema["items"], f"{path}[{index}]"))
    return errors


def json_pointer(value: Any, pointer: str) -> Any:
    for part in [p for p in pointer.split("/") if p]:
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def check_json_schema(spec: dict[str, Any], result: ChatResult, task: Task) -> Outcome:
    try:
        value = extract_json(result.content)
    except ValueError:
        return False, f"no JSON in {_snippet(result.content)!r}"
    errors = validate_schema(value, spec["schema"])
    for pointer, predicate in spec.get("values", {}).items():
        try:
            actual = json_pointer(value, pointer)
        except (KeyError, IndexError, TypeError, ValueError):
            errors.append(f"{pointer}: missing")
            continue
        ok, why = match_value(actual, predicate)
        if not ok:
            errors.append(f"{pointer}: {why}")
    if errors:
        return False, "; ".join(errors[:6])
    return True, "valid JSON matching schema"


# ---- values and tool calls ----------------------------------------------------------------------


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def match_value(actual: Any, predicate: Any) -> Outcome:
    """Match a JSON value against a literal (lenient) or a predicate object.

    Literal strings compare case- and whitespace-insensitively, literal numbers numerically
    (a numeric string is accepted), other literals by equality. Predicate objects use one of:
    eq, ieq, contains, regex, in, num (+tol), min/max, absent.
    """
    if isinstance(predicate, dict) and predicate.keys() & _PREDICATES:
        for key, expected in predicate.items():
            if key == "tol":
                continue
            ok = _PREDICATES_FN[key](actual, expected, predicate)
            if not ok:
                return False, f"{actual!r} fails {key}={expected!r}"
        return True, "ok"
    if isinstance(predicate, str):
        ok = isinstance(actual, str) and normalize(actual, ["case", "whitespace"]) == normalize(
            predicate, ["case", "whitespace"]
        )
    elif isinstance(predicate, (int, float)) and not isinstance(predicate, bool):
        number = _as_number(actual)
        ok = number is not None and abs(number - predicate) <= 1e-9
    else:
        ok = actual == predicate
    return (True, "ok") if ok else (False, f"{actual!r} != {predicate!r}")


_PREDICATES_FN: dict[str, Callable[[Any, Any, dict[str, Any]], bool]] = {
    "eq": lambda a, e, p: a == e,
    "ieq": lambda a, e, p: isinstance(a, str) and a.strip().casefold() == e.strip().casefold(),
    "contains": lambda a, e, p: isinstance(a, str) and e.casefold() in a.casefold(),
    "regex": lambda a, e, p: isinstance(a, str) and re.search(e, a) is not None,
    "in": lambda a, e, p: any(match_value(a, option)[0] for option in e),
    "num": lambda a, e, p: (n := _as_number(a)) is not None and abs(n - e) <= p.get("tol", 1e-9),
    "min": lambda a, e, p: (n := _as_number(a)) is not None and n >= e,
    "max": lambda a, e, p: (n := _as_number(a)) is not None and n <= e,
    "absent": lambda a, e, p: (a is _MISSING) == bool(e),
}
_PREDICATES = frozenset(_PREDICATES_FN)
_MISSING = object()


def _match_call(call: Any, expected: dict[str, Any]) -> Outcome:
    if call.name != expected["name"]:
        return False, f"name {call.name!r} != {expected['name']!r}"
    if not call.arguments_valid or not isinstance(call.arguments, dict):
        return False, f"arguments are not a JSON object: {call.arguments!r}"
    for key, predicate in expected.get("arguments", {}).items():
        actual = call.arguments.get(key, _MISSING)
        if actual is _MISSING and not (isinstance(predicate, dict) and "absent" in predicate):
            return False, f"missing argument {key!r}"
        ok, why = match_value(actual, predicate)
        if not ok:
            return False, f"argument {key!r}: {why}"
    if expected.get("exact_keys", False):
        extra = set(call.arguments) - set(expected.get("arguments", {}))
        if extra:
            return False, f"unexpected arguments {sorted(extra)!r}"
    return True, "ok"


def check_tool_call(spec: dict[str, Any], result: ChatResult, task: Task) -> Outcome:
    calls = result.tool_calls
    summary = [f"{c.name}({json.dumps(c.arguments, ensure_ascii=False)})" for c in calls]
    if spec.get("expect") == "none":
        if calls:
            return False, f"unexpected tool calls: {summary}"
        if "content" in spec:
            return run_check(spec["content"], result, task)
        return True, "no tool call"
    expected_calls = spec.get("calls") or [spec]
    if not calls:
        return False, f"no tool call; content {_snippet(result.content)!r}"
    remaining = list(calls)
    for expected in expected_calls:
        reasons = []
        for call in remaining:
            ok, why = _match_call(call, expected)
            if ok:
                remaining.remove(call)
                break
            reasons.append(why)
        else:
            return False, f"no call matches {expected['name']}: {'; '.join(reasons)}"
    if remaining and spec.get("exclusive", True):
        return False, f"extra tool calls: {summary}"
    return True, f"calls {summary}"


# ---- code execution -----------------------------------------------------------------------------


def extract_code(text: str) -> str | None:
    blocks = _FENCE.findall(text)
    python = [body for lang, body in blocks if lang.lower() in {"python", "py", "python3"}]
    chosen = python or [body for lang, body in blocks if not lang]
    return chosen[-1] if chosen else None


def run_python(code: str, test: str, timeout_s: float) -> Outcome:
    """Run solution + tests in a fresh isolated interpreter; pass iff exit status is zero."""
    with tempfile.TemporaryDirectory(prefix="ninfer-eval-") as workdir:
        script = os.path.join(workdir, "check.py")
        with open(script, "w", encoding="utf-8") as handle:
            handle.write(code + "\n\n# ---- hidden tests ----\n" + test + "\n")
        try:
            completed = subprocess.run(
                [sys.executable, "-I", script],
                cwd=workdir,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return False, f"timed out after {timeout_s:g}s"
    if completed.returncode == 0:
        return True, "tests passed"
    tail = (completed.stderr or completed.stdout).strip().splitlines()[-3:]
    return False, f"exit {completed.returncode}: {' | '.join(tail)}"


def check_python_exec(spec: dict[str, Any], result: ChatResult, task: Task) -> Outcome:
    code = extract_code(result.content)
    if code is None:
        return False, "no fenced python code block"
    test = spec["test"]
    test = "\n".join(test) if isinstance(test, list) else test  # task files store lines
    return run_python(code, test, float(spec.get("timeout", 10)))


# ---- long context -------------------------------------------------------------------------------


def check_needle(spec: dict[str, Any], result: ChatResult, task: Task) -> Outcome:
    if task.needle is None:
        raise ValueError(f"task {task.id} has a needle checker but no needle")
    squash = lambda s: re.sub(r"[\s\-_]", "", s.casefold())  # noqa: E731
    secret, answer = task.needle.secret, squash(result.content)
    wrong = [code for code in task.needle.distractors if squash(code) in answer]
    if wrong:
        return False, f"answered distractor code(s) {wrong!r}; secret is {secret!r}"
    if squash(secret) in answer:
        return True, f"found secret {secret!r}"
    return False, f"secret {secret!r} not in {_snippet(result.content)!r}"


# ---- composition --------------------------------------------------------------------------------


def check_all(spec: dict[str, Any], result: ChatResult, task: Task) -> Outcome:
    details = []
    for sub in spec["checks"]:
        ok, why = run_check(sub, result, task)
        if not ok:
            return False, f"{sub['kind']}: {why}"
        details.append(f"{sub['kind']}: {why}")
    return True, "; ".join(details)


CHECKERS: dict[str, Callable[[dict[str, Any], ChatResult, Any], Outcome]] = {
    "exact": check_exact,
    "contains": check_contains,
    "not_contains": check_not_contains,
    "regex": check_regex,
    "word_count": check_word_count,
    "bullets": check_bullets,
    "language": check_language,
    "number": check_number,
    "json_schema": check_json_schema,
    "tool_call": check_tool_call,
    "python_exec": check_python_exec,
    "needle": check_needle,
    "all": check_all,
}


def run_check(spec: dict[str, Any], result: ChatResult, task: Task) -> Outcome:
    # Everything except tool_call judges the answer text; a tool call where text was expected fails.
    if spec["kind"] not in {"tool_call", "all"} and not result.content and result.tool_calls:
        return False, "answered with a tool call instead of text"
    return CHECKERS[spec["kind"]](spec, result, task)


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else [value]
