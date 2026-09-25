"""Task-level quality suite: checkers, bundled task answers, and the run/compare flow."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

import pytest

from tools.eval import cli
from tools.eval.checkers import final_number, run_check, run_python, validate_schema
from tools.eval.client import ChatClient, ChatResult, ToolCall, parse_completion, strip_think
from tools.eval.runner import RunConfig, run_suite
from tools.eval.tasks import load_tasks, make_needle, parse_task

# ---- mock server --------------------------------------------------------------------------------

Responder = Callable[[dict[str, Any]], tuple[int, dict[str, Any]]]


def completion(content: str = "", tool_calls: list[dict[str, Any]] | None = None,
               reasoning: str = "") -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning_content"] = reasoning
    if tool_calls:
        message["tool_calls"] = [
            {"id": f"call_{i}", "type": "function",
             "function": {"name": name, "arguments": json.dumps(args)}}
            for i, (name, args) in enumerate(tool_calls)
        ]
    return {
        "id": "chatcmpl-1", "object": "chat.completion", "model": "mock",
        "choices": [{"index": 0, "message": message,
                     "finish_reason": "tool_calls" if tool_calls else "stop"}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 7, "total_tokens": 27},
        "timings": {"predicted_n": 7, "predicted_per_second": 50.0},
    }


class MockServer:
    def __init__(self, responder: Responder) -> None:
        self.requests: list[dict[str, Any]] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def _send(self, status: int, body: dict[str, Any]) -> None:
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                self._send(200, {"object": "list", "data": [{"id": "mock-model"}]})

            def do_POST(self) -> None:
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.requests.append(payload)
                self._send(*responder(payload))

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self._server.server_address[1]}/v1"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> MockServer:
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._server.shutdown()
        self._server.server_close()


def last_user(payload: dict[str, Any]) -> str:
    return [m for m in payload["messages"] if m["role"] == "user"][-1]["content"]


# ---- checkers -----------------------------------------------------------------------------------

TASK = parse_task({"id": "t", "category": "c", "messages": [], "check": {"kind": "exact",
                                                                        "value": "x"}})


def check(spec: dict[str, Any], content: str = "", tool_calls: list[ToolCall] | None = None):
    return run_check(spec, ChatResult(content=content, tool_calls=tool_calls or []), TASK)[0]


def test_text_checkers_pass_and_fail() -> None:
    assert check({"kind": "exact", "value": "OK"}, "  ok ")
    assert not check({"kind": "exact", "value": "OK", "normalize": ["whitespace"]}, "ok")
    assert check({"kind": "contains", "values": ["cat", ["under", "beneath"]]}, "The Cat beneath")
    assert not check({"kind": "contains", "values": ["cat", "dog"]}, "the cat")
    assert check({"kind": "contains", "values": ["manana"], "normalize": ["case", "accents"]},
                 "por la Mañana")
    assert not check({"kind": "contains", "values": ["cat"], "words": True}, "concatenate")
    assert check({"kind": "not_contains", "values": ["wave"]}, "waves crash")
    assert not check({"kind": "not_contains", "values": ["wave"]}, "a Wave")
    assert check({"kind": "regex", "pattern": r"^\d{3}$", "fullmatch": True}, "123")
    assert not check({"kind": "regex", "pattern": r"^\d{3}$"}, "1234")
    assert check({"kind": "word_count", "max": 3}, "one two three")
    assert not check({"kind": "word_count", "max": 3}, "one two three four")
    assert check({"kind": "bullets", "min": 2, "max": 2, "only": True}, "- a\n* b\n")
    assert not check({"kind": "bullets", "min": 2, "max": 2, "only": True}, "Intro\n- a\n- b")
    assert check({"kind": "language", "lang": "es"}, "El coche es muy bueno para la ciudad y no contamina.")
    assert not check({"kind": "language", "lang": "es"}, "The car is good for the city and it is clean.")
    both = {"kind": "all", "checks": [{"kind": "contains", "values": ["a"]},
                                      {"kind": "word_count", "max": 1}]}
    assert check(both, "a") and not check(both, "a b")


def test_number_parsing_accepts_marker_and_locale_separators() -> None:
    assert final_number("7 notebooks... Answer: $12.40") == 12.4
    assert final_number("Respuesta: 6,50 euros") == 6.5
    assert final_number("total 1,234.5 then 1.234,5") == 1234.5
    assert final_number("so \\boxed{75} km/h, not 3") == 75
    assert final_number("Answer: 45\nCarl pays 20 ...") == 45
    assert check({"kind": "number", "value": 88}, "It ends at 88.")
    assert not check({"kind": "number", "value": 88}, "Answer: 87")
    assert not check({"kind": "number", "value": 88}, "no digits here")


def test_json_schema_subset_and_values() -> None:
    schema = {"type": "object", "required": ["a", "b"], "additionalProperties": False,
              "properties": {"a": {"type": "integer", "minimum": 1, "maximum": 3},
                             "b": {"type": "array", "items": {"enum": ["x", "y"]}},
                             "c": {"type": ["string", "null"]}}}
    assert validate_schema({"a": 2, "b": ["x"], "c": None}, schema) == []
    errors = validate_schema({"a": 5, "b": ["z"], "d": 1}, schema)
    assert any("maximum" in e for e in errors) and any("not in" in e for e in errors)
    assert any("unexpected property" in e for e in errors)
    assert validate_schema(True, {"type": "integer"}) != []

    spec = {"kind": "json_schema", "schema": schema, "values": {"/b/0": "x"}}
    assert check(spec, 'Here:\n```json\n{"a": 1, "b": ["x"]}\n```')
    assert check(spec, 'Result {"a": 1, "b": ["x"]} done')
    assert not check(spec, '{"a": 1, "b": ["y"]}')  # value mismatch
    assert not check(spec, "no json at all")


def test_tool_call_checker() -> None:
    weather = ToolCall("get_weather", {"city": "Madrid", "unit": "celsius"}, True)
    spec = {"kind": "tool_call", "name": "get_weather",
            "arguments": {"city": {"regex": "(?i)madrid"}, "unit": "Celsius"}}
    assert check(spec, tool_calls=[weather])
    assert not check(spec, tool_calls=[ToolCall("get_time", {}, True)])
    assert not check(spec, "It is sunny.")
    assert not check(spec, tool_calls=[ToolCall("get_weather", "{bad", False)])
    assert not check(spec, tool_calls=[weather, weather])  # exclusive by default
    numeric = {"kind": "tool_call", "name": "f", "arguments": {"n": 45}}
    assert check(numeric, tool_calls=[ToolCall("f", {"n": "45"}, True)])
    parallel = {"kind": "tool_call", "calls": [{"name": "w", "arguments": {"city": "Paris"}},
                                               {"name": "w", "arguments": {"city": "Tokyo"}}]}
    calls = [ToolCall("w", {"city": "Tokyo"}, True), ToolCall("w", {"city": "Paris"}, True)]
    assert check(parallel, tool_calls=calls) and not check(parallel, tool_calls=calls[:1])
    none = {"kind": "tool_call", "expect": "none", "content": {"kind": "contains",
                                                               "values": ["canberra"]}}
    assert check(none, "Canberra") and not check(none, "Sydney")
    assert not check(none, tool_calls=[weather])


def test_parse_completion_tool_calls_and_think_stripping() -> None:
    body = completion("<think>hmm</think>\n\nFinal", [("f", {"x": 1})], reasoning="plan")
    body["choices"][0]["message"]["tool_calls"].append(
        {"id": "c", "type": "function", "function": {"name": "g", "arguments": "{not json"}})
    result = parse_completion(body)
    assert result.content == "Final" and result.reasoning == "plan"
    assert result.tool_calls[0] == ToolCall("f", {"x": 1}, True)
    assert result.tool_calls[1] == ToolCall("g", "{not json", False)
    assert result.timings["predicted_per_second"] == 50.0
    assert strip_think("reasoning leaked</think>answer") == "answer"
    assert strip_think("a <think>x</think> b") == "a  b"


def test_python_exec_pass_fail_and_timeout() -> None:
    assert run_python("def f(x):\n    return x + 1", "assert f(1) == 2", 10)[0]
    passed, detail = run_python("def f(x):\n    return x", "assert f(1) == 2", 10)
    assert not passed and "AssertionError" in detail
    passed, detail = run_python("while True:\n    pass", "", 1)
    assert not passed and "timed out" in detail
    code = {"kind": "python_exec", "test": ["assert g() == 3"]}
    assert check(code, "Sure:\n```python\ndef g():\n    return 3\n```\nDone.")
    assert not check(code, "def g(): return 3")  # no fenced block


def test_needle_is_deterministic_and_rejects_distractors() -> None:
    a, b = make_needle(2000, 0.5, 7), make_needle(2000, 0.5, 7)
    assert a == b and a.haystack.count(a.secret) == 1
    assert 7000 <= len(a.haystack) <= 9000
    task = parse_task({"id": "n", "category": "long_context", "check": {"kind": "needle"},
                       "needle": {"approx_tokens": 2000, "depth": 0.5, "seed": 7},
                       "messages": [{"role": "user", "content": "{haystack}\nVault {needle_key}?"}]})
    assert task.needle is not None and task.needle.key in task.messages[0]["content"]
    ok = ChatResult(content=f"The code is {task.needle.secret.lower().replace('-', ' ')}.")
    assert run_check(task.check, ok, task)[0]
    wrong = ChatResult(content=task.needle.distractors[0])
    assert not run_check(task.check, wrong, task)[0]


# ---- bundled tasks --------------------------------------------------------------------------------

_CODE = {
    "code.fizzbuzz_list": "def fizz(n):\n    return ['FizzBuzz' if i % 15 == 0 else 'Fizz' if i % 3 == 0 else 'Buzz' if i % 5 == 0 else str(i) for i in range(1, n + 1)]",
    "code.to_roman": "def to_roman(n):\n    if not 1 <= n <= 3999:\n        raise ValueError(n)\n    out = ''\n    for v, s in [(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),(90,'XC'),(50,'L'),(40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]:\n        while n >= v:\n            out += s; n -= v\n    return out",
    "code.merge_intervals": "def merge_intervals(xs):\n    out = []\n    for a, b in sorted(xs):\n        if out and a <= out[-1][1]:\n            out[-1] = (out[-1][0], max(b, out[-1][1]))\n        else:\n            out.append((a, b))\n    return out",
    "code.parse_duration": "import re\ndef parse_duration(t):\n    m = re.fullmatch(r'(?:(\\d+)h)?(?:(\\d+)m)?(?:(\\d+)s)?', t)\n    if not t or not m:\n        raise ValueError(t)\n    h, mi, s = (int(g or 0) for g in m.groups())\n    return h * 3600 + mi * 60 + s",
    "code.lru_cache": "from collections import OrderedDict\nclass LRUCache:\n    def __init__(self, c):\n        self.c, self.d = c, OrderedDict()\n    def get(self, k):\n        if k not in self.d:\n            return -1\n        self.d.move_to_end(k)\n        return self.d[k]\n    def put(self, k, v):\n        self.d[k] = v\n        self.d.move_to_end(k)\n        if len(self.d) > self.c:\n            self.d.popitem(last=False)",
    "code.top_words": "import re\nfrom collections import Counter\ndef top_words(text, k):\n    c = Counter(w.lower() for w in re.findall('[A-Za-z]+', text))\n    return sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:k]",
    "code.fix_binary_search": "def binary_search(xs, target):\n    lo, hi = 0, len(xs) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if xs[mid] == target:\n            return mid\n        if xs[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1",
}
_TEXT = {
    "json.extract_person": '{"name": "María López", "age": 34, "city": "Valencia", "languages": ["Spanish", "English", "Catalan"]}',
    "json.classify_tickets": '[{"id": 1, "priority": "high"}, {"id": 2, "priority": "low"}, {"id": 3, "priority": "medium"}]',
    "json.order_total": '{"order_id": "A-1042", "items": [{"sku": "PEN-01", "qty": 3, "unit_price": 1.5}, {"sku": "NB-07", "qty": 2, "unit_price": 4.25}], "total": 13.0}',
    "json.config_types": '```json\n{"host": "api.internal", "port": 8443, "tls": true, "proxy": null}\n```',
    "json.sentiment_enum": '{"r1": "positive", "r2": "negative", "r3": "neutral", "r4": "negative"}',
    "math.change_due": "23.80 + 13.80 = 37.60\nAnswer: 12.40",
    "math.percent_chain": "Answer: 88",
    "math.debt_ledger": "Answer: 45",
    "math.pointer_chase": "A, D, B, F, C.\nAnswer: 417",
    "math.variable_chain": "Answer: 11",
    "math.pipes_rate": "Answer: 4",
    "math.average_speed": "Answer: 75",
    "math.inclusion_exclusion": "Answer: 80",
    "if.three_bullets": "- Catch regressions\n- Document behavior\n- Enable refactoring",
    "if.word_limit": "A hash map stores key-value pairs and uses a hash function to find a value from its key in constant average time.",
    "if.forbidden_words": "The ocean covers most of our planet. Its depths hide countless creatures.",
    "if.only_ok": "OK",
    "if.caps_title": "WHY RUST KEEPS YOUR MEMORY SAFE",
    "if.numbered_steps_done": "1. Boil the kettle.\n2. Put a tea bag in a cup.\n3. Pour the hot liquid.\n4. Steep for three minutes.\nDONE",
    "if.answer_in_spanish": "Los coches eléctricos no emiten gases por el tubo de escape y su mantenimiento es más barato.",
    "es.translate_to_spanish": "La biblioteca abre a las nueve de la mañana y cierra a las seis de la tarde.",
    "es.translate_to_english": "The cat sleeps under the kitchen table.",
    "es.change_word_problem": "Las manzanas cuestan 7,20 euros y las peras 6,30 euros, en total 13,50 euros.\nRespuesta: 6,50",
    "es.ages_equation": "Si Ana tiene a años, Luis tiene 2a y en 5 años la suma es 3a + 10 = 40, por lo que a = 10.\nRespuesta: 20",
    "es.explain_rest_api": "Una API REST es una interfaz que permite a dos sistemas comunicarse mediante el protocolo HTTP. Cada recurso se identifica con una URL y se manipula con los métodos GET, POST, PUT y DELETE. Las respuestas suelen estar en formato JSON y el servidor no guarda el estado de la sesión del cliente.",
    "es.spelling_fix": "Ayer fuimos a la biblioteca y leímos un libro muy interesante sobre la historia de España.",
    "tool.no_tool_needed": "Canberra",
    "tool.use_tool_result": "It is 19 °C and overcast in Lima.",
}
_TOOLS = {
    "tool.weather_single": [("get_weather", {"city": "Madrid", "unit": "celsius"})],
    "tool.calendar_arguments": [("create_calendar_event", {"title": "Q3 budget review", "date": "2026-10-14", "start_time": "15:30", "duration_minutes": 45, "attendees": ["ana@example.com", "luis@example.com"]})],
    "tool.choose_read_file": [("read_file", {"path": "src/utils/date_parser.py"})],
    "tool.choose_search_code": [("search_code", {"query": "def parse_iso_week"})],
    "tool.choose_run_tests": [("run_command", {"command": "python -m pytest tests/test_api.py"})],
    "tool.currency_extraction": [("convert_currency", {"amount": 250, "from_currency": "USD", "to_currency": "EUR"})],
    "tool.parallel_weather": [("get_weather", {"city": "Tokyo"}), ("get_weather", {"city": "Paris"})],
}


def reference_answer(task_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    if task_id in _CODE:
        return completion(f"Here it is:\n```python\n{_CODE[task_id]}\n```")
    if task_id in _TOOLS:
        return completion("", _TOOLS[task_id])
    if task_id.startswith("long."):
        task = load_tasks(task_id)[0]
        return completion(task.needle.secret)
    return completion(_TEXT[task_id])


def test_bundled_tasks_are_balanced_and_reference_answers_pass() -> None:
    tasks = load_tasks()
    assert len(tasks) >= 40
    categories = {t.category for t in tasks}
    assert categories == {"tool_calling", "json_output", "code_python", "reasoning_math",
                          "instruction_following", "spanish", "long_context"}
    for task in tasks:
        result = parse_completion(reference_answer(task.id))
        passed, detail = run_check(task.check, result, task)
        assert passed, f"{task.id}: {detail}"
        # A non-answer must never pass.
        assert not run_check(task.check, parse_completion(completion("I don't know.")), task)[0]


# ---- run -> results -> compare --------------------------------------------------------------------


def test_run_suite_records_failures_and_http_errors() -> None:
    tasks = load_tasks("math.*")

    def responder(payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        if "Pipe A" in last_user(payload):
            return 400, {"error": {"code": "invalid_prompt", "message": "boom"}}
        return 200, completion("Answer: 88")

    with MockServer(responder) as server:
        config = RunConfig(model="m", label="x", base_url=server.base_url, thinking="off",
                           repeats=2, seed=10)
        results = run_suite(ChatClient(server.base_url, 30), tasks, config, log=lambda _: None)
    records = results["results"]
    assert len(records) == 2 * len(tasks)
    assert [r["id"] for r in records if r["passed"]] == ["math.percent_chain"] * 2
    failed = next(r for r in records if r["id"] == "math.pipes_rate")
    assert "HTTP 400 invalid_prompt: boom" in failed["detail"] and failed["error"]
    request = server.requests[0]
    assert request["chat_template_kwargs"] == {"enable_thinking": False}
    assert request["temperature"] == 0 and request["stream"] is False
    assert sorted({r["seed"] for r in server.requests}) == [10, 11]


def test_run_suite_survives_unreachable_server() -> None:
    config = RunConfig(model="m", label="x", base_url="http://127.0.0.1:9/v1")
    results = run_suite(ChatClient(config.base_url, 2), load_tasks("if.only_ok"), config,
                        log=lambda _: None)
    assert results["results"][0]["passed"] is False
    assert "request failed" in results["results"][0]["detail"]


def test_cli_run_and_compare(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    def good(payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        task_id = next(t.id for t in load_tasks("tool.*") if t.messages == payload["messages"])
        return 200, reference_answer(task_id)

    def bad(payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        return 200, completion("<think>no tools</think>I cannot help.")

    outputs = []
    for label, responder in (("good", good), ("bad", bad)):
        with MockServer(responder) as server:
            out = tmp_path / f"results_{label}.json"
            assert cli.main(["run", "--base-url", server.base_url, "--label", label,
                             "--out", str(out), "--tasks", "tool_calling"]) == 0
        # The model id is discovered from /v1/models when --model is omitted.
        assert server.requests[0]["model"] == "mock-model"
        assert "chat_template_kwargs" not in server.requests[0]
        assert server.requests[0]["tool_choice"] == "auto"
        outputs.append(out)

    good_results = json.loads(outputs[0].read_text(encoding="utf-8"))
    assert all(r["passed"] for r in good_results["results"])
    assert good_results["results"][0]["timings"]["predicted_per_second"] == 50.0
    bad_results = json.loads(outputs[1].read_text(encoding="utf-8"))
    assert bad_results["results"][0]["content"] == "I cannot help."

    markdown = tmp_path / "compare.md"
    capsys.readouterr()
    assert cli.main(["compare", str(outputs[0]), str(outputs[1]), "--markdown", str(markdown)]) == 0
    report = markdown.read_text(encoding="utf-8")
    assert report == capsys.readouterr().out.rstrip("\n") + "\n"
    assert "| good | mock-model | server-default | 9/9 | 100.0% |" in report
    assert "| tool_calling | 9/9 | 0/9 |" in report
    assert "| tool.weather_single | tool_calling | 1/1 | 0/1 |" in report
    assert "no tool call" in report  # failure details
