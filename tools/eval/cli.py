"""`python -m tools.eval {run,compare,list}`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tools.eval.client import ChatClient
from tools.eval.compare import compare
from tools.eval.runner import RunConfig, load_results, run_suite, write_results
from tools.eval.tasks import load_tasks


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m tools.eval", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run tasks against a live OpenAI-compatible server")
    run.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    run.add_argument("--model", help="request model id (default: the server's only /v1/models id)")
    run.add_argument("--label", required=True, help="short run name shown by compare")
    run.add_argument("--out", required=True, type=Path, help="results JSON path")
    run.add_argument("--tasks", default="*", help="glob over task ids or categories")
    run.add_argument("--repeats", type=_positive_int, default=1)
    run.add_argument("--thinking", choices=["on", "off"],
                     help="send chat_template_kwargs.enable_thinking (default: server default)")
    run.add_argument("--max-tokens", type=_positive_int, default=4096,
                     help="max_completion_tokens unless a task sets its own")
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--seed", type=int, default=1234, help="seed for attempt 0; +1 per repeat")
    run.add_argument("--timeout", type=float, default=600.0, help="per-request socket timeout, s")
    run.add_argument("--api-key")
    run.add_argument("--opik-project", help="also log each task as an Opik trace")

    cmp = commands.add_parser("compare", help="compare result files")
    cmp.add_argument("results", nargs="+", type=Path)
    cmp.add_argument("--markdown", type=Path, help="also write the table to this file")

    listing = commands.add_parser("list", help="list task ids and categories")
    listing.add_argument("--tasks", default="*")
    return parser


def _run(args: argparse.Namespace) -> int:
    tasks = load_tasks(args.tasks)
    if not tasks:
        print(f"no task matches {args.tasks!r}", file=sys.stderr)
        return 2
    client = ChatClient(args.base_url, args.timeout, args.api_key)
    model = args.model or client.discover_model()
    config = RunConfig(
        model=model, label=args.label, base_url=args.base_url, max_tokens=args.max_tokens,
        thinking=args.thinking, temperature=args.temperature, seed=args.seed,
        repeats=args.repeats, timeout_s=args.timeout,
    )
    on_record = flush = None
    if args.opik_project:
        from tools.eval.opik_log import opik_recorder

        on_record, flush = opik_recorder(args.opik_project, args.label)
    print(f"{len(tasks)} tasks x {args.repeats} against {model} at {args.base_url}")
    results = run_suite(client, tasks, config, on_record=on_record)
    if flush is not None:
        flush()
    write_results(results, args.out)
    print()
    print(compare([results]))
    print(f"wrote {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to a legacy code page
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return _run(args)
    if args.command == "compare":
        report = compare([load_results(path) for path in args.results])
        print(report)
        if args.markdown:
            args.markdown.write_text(report, encoding="utf-8")
        return 0
    for task in load_tasks(args.tasks):
        print(f"{task.category:24} {task.id}")
    return 0
