"""`python -m tools.spec_sim`: replay recorded sequences under speculative proposers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .cost import (DEFAULT_MARGINAL_MS, DEFAULT_POINTS, DEFAULT_WIDTH_BUCKETS, CostModel,
                   describe, parse_floats, parse_ints, parse_points)
from .inputs import load_records, load_tokenizer
from .simulate import LONG_DRAFT, POLICIES, SimConfig, simulate

DEFAULT_POLICIES = ("mtp", "ngram-simple", "ngram-mod", "select:ngram-mod", "chain:ngram-mod")


def _points_text(points) -> str:
    return ",".join(f"{t}:{ms:g}" for t, ms in points)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m tools.spec_sim", description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="JSONL record files")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--tokenizer-json", type=Path, help="tokenizer.json for text input")
    source.add_argument("--artifact", type=Path,
                        help=".ninfer artifact whose embedded tokenizer.json tokenizes text")
    parser.add_argument("--claude-code", action="store_true",
                        help="inputs are Claude Code session transcripts, one session per file")
    parser.add_argument("--policies", default=",".join(DEFAULT_POLICIES),
                        help=f"comma list from: {', '.join(POLICIES)}")
    parser.add_argument("--caps", default="15,32,64", help="draft caps (verify width = cap + 1)")
    parser.add_argument("--ngram-n", default="12", help="comma list of n-gram key lengths")
    parser.add_argument("--ngram-m", type=int, default=64, help="ngram-simple copy length")
    parser.add_argument("--pool-mib", type=float, default=16.0, help="ngram-mod pool size")
    parser.add_argument("--tag-bits", type=int, default=14,
                        help="pool entry tag bits; 0 reproduces llama.cpp's untagged pool")
    parser.add_argument("--pool-scope", choices=("global", "sequence"), default="global",
                        help="global: one pool for all records in order, like a shared server "
                             "pool; sequence: fresh pool per record")
    parser.add_argument("--draft-min", type=int, default=1,
                        help="drop n-gram drafts (or chain extensions) shorter than this")
    parser.add_argument("--mtp-k", type=int, default=3)
    parser.add_argument("--mtp-accept", default="0.86",
                        help="per-position conditional MTP acceptance; last value repeats")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cost-points", default=_points_text(DEFAULT_POINTS),
                        help="round cost anchors T:ms, linear between anchors")
    parser.add_argument("--marginal-ms", type=float, default=DEFAULT_MARGINAL_MS,
                        help="cost per column past the last anchor (1/prefill tok/s)")
    parser.add_argument("--width-buckets", default=",".join(map(str, DEFAULT_WIDTH_BUCKETS)),
                        help="captured verify widths; a round runs the smallest that fits")
    parser.add_argument("--wide-verify", choices=("records", "replay"), default="records",
                        help="cost of widths above 16: widened ReplaySSM records, or prefill "
                             "route plus re-running the committed prefix after a rejection")
    parser.add_argument("--json", type=Path, help="write every summary to this JSON file")
    return parser


def format_table(summaries: Sequence[dict], decode_tok_s: float) -> str:
    header = (f"{'policy':<20} {'n':>3} {'cap':>4} {'tok/rnd':>8} {'acc/rnd':>8} "
              f"{'accept%':>8} {'>15 drf':>8} {'>15 acc':>8} {'>15 tok':>8} "
              f"{'tok/s':>7} {'x dec':>6}")
    lines = [header, "-" * len(header)]
    for s in summaries:
        lines.append(
            f"{s['policy']:<20} {s['n']:>3} {s['cap']:>4} {s['tokens_per_round']:>8.2f} "
            f"{s['accepted_per_round']:>8.2f} {100 * s['acceptance']:>7.1f}% "
            f"{100 * s['draft_over_15_fraction']:>7.1f}% "
            f"{100 * s['accept_over_15_fraction']:>7.1f}% "
            f"{100 * s['token_share_accept_over_15']:>7.1f}% "
            f"{s['tok_s']:>7.1f} {s['tok_s'] / decode_tok_s:>6.2f}")
    return "\n".join(lines)


def format_histograms(summaries: Sequence[dict]) -> str:
    lines = ["accepted drafts per round (share of rounds):"]
    for s in summaries:
        rounds = max(s["rounds"], 1)
        cells = " ".join(f"{k}:{100 * v / rounds:.1f}%" for k, v in s["accept_histogram"].items())
        lines.append(f"  {s['policy']:<20} n={s['n']:<3} cap={s['cap']:<3} {cells}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    cost = CostModel(parse_points(args.cost_points), args.marginal_ms,
                     parse_ints(args.width_buckets),
                     replay_above=16 if args.wide_verify == "replay" else 0)
    tokenizer = load_tokenizer(args.tokenizer_json, args.artifact)
    records = load_records(args.inputs, tokenizer, args.claude_code)
    if not records:
        print("no records", file=sys.stderr)
        return 1
    prompt_tokens = sum(len(r.prompt) for r in records)
    completion_tokens = sum(len(r.completion) for r in records)
    print(f"{len(records)} records, {prompt_tokens} prompt tokens, "
          f"{completion_tokens} completion tokens")
    print(f"cost model: {describe(cost)}; plain decode {cost.decode_tok_s():.1f} tok/s")

    summaries = []
    for policy in policies:
        uses_ngram = policy != "mtp"
        for n in (parse_ints(args.ngram_n) if uses_ngram else parse_ints(args.ngram_n)[:1]):
            for cap in parse_ints(args.caps):
                config = SimConfig(
                    policy=policy, cap=cap, n=n, m=args.ngram_m,
                    pool_bytes=int(args.pool_mib * 1024 * 1024), tag_bits=args.tag_bits,
                    draft_min=args.draft_min, mtp_k=args.mtp_k,
                    mtp_accept=parse_floats(args.mtp_accept), pool_scope=args.pool_scope,
                    seed=args.seed)
                summaries.append(simulate(records, config, cost).summary())
    print()
    print(format_table(summaries, cost.decode_tok_s()))
    print(f"(>15 drf / >15 acc: rounds whose draft / accepted prefix exceeds {LONG_DRAFT} "
          f"tokens; >15 tok: share of tokens committed by such rounds)")
    print()
    print(format_histograms(summaries))
    if args.json is not None:
        args.json.write_text(json.dumps({
            "records": len(records),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_model": {"points": cost.points, "marginal_ms": cost.marginal_ms,
                           "width_buckets": cost.width_buckets,
                           "wide_verify": args.wide_verify,
                           "decode_tok_s": cost.decode_tok_s()},
            "summaries": summaries,
        }, indent=2) + "\n", encoding="utf-8")
    return 0
