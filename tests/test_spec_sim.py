"""Offline speculative-decoding simulator: pool contract, proposers, replay and reporting."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from tools.spec_sim import cli
from tools.spec_sim.cost import CostModel
from tools.spec_sim.inputs import claude_code_messages, load_records, records_from_messages
from tools.spec_sim.ngram import (
    Context,
    NgramModPool,
    NgramSimpleIndex,
    fmix64,
    slot_of,
    tag_of,
    window_hash,
)
from tools.spec_sim.simulate import MtpModel, Record, SimConfig, simulate

# ---- pool hash contract (pinned identically in tests/models/qwen3_5/test_ngram_pool.cpp) -------


def test_hash_slot_and_tag_match_the_cpp_pool_vector() -> None:
    window = window_hash([1, 2, 3])
    assert window == 0x190380FC9ABAAC46
    assert fmix64(window) == 0x2F67E0CD700B3673
    assert slot_of(window, 4 * 1024 * 1024) == 734835
    assert tag_of(window, 14) == 3033
    top = window_hash([248076] * 4)
    assert slot_of(top, 1000003) == 967702
    assert tag_of(top, 14) == 6786


def test_rolling_context_hash_equals_direct_window_hash() -> None:
    rng = random.Random(7)
    tokens = [rng.randrange(248077) for _ in range(200)]
    context = Context(5)
    for index, token in enumerate(tokens):
        context.append(token)
        if index >= 4:
            assert context.hash == window_hash(tokens[index - 4:index + 1])
    assert context.rolled([9, 8]) == window_hash(tokens[-3:] + [9, 8])


# ---- proposers ---------------------------------------------------------------------------------


def test_pool_walk_reproduces_a_repeated_span_and_stops_at_unseen_ngram() -> None:
    pool = NgramModPool(3, 1 << 16)
    context = Context(3)
    block = list(range(100, 140))
    context.extend(block + [7, 7, 7] + block[:3], [pool])
    draft = pool.propose(context, 64)
    assert draft == block[3:] + [7, 7, 7] + block[:3] + block[3:3 + 21]
    assert pool.propose(context, 5) == block[3:8]
    assert pool.propose(Context(3), 5) == []


def test_incremental_pool_insertion_equals_bulk_insertion() -> None:
    rng = random.Random(3)
    tokens = [rng.randrange(50) for _ in range(500)]
    bulk = NgramModPool(4, 997)
    Context(4).extend(tokens, [bulk])
    incremental = NgramModPool(4, 997)
    context = Context(4)
    cursor = 0
    while cursor < len(tokens):
        step = rng.randrange(1, 9)
        context.extend(tokens[cursor:cursor + step], [incremental])
        cursor += step
    assert bulk.table == incremental.table
    assert bulk.occupied == incremental.occupied


def test_pool_tag_rejects_a_colliding_ngram_that_the_untagged_pool_returns() -> None:
    first, second = [1, 2], [3, 4]
    assert tag_of(window_hash(first), 14) != tag_of(window_hash(second), 14)
    for tag_bits, expected in ((14, []), (0, [5])):
        pool = NgramModPool(2, 1, tag_bits)
        Context(2).extend(first + [5], [pool])
        probe = Context(2)
        probe.extend(second)
        assert pool.propose(probe, 1) == expected


def test_ngram_simple_copies_after_the_most_recent_occurrence_with_overlap() -> None:
    index = NgramSimpleIndex(2, 6)
    context = Context(2)
    context.extend([1, 2, 3, 9, 1, 2, 4, 5, 1, 2], [index])
    assert index.propose(context, 10) == [4, 5, 1, 2, 4, 5]
    periodic = NgramSimpleIndex(2, 8)
    context = Context(2)
    context.extend([6, 7, 6, 7], [periodic])
    assert periodic.propose(context, 8) == [6, 7, 6, 7, 6, 7, 6, 7]
    assert periodic.propose(context, 3, prefix=[6]) == [7, 6, 7]


# ---- replay ------------------------------------------------------------------------------------

COST = CostModel()


def run(policy: str, records: list[Record], cap: int = 15, **overrides) -> dict:
    config = SimConfig(policy=policy, cap=cap, n=overrides.pop("n", 3), **overrides)
    return simulate(records, config, COST).summary()


def test_replay_commits_every_recorded_token_within_cap_and_budget() -> None:
    rng = random.Random(11)
    block = [rng.randrange(1000) for _ in range(300)]
    records = [Record(tuple(block), tuple(block + block[:50]))]
    for policy in ("mtp", "ngram-simple", "ngram-mod", "select:ngram-mod", "chain:ngram-simple"):
        for cap in (4, 15, 64):
            config = SimConfig(policy=policy, cap=cap, n=3)
            result = simulate(records, config, COST)
            assert result.tokens == 350
            assert max(result.accept_counts) <= cap
            assert max(result.width_counts) <= cap + 1


def test_unrepeated_text_gets_no_ngram_drafts_and_costs_plain_decode() -> None:
    record = Record((), tuple(range(1000, 1400)))
    summary = run("ngram-mod", [record])
    assert summary["rounds"] == 400
    assert summary["draft_round_fraction"] == 0.0
    assert summary["tok_s"] == pytest.approx(COST.decode_tok_s())


def test_repeated_prompt_content_is_accepted_in_long_rounds() -> None:
    rng = random.Random(5)
    file = tuple(rng.randrange(5000) for _ in range(640))
    record = Record(file, file)
    # The window ending at the prompt boundary has no recorded continuation, so the first n=3
    # tokens are plain rounds; afterwards every round accepts its whole (budget-bounded) draft.
    wide = run("ngram-mod", [record], cap=64)
    assert wide["rounds"] == 3 + 10
    assert wide["accept_over_15_fraction"] == pytest.approx(10 / 13)
    narrow = run("ngram-mod", [record], cap=15)
    assert narrow["rounds"] == 3 + 40
    assert narrow["accept_over_15_fraction"] == 0.0
    assert wide["tok_s"] > narrow["tok_s"]


def test_global_pool_carries_across_records_and_sequence_scope_does_not() -> None:
    rng = random.Random(9)
    text = tuple(rng.randrange(5000) for _ in range(200))
    records = [Record((1,), text), Record((2,), text)]
    shared = run("ngram-mod", records, pool_scope="global")
    private = run("ngram-mod", records, pool_scope="sequence")
    assert shared["rounds"] < private["rounds"]
    assert private["rounds"] == 400


def test_mtp_model_acceptance_extremes_and_select_chain_policies() -> None:
    truth = list(range(50))
    assert MtpModel(3, (1.0,), 0).draft(truth, 4, 3) == [4, 5, 6]
    assert MtpModel(3, (0.0,), 0).draft(truth, 4, 3) == [MtpModel.WRONG] * 3
    assert MtpModel(3, (1.0, 0.0), 0).draft(truth, 4, 2) == [4, MtpModel.WRONG]

    rng = random.Random(2)
    file = tuple(rng.randrange(5000) for _ in range(320))
    record = Record(file, file)
    perfect = run("mtp", [record], mtp_accept=(1.0,))
    assert perfect["tokens_per_round"] == pytest.approx(4.0)
    select = run("select:ngram-mod", [record], cap=32, mtp_accept=(1.0,))
    chain = run("chain:ngram-mod", [record], cap=32, mtp_accept=(1.0,))
    # Chain extends the correct MTP prefix at once; select needs n=3 committed tokens (one MTP
    # round) before the pool window is inside the repeated span.
    assert chain["tokens_per_round"] == pytest.approx(32.0)
    assert select["rounds"] == 1 + 10
    assert select["ngram_round_fraction"] == pytest.approx(10 / 11)


def test_draft_min_drops_short_ngram_drafts() -> None:
    record = Record((), (1, 2, 3, 4, 1, 2, 3, 5, 6, 7))
    assert run("ngram-simple", [record], draft_min=1)["draft_round_fraction"] > 0
    assert run("ngram-simple", [record], draft_min=3)["draft_round_fraction"] == 0


def test_cost_model_anchors_extrapolation_and_width_buckets() -> None:
    assert COST.round_ms(1) == pytest.approx(21.3)
    assert COST.round_ms(4) == pytest.approx(25.9)
    assert COST.round_ms(13) == pytest.approx(33.5)
    assert COST.round_ms(33) == pytest.approx(43.5)
    assert COST.round_ms(65) == pytest.approx(59.5)
    assert COST.width_for(0, 15) == 1
    assert COST.width_for(3, 15) == 4
    assert COST.width_for(9, 15) == 16
    assert COST.width_for(20, 32) == 24
    assert COST.width_for(30, 32) == 33
    assert COST.width_for(50, 64) == 65


def test_replay_wide_verify_charges_rejected_wide_rounds_only() -> None:
    replay = CostModel(replay_above=16)
    assert replay.verify_ms(16, 15, 3) == pytest.approx(COST.round_ms(16))
    assert replay.verify_ms(33, 32, 32) == pytest.approx(COST.round_ms(33))
    assert replay.verify_ms(33, 32, 9) == pytest.approx(COST.round_ms(33) + COST.round_ms(10))
    assert COST.verify_ms(33, 32, 9) == pytest.approx(COST.round_ms(33))


# ---- inputs and CLI ----------------------------------------------------------------------------


class CharTokenizer:
    """Fake tokenizer: one id per character, special markers as single ids."""

    SPECIAL = {"<|im_start|>": 1000, "<|im_end|>": 1001}

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        index = 0
        while index < len(text):
            for marker, token in self.SPECIAL.items():
                if text.startswith(marker, index):
                    out.append(token)
                    index += len(marker)
                    break
            else:
                out.append(ord(text[index]) % 1000)
                index += 1
        return out


def test_messages_become_one_record_per_assistant_turn_with_extending_prompts() -> None:
    tokenizer = CharTokenizer()
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ok",
         "tool_calls": [{"function": {"name": "read", "arguments": "{\"path\": \"a\"}"}}]},
        {"role": "tool", "content": "data"},
        {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
    ]
    first, second = records_from_messages(messages, tokenizer)
    full_first = first.prompt + first.completion
    assert second.prompt[:len(full_first)] == full_first
    rendered = "".join(chr(t) if t < 1000 else "|" for t in first.completion)
    assert '<tool_call>\n{"name": "read", "arguments": {"path": "a"}}\n</tool_call>' in rendered
    assert first.completion[-1] == 1001
    assert tokenizer.encode("<tool_response>\ndata\n</tool_response>")[0] in second.prompt


def test_claude_code_transcript_merges_assistant_events(tmp_path: Path) -> None:
    events = [
        {"type": "user", "message": {"role": "user", "content": "fix it"}},
        {"type": "assistant", "message": {"role": "assistant",
                                          "content": [{"type": "thinking", "thinking": "x"},
                                                      {"type": "text", "text": "Reading."}]}},
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "tool_use", "name": "Read", "input": {"file_path": "a.py"}}]}},
        {"type": "user", "message": {"role": "user", "content": [
            {"type": "tool_result", "content": [{"type": "text", "text": "print(1)"}]}]}},
        {"type": "summary", "summary": "ignored"},
        {"type": "assistant", "isSidechain": True, "message": {"role": "assistant",
                                                               "content": "sidechain"}},
    ]
    path = tmp_path / "session.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    messages = claude_code_messages(path)
    assert [m["role"] for m in messages] == ["user", "assistant", "tool"]
    assert messages[1]["content"] == "Reading."
    assert messages[1]["tool_calls"][0]["function"]["name"] == "Read"
    assert messages[2]["content"] == "print(1)"
    records = load_records([path], CharTokenizer(), claude_code=True)
    assert len(records) == 1


def test_text_records_require_a_tokenizer(tmp_path: Path) -> None:
    path = tmp_path / "text.jsonl"
    path.write_text(json.dumps({"prompt": "a", "completion": "b"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="tokenizer"):
        load_records([path], None)
    assert load_records([path], CharTokenizer())[0].completion == (ord("b"),)


def test_cli_reports_every_policy_cap_and_n(tmp_path: Path, capsys) -> None:
    rng = random.Random(4)
    file = [rng.randrange(3000) for _ in range(400)]
    path = tmp_path / "ids.jsonl"
    path.write_text("\n".join([json.dumps({"prompt_ids": file, "completion_ids": file}),
                               json.dumps(file[:100])]) + "\n", encoding="utf-8")
    out = tmp_path / "report.json"
    assert cli.main([str(path), "--ngram-n", "3,8", "--json", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    keys = {(s["policy"], s["n"], s["cap"]) for s in report["summaries"]}
    assert len(keys) == 3 * (1 + 4 * 2)
    assert all(s["tokens"] == 500 for s in report["summaries"])
    assert "ngram-mod" in capsys.readouterr().out
