# Speculative decoding simulator

`tools/spec_sim` replays recorded token sequences under draft proposers and estimates
tokens per round and decode tok/s. It needs no GPU. It answers one question before any runtime
work: do draft-free n-gram drafts (llama.cpp `ngram-mod` style) help this workload, and is
verifying more than 16 tokens per round worth building? The runtime plan is in
[`docs/maintainer/ngram-speculation-plan.md`](../../docs/maintainer/ngram-speculation-plan.md).

## Model

The target model emits the recorded completion. Each round drafts `D <= cap` tokens, accepts the
longest draft prefix equal to the recorded continuation (`A` tokens), and commits `A + 1` tokens:
the accepted drafts plus the target's correction or bonus token. This is greedy verification,
so the committed text is identical under every policy; only the number of rounds changes. As in
NInfer, a draft never exceeds the remaining budget minus one.

| Policy | Draft |
|---|---|
| `mtp` | `--mtp-k` tokens; draft `j` is correct with conditional probability `--mtp-accept[j]` given drafts `< j` were correct |
| `ngram-simple` | the `--ngram-m` tokens that followed the most recent earlier occurrence of the last `n` tokens in this lane's history (overlapping copies repeat a short period) |
| `ngram-mod` | walk of one shared hash pool (`--pool-mib`, 4-byte entries) mapping an `n`-token window to its latest continuation, until a miss or the cap |
| `select:<ngram>` | the MTP draft, replaced by the n-gram draft when that is longer than the MTP draft |
| `chain:<ngram>` | the MTP draft followed by the n-gram continuation of context + MTP draft |

N-gram drafts or chain extensions shorter than `--draft-min` are dropped. The pool hash, slots,
14-bit entry tags and encoding are the exact contract of the C++ `NgramDraftPool`
(`src/models/qwen3_5/program/speculative/ngram_pool.h`); `--tag-bits 0` reproduces llama.cpp's
untagged pool, where every slot collision becomes a (wrong) draft.

With `--pool-scope global` (default) one pool serves all records in file order, like a pool
shared by server lanes. A record whose prompt extends the previous record's prompt + completion
continues that lane history and only its new suffix enters the pool, as with prefix reuse.

### Cost model

Round cost `C(T)` is a function of verify width `T = D + 1`, linear between single-lane
Qwen3.8-27B RTX 4090 anchors (`--cost-points`, from `WINDOWS_PORT.md`):

| T | ms | Source |
|---|---|---|
| 1 | 21.3 | plain greedy decode, tg128 47 tok/s |
| 4 | 25.9 | MTP 3 round, MTP head included |
| 13 | 33.5 | DFlash2 d12 round, 3.5 ms drafter included |

Past the last anchor each column adds the prefill marginal cost, 0.5 ms (`--marginal-ms`,
1/2000 tok/s), so `C(16) = 35.0`, `C(33) = 43.5` and `C(65) = 59.5` ms. The anchors include an
MTP head or a drafter, so the curve is conservative for pool-only rounds. A round runs at the
smallest captured width that holds its draft (`--width-buckets`, default
`1,2,4,8,16,24,33,48,65` plus `cap + 1`), because CUDA Graphs are captured per width.
`--wide-verify replay` models the prefill-route alternative for widths above 16: a partially
accepted wide round also pays `C(A + 1)` to rebuild the recurrent state.

## Inputs

JSONL files, one record per line:

```text
[151644, 872, 198, ...]                                   token ids; everything is generated
{"prompt_ids": [...], "completion_ids": [...]}            token ids with a prompt
{"prompt": "...", "completion": "..."}                    text (needs a tokenizer)
{"text": "..."}                                           text; everything is generated
{"messages": [{"role": "user", "content": "..."}, ...]}   chat transcript (needs a tokenizer)
```

A `messages` transcript yields one record per assistant message. Its prompt is every earlier
message in the Qwen ChatML layout plus the assistant header; its completion is the assistant body
plus `<|im_end|>`. OpenAI `tool_calls` render as Qwen `<tool_call>` JSON and `tool` messages as
`<tool_response>` user turns. Thinking content is not rendered.

`--claude-code` reads Claude Code session transcripts (`~/.claude/projects/<project>/*.jsonl`),
one session per file, and converts them to the same transcript form.

Text needs the model tokenizer and the `tokenizers` package (`pip install tokenizers`). Pass
`--tokenizer-json path/to/tokenizer.json`, or `--artifact out/qwen3_8_27b.ninfer` to read the
tokenizer embedded in a v3 artifact; only that resource is read, no weights. Token-id input needs
neither.

## Getting your own data

`ninfer-serve --request-log-jsonl` records counts and timings only (`src/serve/request_log.cpp`), not
prompts, outputs or token ids, so it cannot feed the simulator. Useful sources:

- Coding-agent transcripts: Claude Code session files with `--claude-code`, or an OpenCode
  session converted to `{"messages": [...]}` (OpenAI-style roles, `tool_calls`, `tool`). A
  transcript written by another model measures the workload's repetition (file rewrites,
  tool-call JSON, paths), not this model's exact text.
- This model's own outputs: point the agent at `ninfer-serve`, then convert its stored session to
  `messages`. This is the best estimate, because the replayed completion is the text the target
  would generate.

## Running

```bash
python -m tools.spec_sim sessions/*.jsonl --claude-code --artifact out/qwen3_8_27b.ninfer \
  --ngram-n 8,12,24 --caps 15,32,64 --json profiles/spec_sim.json
```

The table has one row per policy, `n` and cap: committed tokens per round, accepted drafts per
round, draft acceptance, the share of rounds whose draft (`>15 drf`) or accepted prefix
(`>15 acc`) exceeds 15 tokens, the share of committed tokens produced by such rounds
(`>15 tok`), projected tok/s and speedup over plain decode. A histogram of accepted drafts per
round follows. `--json` writes every summary, including width histograms.

Reading the result: widening verification past 16 pays only when cap 32/64 rows beat cap 15 for
the same policy by more than the implementation risk, which requires a substantial `>15 tok`
share. Compare `chain`/`select` against `mtp` for the gain of adding the pool to MTP at cap 15,
which needs no Op widening.

## Tests

```bash
python -m pytest tests/test_spec_sim.py
```
