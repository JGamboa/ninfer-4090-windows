# Task-level quality evaluation

`tools/eval` asks a running OpenAI-compatible chat server a fixed set of 45 tasks and checks each
answer deterministically. It shows how well a model handles the work it will actually be given
(coding-agent tool calls, structured output, code, reasoning, Spanish chat). Perplexity and
tokens per second do not show this. The suite is written for `ninfer-serve`, uses only the
Python 3.11 standard library, and sends non-streaming `POST /v1/chat/completions` requests.

| Category | Tasks | What passes |
|---|---|---|
| `tool_calling` | 9 | the right function and arguments, parallel calls, no call when none is needed, using a tool result |
| `json_output` | 5 | the answer's JSON matches a schema and the expected field values |
| `code_python` | 7 | the last fenced Python block passes hidden asserts in a subprocess (10 s limit) |
| `reasoning_math` | 8 | multi-step arithmetic, a debt ledger, and pointer chasing; the final number matches |
| `instruction_following` | 7 | word, bullet, and sentence limits, forbidden words, exact output, reply language |
| `spanish` | 6 | translation key terms, Spanish word problems, Spanish-only explanations, spelling fixes |
| `long_context` | 3 | recalls a vault code planted in a generated ~4K/16K/32K-token ledger, ignoring six decoys |

List them with `python -m tools.eval list`. Tasks are JSON data in [`tasks/`](tasks/). The checker
kinds are `exact`, `contains`, `not_contains`, `regex`, `word_count`, `bullets`, `language`,
`number`, `json_schema`, `tool_call`, `python_exec`, `needle` and `all`, defined in
[`checkers.py`](checkers.py). A `<think>` block that leaks into `content` is removed before
checking. NInfer returns reasoning separately as `reasoning_content`, which the results file keeps
but no checker grades.

## Running on Windows (PowerShell)

Start one server at a time on port 8080, then run the suite from the repository root. Serve
Ternary Bonsai 2 27B first:

```powershell
.\build\apps\ninfer-serve.exe E:\models\bonsai_27b.ninfer --host 127.0.0.1 --port 8080
```

In a second terminal:

```powershell
python -m tools.eval run --base-url http://127.0.0.1:8080/v1 --label bonsai `
  --out E:\eval\results_bonsai.json --thinking off
```

Stop that server, start it again with the Qwen3.8-27B artifact, and run with a different label:

```powershell
python -m tools.eval run --base-url http://127.0.0.1:8080/v1 --label qwen-q4 `
  --out E:\eval\results_qwen.json --thinking off
python -m tools.eval compare E:\eval\results_bonsai.json E:\eval\results_qwen.json `
  --markdown E:\eval\compare.md
```

`compare` prints an overall table, a table per category, and a table per task. Each shows the pass
count, mean latency, mean completion tokens, and the mean of the server's
`timings.predicted_per_second`. After the tables it lists why every failed task failed. Each
`run` also prints the same report for its own results file.

Options for `run`:

- `--model` is the request `model`; it defaults to the only id listed at `/v1/models`.
- `--tasks` is a glob over task ids or categories, such as `tool.*`, `spanish`, or `long.needle_4k`.
- `--thinking on|off` sends `chat_template_kwargs.enable_thinking`. Leave it out to use the server
  default. With thinking on, raise `--max-tokens` (default 4096) so reasoning does not use up
  the answer budget. A failure caused by hitting the limit is marked `[output truncated ...]`.
- `--temperature` (default 0, greedy), `--seed` (default 1234, plus one per repeat) and
  `--repeats N` repeat the whole task list N times.
- `--timeout` is the per-request socket timeout in seconds (default 600).
- `--opik-project NAME` also logs each task as an [Opik](https://www.comet.com/docs/opik/) trace
  with a `passed` feedback score. It needs `pip install opik`; the suite does not need Opik
  otherwise.

For each attempt, the results JSON records whether it passed, the checker's explanation, the
content (truncated to 4000 characters), the reasoning (truncated to 2000 characters), the tool
calls, latency, `usage`, and `timings`. A failed HTTP request or a server error counts as a
failure and does not stop the run.

## Caveats

- One greedy sample per task only indicates quality; it is not a benchmark. With 5 to 9 tasks per
  category, a difference of one task is noise. Look for gaps across several categories, then
  read the failure details. To measure variance, use `--temperature 0.7 --repeats 3`.
- Checkers are deliberately strict about format, because agents such as Junie depend on it. A
  correct answer in the wrong shape, such as a missing `Answer:` line or prose around a code
  block, can still fail.
- NInfer does not accept JSON-constrained output (`response_format` of type `json_schema`), so
  the JSON tasks ask for JSON in the prompt only. They test whether the model follows that request
  unaided.
- `python_exec` runs model-written code on your machine with a timeout but no sandbox. Run the
  suite only against models you trust.
- The long-context sizes are approximate: 4 characters per token. The 32K task needs a server
  context of at least 34K tokens.
