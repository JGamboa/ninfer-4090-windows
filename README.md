# NInfer-4090 for Windows — with Ternary Bonsai 2 27B

A specialized C++20/CUDA inference engine for **one NVIDIA GeForce RTX 4090** (`sm_89`), built
natively on **Windows** (MSVC + CUDA, no WSL, no Docker) and on Linux. It runs two 27B models
of the same architecture:

- **[Prism ML Ternary Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)**
  — Qwen3.8-27B compressed to ternary weights {−1, 0, +1}. This branch adds it: **~141–178
  tok/s decode** (prompt-dependent) from a 6.6 GB artifact with image input. That is 1.8–2.3x the
  decode speed of Prism's own llama.cpp fork on the same card, at the same perplexity.
- **Qwen3.8-27B** (the official NInfer groupwise artifact), inherited from the upstream 4090
  port: 148.6 tok/s code decode, 262K context. See [Qwen3.8-27B on the RTX 4090](#qwen38-27b-on-the-rtx-4090).

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4090, 24 GB (`sm_89`). Other GPUs are not supported by this build. |
| OS | Windows 11 x64 (MSVC, CUDA 13.4 validated) or Linux |
| Models | Ternary Bonsai 2 27B (text, vision, MTP), Qwen3.8-27B |
| Serving | CLI, OpenAI- and Anthropic-compatible HTTP server |
| Engine | [Neroued/ninfer](https://github.com/Neroued/ninfer) lineage; see [credits](#upstream-and-credits) |

## Ternary Bonsai 2 27B

### Results on the RTX 4090

Same machine for every row: RTX 4090 at stock clocks, Core i9-13900K, Windows 11, driver
595.97, CUDA 13.4, display at 60 Hz. Greedy decoding. NInfer uses MTP speculative decoding with
2 draft tokens and `--lm-head-draft`. The Prism fork is llama.cpp build b10709
(`PrismML-Eng/llama.cpp`) on `Ternary-Bonsai-2-27B-PTQ1_0.gguf`.

| Measurement | NInfer (this branch) | Prism llama.cpp fork |
|---|---:|---:|
| Decode, short story (MTP) | **142 tok/s** | — |
| Decode, Python code (MTP) | **178 tok/s** | — |
| Decode, mean of six prompts (MTP) | **163 tok/s** | — |
| Decode, no speculation (`tg128`) | **101 tok/s** | 77 tok/s |
| Prefill (`pp512`) | **2,428 tok/s** | 1,363 tok/s |
| Perplexity, wikitext / code corpus | 8.087 / 1.895 | 8.178 / 1.899 |
| Weights in VRAM, text only | 5.52 GiB | 5.53 GiB |
| Weights in VRAM with the MTP head | ~6.5 GiB | — |
| Vision tower | 22 ms per image | via `--mmproj` |

- The decode rows depend on the text: MTP drafts are accepted more often in predictable output
  (code, math: 170–178 tok/s) than in free prose (141–142 tok/s).
- Perplexity: wikitext and code are the two corpora where both tools score comparable text.
  NInfer's quick run over four corpora gives an overall 5.8556 (Qwen3.8-27B Q4/Q5: 4.80).
- The fork's prefill figure uses the PTQ1_0 packing; Prism's model card says its PQ2_0 packing
  processes prompts faster, so part of the prefill gap is the file format, not the engine.
- The RTX 4090 in these measurements also drives a 4K desktop. See
  [When the 4090 also drives the display](#when-the-4090-also-drives-the-display).

### Why it is fast

At one token per step, decode reads every weight once per token, so the speed is set by memory
bandwidth. Without speculation NInfer reads the same bytes as the fork (both store about 1.75
bits per weight) and decodes 101 against 77 tok/s. With speculation the difference grows, for
three reasons:

1. **MTP speculative decoding.** The Bonsai GGUF has no MTP head, so the converter copies the one
   from Qwen3.8-27B (same architecture). Each round, the MTP layer proposes two tokens and the
   ternary model verifies three positions while reading its weights once. A round costs about
   13 ms and yields about 1.8 tokens.
2. **Kernels specialized for this model and this card.** The ternary weights are stored as
   scaled base 3, five weights per byte, with one FP16 scale per 128 weights (`t5_g128_fp16`).
   A few integer operations turn four bytes into `dp4a` operands. Decode quantizes activations
   to int8 and multiplies with `dp4a`; prefill uses int8 tensor cores. Prism's Hadamard rotation
   is fused into the activation quantization, and the whole decode round replays as one CUDA
   Graph.
3. **Prefill in large tiles.** The int8 tensor-core GEMM reads the weights once per 64 tokens.

The costs: about 1 GB of VRAM for the MTP head, and an MTP head that was trained for Qwen3.8, not
for Bonsai, so acceptance varies with the content.

### Quick start (Windows)

**1. Build.** Requirements, vcpkg setup and details are in [WINDOWS_PORT.md](WINDOWS_PORT.md).
From a `vcvars64` shell with CUDA on `PATH`:

```bat
git clone -b feat/bonsai-ternary https://github.com/JGamboa/ninfer-4090-windows
cd ninfer-4090-windows
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_TOOLCHAIN_FILE=<path>/vcpkg/scripts/buildsystems/vcpkg.cmake ^
  -DVCPKG_TARGET_TRIPLET=x64-windows -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j
```

**2. Download** into `E:\LLM` (any folder works; adjust the paths below):

- `Ternary-Bonsai-2-27B-PTQ1_0.gguf` and, for images, `Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf`
  from [prism-ml/Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf);
- `qwen3_8_27b.ninfer` from [neroued/Qwen3.8-27B-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-NInfer).
  The converter takes the tokenizer, chat template and configuration from it, and copies its MTP head.

**3. Convert** to a `.ninfer` artifact. The converter is Python (3.11 or 3.12) with NumPy,
safetensors and PyTorch. The conversion takes a few minutes with `--device cuda`:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install numpy safetensors
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m tools.convert.bonsai_base --gguf E:\LLM\Ternary-Bonsai-2-27B-PTQ1_0.gguf ^
  --reference E:\LLM\qwen3_8_27b.ninfer ^
  --mmproj E:\LLM\Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf --out E:\LLM\bonsai2-27b-vl
python -m tools.convert --model E:\LLM\bonsai2-27b-vl --recipe bonsai2_27b ^
  --components text,vision,mtp --source gguf=E:\LLM\Ternary-Bonsai-2-27B-PTQ1_0.gguf ^
  --source mmproj=E:\LLM\Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf ^
  --source mtp=E:\LLM\qwen3_8_27b.ninfer --proposal --name bonsai2-27b ^
  --out E:\LLM\bonsai2_27b_vl.ninfer --device cuda
```

Leave out `--mmproj`, `vision` and `--source mmproj=...` for a text-only artifact. The
[conversion guide](docs/maintainer/bonsai-ternary-conversion.md) lists every tensor mapping
and a mapping check you can run before converting.

**4. Run.**

```bat
build\apps\ninfer.exe E:\LLM\bonsai2_27b_vl.ninfer --prompt "Write a short story about a lighthouse keeper." ^
  --max-context 4096 --max-new 512 --greedy --spec mtp --draft-tokens 2 --lm-head-draft

build\apps\ninfer-serve.exe E:\LLM\bonsai2_27b_vl.ninfer --host 127.0.0.1 --port 8080 ^
  --max-context 32768 --spec mtp --draft-tokens 2 --lm-head-draft --vision
```

The model thinks by default. Prism recommends `temperature 1.0, top_p 0.95, top_k 20,
min_p 0.05` in thinking mode. `--draft-tokens 3` is faster on code and math (up to +18 %) and
slower on prose (about −10 %); 2 is the better default. `ninfer.exe --help` and
[docs/cli.md](docs/cli.md) cover images, reasoning effort and the rest of the options.

### What was built for Bonsai

- **Converter** (`tools/convert`): a Prism GGUF reader that presents the model under the
  Hugging Face Qwen3.8 names. It handles the PTQ1_0 and PQ2_0 packings, the GDN head order,
  the norm offsets and the Hadamard sign vectors. The `bonsai2_27b` recipe keeps Prism's
  ternary codes bit-exact, including the output head and the token embedding. A reader for
  Prism's Qwen3-VL `mmproj` adds the vision tower (the Qwen3.8 tower, confirmed by
  `bonsai_vision_check`).
- **Format** `t5_g128_fp16`: scaled base-3 ternary codes (13 bytes per 64 weights), one FP16
  scale per 128 weights, rows of fused projections in one parent object. The converter repacks
  Prism's trits exactly; perplexity equals that of an earlier 2-bit layout to every printed digit.
- **Kernels** (`src/ops/linear/t5`): an int8 activation path (`dp4a` GEMV for decode,
  `m16n8k32` tensor-core GEMM for prefill), the Prism Hadamard rotation fused into activation
  quantization, and a ternary embedding gather with the inverse rotation. Each kernel is tested
  against an independent FP64 oracle (`ninfer_linear_t5_test`) and benchmarked with
  `ninfer_t5_bench`.
- **Runtime**: every ternary weight carries its Hadamard sign vector, so the model code passes
  ordinary activations. The loader checks that the artifact's rotation metadata matches the
  formats it binds.

The design, every measurement and the reasoning behind each decision are in the
[Bonsai design notes](docs/maintainer/bonsai-ternary-design.md) (section 9).

### Limits

- Bonsai is measured at up to 4K context on this branch; longer contexts use the same
  attention and KV code as Qwen3.8 but were not re-measured with Bonsai.
- DFlash2 with the full ternary output head is not supported. Use MTP.
- Converting requires the Qwen3.8-27B `.ninfer` artifact (tokenizer, configuration, MTP head).
- The rest of the engine's limits apply: one process, one GPU, one resident model.

## When the 4090 also drives the display

If the RTX 4090 also drives your monitor, the Windows desktop compositor takes the GPU from CUDA
on every display frame, and decode slows down. With a 3840x2160 desktop this cost 18 % of each
MTP decode round at 120 Hz and 15 % at 60 Hz. For the best decode speed:

- connect the monitor to the motherboard (integrated graphics) or to another GPU, so the 4090
  renders nothing;
- otherwise lower the refresh rate to 60 Hz and keep animated windows (browsers, video, chat
  apps) still while generating.

Compare tok/s figures only between runs taken with the same display setup.

## Qwen3.8-27B on the RTX 4090

This section and its measurements come from the Linux 4090 port this repository builds on
([sergiuszm/ninfer-4090](https://github.com/sergiuszm/ninfer-4090)); the native Windows build
reproduces the headline decode figure (149 tok/s, see [WINDOWS_PORT.md](WINDOWS_PORT.md)).

NInfer-4090 runs **Qwen3.8-27B** on one 24 GB NVIDIA GeForce RTX 4090. It is an `sm_89` port of
[NInfer-3090](https://github.com/Don-Chad/ninfer-3090), which derives from
[Neroued/ninfer](https://github.com/Neroued/ninfer). The engine loads the official groupwise
`.ninfer` artifact, serves OpenAI- and Anthropic-compatible APIs, and supports paged KV,
compatible-prefix reuse, CUDA Graphs, MTP speculative decoding, reasoning-effort control, and
ReplaySSM state transactions. Blackwell-only NVFP4/W4A4 execution is unavailable on `sm_89`;
the engine uses the same groupwise-int path as the 3090 base.

### Measured results on the RTX 4090

Conditions: single request, greedy decoding, CUDA Graphs on, INT8 KV, `--prefill-chunk 1024`,
official 16.96 GiB Qwen3.8-27B artifact. The code-generation decode row and the prefill rows
are measured from the `ninfer-serve` `/metrics` counters (computed prefill only); the other
decode rows use the `ninfer` CLI.

| Test | Result |
|---|---|
| Decode, code generation, MTP3 | **148.6 tok/s** at 81.0% draft acceptance |
| Decode, bench corpus, MTP3 | 106.5 tok/s at 48.7% acceptance |
| Decode, no speculation | 50.5 tok/s |
| Decode at 128K depth, no speculation | 39.6 tok/s |
| 64K needle-in-a-haystack | exact answer, 1,849 tok/s prefill |
| 128K needle-in-a-haystack | exact answer, 1,561 tok/s prefill |
| Vision, chart reading | 3 of 3 oracle facts, 22 ms vision tower |
| Ops test suite | 78 of 78 runnable tests pass on `sm_89` |

MTP acceptance, and with it the decoded rate, tracks how predictable the output is: structured
code accepts about 81% of draft tokens, the mixed bench corpus about 49%.

The shipping default has since moved from INT8 KV to the E8 4-bit KV mode, which serves the
model's full native 262,144-token context on this card. Retrieval stays exact through 260K
(single-needle, 5-needle, and exact-code-detail probes), MTP acceptance at depth is unchanged,
and the costs against the INT8 numbers above are a 5.7% decode tax and 1-2% of prefill; see
[Quick start](#text-only-full-262k-native-context-e8-4-bit-kv-default) for the measured deltas.

For scale: llama.cpp on the same card decodes the Qwen3.8-27B `UD-Q4_K_XL` GGUF at about
46 tok/s in a 144K-context configuration where the MTP buffers do not fit. The upstream engine
on an RTX 5090 measures 172 tok/s on the same code-generation prompts with a 400 W power cap
(the upstream README quotes about 200), so this card lands within 14% of it under MTP.

#### Depth sweep against llama.cpp

Both engines were measured on the same card. llama.cpp build 10358 ran `llama bench` on the
`UD-Q4_K_XL` GGUF (16.68 GiB) with q8_0 KV cache, flash attention, and `-ub 1024 -b 4096`,
which matches its deployed configuration, on 2026-08-15. The NInfer side was re-measured on
2026-08-17 on the deployed E8 262K configuration through the `/metrics` counters; the
llama.cpp configuration did not change between the dates. Two caveats: the artifacts differ
by about 2% in size, and `llama bench` is a bare kernel loop while the NInfer numbers
include the full server path.

Marginal rates at depth:

| Depth | llama.cpp pp2048 | llama.cpp tg32 | NInfer decode, no speculation |
|---:|---:|---:|---:|
| 0 | 3,024 tok/s | 45.9 tok/s | 50.4 tok/s |
| 32K | 2,327 | 42.0 | - |
| 64K | 1,866 | 38.6 | - |
| 128K | 1,336 | 33.1 | 42.1 |
| 256K | no entry | no entry | 36.6 |

Wall time to prefill one full prompt (llama.cpp integrated from the marginal rates, NInfer
measured):

| Prompt | llama.cpp | NInfer |
|---:|---:|---:|
| 32K | 12.5 s (2,630 tok/s) | 14.5 s (2,027 tok/s) |
| 64K | 28.3 s (2,317 tok/s) | 31.7 s (1,857 tok/s) |
| 128K | 70.4 s (1,862 tok/s) | 74.5 s (1,581 tok/s) |
| 192K | no entry | 127.9 s (1,381 tok/s) |
| 256K | no entry | 191.7 s (1,228 tok/s) |

The llama.cpp prefill lead narrows with depth. Server-measured, it prefills a 64K prompt in
28.7 s against 31.7 s (a 10% lead) and a 128K prompt in 71.6 s against 74.5 s (4%); the
server path costs llama.cpp 2-4% over the bare-loop estimates above. Everything past its
144K ceiling is NInfer-only. Decode inverts the shallow picture. NInfer leads by 10%
shallow and by 27% at 128K without speculation, and the MTP3 gap grows with depth:

| Workload | llama.cpp `draft-mtp` | NInfer MTP3 (E8) |
|---|---:|---:|
| Code, shallow | 118.8 tok/s at 85.9% acceptance | 142.9 tok/s at 78.0% |
| Prose, 64K depth | 55.5 tok/s at 45.3% | 86.1 tok/s at 42.3% |
| Prose, 128K depth | 42.3 tok/s at 45.4% | 77.5 tok/s at 41.6% |
| Prose, 256K depth | no entry | 65.4 tok/s at 41.1% |
| Code, 256K depth | no entry | 91.2 tok/s at 72.1% |

The NInfer rows in this table use the 2026-08-17 generated corpora; acceptance on them runs
a few points below the 2026-08-15 payloads (code 78% against 81%), which accounts for the
difference from the headline 148.6 tok/s. The llama.cpp MTP rows required a reduced
131,584-token context; the draft buffers push VRAM
to 23.8 of 24 GiB, and the deployed 144K llama.cpp configuration cannot fit them at all.
NInfer serves 172,032 tokens with MTP in the same VRAM at INT8 KV, and the full native
262,144 with the E8 4-bit KV default. Acceptance matches per content type, so the decode gap
is engine time, not draft quality.

Full configurations, method, and raw numbers:
[NInfer against llama.cpp](docs/llamacpp-comparison.md).

### Quick start (Linux)

Requirements: an RTX 4090, a recent NVIDIA driver, Docker with the NVIDIA Container Toolkit.

Build the image and download the model once:

```bash
docker build --tag ninfer-4090:sm89 .
NINFER_MODEL_DIR="$PWD/models" bash scripts/download-qwen38.sh
```

Then start one of the three profiles. The API is available at `http://127.0.0.1:8080/v1`.

The profiles as written run one generation slot. `--max-concurrency 2` is measured
and worthwhile on the 4090: the second lane costs about 390 MiB (state pools plus a
doubled CUDA-graph allowance) while the KV page pool stays shared, so a lone session
still uses the full context; single-stream decode is unregressed and two sessions
decode batched at roughly 1.5x aggregate throughput, each lane keeping its own
resident prefix. Prefill still serializes across lanes, so a deep cold prefill
delays the other lane's first token.

When clients edit recent conversation history (a re-serialized reply, tool results
folded into the previous turn, an updated agent memory block), the server proposes a
private long anchor at each of the last N message boundaries of every prompt
(`--auto-long-anchors N`, on by default at the `--max-long-anchors-per-continuation`
cap of 2) and re-prefills from the anchor below the edit instead of from zero.
Coverage reaches back exactly as many message boundaries as the retention cap:
when the anchor set is full the shallowest is evicted, so an edit deeper than the
cap still re-prefills from zero. Raise `--max-long-anchors-per-continuation` and
`--auto-long-anchors` together for deeper reach; each retained anchor holds one GDN
state image (about 147 MiB of host memory on Qwen3.8-27B), so size `--host-state-slots`
for `continuations x (2 + anchors)`. The old `--turn-checkpoints` ring is retired
and ignored; see [docs/turn-checkpoint-ring.md](docs/turn-checkpoint-ring.md).

Extra requests beyond the slots wait in the admission queue, and the queue deadline
defaults to 30 seconds. A deep prefill can hold a slot longer than that, so
parallel agent clients would fail with `request_queue_timeout`. The
`--pending-timeout-ms 600000` line raises the deadline to 10 minutes. On a
streaming request the timeout arrives as an in-band SSE error event after HTTP 200;
a client that does not parse error events sees a stream that ends without a
`finish_reason`. See [docs/serving.md](docs/serving.md) for the full queue
contract.

#### Text-only, full 262K native context (E8 4-bit KV, default)

The E8 Conway-Sloane lattice KV mode (`rk4v4-e8`, ported from
[UDPSendToFailed/ninfer-4090](https://github.com/UDPSendToFailed/ninfer-4090); see
[the fork comparison](docs/udp-fork-comparison.md)) fits the model's entire native
262,144-token context on 24 GB with 1.4 GiB to spare:

```bash
docker run --rm --gpus all --publish 8080:8080 \
  --volume "$PWD/models:/workspace/models:ro" \
  ninfer-4090:sm89 \
  ninfer-serve models/qwen3_8_27b.ninfer \
  --host 0.0.0.0 --port 8080 \
  --max-context 262144 --kv-capacity 262144 \
  --max-concurrency 1 --max-pending-requests 16 \
  --pending-timeout-ms 600000 \
  --prefill-chunk 1024 --kv-dtype rk4v4-e8 \
  --spec mtp --draft-tokens 3 --lm-head-draft \
  --preserve-thinking
```

Measured against INT8 KV on this build: identical MTP acceptance at 111K depth
(78.8% vs 78.4%), a 5.7% decode tax (126.6 vs 134.2 tok/s on a shallow greedy code
probe), prefill within 1-2% at matched depth, and exact single-needle, 5-needle, and
code-detail retrieval through 260K tokens.

#### Text-only, 168K context (INT8 KV, maximum precision)

```bash
docker run --rm --gpus all --publish 8080:8080 \
  --volume "$PWD/models:/workspace/models:ro" \
  ninfer-4090:sm89 \
  ninfer-serve models/qwen3_8_27b.ninfer \
  --host 0.0.0.0 --port 8080 \
  --max-context 172032 --kv-capacity 172032 \
  --max-concurrency 1 --max-pending-requests 16 \
  --pending-timeout-ms 600000 \
  --prefill-chunk 1024 --kv-dtype int8 \
  --spec mtp --draft-tokens 3 --lm-head-draft \
  --preserve-thinking
```

#### With vision, full 262K context (E8 4-bit KV)

The vision scratchpad defaults to 8192 tokens (`--vision-max-tokens`, ported from
the same fork as the E8 KV modes) instead of the former hardcoded 32768. The
smaller scratchpad frees about 1.5 GiB, so the full native context fits next to
vision on 4-bit keys:

```bash
docker run --rm --gpus all --publish 8080:8080 \
  --volume "$PWD/models:/workspace/models:ro" \
  ninfer-4090:sm89 \
  ninfer-serve models/qwen3_8_27b.ninfer \
  --host 0.0.0.0 --port 8080 \
  --max-context 262144 --kv-capacity 262144 \
  --max-concurrency 1 --max-pending-requests 16 \
  --pending-timeout-ms 600000 \
  --prefill-chunk 1024 --kv-dtype rk4v4-e8 \
  --spec mtp --draft-tokens 3 --lm-head-draft \
  --vision --preserve-thinking
```

The scratchpad bounds the image tokens per request, not the conversation depth:
a 51K-token conversation with an attached image completes normally. One
1024x1024 image costs 1026 vision tokens, so the default fits about seven
maximum-size images per request. The server rejects a request over the limit
with `media_budget_exceeded` before the request reaches the encoder. For dense
video workloads, raise the limit with `--vision-max-tokens`. Each additional
1024 tokens of scratchpad costs about 62 MiB of VRAM.

#### The tradeoff

KV precision, vision, and maximum context trade against each other on a 24 GB card:

| Profile | KV mode | Context | KV runtime | Startup slack |
|---|---|---:|---:|---:|
| Text-only, MTP3 | `rk4v4-e8` | 262144 (256K) | 5.08 GiB | 1.37 GiB |
| Text-only, MTP3 | `rk2v4-e8` | 262144 (256K) | 4.01 GiB | 2.43 GiB |
| Text-only, MTP3 | `int8` | 172032 (168K) | 6.31 GiB | 136 MiB |
| With `--vision`, MTP3 | `rk4v4-e8` | 262144 (256K) | 5.41 GiB | 780 MiB |
| With `--vision` (32K scratchpad), MTP3 | `rk2v4-e8` | 262144 (256K) | 5.85 GiB | 329 MiB |
| With `--vision` (32K scratchpad), MTP3 | `rk4v4-e8` | 212992 (208K) | 6.06 GiB | 108 MiB |
| With `--vision` (32K scratchpad), MTP3 | `int8` | 98304 (96K) | - | ~1 GiB |

262,144 is the model's own context limit, so `rk2v4-e8` (2-bit keys, 96.2% cosine)
buys no additional context over `rk4v4-e8` in the text-only profile - only slack.
That slack is what pays for vision. With the former hardcoded 32,768-token vision
scratchpad, vision cost about 2.1 GiB (1.83 GiB of runtime buffers plus a
0.28 GiB tower): INT8 could only afford it at 96K, 4-bit keys topped out at
212992, and only 2-bit keys fit the full 262,144. The default 8192-token
scratchpad cuts the cost to about 0.6 GiB, and the full native 262,144 now fits
alongside vision on 4-bit keys with 780 MiB of slack. The vision
modes answer a two-swatch color oracle exactly at temperature 0, including with
the image buried under 52,700 tokens of text on `rk2v4-e8`. `rk2v4-e8` also passes
the text retrieval gates (single-needle at 260K, 5-needle at 118K, exact code
details at 168K) at a 10% decode tax (120.5 tok/s on the shallow code probe). The
INT8 text-only ceiling is near 176K: 172032 starts, and 196608 is rejected at
startup with a byte-exact deficit. The server validates memory before it listens,
so an oversized context fails fast instead of at request time.

For a native build, follow the [Linux build guide](docs/rtx-3090-linux.md) with
`CMAKE_CUDA_ARCHITECTURES=89` (the default in this fork). The build requires CUDA 12.8 or newer,
GCC 13, and CMake 3.28 or newer; the Docker image builds with CUDA 13.1.

Tests and benchmarks are excluded from the default build. `cmake --preset release` configures
the same product build; `cmake --preset dev` also enables tests and benchmarks and finds a
Python 3 interpreter. Both presets use `build/` and explicitly reset the build options.
Machine-specific compiler and Python paths belong in the ignored `CMakeUserPresets.json`. See
[build organization and configuration](docs/maintainer/build-system.md) for details.

There is no install target or packaged binary distribution; run NInfer from its source build tree.
Python tools run independently of CMake; the standalone HBM probe has its own
[build command](tools/README.md#standalone-hbm-probe).

### Documentation

- [Documentation index](docs/README.md)
- [Ternary Bonsai design notes and measurements](docs/maintainer/bonsai-ternary-design.md)
- [Ternary Bonsai conversion guide](docs/maintainer/bonsai-ternary-conversion.md)
- [Native Windows port](WINDOWS_PORT.md)
- [CLI](docs/cli.md)
- [HTTP serving](docs/serving.md)
- [Performance](docs/performance.md)
- [Perplexity evaluation](docs/perplexity.md)
- [Weight conversion and custom recipes](docs/weight-conversion.md)
- [Resource scheduling and context cache](docs/maintainer/resource-scheduling-and-context-cache.md)
- [Serve TTFT benchmark](tools/bench/ttft/)
- [CLI examples](examples/cli/)
- [Contributing](CONTRIBUTING.md)

### What this fork changes

- **`sm_89` retarget.** The CMake architecture pin, the runtime compute-capability check, and the
  NVFP4 stub gate now select `sm_89`. Most SM86 kernel schedules run unmodified on Ada; the
  INT8 attention prefill schedule is retuned (below).
- **Ada-retuned INT8 attention prefill.** The SM120 schedule spills registers on Ada and pays the
  consumer half-rate penalty for f32-accumulate HMMA. Arch-gated for `sm_89`: the full
  128-register budget, eight paired producer warps over `Bc` column halves with one named-barrier
  exchange per key tile, byte-permute V dequantization (bit-identical), and fp16-accumulated PV
  tiles folded into the fp32 running accumulator each tile. The kernel gains 30% at 64K depth
  (109 to 143 TFLOP/s on the `d256-h24-kv4` INT8 append shape); serve prefill gains 5-7% at
  88K-128K. Needle-in-a-haystack retrieval stays exact at both depths and all 84 suite tests
  pass, which bounds the fp16-accumulation numerics change.
- **Causal-tile partitioned key-block traversal.** Interior key blocks (wholly below the causal
  diagonal for the whole CTA tile) run a separate instantiation of the key-block body: KV stages
  with unconditional copies and the softmax drops its masking selects; boundary blocks keep the
  exact masked path. The idea comes from the
  [UDPSendToFailed fork](https://github.com/UDPSendToFailed/ninfer-4090) (c5f70526),
  re-implemented inside the retuned schedule above. Kernel: 144 to 165 TFLOP/s at 32K-224K
  context on the INT8 append shape (-12 to -13% latency), register count unchanged, bit-exact.
  End-to-end this is bounded by the attention wall share of this hybrid-GDN model: about +1%
  serve prefill at 51K on INT8 KV, within noise on the E8 modes, whose staging time is dominated
  by lattice decode rather than the removed guards.
- **`/v1/models` reports `context_window`.** Clients without access to a llama.cpp `/props` or a
  vLLM `max_model_len` can size prompts from the models payload.
- **llama.cpp-compatible `timings` on chat completions.** Responses and final stream chunks carry
  a top-level `timings` block (`prompt_n`/`predicted_n`, per-second rates, `ttft_ms`, `cache_n`,
  `draft_n`/`draft_n_accepted`), so proxies such as llama-swap show per-request prefill and decode
  rates, MTP draft acceptance, and prefix-cache hits. Contributed by the
  [shantanusingh16 fork](https://github.com/shantanusingh16/ninfer-4090) of this repository.
- **`GET /metrics`.** Prometheus counters under llama.cpp-compatible names
  (`llamacpp:prompt_tokens_total`, `llamacpp:prompt_seconds_total`,
  `llamacpp:tokens_predicted_total`, `llamacpp:tokens_predicted_seconds_total`,
  `llamacpp:requests_processing`, `llamacpp:requests_deferred`), so existing scrapers read this
  server without changes. Prompt tokens count only computed prefill; prefix-cache hits are
  excluded, as in llama.cpp. Additional `ninfer:` series report request totals, prefix-cache
  hits, and MTP draft/acceptance totals.
- **`GET /slots`.** A llama.cpp-shaped slot table read from the engine's real lane state: busy
  slots report their request's prompt and reused-prefix sizes, idle retained slots report the
  resident session's depth and its identifying `session_digest`. Truthful per-slot attribution
  holds at any `--max-concurrency`.
- **Slot session save/restore.** `--slot-save-path DIR` (off by default) enables llama.cpp-style
  `POST /slots/{id}?action=save|restore|erase`: one idle slot's complete resident session -
  paged Text and MTP KV, GDN linear-attention state, rewrite checkpoint, long anchors, and
  prefix identity - moves to or from disk, and a restored slot reuses the cache across server
  restarts instead of re-prefilling (a 6.9k-token session restores in about 0.1 s against a
  multi-second reprefill).
  Sessions are identified by a stable `session_digest`; chat completions carry `id_slot` and the
  digest next to `timings`, and `save`/`erase` accept an `if_digest` precondition checked
  atomically, so a client always persists exactly the session it means. A restored session is
  reusable from its endpoint, its rewrite checkpoint, or any retained long anchor; the GDN
  state cannot rewind below the deepest retained checkpoint, and the DFlash backend is not
  supported. Details in [docs/serving.md](docs/serving.md).
- **Reuse-aware lane choice.** When prefix reuse ties (typically zero for a fresh session),
  admission picks the lane whose occupation costs least to replace - an empty lane before any
  retained session, then the shallowest - so a burst request no longer evicts a deep resident
  session while a free lane exists.
- **Automatic long anchors.** `--auto-long-anchors N` (default: the
  `--max-long-anchors-per-continuation` cap) has the server propose a private long anchor at
  each of the last N message boundaries of every prompt. Upstream's long anchors exist only
  where a client places an explicit `PrivateLongAnchor` marker, which no OpenAI or Anthropic
  request can express, so without this flag a rewrite deeper than the last assistant reply
  has no reuse candidate at all and re-prefills from token zero. With it, the prompt restores
  at the anchor below the edit. A full anchor set replaces its shallowest entry, so the
  retained anchors track the most recent boundaries and coverage reaches back exactly the
  cap: an edit deeper than `--max-long-anchors-per-continuation` boundaries still re-prefills
  from zero, so raise the cap and this flag together (and `--host-state-slots` with them) for
  deeper history. Measured on agent traffic, 94% of consecutive prompts are pure appends and
  96.5% of the remaining history rewrites are two messages deep or less, so the default cap of
  2 covers 99.8% of turns. The rare deeper edits replace the whole history from message one or
  two, where no anchor can help. Anchors ride the existing catalog, pressure planner and slot
  snapshots. This replaces the retired `--turn-checkpoints` ring
  ([docs/turn-checkpoint-ring.md](docs/turn-checkpoint-ring.md)).
- **Auto-save on eviction.** `--auto-save-evicted` (off by default, requires
  `--slot-save-path`) spills an involuntarily evicted session - endpoint, rewrite checkpoint
  and long anchors included - back to the slot file it was last saved to or restored from,
  before the eviction destroys it. Rotating more sessions than slots then loses nothing: the
  next restore recovers the session at its latest frontier. Explicit `erase` never auto-saves.
  Two rules keep a spill from losing data. First, a slot file is bound to at most one slot at
  a time: the most recent `save` or `restore` of a path owns it, and every other slot that held
  the same path is unbound. A stale copy of a session, left behind when a restore retains its
  source, therefore cannot write the file when it is evicted. Second, a spill never rolls a
  file back: a spill with fewer tokens than the file already holds is refused and logged as
  `slot auto-save SKIPPED`, while an explicit `save` always wins. Without these rules a
  two-day-old copy of a live session once overwrote its 78k-token file, and the client resumed
  the rolled-back state.
- **Planner diagnostics in the request JSONL.** `--request-log-jsonl FILE` records, per
  request, the reuse path the planner chose (`prefix_reuse_path`), the prefix tokens it reused,
  and the materialization search behind the choice: `stop_reason`, `budget_exhausted`,
  `selected_maximal_fallback`, `targets_evaluated`, and `best_reuse_prompt_tokens`, the most
  reuse any candidate offered. That last field separates the two causes of a cold prefill. A
  value of `0` means that no reuse candidate existed, so the cause sits upstream of the planner:
  a missing anchor or a changed prefix. A large value beside `prefix_reuse_path=root` means
  that a candidate existed and the planner rejected it, which points at the search itself.
  `/metrics` carries only the llama.cpp-compatible subset, so this file is the only place these
  fields appear. Field reference in [docs/serving.md](docs/serving.md).
- **NVFP4-A4 test gating.** The A4 activation tests skip on hardware without FP4 tensor cores
  instead of aborting. The full remaining suite passes on the RTX 4090.
- **E8 lattice KV quantization (ported).** The `rk8v4`/`rk4v4`/`rk4v4-e8`/`rk2v4-e8` KV modes
  and the 262K-to-1M visible-keys envelope lift from the
  [UDPSendToFailed/ninfer-4090](https://github.com/UDPSendToFailed/ninfer-4090) sibling fork,
  merged under this fork's retuned `sm_89` attention prefill schedule. The E8 codec verifies
  bit-exactly against the upstream microbenchmark (96.155% / 98.678% cosine); their 1 GiB
  CUDA-graph allowance bump was deliberately not taken (it would evict the INT8 168K profile).
  Method and measurements in [docs/udp-fork-comparison.md](docs/udp-fork-comparison.md).
- **Configurable vision scratchpad (ported).** `--vision-max-tokens` comes from the same fork
  and sizes the vision encode workspace (default 8192 tokens, formerly hardcoded 32768). This
  fork additionally wires the processor media budget to the same limit, so an over-limit
  request fails as `media_budget_exceeded` instead of reaching an undersized encoder.

### Known limits on the RTX 4090

- Prefill trails llama.cpp by 16-24% on full 32K-128K prompts under matched conditions (see
  the depth sweep above). The rate is flat across `--prefill-chunk` 1024 to 2688, so the
  chunk size is not the lever. With the attention schedule retuned, the remaining gap sits in
  the custom quantized GEMMs, which run about 10% below cuBLAS on Ada. Decode is where this
  engine leads.
- Keep `--prefill-chunk` at 2688 or below. This fork carries measured `sm_89` cooperative
  residency tables (the former hard abort above chunk 1024 is fixed), and chunks through 2688
  stay on split-K. Larger chunks route to the unsplit schedule, which is marginally less
  accurate at its onset (about 1e-5 relative).
- `--max-concurrency 2` is measured on the 4090 (see Quick start); higher lane counts are
  untested here, and the published cohort results in the
  [3090 base](https://github.com/Don-Chad/ninfer-3090) do not transfer directly.
- Prefill is strictly serialized across lanes with no chunk-level interleaving, and decode
  starves while any prefill runs: a short request submitted behind a 31k-token cold prefill
  measured a 13.5 s first token. Concurrency pays off for decode and for per-lane resident
  prefixes, not for prefill fairness.
- The limits of the base engine apply: one process, one GPU, one model, bounded FIFO admission,
  no multi-GPU execution, no weight offload.

The product boundary stays intentionally small: one RTX 4090 and one resident model per Engine;
a startup-fixed capacity of one to eight active requests with bounded FIFO ingress; no request
preemption, priority/QoS, active-request swapping, weight offload, multi-GPU, or distributed
serving; one shared startup-fixed KV pool across active requests and retained prefixes; model
architectures and format/shape combinations use explicitly implemented native paths; parsed tool
calls are returned to the client, and NInfer does not execute tools; and the in-tree C++ headers
are not distributed as an installed SDK.

### Artifact

| Model | Artifact | Size |
|---|---|---:|
| Qwen3.8-27B | [official NInfer groupwise artifact](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | 16.96 GiB |

The artifact is architecture-independent; the model card's RTX 5090 requirement describes the
upstream engine, not the file. Verify the download against the SHA-256 published on the card.

The v3 `.ninfer` container carries model configuration, encoded weights, logical bindings and
frontend resources; the engine loads those facts against the implemented model and Op
capabilities. You can also [convert your own weights](docs/weight-conversion.md), reuse an
official recipe, or choose another supported mixture of formats. The engine requires v3
artifacts; existing official v2 downloads can be
[upgraded locally](docs/weight-conversion.md#upgrade-an-existing-v2-artifact) without downloading
the weights again.

### Reasoning effort

Qwen3.8-27B has three trained reasoning depths plus an off switch. OpenAI Chat Completions
accepts a top-level `reasoning_effort` field (`low`, `medium`, `xhigh`) and a top-level
`enable_thinking` boolean; hidden reasoning returns separately as `message.reasoning_content`.
Only those three levels are accepted, plus `none` to turn thinking off. `high`, `minimal`, and
`max` are rejected as `reasoning_effort_not_supported`, so a client that offers a `high` setting
must map it to `xhigh`. A token budget for reasoning is separate from the effort level. It is
set only through the Anthropic Messages path (`thinking.budget_tokens`) or server-wide with
`--default-thinking-budget N`; the OpenAI paths have no field for it. Without a budget,
reasoning is bounded only by the request's `max_tokens`, which is what the model card
recommends, and the `model_thinking_tokens` field of the request JSONL reads zero, because that
counter runs only under a budget. The `chat_template_kwargs` request field of llama.cpp is not
supported and is rejected. For the CLI, pass `--reasoning-effort` or `--no-thinking`. Sampling
defaults come from the model card and switch with the thinking mode: `temperature=1.0`,
`top_p=0.95`, `top_k=20` in thinking mode; `temperature=0.7`, `top_p=0.80`, `top_k=20`,
`presence_penalty=1.5` in non-thinking mode.

### Serving APIs

OpenAI Chat Completions, OpenAI Responses with streaming and local continuation state, Anthropic
Messages, prompt-rendered function tools with parsed tool calls, compatible-prefix reuse, and
JSONL request logs. See [HTTP serving](docs/serving.md) and [CLI usage](docs/cli.md).

## Upstream and credits

- [Prism ML](https://huggingface.co/prism-ml) - Ternary Bonsai 2 27B, its ternary packings and
  Hadamard rotation, and the [llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp) whose
  conversion and dequantization code defined the formats this branch reads.
- [fraserprice/bonsai-vllm](https://github.com/fraserprice/bonsai-vllm) - a CUDA reference for
  the Hadamard kernel and a ternary tensor-core GEMM.
- The Bonsai port (converter, ternary kernels, runtime integration, vision, tests and
  measurements) was developed with [Claude Code](https://claude.com/claude-code), with every
  kernel checked against FP64 oracles and every performance claim measured on the RTX 4090.
- [Neroued/ninfer](https://github.com/Neroued/ninfer) - the engine, developed for the RTX 5090
  (`sm_120a`).
- [Don-Chad/ninfer-3090](https://github.com/Don-Chad/ninfer-3090) - the SM86 compatibility layer,
  ReplaySSM integration, and Qwen3.8 runtime support this fork builds on. Its
  [v0.6.1 release notes](RELEASE_NOTES_0.6.1.md) describe the inherited state.
- [sergiuszm/ninfer-4090](https://github.com/sergiuszm/ninfer-4090) - the `sm_89` RTX 4090
  port (Ada-tuned attention prefill, serving features) this repository builds on; the native
  Windows (MSVC) port is described in [WINDOWS_PORT.md](WINDOWS_PORT.md).
- [UDPSendToFailed/ninfer-4090](https://github.com/UDPSendToFailed/ninfer-4090) - a sibling
  RTX 4090 port from the same 3090 base. The rotated and E8-lattice KV-cache quantization
  modes (`rk8v4`, `rk4v4`, `rk4v4-e8`, `rk2v4-e8`), the E8 codecs, and the 1M visible-keys
  envelope are their work, cherry-picked here with authorship preserved. The full 262K
  default profile exists because of it; see
  [the fork comparison](docs/udp-fork-comparison.md).
- [jram4/ninfer-4090](https://github.com/jram4/ninfer-4090) - an earlier RTX 4090 port of a July
  2026 snapshot. Its Ada dispatch tuning targets a kernel organization that upstream has since
  replaced, so this fork starts from the current 3090 base instead.

## Support

NInfer is a personal project that I develop out of interest. If you find it useful and would like
to support its continued development, you can [support the project on Ko-fi](https://ko-fi.com/neroued).

Support is entirely voluntary. It is not a purchase or investment and does not come with financial
returns, promised services or features, or a role in project decisions. The project's direction,
priorities, technical choices, and release schedule remain independently determined by the
maintainer.

## License

Apache License 2.0. See [LICENSE](LICENSE).
