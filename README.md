# NInfer-4090 for Windows

A C++20/CUDA inference engine specialized for **one NVIDIA GeForce RTX 4090** (`sm_89`), built
and run natively on **Windows 11** (MSVC + CUDA; no WSL, no Docker). It serves two 27B models of
the same architecture through a CLI and an OpenAI- and Anthropic-compatible HTTP server:

| Model | Artifact | Size | Decode (MTP) | Best case |
|---|---|---:|---:|---:|
| **Ternary Bonsai 2 27B** (Prism ML, ternary weights, text + vision) | [jgamboa/Ternary-Bonsai-2-27B-NInfer-4090](https://huggingface.co/jgamboa/Ternary-Bonsai-2-27B-NInfer-4090) | 6.4 GiB | **188 tok/s** | **532 tok/s** (MTP + n-gram, edit-style prompts) |
| **Qwen3.8-27B** (official NInfer groupwise Q4/Q5, text + vision) | [neroued/Qwen3.8-27B-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | 19.0 GiB | **107 tok/s** | **289 tok/s** (MTP + n-gram), **211 tok/s** (DFlash2, code) |

Every figure in this README was measured on the same RTX 4090 under Windows unless it says
otherwise; the conditions are next to each table. Both models run the full 262K-token context on
24 GB.

- [Quick start](#quick-start)
- [Performance](#performance)
- [Choosing settings](#choosing-settings)
- [Benchmarking and validation](#benchmarking-and-validation)
- [How it works](#how-it-works)
- [Converting models](#converting-models)
- [Serving](#serving)
- [Limits](#limits)
- [Documentation](#documentation)
- [Lineage and credits](#lineage-and-credits)

## Quick start

### Requirements

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4090, 24 GB (`sm_89`). The build targets this card only. |
| OS | Windows 11 x64 |
| Toolchain | Visual Studio Build Tools with MSVC (2026 validated), CUDA 12.8 or newer (13.4 validated), CMake 3.28+, Ninja |
| Libraries | [vcpkg](https://github.com/microsoft/vcpkg) (`curl`, `ffmpeg`, `pkgconf`, installed from the manifest) |
| Download tool | [`hf`](https://huggingface.co/docs/huggingface_hub/guides/cli) (`pip install -U huggingface_hub`) or a browser |

### 1. Build

From a "x64 Native Tools" prompt (or after running `vcvars64.bat`) with CUDA on `PATH`:

```bat
git clone -b feat/bonsai-ternary https://github.com/JGamboa/ninfer-4090-windows
cd ninfer-4090-windows
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake ^
  -DVCPKG_TARGET_TRIPLET=x64-windows -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j
```

This produces `build\apps\ninfer.exe` (CLI), `build\apps\ninfer-serve.exe` (HTTP server) and
`build\apps\ninfer-perplexity.exe`. Add `-DNINFER_BUILD_BENCHMARKS=ON -DBUILD_TESTING=ON` to also
build the benchmarks and tests. At runtime, put `build\vcpkg_installed\x64-windows\bin` and the
CUDA `bin` directory on `PATH`. Build details and the Windows-specific changes are in
[WINDOWS_PORT.md](WINDOWS_PORT.md).

### 2. Download a model

```bat
:: Ternary Bonsai 2 27B: text + vision + MTP head (6.4 GiB)
hf download jgamboa/Ternary-Bonsai-2-27B-NInfer-4090 bonsai2_27b_vl_mtp_q4q5.ninfer --local-dir E:\LLM

:: Qwen3.8-27B: text + vision + MTP + DFlash2 (19.0 GiB)
hf download neroued/Qwen3.8-27B-NInfer qwen3_8_27b.ninfer --local-dir E:\LLM
```

Any folder works; the examples below use `E:\LLM`. The Bonsai repository also has
`bonsai2_27b_vl.ninfer`, the same model with a Q8 MTP layer (3.5-4.8 % slower decode). Verify
each download against the checksums on its model card.

### 3. Run from the command line

```bat
build\apps\ninfer.exe E:\LLM\bonsai2_27b_vl_mtp_q4q5.ninfer ^
  --prompt "Write a Python function that merges two sorted lists." ^
  --max-context 8192 --max-new 1024 --spec mtp --draft-tokens 2 --lm-head-draft

build\apps\ninfer.exe E:\LLM\qwen3_8_27b.ninfer ^
  --prompt "Write a Python function that merges two sorted lists." ^
  --max-context 8192 --max-new 1024 --spec mtp --draft-tokens 3 --lm-head-draft --ngram chain
```

The CLI prints the answer, then a summary with prefill and decode speed, MTP acceptance and memory
use. Both models think by default; add `--no-thinking` or `--reasoning-effort low|medium|xhigh`.
Chat histories, images and video go through `--messages FILE.json` (examples in
[examples/cli/messages](examples/cli/messages)). `ninfer.exe --help` and [docs/cli.md](docs/cli.md)
list every option.

### 4. Run the server

Ternary Bonsai 2 27B, full 262K context per request, three concurrent requests, vision on:

```bat
build\apps\ninfer-serve.exe E:\LLM\bonsai2_27b_vl_mtp_q4q5.ninfer ^
  --host 127.0.0.1 --port 8080 --model-id bonsai-27b ^
  --max-context 262144 --kv-capacity auto --kv-dtype rk4v4-e8 --max-concurrency 3 ^
  --spec mtp --draft-tokens 2 --lm-head-draft --ngram chain --vision
```

Qwen3.8-27B, 100K context, three concurrent requests:

```bat
build\apps\ninfer-serve.exe E:\LLM\qwen3_8_27b.ninfer ^
  --host 127.0.0.1 --port 8080 --model-id qwen3.8-27b ^
  --max-context 100000 --kv-capacity 100000 --kv-dtype rk4v4-e8 --max-concurrency 3 ^
  --max-pending-requests 10 --pending-timeout-ms 600000 --prefill-chunk 1408 ^
  --spec mtp --draft-tokens 3 --lm-head-draft --ngram chain --preserve-thinking ^
  --device-state-slots 3 --host-state-slots 4 --host-kv-mib 4096
```

These are the configurations the author runs daily. The server checks the memory plan before it
listens, so a configuration that does not fit fails at startup, not at request time.

- API base URL: `http://127.0.0.1:8080/v1` (OpenAI Chat Completions and Responses) and
  `http://127.0.0.1:8080/v1/messages` (Anthropic Messages). Use `--host 0.0.0.0` to serve the
  local network and `--api-key KEY` to require a key.
- Live monitor: open `http://127.0.0.1:8080/monitor` in a browser for decode and prefill speed,
  draft acceptance, prefix-cache reuse, KV occupancy, slots and recent requests.
- Prometheus metrics at `/metrics`, a llama.cpp-style slot table at `/slots`.

A first request from PowerShell:

```powershell
$body = @{ model = "bonsai-27b"; max_tokens = 512
           messages = @(@{ role = "user"; content = "Explain CUDA graphs in three sentences." }) } | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri http://127.0.0.1:8080/v1/chat/completions -Method Post `
  -ContentType "application/json" -Body $body | Select-Object -ExpandProperty choices
```

Any OpenAI- or Anthropic-compatible client (OpenCode, Claude Code through a proxy, LiteLLM, Open
WebUI, the `openai` Python package) works with the same URL.

## Performance

Conditions for every table: RTX 4090 at stock clocks, Core i9-13900K, Windows 11, driver 595.97,
CUDA 13.4, greedy decoding unless noted. The same 4090 drives a 4K desktop at 60 Hz, which costs
about 15 % of decode ([below](#when-the-4090-also-drives-the-display)).

### Ternary Bonsai 2 27B

Against Prism's own llama.cpp fork (build b10709, `Ternary-Bonsai-2-27B-PTQ1_0.gguf`) on the same
machine:

| Measurement | NInfer | Prism llama.cpp fork |
|---|---:|---:|
| Decode, no speculation (`tg128`) | **101 tok/s** | 77 tok/s |
| Decode, MTP 2, mean of six mixed prompts | **188 tok/s** | — |
| Decode, MTP 2, prose / edit-style prompts | 167 / 250 tok/s | — |
| Decode, MTP 2 + n-gram, edit-style prompts | **532 tok/s** | — |
| Decode, three concurrent requests, aggregate | **360 tok/s** | — |
| Prefill, `pp512` / `pp2048` | **4,180-4,250 / 4,460-4,500 tok/s** | 1,363 tok/s / — |
| Prefill, 64K / 128K-token prompt (needle test, answer exact) | 18.6 s / 46.2 s | — |
| Perplexity, wikitext / code corpus | 8.087 / 1.895 | 8.178 / 1.899 |
| Task quality, 45 deterministic tasks (`tools/eval`) | 43/45 | — |
| Weights in VRAM (text / + vision) | 6.11 / 6.39 GiB | 5.53 GiB (text) |

- Decode depends on the text: drafts are accepted more often in predictable output. Edit-style
  prompts (return a file, a JSON list or a document with a small change) restate their input.
- Prism's model card says its PQ2_0 packing processes prompts faster than PTQ1_0, so part of the
  prefill gap is the file format.
- Sources: [Bonsai design notes](docs/maintainer/bonsai-ternary-design.md), section 9.1
  (items 14-25).

### Qwen3.8-27B

| Measurement | Result |
|---|---:|
| Decode, no speculation (`tg128`) | 47.0 tok/s (~95 % of the card's measured read bandwidth) |
| Decode, MTP 3, mean of six mixed prompts, thinking on | 107 tok/s |
| Decode, MTP 3, code prompt, thinking off (server) | 149 tok/s |
| Decode, DFlash2 draft 12, code prompt, thinking off | **211 tok/s** |
| Decode, MTP 3 + n-gram, edit-style prompts, thinking off | **289 tok/s** |
| Decode, MTP 3 + n-gram, 7K-11K-token file edits through the server, thinking on | 258 tok/s |
| Prefill, `pp512` / `pp2048`, official artifact / int8 artifact | 2,536 / 2,762 tok/s, **4,436 / 5,008 tok/s** |
| Prefill, 8K / 64K / 128K-token prompt, official artifact (needle test, answer exact) | 2.9 s / 27.4-27.6 s / 64.0 s |
| Prefill, 8K / 64K / 128K-token prompt, int8 artifact (needle test, answer exact) | **1.6 s / 17.2 s / 43.0 s** |
| Perplexity, quick four-corpus run, official / int8 artifact | 4.8007 / 4.7944 |
| Task quality, 45 deterministic tasks (`tools/eval`) | 44/45 (2026-09-25); 45/45 on both artifacts (2026-09-26) |

The **int8 artifact** (`qwen3_8_27b_a8.ninfer`) holds the same weights as the official one, byte
for byte, and allows the prefill GEMMs to quantize their activations to int8 (per token and per 64
channels) and run on int8 tensor cores; decode is unchanged. It is built from the official file in
a few minutes, without a BF16 checkpoint ([conversion](#converting-models)).

Against the official llama.cpp on the same machine and the same Qwen3.8-27B fine-tune (ColdFusion;
llama.cpp `a894dae` on the Q4_K_S GGUF with q8_0 KV and flash attention, NInfer on the Q4/Q5
artifact with int8 KV; same needle prompts, thinking off, no speculation, two runs each):

| Prompt prefill | llama.cpp | NInfer |
|---|---:|---:|
| 8K tokens | 3.0 s (2,558-2,585 tok/s) | 3.0-3.1 s (2,470-2,530 tok/s) |
| 64K tokens | 29.8-29.9 s | **28.9-29.4 s** |
| 128K tokens | 73.6-73.8 s | **66.8-66.9 s** (-9 %) |
| `pp512` / `pp2048` (bench tools) | **2,756 / 2,729 tok/s** | 2,334 / 2,594 tok/s |

With the official artifact, llama.cpp leads on short prompts and NInfer ties at 8K-64K and leads
at 128K. The int8 artifact prefills `pp512` / `pp2048` at 4,436 / 5,008 tok/s, 1.6-1.8x the
llama.cpp rates above (the int8 runs used the base Qwen3.8 artifact; the ColdFusion files were
removed). Measurements and methods: [WINDOWS_PORT.md](WINDOWS_PORT.md).

### Speculative decoding by workload

Decode tok/s, KV `rk4v4-e8`:

| Workload | Bonsai MTP 2 | Bonsai MTP 2 + n-gram | Qwen3.8 MTP 3 | Qwen3.8 MTP 3 + n-gram | Qwen3.8 DFlash2 d6 |
|---|---:|---:|---:|---:|---:|
| CLI, edit-style prompts, thinking off | 250 | **532** | 152 | **289** | 215 |
| CLI, prose, thinking off | 167 | 167 | 97 | 97 | 96 |
| Server, edit-style prompts, thinking on | 212 | **319** | — | 165 | 169 |
| Server, edit of a 7K-11K-token file, thinking on | 215 | **378** | — | **258** | 139 |

CLI rows: one request, greedy, up to 1024 tokens. Server rows: `ninfer-serve` with the launcher
flags above (three lanes, sampling and thinking defaults), one request at a time.

## Choosing settings

**Speculation.** Every drafted token is verified by the model, so the output distribution does
not change; only the speed does.

| Model | Recommended | When to change |
|---|---|---|
| Bonsai | `--spec mtp --draft-tokens 2 --lm-head-draft --ngram chain` | `--draft-tokens 3` is up to 18 % faster on code and math and about 10 % slower on prose |
| Qwen3.8 | `--spec mtp --draft-tokens 3 --lm-head-draft --ngram chain` | `--spec dflash2 --draft-tokens 6` is faster on free-form prose (99 against 85 tok/s through the server); `--draft-tokens 12` peaks at 211 tok/s on code. DFlash2 loads 1.6 GiB more weights; with three lanes, `--no-cuda-graph` frees about 1.1 GiB of graph memory for about 4 % of decode speed |

`--ngram chain` extends each MTP proposal with text copied from the context, so rounds that
reproduce code, JSON, tool calls or documents accept 10 or more tokens. It costs a second set of
CUDA graphs (about 2.5 % of the KV pool) and changes nothing on text with nothing to copy.
[docs/cli.md](docs/cli.md#speculative-decoding) lists the `--ngram-*` options.

**KV cache.**

| `--kv-dtype` | Use it for |
|---|---|
| `rk4v4-e8` (4-bit E8-lattice keys, 4-bit values) | Default. Full 262K context with vision; retrieval exact through 260K |
| `int8` | Maximum precision; Qwen3.8 fits about 168K tokens text-only |
| `rk2v4-e8` | More headroom at 262K (2-bit keys); about 10 % slower decode |

**Context and lanes.** `--max-context` bounds one request; `--kv-capacity` sizes the KV pool
shared by all lanes (`auto` takes what fits). `--max-concurrency` (one to eight) sets the lanes;
three are measured on both models. Prefill runs one request at a time, so a long prompt delays the
first token of other requests.

**Sampling.** The server applies the model card defaults: `temperature 1.0, top_p 0.95, top_k 20`
with thinking, and `temperature 0.7, top_p 0.80, top_k 20, presence_penalty 1.5` without.
Requests and startup flags (`--temperature`, `--top-p`, ...) override them.

### When the 4090 also drives the display

If the RTX 4090 also drives your monitor, the Windows desktop compositor takes the GPU from CUDA on
every frame. With a 3840x2160 desktop this cost 18 % of each MTP decode round at 120 Hz and 15 % at
60 Hz. For the best decode speed, connect the monitor to the motherboard (integrated graphics) or
another GPU; otherwise use 60 Hz and keep animated windows still while generating. Compare tok/s
only between runs with the same display setup.

## Benchmarking and validation

Build with `-DNINFER_BUILD_BENCHMARKS=ON -DBUILD_TESTING=ON`. Close `ninfer-serve` first, and run
each A/B comparison alternating the two binaries (old, new, new, old) so thermal and display
effects cancel. All commands run from the repository root.

**Throughput** (`ninfer_bench`, the public Engine route: `pp` = prefill, `tg` = decode):

```bat
build\bench\ninfer_bench.exe --weights E:\LLM\qwen3_8_27b.ninfer -p 512,2048 -n 128 -r 3 --kv-dtype int8
build\bench\ninfer_bench.exe --weights E:\LLM\bonsai2_27b_vl_mtp_q4q5.ninfer -n 128 -r 3 ^
  --spec mtp --draft-tokens 2 --lm-head-draft
```

`-o json --output-file FILE` writes a machine-readable report; see [bench/README.md](bench/README.md).

**Decode on real prompts** (the six-prompt protocol behind the decode means above; read `decode
speed` and `mtp acceptance rate` from the summary):

```bat
build\apps\ninfer.exe E:\LLM\qwen3_8_27b.ninfer --prompt "Explain how a transformer language model generates text, step by step." ^
  --max-context 4096 --max-new 512 --greedy --spec mtp --draft-tokens 3 --lm-head-draft
```

**Long-context prefill and retrieval** (needle in a haystack; the answer must contain
`ORCHID=493817`; prompts at 8K, 64K, 128K and 256K):

```bat
build\apps\ninfer.exe E:\LLM\bonsai2_27b_vl_mtp_q4q5.ninfer --messages examples\cli\messages\long_niah_64k.json ^
  --max-context 262144 --prefill-chunk 1024 --kv-dtype rk4v4-e8 --no-thinking --max-new 16 --greedy ^
  --spec mtp --draft-tokens 2 --lm-head-draft
```

**Perplexity** (four corpora; `--quick` takes one stream per corpus, about two minutes):

```bat
build\apps\ninfer-perplexity.exe E:\LLM\qwen3_8_27b.ninfer --corpus eval\corpora\perplexity-1m\manifest.json ^
  --quick --kv-dtype bf16 --output E:\eval\ppl_qwen38
```

See [docs/perplexity.md](docs/perplexity.md) for full runs and long-context scoring.

**Task quality** (45 deterministic tasks against a running server: tool calls, JSON, Python,
math, instruction following, Spanish, 4K-32K retrieval):

```powershell
python -m tools.eval run --base-url http://127.0.0.1:8080/v1 --label qwen38 --out E:\eval\qwen38.json --thinking off
python -m tools.eval compare E:\eval\bonsai.json E:\eval\qwen38.json --markdown E:\eval\compare.md
```

See [tools/eval/README.md](tools/eval/README.md).

**Tests.** `ctest --test-dir build --output-on-failure` runs the suite. Every CUDA kernel is
checked against an independent FP32/FP64 oracle; for example `build\tests\ninfer_linear_t5_test.exe`
(ternary GEMM/GEMV) and `build\tests\ninfer_softmax_attention_test.exe` (attention, all KV
modes). Tests that load a real model are opt-in; see [tests/README.md](tests/README.md).

**Profiling.** `nsys profile --trace=cuda,nvtx` on `ninfer_bench` or the CLI attributes time by
kernel; Nsight Compute (`ncu`) answers per-kernel questions. The Op benchmarks under
`build\bench\ninfer_*_bench.exe` (for example `ninfer_t5_bench`, `ninfer_q4_linear_swiglu_bench`)
time one kernel family at real shapes.

## How it works

NInfer is a from-scratch engine for one architecture family (Qwen3.5/3.6/3.8: 48 Gated DeltaNet
linear-attention layers and 16 full-attention layers at 27B). There is no generic graph: each
layer runs explicit kernels selected for its weight format, and every decode round replays as one
CUDA graph.

- **Ternary weights (Bonsai).** Prism's ternary codes are stored as scaled base 3, five weights
  per byte, with one FP16 scale per 128 weights (`t5_g128_fp16`, 1.75 bits per weight). Decode
  quantizes activations to int8 and multiplies with `dp4a`; prefill and multi-token verification
  use int8 tensor cores (`m16n8k32`), with a pipelined kernel for the tall projections that
  overlaps tensor-core work with the ternary decode. Prism's Hadamard rotation is fused into the
  activation quantization.
- **Groupwise weights (Qwen3.8).** Q4/Q5 codes with one FP16 scale per 64 weights, dequantized in
  shared memory for BF16 tensor-core GEMMs, with K-split routes for the 5-16-token verification
  band of DFlash2. Prefill runs pipelined one-CTA-per-SM GEMMs; with the int8 artifact the codes
  feed int8 tensor cores directly against per-token, per-64-channel int8 activations.
- **Speculative decoding.** MTP uses the model's own multi-token-prediction layer (Bonsai borrows
  Qwen3.8's, which shares its architecture); DFlash2 is a separate drafter; the n-gram pool
  (16 MiB, 8-token keys) extends MTP drafts with text from the context, up to 15 tokens per round.
  The linear-attention state rolls back rejected tokens through ReplaySSM records.
- **Long context.** Paged KV in int8 or E8-lattice 4-bit/2-bit codes, a warp-specialized prompt
  attention kernel (producer warps score, worker warps accumulate), and decode attention splits
  sized for whole waves of the 128 SMs.
- **Serving.** A fixed number of lanes (one to eight) decode as one batch; prefixes of previous
  requests are reused from device memory, host memory or disk.

The ternary work (converter, format, kernels, measurements and every design decision) is
documented in the [Bonsai design notes](docs/maintainer/bonsai-ternary-design.md) and the
[conversion guide](docs/maintainer/bonsai-ternary-conversion.md). Engine internals are in
[docs/maintainer/engine-architecture.md](docs/maintainer/engine-architecture.md).

## Converting models

The converter is Python (3.11 or 3.12) with NumPy, safetensors and PyTorch; it runs on Windows and
uses the GPU when `--device cuda` is given:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install numpy safetensors
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

**Ternary Bonsai 2 27B from Prism's GGUF.** Download `Ternary-Bonsai-2-27B-PTQ1_0.gguf` and, for
images, `Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf` from
[prism-ml/Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf),
and `qwen3_8_27b.ninfer` (it supplies the tokenizer, chat template and MTP head). About three
minutes:

```bat
python -m tools.convert.bonsai_base --gguf E:\LLM\Ternary-Bonsai-2-27B-PTQ1_0.gguf ^
  --reference E:\LLM\qwen3_8_27b.ninfer ^
  --mmproj E:\LLM\Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf --out E:\LLM\bonsai2-27b-vl
python -m tools.convert --model E:\LLM\bonsai2-27b-vl --recipe bonsai2_27b_mtp_q4q5 ^
  --components text,vision,mtp --source gguf=E:\LLM\Ternary-Bonsai-2-27B-PTQ1_0.gguf ^
  --source mmproj=E:\LLM\Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf ^
  --source mtp=E:\LLM\qwen3_8_27b.ninfer --proposal --name bonsai2-27b ^
  --out E:\LLM\bonsai2_27b_vl_mtp_q4q5.ninfer --device cuda
```

Leave out `--mmproj`, `vision` and `--source mmproj=...` for a text-only artifact. The recipe keeps
Prism's ternary codes bit-exact.

**Qwen3.8-27B fine-tunes from BF16 safetensors.** Any checkpoint with the Qwen3.8-27B
configuration converts with the official recipe, including its MTP layer; add the DFlash2
drafter from [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) if you want `--spec
dflash2`. A 55 GB BF16 checkpoint converts in about 100 s:

```bat
python -m tools.convert --model E:\LLM\my-qwen38-finetune --recipe qwen3_8_27b ^
  --source dflash2=E:\LLM\dflash2-src --components text,vision,mtp,dflash2 ^
  --resource chat_template.jinja=tools/chat_templates/qwen3_8.jinja --proposal ^
  --name my-qwen38 --out E:\LLM\my_qwen38.ninfer --device cuda
```

Use NInfer's `qwen3_8.jinja` template: it adds developer and mid-conversation system messages
and tool-result handling that agent clients need.

**Qwen3.8 int8 prefill artifact.** The `qwen3_8_27b_a8` recipe copies an existing
`qwen3_8_27b.ninfer` word for word and only grants int8 activations to the prefill GEMMs. It needs
a configuration directory (the Qwen3.8 `config.json` and tokenizer/template files) and takes about
four minutes on the CPU:

```bat
python -m tools.convert --model E:\LLM\qwen38-config --recipe qwen3_8_27b_a8 ^
  --source reference=E:\LLM\qwen3_8_27b.ninfer --source dflash2=E:\LLM\dflash2-src ^
  --components text,vision,mtp,dflash2 --name qwen3.8-27b --out E:\LLM\qwen3_8_27b_a8.ninfer
```

The [conversion guide](docs/weight-conversion.md) lists the files the configuration directory
needs. Custom recipes and every option are in the
[weight conversion guide](docs/weight-conversion.md).

## Serving

- **APIs.** OpenAI Chat Completions, OpenAI Responses (streaming, stored continuation state) and
  Anthropic Messages; function tools are rendered into the prompt and parsed back as tool calls
  (NInfer does not execute tools).
- **Reasoning.** `reasoning_effort` (`low`, `medium`, `xhigh`, or `none`) and `enable_thinking`
  on OpenAI routes; `thinking.budget_tokens` on Anthropic Messages or `--default-thinking-budget`
  server-wide. Reasoning returns separately as `reasoning_content`.
- **Prefix reuse.** Requests that extend or edit a previous conversation resume from the longest
  compatible prefix. Automatic long anchors at the last message boundaries
  (`--auto-long-anchors`) let an edited history restart below the edit instead of from zero.
- **Session persistence.** `--slot-save-path DIR` enables llama.cpp-style
  `POST /slots/{id}?action=save|restore|erase`, so a long session survives a server restart
  (a 6.9K-token session restores in about 0.1 s); `--auto-save-evicted` spills evicted sessions.
- **Observability.** `/monitor` (browser dashboard), `/metrics` (Prometheus, llama.cpp-compatible
  names), `/slots`, llama.cpp-style `timings` on every response, and a per-request JSONL log with
  the prefix-reuse decisions (`--request-log-jsonl FILE`).
- **Admission.** Requests beyond the lanes wait in a bounded FIFO queue (`--max-pending-requests`,
  `--pending-timeout-ms`); the default 30 s deadline is short for deep prefills.

The full protocol reference, including every field and error code, is in
[docs/serving.md](docs/serving.md).

## Limits

- One RTX 4090, one process, one resident model. No multi-GPU, no weight offload, no request
  preemption or priorities.
- Prefill runs one request at a time; decode of other lanes waits while it runs.
- With the official Qwen3.8 artifact, prefill trails llama.cpp on short prompts (`pp512` 2,334
  against 2,756 tok/s on the same machine); the int8 artifact leads at every length
  ([Qwen3.8 performance](#qwen38-27b)). Bonsai prefill is about 3x Prism's llama.cpp fork.
- Long-context decode slows with depth: a Bonsai MTP round costs 18.5-20.8 ms at 128K against
  12.8 ms at short context, all of it in attention.
- DFlash2 is not supported with Bonsai's ternary output head; use MTP.
- NVFP4/W4A4 execution needs Blackwell tensor cores and is unavailable on `sm_89`.
- This branch is developed and validated on Windows. The Linux build and `Dockerfile` are
  inherited from the upstream 4090 port and are not re-validated here; see
  [docs/rtx-3090-linux.md](docs/rtx-3090-linux.md) with `CMAKE_CUDA_ARCHITECTURES=89`.

## Documentation

| Topic | Document |
|---|---|
| All documentation | [docs/README.md](docs/README.md) |
| CLI options | [docs/cli.md](docs/cli.md) |
| HTTP server and protocols | [docs/serving.md](docs/serving.md) |
| Windows build, Qwen3.8 measurements and experiments | [WINDOWS_PORT.md](WINDOWS_PORT.md) |
| Ternary Bonsai design, kernels and measurements | [docs/maintainer/bonsai-ternary-design.md](docs/maintainer/bonsai-ternary-design.md) |
| Ternary Bonsai conversion and tensor mapping | [docs/maintainer/bonsai-ternary-conversion.md](docs/maintainer/bonsai-ternary-conversion.md) |
| Weight conversion and custom recipes | [docs/weight-conversion.md](docs/weight-conversion.md) |
| Perplexity evaluation | [docs/perplexity.md](docs/perplexity.md) |
| Benchmarks and tests | [bench/README.md](bench/README.md), [tests/README.md](tests/README.md) |
| Engine architecture | [docs/maintainer/engine-architecture.md](docs/maintainer/engine-architecture.md) |
| Context cache and scheduling | [docs/maintainer/resource-scheduling-and-context-cache.md](docs/maintainer/resource-scheduling-and-context-cache.md) |
| Comparison with llama.cpp (Linux, upstream port) | [docs/llamacpp-comparison.md](docs/llamacpp-comparison.md) |

## Lineage and credits

This repository descends from a chain of ports; each added what the next one builds on:

- [Neroued/ninfer](https://github.com/Neroued/ninfer) — the engine, developed for the RTX 5090
  (`sm_120a`).
- [Don-Chad/ninfer-3090](https://github.com/Don-Chad/ninfer-3090) — the `sm_86` compatibility
  layer, ReplaySSM integration and Qwen3.8 runtime support
  ([v0.6.1 release notes](RELEASE_NOTES_0.6.1.md)).
- [sergiuszm/ninfer-4090](https://github.com/sergiuszm/ninfer-4090) — the `sm_89` RTX 4090 port:
  Ada-tuned attention prefill, serving features (`/metrics`, `/slots`, slot save/restore,
  automatic long anchors, request diagnostics).
- [UDPSendToFailed/ninfer-4090](https://github.com/UDPSendToFailed/ninfer-4090) — the rotated and
  E8-lattice KV modes (`rk8v4`, `rk4v4`, `rk4v4-e8`, `rk2v4-e8`) and the configurable vision
  scratchpad, cherry-picked with authorship preserved
  ([comparison](docs/udp-fork-comparison.md)).
- [shantanusingh16/ninfer-4090](https://github.com/shantanusingh16/ninfer-4090) — llama.cpp-style
  `timings` on chat completions.
- This repository — the native Windows build, Ternary Bonsai 2 27B (converter, ternary format and
  kernels, vision), n-gram speculation, the concurrent-lane and prefill work, and the Qwen3.8
  DFlash2 verification routes.

Ternary Bonsai 2 27B, its packings and Hadamard rotation are by [Prism ML](https://huggingface.co/prism-ml);
their [llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp) defined the formats this branch
reads. [fraserprice/bonsai-vllm](https://github.com/fraserprice/bonsai-vllm) was a CUDA reference
for the Hadamard kernel and a ternary tensor-core GEMM. The Bonsai port was developed with
[Claude Code](https://claude.com/claude-code), with every kernel checked against FP64 oracles and
every performance claim measured on the RTX 4090.

## License

Apache License 2.0. See [LICENSE](LICENSE).
