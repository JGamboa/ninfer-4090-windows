# NInfer-4090 — Native Windows (MSVC) Port

This branch makes [NInfer-4090](https://github.com/sergiuszm/ninfer-4090) build and
run on **native Windows** with MSVC + CUDA, without WSL or Docker. The upstream fork
targets `sm_89` + Linux; the Windows path was inherited but untested. This work makes
it actually compile, link, and run on Windows.

Verified on: **RTX 4090 (sm_89), Windows, CUDA 13.4, Visual Studio Build Tools 2026
(MSVC 14.51), CMake 4.4, Ninja**, against the `qwen3_8_27b` groupwise **v2** artifact.

Measured `ninfer-serve` on this machine (RTX 4090, INT8 KV, MTP3):
- Code generation, thinking off: **149 tok/s** decode at 98% MTP acceptance.
- With reasoning/thinking on: ~96 tok/s (acceptance drops on unpredictable text, as expected).
- Prefill: 280–740 tok/s depending on prompt.

These match the upstream fork's published RTX 4090 numbers (~148.6 tok/s code decode).

---

## Credits

This is a downstream port. Full credit to the original authors:

- **[Neroued/ninfer](https://github.com/Neroued/ninfer)** — the from-scratch C++/CUDA
  inference engine (developed for the RTX 5090, `sm_120a`).
- **[Don-Chad/ninfer-3090](https://github.com/Don-Chad/ninfer-3090)** — the `sm_86`
  compatibility layer, ReplaySSM integration, and Qwen3.8 runtime support.
- **[sergiuszm/ninfer-4090](https://github.com/sergiuszm/ninfer-4090)** — the `sm_89`
  RTX 4090 retarget (Ada-tuned attention prefill, E8 lattice KV, etc.) that this port
  builds directly on.
- **[UDPSendToFailed/ninfer-4090](https://github.com/UDPSendToFailed/ninfer-4090)** —
  the E8-lattice KV modes cherry-picked into the sergiuszm fork.

All original work remains under the Apache-2.0 license of the upstream projects. This
port only adds Windows-compatibility shims; it does not change the engine's algorithms,
kernels, or numerics.

---

## What this port changes (Windows compatibility only)

The engine assumed a Linux/GCC toolchain. The changes are POSIX-to-Win32 shims, an MSVC
128-bit integer path, an MSVC-specific compiler flag, and one MSVC template-linkage fix.
Every change is guarded with `#ifdef _WIN32` so the Linux build is unaffected.

### 1. POSIX headers / functions → Win32 equivalents

| File | POSIX use | Windows replacement |
|------|-----------|---------------------|
| `src/product/logging/startup_log.cpp` | `<sys/ioctl.h>`, `ioctl(TIOCGWINSZ)` for terminal width | `<windows.h>` + `GetConsoleScreenBufferInfo` |
| `src/product/logging/logging.cpp` | `<unistd.h>`, `isatty(STDERR_FILENO)` | `<io.h>` + `_isatty(_fileno(stderr))` |
| `src/product/logging/logging.cpp` | `localtime_r(&t, &tm)` | `localtime_s(&tm, &t)` (note: arg order is swapped on MSVC) |
| `apps/perplexity/main.cpp` | `gmtime_r(&t, &tm)` | `gmtime_s(&tm, &t)` |
| `src/runtime/engine/context_cost.cpp` | `<unistd.h>`, `unsigned __int128` | `<process.h>`/`<intrin.h>` + a small `U128` helper using `_umul128` |
| `src/artifact/reader.cpp` | `<fcntl.h>`, `<sys/mman.h>`, `<sys/stat.h>`, `<unistd.h>` (mmap) | Win32 file-mapping equivalents |
| `src/product/media_acquire/acquire.cpp` | `<sys/socket.h>`, `<arpa/inet.h>` | Winsock shims |
| `src/serve/request_log.cpp` | `<unistd.h>` | guarded include |

(Several of these were started in an earlier pass; this branch completes them.)

### 2. MSVC lacks `unsigned __int128` / `__uint128_t`

Two hot paths used native 128-bit integers, which MSVC does not provide:

- `src/runtime/contract/types.h` — attention-work saturating product. Uses `_umul128`
  to compute the 64×64→128 product as (hi, lo) halves.
- `src/runtime/engine/materialization_planner.h` — cross-product comparison
  (`delta*b` vs `delta*a`) to compare two fractions without dividing. Ported to compare
  the two products via their `_umul128` (hi, lo) halves. Added `#include <intrin.h>`.
- `src/runtime/engine/context_cost.cpp` — a full `U128` helper struct (multiply / shift /
  compare) backed by `_umul128`.

### 3. `/utf-8` compiler flag (spdlog / fmt)

bundled `fmt` (via spdlog) hard-fails on MSVC without `/utf-8`
(`static assertion failed: 'Unicode support requires compiling with /utf-8'`).
Added to `CMakeLists.txt` for both C/CXX and CUDA host compilation:

```cmake
if(MSVC)
  add_compile_options(
    $<$<COMPILE_LANGUAGE:C,CXX>:/Zc:preprocessor>
    $<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
```

This single flag unblocked the bulk of the object files (spdlog is pervasive).

### 4. MSVC template-member linkage (`= default` specializations)

`src/targets/qwen3_6/impl/runtime/program.h` defines explicit specializations of
`AdmissionCandidate<Variant>::operator=(&&)` and
`CapturePressureCandidate<Variant>::operator=(&&)` as `= default` out of line. MSVC does
**not** emit external symbols for defaulted special members declared this way in an
explicit specialization, so linking `ninfer.exe` failed with 4 unresolved externals
(LNK2019) for the 27b and 35b variants. Fixed by giving each move-assignment an explicit
body (`impl_ = std::move(other.impl_); return *this;`), which MSVC emits normally. GCC/Clang
accepted the `= default` form, so this is MSVC-specific and Linux is unaffected.

---

## Building on Windows

Requirements: RTX 4090 (sm_89), CUDA 12.8+ (13.4 validated), Visual Studio Build Tools
(MSVC), CMake 3.28+, Ninja, and vcpkg for `curl`/`ffmpeg`/`pkgconf`.

```bat
:: from a shell with MSVC env (vcvars64.bat) and CUDA on PATH
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_TOOLCHAIN_FILE=<path>/vcpkg/scripts/buildsystems/vcpkg.cmake ^
  -DVCPKG_TARGET_TRIPLET=x64-windows ^
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j
```

Products: `build/apps/ninfer.exe`, `build/apps/ninfer-serve.exe`,
`build/apps/ninfer-perplexity.exe`. At runtime, put the vcpkg `bin` and CUDA `bin` on PATH.

## Model artifact — v2 and v3 both work (on `main`)

This branch reads both **v2** (`NINFER\0\2`) and **v3** (`NINFER\0\3`) `.ninfer`
artifacts — the reader auto-detects the container version from the magic bytes, no build
flag needed. HuggingFace repos originally shipped v2 and have since migrated their `main`
to v3; either works here now. See
[docs/artifact-v3-port-notes.md](docs/artifact-v3-port-notes.md) for what v3 changed and
how this port reads it.

If you're on the `winport-v2` branch instead (plain Windows port, no v3 support), that
build only reads v2 and rejects v3 with `artifact magic is not NInfer v2` — download a v2
revision instead of `main` from the HuggingFace repo. For Qwen3.8-27B the pre-v3 commit is
`dc370fb`:

```
https://huggingface.co/neroued/Qwen3.8-27B-NInfer/resolve/dc370fb/qwen3_8_27b.ninfer
```

## Running

```bat
ninfer-serve.exe qwen3_8_27b.ninfer ^
  --host 0.0.0.0 --port 8080 ^
  --max-context 168000 --kv-capacity 168000 ^
  --prefill-chunk 1024 --kv-dtype int8 ^
  --spec mtp --draft-tokens 3 --lm-head-draft --preserve-thinking
```

For maximum decode speed on code, send requests with thinking disabled
(`"enable_thinking": false` / `"reasoning_effort": "none"` on the OpenAI route): MTP
acceptance rises to ~98% and decode reaches ~149 tok/s. For the full native 262K context
on 24 GB, use `--kv-dtype rk4v4-e8`.

### Faster still: `--spec dflash2` beats MTP, and its sweet spot isn't its max

Swept `--draft-tokens` on this RTX 4090 with a fixed code-generation prompt
(`enable_thinking:false`, greedy, same seed) to find the actual optimum instead of
guessing. `--spec mtp` accepts `--draft-tokens` in `[1,5]`; `--spec dflash2` (a real,
separate small autoregressive draft model — not a single-shot head like MTP, so it can
speculate deeper before its accuracy collapses) accepts `[1,15]`:

| Backend | draft-tokens | Decode | MTP/DFlash acceptance |
|---|---:|---:|---:|
| mtp | 3 (this doc's old default) | 137.8 tok/s | 96.2% |
| mtp | 5 (max) | 160.6 tok/s | 89.8% |
| dflash2 | 8 | 194.4 tok/s | 97.2% |
| **dflash2** | **12** | **210.9 tok/s** | 85.7% |
| dflash2 | 15 (max) | 201.5 tok/s | 76.8% |

**`--spec dflash2 --draft-tokens 12` is the fastest configuration found — 210.9 tok/s,
+53% over this doc's previous `mtp --draft-tokens 3` default.** Note the optimum is *not*
the maximum allowed value for either backend: acceptance keeps falling as draft-tokens
rises, and past a point the extra verification cost outweighs the extra accepted tokens
(dflash2 peaks at 12, then drops back down by 15). Requires the artifact to actually ship
DFlash2 weights (adds ~1.6 GiB to the load); the server auto-detects and requires them
when `--spec dflash2` is passed.

## Memory bandwidth ceiling on this RTX 4090 (2026-09-25)

`tools/hbm_bandwidth_probe.cu`, built with `nvcc -O3 -std=c++17 -arch=sm_89
tools/hbm_bandwidth_probe.cu -o build\hbm_bandwidth_probe.exe` and run with no arguments. The
monitor was on the RTX 4090 at 60 Hz and `ninfer-serve` was off. The buffers were 4 GiB each
(57x L2), with 768 resident blocks of 256 threads. Bus GB/s counts N bytes for a read or a
write and 2N for a copy. Two runs, best of 5 trials (median in parentheses):

| Method | Run 1 bus GB/s | Run 2 bus GB/s | Of the 4090's 1008 GB/s |
|---|---:|---:|---:|
| `kernel uint4 read` (pure read) | 839.1 (802.5) | 848.1 (844.2) | 83-84 % |
| `kernel uint4 copy` | 785.5 (774.4) | 791.0 (785.1) | 78 % |
| `kernel uint4x4 copy` | 785.8 (782.8) | 783.3 (783.0) | 78 % |
| `cudaMemcpyAsync` D2D | 820.6 (811.9) | 831.1 (825.6) | 81-82 % |
| `kernel uint4 write` | 748.9 (742.4) | 747.0 (743.1) | 74 % |
| `cudaMemsetAsync` (write) | 826.3 (722.3) | 826.0 (819.5) | 82 % |

These runs used the probe's former `--peak-gbps` default of 1792 (the RTX 5090's figure, 46-47 %
for the read here); the last column above is recomputed against the 4090's advertised
1008 GB/s, which is now the probe's default.

**Ceiling: pure read ~845 GB/s, copy ~785 GB/s.** Decode is a weight-read stream, so the GB/s
figures in this document are read against **~845 GB/s**, not the advertised 1008. On that
scale:
- Qwen3.8 tg128 (~800 GB/s) runs at ~95 % of the ceiling.
- The best K-split MMA instances in the DFlash2 verification (~680-690 GB/s) run at ~81 %.
- The Q4 gate+up at T = 13 (608 GB/s) runs at ~72 %.

## Qwen3.8 baseline on the Bonsai branch (2026-09-24)

Measured on `feat/bonsai-ternary` at `15df903` with `E:\LLM\qwen3_8_27b.ninfer` (15.92 GiB of
Q4/Q5 weights). The RTX 4090 also drives the desktop, set to 60 Hz; `ninfer-serve` was off.
The desktop compositor still preempts the GPU: a few kernels run 2-30x long, so profiled
averages sit slightly above the medians. These numbers are the baseline for prefill and
decode work on this model.

**Throughput** (`ninfer_bench --weights E:\LLM\qwen3_8_27b.ninfer -p 512,2048 -n 128 -r 3
--kv-dtype int8`, no speculation):

| Test | Result |
|---|---:|
| pp512 | 1821 tok/s |
| pp2048 | 2035 tok/s |
| tg128 | 47.0 tok/s (46.9 with the default bf16 KV) |

tg128 reads ~16 GiB of weights per token, about 800 GB/s. That is ~95 % of the measured
~845 GB/s read ceiling (see the section above); the advertised figure is 1008 GB/s.

**Prefill profile** (`nsys profile --trace=cuda,nvtx` of `ninfer_bench -p 2048 -r 3
--kv-dtype int8`, saved as `profiles/nsys/qwen38_pp2048`). pp2048 runs as two 1024-token
chunks; each chunk takes 512 ms wall and 510 ms of kernels. Share of GPU time:

| Kernel (route) | Shape (N x K) | Grid | Per call | Calls per chunk | Share |
|---|---|---|---:|---:|---:|
| `q4_linear_swiglu_mma_split_half_pair_kernel` (Q4 gate+up with SwiGLU) | 34816 x 5120 | 544 x 8 | 3.59 ms | 64 | 45.1 % |
| `q5_rowsplit_gemm_mma_kernel` (Q5 `linear_add`: mlp down, attention o_proj and GDN out_proj) | 5120 x 17408 / 6144 | 80 x 8 | 1.50 / 0.55 ms (medians) | 64 + 64 | 30.3 % |
| `rowsplit_grouped_mma_kernel` (mixed Q4/Q5 GDN in_proj) | 16384 x 5120 | 256 x 8 | 1.72 ms | 48 | 16.2 % |
| `rowsplit_grouped_mma_kernel` (mixed Q4/Q5 attention qkvg) | 14336 x 5120 | 224 x 8 | 1.32 ms | 16 | 4.1 % |
| GDN (`state_passing` 1.1 %, `prepare_wy_wu` 0.7 %, `output` 0.4 %, conv 0.4 %, gating GEMM 0.1 %, l2norm 0.1 %) | | | | | 2.2 % |
| Attention (`causal_attention_prompt_i8_kernel`) | | | 0.28 ms | 16 | 0.9 % |
| rmsnorm, sigmoid gate and other elementwise kernels | | | | | 0.6 % |
| Last-token head (`q8_ksplit_mma`, once per prefill) and bookkeeping | | | | | 0.4 % |

The GEMMs take 95.9 % of prefill time, all on bf16 `mma` with dequantization in shared
memory. Their rates are 50-60 T MAC/s: gate+up 182.5 G MAC per chunk in 3.59 ms, down 91.3 G
in 1.50 ms. For comparison, Bonsai's int8 t5 gate+up does 2048 tokens in 3.76 ms, about half
the time per token.

**Nsight Compute of one gate+up launch** (`ncu --set full` on the 21st
`q4_linear_swiglu_mma_split_half_pair_kernel` launch of `ninfer_bench -p 2048 -r 1 --warmup 0
--kv-dtype int8`, saved as `profiles/ncu/qwen38_q4_gateup_pp2048`). pp2048 launches this kernel
with T = 1024, grid 544 x 8, 128 threads, `GemmCfg<64, 128, 64, 64, 32, 2, 1, 0, 1, 1>`.

- Duration 3.31 ms at a locked 2.60 GHz SM clock.
- Memory: DRAM throughput 24 %; L2 throughput 38 % with an 88.6 % hit rate; 747 MB read from
  DRAM. The ~100 MB of Q4 weights are re-read once per 128-token tile, because they exceed
  the 72 MB L2.
- Compute: SM 32 %, tensor pipe (HMMA) active 32.5 % of cycles, LSU 30.9 %, ALU 20.1 %,
  issue slots 22.9 % busy, 0.92 IPC.
- Stalls: 8.67 warp cycles per issued instruction, made up of math-pipe throttle 4.43 (51 %),
  wait 1.86, selected 1.00, short scoreboard 0.56, not selected 0.24, barrier 0.23, branch
  0.12, MIO 0.11 and long scoreboard 0.03.
- Occupancy: 16.7 % theoretical and achieved, i.e. 8 warps per SM from 2 CTAs of 4 warps.
  **Shared memory limits it** (45.6 KB static plus 1 KB reserved per CTA, 2 CTAs per SM);
  the 157 registers per thread would allow 3 CTAs.
- No register spills and no local memory.
- ncu also estimates smaller gains: global stores use 16 of 32 bytes per sector (up to
  15 %), 9 % of global sectors and 8 % of shared wavefronts are excess (uncoalesced), and
  there are 8.0 M shared bank conflicts, mostly on stores.

The math-pipe throttle does not mean the kernel is compute-bound: the tensor pipe is busy
only a third of the time. With two warps per scheduler, back-to-back HMMAs from the same warp
wait on the pipe and no other warp can fill the gap. The levers, in order, are occupancy
(less shared memory per CTA, or a third CTA), then the coalescing of the epilogue stores.

**MTP decode, six prompts** (`ninfer.exe --prompt <p> --max-context 4096 --max-new 512
--greedy --spec mtp --draft-tokens 3 --lm-head-draft`, thinking on by default, default bf16
KV):

| Prompt | tok/s | Acceptance | Tokens per round |
|---|---:|---:|---:|
| Lighthouse story | 81.0 | 37.3 % | 2.12 |
| Python merge | 119.1 | 70.8 % | 3.12 |
| Transformer explanation | 109.3 | 61.8 % | 2.85 |
| Historia de Chile (Spanish) | 95.5 | 49.9 % | 2.49 |
| Energy tips | 107.2 | 60.3 % | 2.81 |
| Train problem | 127.2 | 78.4 % | 3.35 |
| Mean | 106.6 | 59.8 % | 2.79 |

### DFlash2 round profile, draft 12 (2026-09-24)

This profile uses the "fast" configuration from the table above on a harder prompt, to see
where a round's time goes. Command: `nsys profile --trace=cuda,nvtx --cuda-graph-trace=node
build\apps\ninfer.exe E:\LLM\qwen3_8_27b.ninfer --messages
examples\cli\messages\scenario_code_python.json --max-context 8192 --max-new 512 --greedy
--no-thinking --spec dflash2 --draft-tokens 12 [--lm-head-draft]` (`profiles/nsys/
qwen38_dflash2_d12`, `_d12_fullhead`). Same build, desktop at 60 Hz, server off.

The prompt asks for a whole Python package with tests: 122 prompt tokens, 512 generated. It
is much less predictable than the quicksort request behind the 210.9 tok/s figure.

| Run | tok/s | Acceptance | Tokens per round | Rounds | ms per round (plain run / nsys) |
|---|---:|---:|---:|---:|---:|
| d12, `--lm-head-draft` | 77.7 / 78.7 (two runs) | 24.8 % | 3.96 | 129 | 51.2 / 51.4 |
| d12, full proposal head | 79.6 | 26.2 % | 4.12 | 124 | 51.6 / 52.1 |
| d6, `--lm-head-draft` (reference) | 101.9 | 35.6 % | 3.13 | 163 | 30.7 / - |

With d12 and the proposal head, accepted tokens by draft position are 100, 76, 59, 40, 31,
23, 15, 13, 10, 6, 6, 3 over 129 rounds. Positions 7 to 12 add 53 of the 382 accepted tokens
but almost double the verification width (T = 7 to T = 13). On this prompt d6 is 30 % faster.

**Where a d12 round goes** (`--lm-head-draft`; per round, 50.6 ms of kernels in 51.4 ms wall,
735 launches, all graphed). Phases are separated by `speculative_prepare_verify_inputs`
(drafter -> verify) and `speculative_select_accepted_hidden` (verify -> commit):

| Phase / kernel (route) | ms per round | % | Per call |
|---|---:|---:|---:|
| **Drafter** (DFlash2 draft model, 5 layers at T = 13, proposal head, top-k, lattice selector) | **3.51** | **6.9** | |
| of which Q8 draft-layer MLP and projections (`q8_ksplit_mma`, grids 2176 / 320 / 384) | 2.47 | | |
| of which proposal head (`q4_ksplit_mma`, grid 8192) + top-k merge | 0.64 | | 511 us |
| of which conv prepare, context KV, sliding-window attention, norms | 0.40 | | |
| **Target verification, T = 13** | **46.78** | **92.4** | |
| mlp down 5120 x 17408, Q5 (`q5_rowsplit_gemm_simt_split2`) | 12.87 | 25.4 | 169 us x 64 |
| mlp gate+up 34816 x 5120, Q4 (`q4_ksplit_mma`) | 11.53 | 22.8 | 155 us x 64 |
| GDN in_proj 16384 x 5120, mixed Q4/Q5 (`rowsplit_grouped_mma`) | 11.17 | 22.1 | 201 us x 48 |
| attention o_proj and GDN out_proj 5120 x 6144, Q5 (`q5_rowsplit_gemm_simt_split2`) | 3.99 | 7.9 | 64 us x 64 |
| attention qkvg 14336 x 5120, mixed Q4/Q5 (`rowsplit_grouped_mma`) | 3.21 | 6.3 | 180 us x 16 |
| verify head 248320 x 5120, Q8 (`q8_ksplit_mma`) | 1.69 | 3.3 | 1.43 ms |
| GDN gating, conv, recurrence record | 1.42 | 2.8 | |
| attention (bf16 KV prompt kernel, rope, KV append, output gate) | 0.60 | 1.2 | 27 us x 16 |
| rmsnorm and elementwise | 0.24 | 0.5 | |
| sampling (argmax, top-k, finalize) | 0.04 | 0.1 | |
| **Commit tail** (`recurrent_fold` 0.33 ms, select, counters) | **0.34** | **0.7** | |

Without `--lm-head-draft`, the drafter takes 4.58 ms (`q8_grouped_ksplit_topk` over the full
vocabulary: 1.67 ms instead of 0.64). Verification is unchanged at 46.54 ms. The short
proposal head saves 1.07 ms per round (2 %). Here that is within the acceptance difference
between the two runs (3.96 against 4.12 tokens per round).

The drafter is cheap; the round is set by the T = 13 verification. That pass takes 46.8 ms,
2.2x a single-token decode (21.3 ms at tg128's 47 tok/s), while reading the same ~17 GB of
weights. That is ~365 GB/s, against ~800 GB/s at T = 1. At T = 13 the Q4/Q5 "few tokens"
routes are below the bandwidth roof. Estimates at ~4.5 bits per Q4 weight and ~5.5 per Q5:
- gate+up (`q4_ksplit_mma`) reads ~100 MB in 155 us, ~650 GB/s.
- The Q5 SIMT routes read down (~61 MB) and o_proj/out_proj (~22 MB) at ~340-360 GB/s.
- The grouped mixed Q4/Q5 mma for in_proj and qkvg reads at ~230-290 GB/s.

These three families are where a wider tensor-core small-T route would pay back. The GEMMs
make up 88 % of the round.

### Q5 tensor-core route for the DFlash2 verification band (`a29aed7`, 2026-09-24)

`a29aed7` routes the Q5 `linear_add` shapes to a new `q5_ksplit_mma_kernel`, with the
residual added in its epilogue:
- mlp down, 5120 x 17408, at T = 7..16;
- attention o_proj and GDN out_proj, 5120 x 6144, at T = 7..13.

T <= 6 keeps the SIMT split2 route. Validated on the RTX 4090, desktop at 60 Hz, server off.
Before and after were measured back to back with the same scripts.

- `ninfer_linear_add_q5_a16_test` passes (`OK Q5_A16 LinearAdd`). It includes the new route
  start at T = 7, the 13/14 and 16/17 route boundaries, and interior T = 11.
- `ninfer_q5_linear_add_bench --execution graph --repeat 100` (median us, and GB/s as the
  bench reports it):

  | T | 5120 x 17408 before | after | 5120 x 6144 before | after |
  |---:|---:|---:|---:|---:|
  | 6 | 108.5 (542) | 110.6 (532) | 44.0 (473) | 45.1 (463) |
  | 7 | 119.8 (491) | 104.4 (564) | 51.2 (408) | 42.0 (497) |
  | 8 | 127.0 (464) | 104.4 (564) | 52.2 (400) | 42.0 (498) |
  | 13 | 182.3 (325) | 111.6 (530) | 79.9 (264) | 46.3 (455) |
  | 14 | 221.2 (268) | 114.7 (517) | 95.2 (222) | 96.3 (219) |
  | 16 | 246.8 (241) | 117.8 (504) | 93.2 (227) | 94.2 (225) |
  | 17 | 300.0 (198) | 312.3 (190) | 102.4 (207) | 102.5 (207) |

  Two route cliffs remain next to the band:
  - 5120 x 6144 at T = 14..16 (DFlash2 d13-d15): 96 us, against 46 us for the new route at
    T = 13.
  - 5120 x 17408 at T = 17 (`MmaResidualR64C16`, T = 17..32): 300 us, against 118 us at
    T = 16.
- DFlash2 on the profile prompt (`scenario_code_python`, `--no-thinking --greedy
  --lm-head-draft`), with MTP as the regression check:

  | Run | tok/s before -> after | Acceptance | Tokens per round | ms per round | Text |
  |---|---|---|---|---|---|
  | DFlash2 d6 | 102.0 -> 129.6 | 35.6 -> 46.5 % | 3.13 -> 3.79 | 30.7 -> 28.9 (-6 %) | differs from char 21 |
  | DFlash2 d12 | 79.1 -> 99.5 | 24.8 -> 26.8 % | 3.96 -> 4.19 | 50.4 -> 41.8 (-17 %) | differs from char 220 |
  | Qwen3.8 MTP 3 | 124.5 -> 126.6 | 74.6 % both | 3.23 both | 25.9 -> 25.3 | identical (md5) |
  | Bonsai MTP 2 (lighthouse prompt) | 149.8 -> 153.9 | 42.4 % both | 1.85 both | 12.2 -> 11.8 | identical (md5) |

  The DFlash2 texts change because the verification sums in a different order, and this
  prompt has a near-tie at the fifth token ("complete, self-contained" against "complete,
  runnable"). Every speculative run leaves plain greedy decoding (no speculation, T = 1)
  there, including MTP 3, whose route did not change. After the change, d6 takes the
  "runnable" branch that d12 and MTP 3 already took. d12 diverges at char 220
  ("asyncio semaphore" against "asyncio lock"). Both texts are valid and of the same length
  (1988 against 1953-1973 chars). Most of d6's tok/s gain therefore comes from a more
  predictable text. The per-round time is the comparable figure: -6 % at d6 (T = 7) and
  -17 % at d12 (T = 13).
- d12 round (`profiles/nsys/qwen38_dflash2_d12_q5band`, same analysis as above), per round:
  - wall 51.4 -> 43.9 ms; verification 46.8 -> 39.5 ms; drafter unchanged (3.5 ms).
  - mlp down 12.87 -> 7.78 ms (169 -> 102 us per call; ~59 MB at ~350 -> ~580 GB/s).
  - o_proj and out_proj 3.99 -> 2.41 ms (64 -> 38 us; ~21 MB at ~330 -> ~560 GB/s).

  Byte counts are the bench's for the same shapes. The verification's largest costs are now
  the two other small-T families: Q4 gate+up (11.5 ms, 155 us per call) and the grouped
  mixed Q4/Q5 in_proj and qkvg (10.9 + 3.2 ms, ~180-200 us per call, ~230-290 GB/s).

### K-split GDN in_proj and the k = 6144 route through T = 16 (`58b5683`, `fa5dad0`, 2026-09-25)

The two commits extend the small-T K-split routes to the rest of the DFlash2 band:
- `58b5683`: the mixed Q4/Q5 GDN input projection runs its two sides as separate K-split
  MMAs up to T = 16. The Q4 side is 4096 rows (`q4_ksplit_mma`, grid 256); the Q5 side is
  12288 rows (`q5_ksplit_mma`, grid 768). They replace `rowsplit_grouped_mma`.
- `fa5dad0`: the 5120 x 6144 Q5 `linear_add` takes the K-split route through T = 16, closing
  the T = 14..16 cliff.

Validated on the RTX 4090 at HEAD `fa5dad0` against the `a29aed7` build, with the same
scripts, desktop at 60 Hz and server off.

- `ninfer_linear_add_q5_a16_test`, `ninfer_gdn_input_proj_test`,
  `ninfer_gdn_input_proj_conv_snapshot_test` and `ninfer_gdn_input_proj_conv_record_test` all
  pass (the snapshot test compares against its sampled FP64 reference with 0 failures).
- Microbenches, graph execution, 100 repeats, median us (bench-reported GB/s):

  | T | Q5 `linear_add` 5120 x 6144, a29aed7 -> fa5dad0 | GDN in_proj Q4/Q5 16384 x 5120 (cold L2), a29aed7 -> fa5dad0 |
  |---:|---|---|
  | 7 | 43.0 -> 44.0 | 97.3 -> 98.3 (542 -> 536) |
  | 12 | 45.1 -> 47.1 | 157.7 -> 98.3 (336 -> 539) |
  | 13 | 46.1 -> 48.1 | 225.3 -> 97.3 (235 -> 545) |
  | 14 | 95.3 -> 48.1 (222 -> 439) | - |
  | 15 | 95.2 -> 49.2 | - |
  | 16 | 95.2 -> 48.1 (222 -> 440) | 226.3 -> 104.4 (235 -> 509) |
  | 17 | 101.4 -> 101.4 | 226.3 -> 227.3 |

  5120 x 17408 is unchanged (T = 7 / 13 / 16 / 17: 103 / 113 / 117 / 299 us). T <= 6 is
  unchanged in both benches.
- DFlash2 on the profile prompt (`scenario_code_python`, `--no-thinking --greedy
  --lm-head-draft`), per-round time against `a29aed7`:

  | Draft | ms per round a29aed7 -> fa5dad0 | tok/s | Acceptance | Tokens per round | Text vs a29aed7 |
  |---|---|---|---|---|---|
  | d6 (T = 7) | 28.9 -> 29.6 (noise; T = 7 was already on the new routes) | 129.6 -> 129.3 | 46.5 % | 3.79 | identical |
  | d12 (T = 13) | 41.8 -> 36.4 (-13 %) | 99.5 -> 116.5 | 26.8 -> 27.1 % | 4.19 -> 4.22 | now identical to d6's text |
  | d15 (T = 16) | 46.8 -> 37.6 (-20 %) | 86.0 -> 108.7 (+26 %) | 20.7 -> 20.9 % | 4.06 -> 4.09 | identical |
  | Qwen3.8 MTP 3 | 25.3 -> 25.3 | 126.6 -> 128.5 | 74.6 % | 3.23 | identical (md5) |
  | Bonsai MTP 2 | 11.8 -> 11.8 | 153.9 -> 154.6 | 42.4 % | 1.85 | identical (md5) |

  d15's text is unchanged, so its +26 % is a like-for-like speedup. d12 now produces exactly
  d6's text. On this prompt d6 is still the fastest configuration (129 tok/s), but d12 is
  within 10 % of it (116.5); it was 23 % behind after `a29aed7` and 22 % behind before it.
- d12 round (`profiles/nsys/qwen38_dflash2_d12_ksplit16`), per round:
  - wall 43.9 -> 37.7 ms; verification 39.5 -> 33.3 ms; drafter 3.55 ms.
  - 782 launches (+48: the two in_proj sides).
  - GDN in_proj 10.85 -> 4.51 ms: 201 -> ~94 us per layer (Q4 side ~20 us + Q5 side ~61 us,
    plus launch gaps), ~53 MB at ~565 GB/s.
  - mlp down 7.86 ms (102 us) and o_proj/out_proj 2.46 ms (39 us), unchanged.

  What remains of the T = 13 verification:
  - Q4 gate+up: 11.3 ms, 155 us per call, ~650 GB/s.
  - attention qkvg: 3.2 ms, still `rowsplit_grouped_mma` at 179 us per call, ~245 GB/s. It
    is the last grouped route in the band; the GDN treatment would save ~1.5 ms per round.
  - head: 1.7 ms.

Across the three commits, the d12 round on this prompt went from 51.4 to 37.7 ms (-27 %),
and d12 decode from 78.7 to 116.5 tok/s.

### K-split qkvg (`cccaace`, 2026-09-25)

`cccaace` moves the mixed Q4/Q5 attention input projection (qkvg, 14336 x 5120) to separate
K-split MMA sides up to T = 16, as `58b5683` did for the GDN in_proj. The Q4 side is 7168
rows (`q4_ksplit_mma`, grid 448); the Q5 side is 7168 rows (`q5_ksplit_mma`, grid 448). They
replace `rowsplit_grouped_mma`, the last grouped route in the DFlash2 band. Validated against
the `fa5dad0` build with the same method.

- `ninfer_attn_input_proj_test` passes (`OK attn_input_proj`, 25 s). It covers T = 1..128
  and graph replay at 12, 13, 16 and 17.
- `ninfer_attn_input_proj_bench --format q4q5 --cache cold --execution graph --repeat 100`,
  median us (bench-reported GB/s), `fa5dad0` -> `cccaace`:

  | T | 1 | 6 | 7 | 12 | 13 | 16 | 17 |
  |---|---|---|---|---|---|---|---|
  | qkvg | 72.7 -> 72.7 | 114.7 -> 114.7 | 85.0 -> 86.0 | 123.9 -> 97.3 (356 -> 453) | 195.6 -> 90.1 (225 -> 489) | 195.6 -> 103.4 (226 -> 427) | 196.6 -> 197.6 |

  A pre-existing cliff shows up next to the band: T = 6 (114.7 us) is slower than T = 7
  (85.0 us). It matters for MTP 5 (T = 6) and for DFlash2 d5.
- DFlash2 on the profile prompt, per-round time against `fa5dad0`:

  | Draft | ms per round | tok/s | Acceptance | Text |
  |---|---|---|---|---|
  | d6 | 29.6 -> 29.6 | 129.3 -> 126.8 | 46.5 % | identical |
  | d12 | 36.4 -> 35.5 | 116.5 -> 119.7 | 27.1 % | identical |
  | d15 | 37.6 -> 37.1 | 108.7 -> 111.0 | 20.9 -> 21.1 % | changes (a tie; now `a29aed7`'s d12 text) |
  | Qwen3.8 MTP 3 | 25.3 -> 25.9 | 128.5 -> 124.6 | 74.6 % | identical (md5) |
  | Bonsai MTP 2 | 11.8 -> 12.2 | 154.6 -> 149.6 | 42.4 % | identical (md5) |

  The MTP rows and d6 use none of the changed routes. Their ±0.5 ms is the run-to-run noise
  of these 3-5 s decodes at 60 Hz.
- d12 round (`profiles/nsys/qwen38_dflash2_d12_qkvg16`), per round:
  - wall 37.7 -> 36.0 ms; verification 33.3 -> 31.4 ms.
  - 798 launches (+16: the two qkvg sides).
  - qkvg 3.23 -> 1.34 ms: 179 -> ~76 us per layer (Q4 side 33.8 + Q5 side 41.7 us),
    ~577 GB/s.

  Across the four commits the d12 round went from 51.4 to 36.0 ms (-30 %).

K-split MMA instances in the d12 round. Medians over 121 rounds. Bytes are the stored
weights: Q4_G64 34 bytes and Q5_G64 42 bytes per 64 weights. Activations and outputs are
under 0.2 MB and are ignored.

| Instance | Kernel, grid | Rows x K | MB | Calls per round | Median us | GB/s |
|---|---|---|---:|---:|---:|---:|
| mlp gate+up (verify) | `q4_ksplit_mma`, 2176 | 34816 x 5120 | 94.7 | 64 | 155.7 | 608 |
| GDN in_proj Q4 side (verify) | `q4_ksplit_mma`, 256 | 4096 x 5120 | 11.1 | 48 | 20.9 | 532 |
| attn qkvg Q4 side (verify) | `q4_ksplit_mma`, 448 | 7168 x 5120 | 19.5 | 16 | 33.8 | 576 |
| DFlash2 proposal head (drafter) | `q4_ksplit_mma`, 8192 | 131072 x 5120 | 356.5 | 1 | 518.8 | 687 |
| GDN in_proj Q5 side (verify) | `q5_ksplit_mma`, 768 | 12288 x 5120 | 41.3 | 48 | 60.9 | 678 |
| attn qkvg Q5 side (verify) | `q5_ksplit_mma`, 448 | 7168 x 5120 | 24.1 | 16 | 41.7 | 577 |
| mlp down (verify) | `q5_ksplit_mma`, 320 | 5120 x 17408 | 58.5 | 64 | 102.8 | 569 |
| o_proj/out_proj (verify) | `q5_ksplit_mma`, 320 | 5120 x 6144 | 20.6 | 64 | 38.2 | 540 |

With the exact byte counts, gate+up reaches 608 GB/s, not the ~650 estimated above at
4.5 bits per weight. The largest instances reach ~680-690 GB/s (proposal head, GDN Q5 side),
so gate+up (11.4 ms per round, a third of the verification) is ~12 % below what the same
kernel family already reaches. The small sides (GDN Q4 at 21 us, o_proj at 38 us) pay a
fixed launch-and-tail cost for their size.

### Phase 3: `q4_ksplit_mma` launch bound (`1acd5db`, 2026-09-25)

`1acd5db` sizes the launch bound of `q4_ksplit_mma` from its shared-memory tile, with no
register spills. Validated at HEAD `e710b6b` against the `cccaace` build, with the same
method, desktop at 60 Hz and server off.

- Six tests pass: `ninfer_linear_swiglu_q4_a16_test`, `ninfer_linear_q4_a16_test`,
  `ninfer_attn_input_proj_test`, `ninfer_gdn_input_proj_test`,
  `ninfer_gdn_input_proj_conv_snapshot_test` (0 failures against the sampled FP64 reference)
  and `ninfer_gdn_input_proj_conv_record_test`.
- d12 round (`profiles/nsys/qwen38_dflash2_d12_phase3`), `q4_ksplit_mma` instances, median
  over 121 rounds, `cccaace` -> `1acd5db`. GB/s is against the ~845 GB/s read ceiling:

  | Instance | Median us | GB/s | Of the ceiling |
  |---|---|---|---|
  | mlp gate+up, 34816 x 5120 (x64 per round) | 155.7 -> 118.2 | 608 -> 801 | 72 -> 95 % |
  | GDN in_proj Q4 side, 4096 x 5120 (x48) | 20.9 -> 19.0 | 532 -> 585 | 63 -> 69 % |
  | attn qkvg Q4 side, 7168 x 5120 (x16) | 33.8 -> 31.2 | 576 -> 624 | 68 -> 74 % |
  | DFlash2 proposal head, 131072 x 5120 (drafter, x1) | 518.8 -> 421.8 | 687 -> 845 | 81 -> 100 % |

  The Q5 instances are unchanged: down 102.0 us, o_proj/out_proj 38.4, GDN Q5 side 61.0,
  qkvg Q5 side 41.3.
- Per round: wall 36.0 -> 33.5 ms; verification 31.4 -> 29.0 ms; gate+up 11.4 -> 9.0 ms;
  drafter 3.53 -> 3.48 ms.
- Plain runs (`scenario_code_python`, `--no-thinking --greedy --lm-head-draft`), ms per round
  `cccaace` -> `1acd5db`, and text md5 against `cccaace`:

  | Run | ms per round | tok/s | Text |
  |---|---|---|---|
  | DFlash2 d6 | 29.6 -> 30.4 | 126.8 -> 125.4 | identical |
  | DFlash2 d12 | 35.5 -> 33.1 (-7 %) | 119.7 -> 128.9 | identical |
  | DFlash2 d15 | 37.1 -> 33.9 (-9 %) | 111.0 -> 122.2 | identical |
  | Qwen3.8 MTP 3 | 25.9 -> 26.6 | 124.6 -> 122.2 | identical (md5) |
  | Bonsai MTP 2 | 12.2 -> 12.2 | 149.6 -> 148.8 | identical (md5) |

  d6 and MTP 3 move by about 0.8 ms, within the run-to-run spread of these 3-5 s decodes at
  60 Hz (±0.5-0.9 ms in the earlier rounds). On this prompt d12 (128.9 tok/s) is now the
  fastest DFlash2 window, ahead of d6 (125.4).

Across the five commits the d12 round went from 51.4 to 33.5 ms (-35 %), and d12 decode from
78.7 to 128.9 tok/s.
