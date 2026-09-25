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

tg128 reads ~16 GiB of weights per token, about 800 GB/s, which is ~80 % of the 4090's
1008 GB/s.

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
