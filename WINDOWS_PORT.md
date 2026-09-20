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

## Model artifact — use a v2 artifact

**Important:** this engine reads **v2** `.ninfer` artifacts (magic `NINFER\0\2`). The
HuggingFace repos have since migrated their `main` to **v3** (`NINFER\0\3`), which this
build rejects with `artifact magic is not NInfer v2`.

Download a v2 revision instead of `main`. For Qwen3.8-27B the pre-v3 commit is `dc370fb`:

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
