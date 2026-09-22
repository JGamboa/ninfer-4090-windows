# TODO — Plan de optimización de velocidad (RTX 4090, Windows 11, MSVC)

Fecha: 2026-09-22. Rama: `main`. Escrito para ejecutarse paso a paso con un agente de IA
en la máquina Windows que tiene la RTX 4090. Cada tarea dice qué tocar, por qué, cómo
validar y qué hacer si empeora.

Estado de partida medido en este port (README.md y WINDOWS_PORT.md en `main`):

| Métrica | Valor actual | Config |
|---|---:|---|
| Decode, código, thinking off | 210.9 tok/s (85.7% aceptación) | `--spec dflash2 --draft-tokens 12` |
| Decode, MTP3 (config vieja) | 137.8 a 149 tok/s | `--spec mtp --draft-tokens 3` |
| Prefill (Linux, fork base) | ~2,000 tok/s a 2K-32K, 1,561 a 128K | INT8 / rk4v4-e8 |
| Prefill (este port, sin confirmar) | 280 a 740 tok/s "según prompt" | prompts cortos probablemente |

Diagnóstico resumido:

- Decode con dflash2 verifica **13 tokens por ronda** (1 + 12 drafts). A 210.9 tok/s con 11.3
  tokens aceptados por ronda son ~19 rondas/s, es decir **~53 ms por ronda**. El piso de ancho de
  banda (17 GB de pesos a ~1 TB/s) es ~18 ms. Hay margen y está en las rutas de kernel para T=9..16.
- Prefill: el fork hermano `soohl/ninfer` midió +68% en la misma tarjeta con activaciones INT8.
  El ledger del fork (`docs/maintainer/port-ledger.md`) ya lo tiene como item #1 pendiente.
- Windows: cuatro ajustes baratos que el fork base descartó "por no aplicar en Linux".

---

## 0. Reglas para el agente que ejecute esto

1. **Una tarea = un commit.** Mensaje en inglés, primera línea corta, cuerpo con qué se midió.
2. **Medir antes y después de cada tarea** con el procedimiento de la sección 1. Si empeora o
   queda igual dentro del ruido (±1.5%), revertir el commit y anotar el resultado en este TODO.
3. **Nunca cambiar dos cosas de kernel a la vez.** El fork base rechazó micro-optimizaciones
   que perdían 52% en Linux; en MSVC el SASS es distinto, así que aquí hay que re-medir todo.
4. **Tests siempre verdes.** Configurar con `-DBUILD_TESTING=ON` y correr `ctest` tras cada
   cambio de kernel. Los tests de ops son bit-exactos contra oráculos.
5. **Bit-exactitud:** un cambio de kernel que altere numéricos (orden de acumulación, tiles)
   debe pasar además una prueba de needle-in-a-haystack a 60K+ tokens y un prompt de código
   fijo con `temperature 0` comparado token a token contra el build anterior.
6. Marcar cada tarea aquí como `[x]` con el número medido y el hash del commit.

Comandos base (desde una shell con `vcvars64.bat`, ver `winport_configure.bat`):

```bat
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_TOOLCHAIN_FILE=E:/LLM/vcpkg/scripts/buildsystems/vcpkg.cmake ^
  -DVCPKG_TARGET_TRIPLET=x64-windows -DCMAKE_CUDA_ARCHITECTURES=89 ^
  -DBUILD_TESTING=ON -DNINFER_BUILD_BENCHMARKS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

---

## 1. Baseline reproducible (hacer primero, sin tocar código)

- [ ] **1.1 Servidor en la config de producción**

```bat
build\apps\ninfer-serve.exe qwen3_8_27b.ninfer --host 127.0.0.1 --port 8080 ^
  --max-context 262144 --kv-capacity 262144 --prefill-chunk 1024 --kv-dtype rk4v4-e8 ^
  --spec dflash2 --draft-tokens 12 --preserve-thinking --max-shared-prefixes 8
```

  (`--max-shared-prefixes 8` es la corrección del PR #300 de upstream: con
  `max-concurrency 1` el catálogo por defecto tiene 4 slots y un request genera hasta 7
  candidatos, así que el reuso de prefijo se agota tras pocos turnos de Claude Code.)

- [ ] **1.2 Script de medición** `tools/bench/win_baseline.py` (crearlo). Debe:
  1. Mandar un prompt de código fijo de ~200 tokens con `enable_thinking:false`, `temperature 0`,
     `max_tokens 512`, y leer `timings` de la respuesta: decode tok/s y aceptación.
  2. Mandar un prompt largo fijo de **32K tokens** (needle-in-a-haystack) y un segundo de **120K**,
     y leer de `/metrics` el prefill tok/s computado (no wall).
  3. Repetir 3 veces cada uno, reportar mediana. Guardar en `docs/bench/win-<fecha>-<hash>.md`.
  4. Verificar que la aguja se recupera exacta en ambos prompts largos (gate de calidad).

- [ ] **1.3 Resolver la duda del prefill.** WINDOWS_PORT.md dice 280 a 740 tok/s. Si el 32K da
  menos de 1,800 tok/s hay un problema específico de Windows y hay que perfilarlo con Nsight
  Systems antes de seguir (sospechosos: paginación WDDM, tarea 2.4; flags MSVC, tarea 2.3).

- [ ] **1.4 Cuánto pesa cada parte de una ronda de decode.** Correr `nsys profile` sobre 20
  rondas de decode a 8K de contexto y a 100K. Anotar en el reporte: tiempo del draft dflash2
  (12 pasos AR del modelo chico), tiempo de verify (T=13) desglosado por kernel, tiempo de
  atención, tiempo entre el fin del graph y el siguiente launch (gap de host). Esto decide el
  orden real de la sección 4.

---

## 2. Windows: ajustes baratos (bajo riesgo, hacer en este orden)

- [ ] **2.1 TCP_NODELAY en el servidor HTTP.** `src/serve/http_server.cpp:237`, justo después
  de `server_.set_socket_options(...)`, añadir `server_.set_tcp_nodelay(true);`. cpp-httplib
  lo trae en `false` (`third_party/cpp-httplib/httplib.h:163`). Sin esto cada evento SSE
  espera el delayed-ACK del cliente. Afecta lo que ve Claude Code, no el engine.
  Validar: comparar tok/s medido por el cliente (script 1.2) antes y después.

- [ ] **2.2 `timeBeginPeriod(1)` al arrancar `ninfer-serve`.** En `apps/serve/main.cpp` dentro
  de `main()`, bajo `#if defined(_WIN32)`: `#include <timeapi.h>`, llamar `timeBeginPeriod(1)`
  al inicio y `timeEndPeriod(1)` al salir; enlazar `winmm` en `apps/CMakeLists.txt` para el
  target serve. Razón: `src/runtime/engine/engine_core.h:2326` espera 1 ms cuando hay lanes
  activos pero nada ejecutable (transacciones de context cache), y Windows redondea a 15.6 ms.
  Validar: en un turno de Claude Code con contexto grande, el tiempo hasta el primer token
  tras un cache hit debería bajar; el decode estable no cambia.

- [ ] **2.3 Flags de optimización MSVC.** En `CMakeLists.txt` bloque `if(MSVC)`, añadir:

```cmake
    $<$<COMPILE_LANGUAGE:C,CXX>:/O2>
    $<$<COMPILE_LANGUAGE:C,CXX>:/Ob3>
    $<$<COMPILE_LANGUAGE:C,CXX>:/Oi>
    $<$<COMPILE_LANGUAGE:C,CXX>:/Ot>
    $<$<COMPILE_LANGUAGE:C,CXX>:/arch:AVX2>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/O2>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Ob3>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Oi>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Ot>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/arch:AVX2>
```

  Origen: commits `aa8a1c98` y `60ff23d5` de `UDPSendToFailed/ninfer-4090`. El fork base los
  marcó "no aplicables" porque estaba en Linux. No copiar `-Xptxas=-O3` (ptxas ya usa O3).
  Validar: build limpio, `ctest` verde, baseline igual o mejor. Afecta al código host
  (tokenizer, planner, parser), no a los kernels.

- [ ] **2.4 Bloqueo de residencia D3D12 para la arena de VRAM.** Portar el commit `a35acf6a` de
  `UDPSendToFailed/ninfer-4090` (`src/core/arena.cu` +151 líneas, `src/targets/registry.cpp`,
  `src/CMakeLists.txt` enlaza `d3d12` y `dxgi`). Hoy `DeviceArena` usa `cudaMalloc` plano
  (`src/core/arena.cu:147`). Con 262K de contexto quedan ~1.4 GiB libres y WDDM puede paginar
  VRAM a RAM del sistema sin avisar; el commit crea un heap D3D12 compartido con
  `D3D12_RESIDENCY_FLAG_DENY_OVERBUDGET`, lo importa con `cudaImportExternalMemory` y cae a
  `cudaMalloc` si falla. El `arena.cu` de ellos difiere del nuestro: aplicar a mano, no
  cherry-pick. Validar: Task Manager > GPU > "Memoria GPU compartida" debe quedarse en ~0 durante
  un prompt de 200K; el decode a 200K de profundidad no debe caer respecto a 8K más de lo que
  cae en Linux (~25%). Riesgo: si D3D12 no está disponible (RDP, sesión sin escritorio) el
  fallback debe funcionar; probarlo.

- [ ] **2.5 Modo de sincronización CUDA (experimento A/B de una línea).** En
  `src/core/device.cu` constructor de `DeviceContext`, antes de `bind_to_current_thread()`,
  probar `cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync)` y por separado
  `cudaDeviceScheduleSpin`. Hoy no se llama y el hilo del engine hace spin por defecto mientras
  compite con 18 hilos de httplib. Medir decode con el script 1.2. Dejar el que gane, o nada.

- [ ] **2.6 Ajustes de Windows sin código.** Documentar en WINDOWS_PORT.md y medir cada uno:
  HAGS (Hardware-accelerated GPU scheduling) activado vs desactivado; plan de energía "Alto
  rendimiento"; cerrar navegadores y apps que retengan VRAM; en NVIDIA Control Panel, "CUDA
  - Sysmem Fallback Policy" en "Prefer No Sysmem Fallback" para `ninfer-serve.exe` (evita la
  misma paginación que 2.4 desde el driver).

- [ ] **2.7 Lector de artefacto: cola de lectura más profunda.** `src/artifact/reader.cpp:307-336`
  hace `ReadFile` overlapped y luego `GetOverlappedResult(..., TRUE)` en cada chunk de 1 GiB:
  profundidad 1. Emitir 2 a 4 lecturas en vuelo. Solo mejora el tiempo de carga del modelo,
  no la inferencia. Baja prioridad.

---

## 3. Prefill (la ganancia grande)

Orden por tamaño esperado de ganancia sobre esfuerzo. Todos con `ninfer_*_bench` antes.

- [ ] **3.1 GDN state passing: encajar el grid en una oleada.**
  `src/ops/linear_attention/gated_delta_net/chunked/state_passing.cu:19` lanza
  `H_v * D_STRIPS = 48 * 8 = 384` CTAs con `MIN_BLOCKS = 2` por SM
  (`state_passing.cuh:35`). En 128 SMs son 256 slots: dos oleadas, la segunda medio vacía.
  El kernel es secuencial sobre chunks (`state_passing.cuh:313`) y corre en 48 de 64 capas.
  Opciones, medir con `ninfer_gated_delta_net_bench` a T=1024 y 2688:
  a) Verificar con `cuobjdump -res-usage` si registros y smem permiten `MIN_BLOCKS = 3`
     (384 slots, una oleada). Requiere ≤85 regs/thread y ≤33 KB smem por CTA.
  b) Si no, repartir los 384 pares (head, strip) en un grid persistente de 128 CTAs con 3
     items cada una (una oleada exacta, mismo tiempo que hoy por item pero sin cola).
  Ganancia teórica: 25% de ese kernel. Bit-exacto si solo cambia el mapeo de bloques.

- [ ] **3.2 Tile K=128 en los GEMM MMA Q4 y Q5 de prefill.**
  `src/ops/linear/q4/q4_rowsplit_gemm_mma.cuh:71` y `q5/q5_rowsplit_gemm_mma.cuh:87` tienen
  `static_assert(kBlockK == kGroupK)` que fija un grupo de cuantización (64) por tile K. Q6
  (`q6_rowsplit_gemm_mma.cuh:75`) y W8 (`w8_rowsplit_gemm_mma.cuh:54`) ya admiten K=128.
  Con K=128 se pagan la mitad de barreras y decodes por tile en los dos GEMM más pesados
  (`mlp/gate_up` K=5120 y `mlp/down` K=17408). Pasos: generalizar `decode_weight` a dos
  grupos por tile, añadir schedule `*R64C128K128`, enrutarlo en `q4_dispatch.cpp` y
  `q5_dispatch.cpp` para T≥64, medir con `ninfer_linear_bench` y `ninfer_q5_linear_add_bench
  --k 17408` a T=1024. Referencia actual: Q5 down 1494.9 µs (122 TFLOP/s), fused Q4 swiglu
  3166 µs (115 TFLOP/s) en Linux; primero reproducir esos números en MSVC.

- [ ] **3.3 Sweep de ocupación en los GEMM de prefill.** Los schedules R64C128 de Q4/Q5/Q6
  usan 4 warps (128 threads), 45.5 KB smem estática, `MinBlocks=1`
  (`q4_rowsplit_gemm_mma.cu:65`). La proyección QKV con el mismo tile usa 8 warps y
  `MinBlocks=2` (`attn_input_proj/q4_q5/q4_q5_attn_input_gemm_mma.cu:137`). Barrer WN16 vs
  WN32 × MinBlocks 1/2 × Stages 2/3 con `ninfer_q4_linear_swiglu_variants_bench` (el mismo
  que decidió WN32 en sm_89) y `ninfer_linear_bench`. También probar el pipeline `PingPong`
  + `Cache::ca`, ya implementado pero solo cableado a los schedules C64.

- [ ] **3.4 Activaciones INT8 para los GEMM de prefill (el cambio grande).**
  Hoy Q4/Q5/Q6 decodifican a BF16 y usan `mma m16n8k16` BF16 (`q4_dispatch.cpp:93-102`
  ignora `LinearPolicy::AllowA8`). sm_89 tiene `mma.s8` y `mma e4m3`, ambos ya en
  `src/ops/common/mma.cuh:62-85` y usados por la atención. `soohl/ninfer` (port independiente
  del 4090) midió **3,548 a 3,684 tok/s a 8K vs 2,111 A16 (+68%)** con activaciones INT8
  grupo-64 solo en el prefill denso; decode y verify siguen en A16; artefacto sin cambios.
  Pasos: (1) leer su implementación y `docs/ada.md` de su repo; (2) implementar la ruta A8
  solo para T≥64 en Q4 y Q5; (3) `ninfer_linear_bench --policy a8`; (4) gate de calidad:
  perplexity (`ninfer-perplexity.exe`) vs A16 con diferencia ≤0.5%, needle a 60K y 120K,
  prompt de código a temp 0 comparado; (5) flag de CLI `--prefill-activations a8|a16` con
  A16 por defecto hasta que el gate pase. Es lossy: es una decisión de producto, no solo de
  velocidad. Impacto: TTFT de 131K de 89 s a ~53 s.

- [ ] **3.5 Constantes de 170 SMs que quedan.** El commit `4e53fcc` corrigió rope, gdn output,
  sparse_moe prefill y el cap de splits 5000-8198. Falta `src/ops/launcher/rmsnorm.cu:20`
  (`kRmsPrefetchBlocks = 170`, usar `device_sm_count()`), y en
  `src/ops/softmax_attention/dense/causal_cache/small_t.cu:239-242` los `160/320` (solo
  batch>1, irrelevante con `max-concurrency 1`; corregir por consistencia a
  `kTargetSmCount` y `2*kTargetSmCount`).

- [ ] **3.6 Sync por chunk de prefill.** `src/targets/qwen3_6/impl/runtime/text_context_impl.h:1327`
  hace `ctx_.synchronize()` al final de cada chunk. A chunk 1024 cuesta <0.1%. Solo vale si
  se implementa interleaving de prefill entre lanes (README "Known limits"). Baja prioridad.

---

## 4. Decode con dflash2 (T=13 por ronda)

Contexto: con `--draft-tokens 12` el verify corre con `width = 13`. Las tablas de rutas se
barrieron en el RTX 5090 (1.8 TB/s, 170 SMs, ~105 TFLOPS FP32). En el 4090 el cruce SIMT↔MMA
se mueve. Rutas activas hoy a T=13 (verificadas en `main`):

| GEMM (por capa) | Formato | Ruta a T=13 | Unidad |
|---|---|---|---|
| attention/query_key 7168×5120 | Q4 | `simt_r8_c4` (`q4_dispatch.cpp` caso 7168, t≤15) | CUDA cores, tile de 4 cols |
| attention/gate_value 7168×5120 | Q5 | `simt_r8_c4` (t≤16) | CUDA cores |
| attention/output 5120×6144 | Q5 | `simt_r8_c8` (t≤24) | CUDA cores |
| gdn/query_key 4096×5120 | Q4 | `simt_r8_c8` (t≤16) | CUDA cores |
| gdn/output 5120×6144 | Q5 | `simt_r8_c8` (t≤24) | CUDA cores |
| mlp/gate_up 34816×5120 fused | Q4 | `SmallTTiled` (`q4_linear_swiglu_plan.cpp` {2,32}) | Tensor cores |
| mlp/down 5120×17408 | Q5 | `simt_r8_c8` (t≤24) | CUDA cores |
| lm_head 248320×5120 | Q6 | `mma_r64_c16_k128` | Tensor cores |

- [ ] **4.1 Re-barrer los cruces SIMT↔MMA para T=9..16 en Q4 y Q5.** Con
  `ninfer_linear_bench` medir a T=13 (y 9, 12, 16) cada shape de la tabla con la ruta SIMT
  actual y con `mma_r64_c128` (existe) y, si se añade, un `mma_r64_c16_k128` como el de Q6.
  Nota: el caso Q4 7168 usa `c4` a T=9..15 pero `c8` a T=8 y 16, lo que huele a agujero de la
  tabla más que a decisión medida. Cambiar solo los umbrales en `q4_dispatch.cpp` y
  `q5_dispatch.cpp`; son bit-exactos por construcción (mismo FP32 accumulate, distinto orden:
  verificar con el test de ops). Es el lever de decode más probable para esta config.

- [ ] **4.2 Split-K para Q4 en decode.** Q5 tiene `simt_split4_exact`/`split2_exact` para T≤6
  (`q5_rowsplit_gemm_simt.cu:71-82`); Q4 no tiene equivalente y va sin split. Vale para MTP
  (T=4), no para dflash2 a T=13. Solo si se vuelve a MTP.

- [ ] **4.3 Perfil del draft dflash2.** 12 pasos autoregresivos del modelo chico por ronda
  (`dflash_impl.h:238-341`). Con el nsys de 1.4, si el draft pesa >30% de la ronda: revisar
  que sus GEMM W8 usen las rutas exact-T de `w8_config.h` (sí tienen tabla propia, no la rama
  `NINFER_SM86`) y si la cadena de 12 pasos tiene gaps entre kernels dentro del graph.

- [ ] **4.4 Atención de decode a width 13.** Ruta `ChunkedSmallT`
  (`causal_softmax_attention.cpp:377`) con rk4v4-e8. Medir con
  `ninfer_causal_softmax_attention_bench` a width 13 y ventanas 8K/32K/128K/256K. Cosas a
  revisar: tope `SmallTMaximumSplits = 85` en `geometry.cuh:12` (viene de 2×170 SMs; con
  4 KV heads son 340 CTAs, que sobre 128 SMs es una oleada y media si caben 2 por SM); la
  tabla de warps del tile en `small_t.cu:170-215`; y que los profile ends de
  `src/targets/qwen3_6_27b/impl/variant.cpp:124,136` sigan a cualquier cambio de tiers.

- [ ] **4.5 Fusionar la rotación inversa en el reduce.** Con `rk4v4-e8`,
  `small_t.cu:405` lanza `kv_cache_inverse_rotate_output_kernel` aparte tras el reduce, en
  cada una de las 16 capas de atención por ronda: 16 launches extra de CTAs de 32 hilos.
  Moverlo al epílogo del kernel reduce (`small_t.cuh:197`). Bit-exacto si se conserva el
  orden de operaciones. Ganancia pequeña (launch overhead dentro del graph), pero segura.

- [ ] **4.6 Separar `NINFER_SM86` de sm_89.** `src/CMakeLists.txt:323` define
  `NINFER_SM86=1` para el build 89 y activa rutas afinadas para el 3090 (82 SMs): por ejemplo
  `w8_config.h:59` y `w8_rowsplit_gemm_splitk.cu:33` bajan `KWarps` a 4 aunque 16 caben en
  48 KB con tile de 8 tokens. Para el 27B las geometrías MTP y DFlash2 tienen tablas propias
  sin esa rama, así que el impacto en esta config es bajo. Hacerlo introduciendo
  `NINFER_SM89` y decidiendo sitio por sitio (hay ~12). Prioridad baja salvo que se use el 35B.

- [ ] **4.7 Re-medir en MSVC las micro-optimizaciones de dequant rechazadas en Linux.**
  `docs/udp-fork-comparison.md:154-180`: commits `73f3d7be`, `8f298555`, `b8ddda48`,
  `d9d701bc` del fork UDP perdieron hasta 52% en GCC/CUDA 13.1, pero el propio doc sospecha
  que en MSVC el SASS de `bfe.s32` y shuffles es distinto. Solo `ninfer_q5_linear_add_bench`
  y `ninfer_q4_linear_swiglu_bench`; adoptar solo si ganan en los shapes de producción.

---

## 5. Mejoras del PR #300 de upstream (Neroued/ninfer) que sí aplican

Analizado el 2026-09-21. Rutas remapeadas: `src/models/qwen3_5/frontend/` →
`src/targets/qwen3_6/impl/frontend/`, `src/runtime/engine/context_cache/` →
`src/runtime/engine/`. 23 de 33 archivos aplican limpios tras el remapeo.

- [x] **5.1 `--max-shared-prefixes 8`** en la línea de arranque (ver 1.1). Sin código.
- [ ] **5.2 Parser de tool calls tolerante a XML estilo Anthropic.** Aplicar los hunks de
  `tool_call_parser.cpp/.h` y `tests/test_tool_call_parser.cpp` (188 líneas de tests). Hoy el
  parser solo acepta `<function=nombre>`; cuando Claude Code habla con Qwen por la ruta
  Anthropic, el modelo imita `<function_calls><invoke name="...">` y la llamada se pierde.
  Ojo con el decoder de streaming: con el cambio, cualquier `<function>` o `<invoke ` en
  prosa o código dispara modo tool-call hasta el final del stream. Aceptable, pero documentar.
- [ ] **5.3 Planner de context cache:** hunks de `resource_manager.h`,
  `materialization_planner.h`, `context_portfolio_value.h`, `shared_capture_planner.h`,
  `include/ninfer/types.h`, `engine.cpp:79` (default a 7) y `tests/test_resource_manager.cpp`.
  Aplican limpios salvo un hunk de contexto en `materialization_planner.h`.
- [ ] **No portar:** NVFP4 ragged SwiGLU (sm_89 no compila NVFP4), los guards eliminados de
  continuación de assistant (`translate.cpp`, `anthropic_messages_request.cpp`, chat template)
  y `response_format: json_schema` aceptado sin aplicarse.

---

## 6. Cosas verificadas que ya están bien (no tocar)

- Sampling en GPU; logits nunca se copian al host; un H2D y un D2H por ronda, ambos dentro
  del CUDA graph; un solo `cudaGraphLaunch` y un `cudaStreamSynchronize` por ronda.
- Detokenización incremental O(1) por token; sin logging ni disco por token.
- El `wait_for(1 ms)` de `engine_core.h:2326` **no** corre en decode estable (las ramas de
  decode/prefill hacen `continue` antes). Solo importa en transacciones (tarea 2.2).
- Los grids con literal 170 en rope, gdn output y sparse_moe prefill ya usan
  `device_sm_count()` (commit `4e53fcc`).
- PDL (programmatic dependent launch) está correctamente desactivado: requiere sm_90.

---

## 7. Referencias

- Fork hermano con trabajo Windows/WDDM: https://github.com/UDPSendToFailed/ninfer-4090
  (commits `a35acf6a`, `aa8a1c98`, `60ff23d5`, `3482bc46` L2 persistence para MTP).
- Fork hermano con prefill INT8: `soohl/ninfer` (ver `docs/maintainer/port-ledger.md:395-425`).
- PR upstream analizado: https://github.com/Neroued/ninfer/pull/300
- Ledger del fork base con la regla "kernel-bench antes de cualquier pick":
  `docs/maintainer/port-ledger.md:105-108` y `docs/udp-fork-comparison.md`.
