# Traspaso: trabajo de rendimiento de NInfer en la RTX 4090 (Bonsai y Qwen3.8)

Este documento es para la sesión de Claude Code que trabaja en el PC del usuario (Windows, RTX 4090).
Hasta ahora el trabajo se repartía entre dos sesiones. Una sesión en la nube, sin GPU, escribía el
código, lo compilaba parcialmente y lo subía. Esta sesión local compilaba, testeaba y medía en la GPU.
Desde ahora esta sesión hace las dos cosas. Es un plan temporal de trabajo activo: bórralo cuando
todo lo pendiente esté cerrado (AGENTS.md: "Temporary plans are useful only for active work").

El usuario escribe en español; respóndele en español, con claridad y sin exagerar resultados.

---

## 1. Cómo trabajar (lo más importante)

Lee `AGENTS.md` completo antes de tocar código. Además, estas reglas salen de errores reales de este
proyecto:

1. **Medido o estimado, siempre dicho.** Toda cifra que no midas en la GPU es una estimación y hay
   que decirlo. Las estimaciones de la sesión en la nube fallaron varias veces:
   - La optimización cp.async de la atención `rk4v4-e8` iba a bajar de 284 a ~180 µs y bajó a 260 µs.
   - El reparto de QK en parejas de warps (`0354c53`) empeoró el kernel de 260 a 326 µs.
   - La fusión A8 (`c1eb910`) no dio ganancia medible en la ronda.

   Ninguna ganancia es real hasta medirla.
2. **Correctitud antes que velocidad.** Todo cambio numérico se valida contra el oráculo FP64
   independiente de su test. No aflojes un criterio de test para que pase: si un test falla, busca la
   causa. Ejemplo real: el test nuevo de `rk4v4-e8` falló por un doble redondeo a BF16, que era un bug
   real (arreglado en `787766f`), no un criterio demasiado estricto.
3. **Un cambio a la vez cuando mides rendimiento.** Hay ruido de ±1 ms por ronda entre corridas: el
   escritorio de Windows usa la misma 4090 a 60 Hz, más deriva térmica y de reloj. Para atribuir una
   ganancia:
   - compara binarios viejo y nuevo en la misma sesión y uno tras otro;
   - mira el tiempo por kernel en nsys/ncu, no solo el ms por ronda;
   - si la diferencia es menor que el ruido, dilo: "dentro del ruido".
4. **Reporta los fracasos tal cual.** Si algo da peor, se escribe peor. Si algo se revierte, se
   escribe por qué.
5. **Registra cada resultado** en `docs/maintainer/bonsai-ternary-design.md`, sección 9.1 (lista
   numerada "Current state and next steps"). Anota el commit medido, el comando, el hardware y los
   números. Los resultados de Qwen3.8 van en `WINDOWS_PORT.md`.
6. **Commits:** usa el formato Conventional Commits (`perf(...)`, `fix(...)`, `test(...)`,
   `docs(...)`). Trabaja en la rama `feat/bonsai-ternary` del repo `JGamboa/ninfer-4090-windows`.
   No subas un cambio de rendimiento sin haber pasado sus tests.

### Trampas de CUDA que ya nos mordieron en este proyecto

- **48 KiB de memoria compartida estática por kernel.** Con `-rdc=true` el exceso no lo detecta el
  compilador: aparece recién en el enlace de dispositivo (`nvlink error: uses too much shared data
  (0xc040 bytes, 0xc000 max)`). Si agregas un `__shared__`, calcula el total de **cada**
  instanciación de la plantilla, sobre todo las variantes anchas (Br = 48).
- **`bar.sync` con el ID en un registro reserva las 16 barreras del hardware** y deja 1 CTA por SM.
  Pasó en `0354c53`: la ocupación bajó de 15 a 8 warps activos. Usa IDs inmediatos
  (`bar.sync 1, 64;`). Para verificarlo en el binario:
  `cuobjdump -sass <obj> | findstr "BAR.SYNC"` no debe mostrar `BAR.SYNC R..`.
- **Índices dinámicos en arrays locales** (`a[2*part]` con `part` variable) mandan el array a memoria
  local (stack). Revisa `-Xptxas -v` o `cuobjdump --dump-resource-usage` (STACK debe ser 0).
- **`__launch_bounds__(threads, minBlocks)`** limita los registros. Un límite mal puesto causa spills
  (caso real: Q4 K-split con 40 registros y spills, arreglado con `q4_ksplit_resident_ctas`).
- **La 4090 tiene 128 SMs** (`kTargetSmCount`). Revisa el tamaño de los grids y cuántas olas
  (waves) salen; el código original estaba pensado para los 170 SMs de la 5090.
- **Techo de lectura medido: ~845 GB/s**, no los 1008 del datasheet. Compara los GB/s contra 845.

---

## 2. Entorno

- **Repo:** `E:\LLM\ninfer-4090-bonsai`, rama `feat/bonsai-ternary`. Hay otro checkout Qwen-only en
  `E:\LLM\ninfer-4090-winport`.
- **Build:** abre una shell con MSVC (`vcvars64.bat`) y CUDA en el PATH, y corre
  `cmake --build build -j`. Para configurar desde cero, sigue `WINDOWS_PORT.md`, sección "Building on
  Windows" (`-DCMAKE_CUDA_ARCHITECTURES=89`).
- **Python:** `.venv\Scripts\python.exe` (3.12; on this machine it has always been 3.12).
- **Artefactos:**
  - `E:\LLM\bonsai2_27b_vl.ninfer`: Bonsai 2 27B ternario t5, con MTP en Q8 y visión.
  - `E:\LLM\qwen3_8_27b.ninfer`: Qwen3.8 27B Q4/Q5.
  - Fuentes de Bonsai: `E:\LLM\Ternary-Bonsai-2-27B-PTQ1_0.gguf` y
    `E:\LLM\bonsai\Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf`.
- **Servidores (.bat del usuario):** `start-bonsai-server - ninfer.bat` (Bonsai) y
  `E:\LLM\ninfer\start-ninfer-server.bat` (Qwen3.8: DFlash2 d6, 3 lanes, 100K de KV `rk4v4-e8`,
  `--no-cuda-graph`).
- **Perfiles:** `profiles\nsys\`, `profiles\ncu\` y `profiles\bench\`.
- **Hardware:** RTX 4090 (sm_89, 128 SMs, 24 GB, 72 MB de L2). El monitor está conectado a la 4090 a
  60 Hz, así que el compositor de Windows le quita tiempo a la GPU.

---

## 3. Estado de la rama (de lo más nuevo a lo más viejo)

| Commit | Qué | Estado |
|---|---|---|
| `1515b53` | docs: la lentitud del reparto en parejas se reproduce con la GPU libre | registro |
| `2f4789e` | fix: IDs inmediatos en la barrera de la pareja de warps (decode small-T) | **sin compilar ni medir** |
| `e319660` | perf: el kernel de prefill int8/`rk4v4` (`prompt_i8.cuh`) reescrito con pipeline entre tiles (~1000 líneas) | **sin compilar ni medir** (ptxas OK en la nube) |
| `ae421af` | perf: ruta con tensor cores de t5 para T = 5..32 (varias lanes) | **sin compilar ni medir** (ptxas OK en la nube) |
| `1bde24c` | feat: simulador `ngram-mod` (`tools/spec_sim/`), `NgramDraftPool` y plan | tests de Python y C++ pasados en la nube |
| `531f09b` | feat: capa MTP en Q5/Q4/mezcla, más recetas del conversor | **sin compilar** (tests de Python pasados) |
| `19d7567` | fix: el reparto por parejas dentro de 48 KiB | validado: correcto, pero **326 µs contra 260 µs** (peor) |
| `0354c53` | perf: QK repartido en parejas de warps, prefetch de `block_table`, `pos[]` fuera del loop | peor por la barrera con registro; ver `2f4789e` |
| `b14ed06` | perf: cp.async de los códigos KV empaquetados (decode small-T) | validado: −8,4 % en el kernel |
| `787766f` | fix: rotación inversa de V en FP32 antes del único redondeo BF16 | validado: el oráculo pasa |
| `8b202c4` | test: oráculo FP64 para `rk4v4-e8` (`--rk4v4-e8-only`) | validado |
| `c1eb910` | perf: rmsnorm y SwiGLU fusionados en la cuantización A8 | validado: correcto, ganancia dentro del ruido |

`build\` todavía está compilado en `19d7567`.

---

## 4. Tarea inmediata: compilar, testear y medir todo lo pendiente

Hazlo en este orden. Si algo no compila o un test falla, para ahí, busca la causa y arréglala (secciones 5 y 6).

**0. Guarda los binarios viejos para las comparaciones.** Antes de recompilar, copia a
`build_19d7567\` estos binarios: `ninfer.exe`, `ninfer-serve.exe`, `ninfer_t5_bench` y
`ninfer_causal_softmax_attention_bench`.

**1. Build completo:** `cmake --build build -j`.

**2. Tests.** Todos deben dar PASS/OK:
- `ninfer_linear_t5_test` (casos nuevos T = 5..72) y `ninfer_attn_input_proj_test`
- `ninfer_linear_q4_a16_test` y `ninfer_linear_q5_a16_test`
- `ninfer_softmax_attention_test`, completo y con `--rk4v4-e8-only`
- `ninfer_kv_cache_append_test`
- `ctest -R "prism_loading|ngram_pool"`
- `python -m pytest tests/test_spec_sim.py tests/convert/test_bonsai_recipe.py`

**3. Regresión rápida:**
- Perplejidad quick de Bonsai (sección 9.1, punto 5). Debe dar ~5.8549.
- Los 6 prompts de Bonsai con MTP draft 2, `--lm-head-draft` y greedy, con 1 lane. No debe haber
  regresión.
- md5 del texto de Qwen3.8 con KV int8 a contexto corto. Debe salir igual al anterior. Con `rk4v4-e8`
  sí puede cambiar, y es esperable.

**4. Atención de decode a 128K (`2f4789e`).**
- ncu del mismo lanzamiento de siempre: Bonsai 128K `rk4v4-e8`, `--launch-skip 200`.
- Esperado: vuelve a 2 CTAs por SM (~15 warps activos), menos espera en barreras que `b14ed06` y menos
  de 260 µs.
- Si sigue en 1 CTA por SM o más lento que 260 µs, revierte el reparto por parejas (sección 5).

**5. Varias lanes (`ae421af`).**
- `ninfer_t5_bench`, binario viejo contra nuevo, en T = 6, 8, 9, 12, 16, 24 y 32 por forma. Con
  T = 1..4 no debe cambiar nada.
- Corre esto con el binario nuevo y con `build_19d7567\ninfer-serve.exe`:

  ```
  python tools/bench/run_serve_concurrency.py --serve build\apps\ninfer-serve.exe --artifact bonsai=E:\LLM\bonsai2_27b_vl.ninfer --mode mtp2 --sampling greedy --suite decode-saturation --concurrency 1 --concurrency 2 --concurrency 3 --kv-capacity auto --output profiles\bench\bonsai_lanes_after
  ```

  Antes de medir, confirma que el script pasa `--lm-head-draft`.
- Punto de partida medido: 143,6 tok/s y 14,49 ms por ronda con 1 lane; 166,9 tok/s en total y
  37,09 ms por ronda con 3 lanes.
- Estimación (sin medir): ~2× el total con 3 lanes.

**6. Prefill a contexto largo (`e319660`).**
- Corre esto con el binario viejo y con el nuevo:

  ```
  ninfer_causal_softmax_attention_bench --entry append --geometry d256-h24-kv4 --kv-dtype int8 --batch 1 --tokens 1024 --context 8192,32768,65536,131072 --mapping fragmented --execution eager --cache cold --warmup 5 --repeat 21
  ```

  y lo mismo con `--kv-dtype rk4v4-e8`.
- Corre `long_niah_64k` y `long_niah_128k`, con int8 y con `rk4v4-e8`:
  `--max-context 262144 --prefill-chunk 1024 --no-thinking --max-new 128 --greedy`.
  - La respuesta debe ser exactamente `ORCHID=493817; COLOR=COBALT`.
  - Tiempos de prefill antes: 24,9 s (int8) y 25,7 s (`rk4v4-e8`) a 64K; 61,0 s y 65,5 s a 128K.
- Estimación del agente: −13 a −23 % a 128K.

**7. MTP en Q4/Q5 (`531f09b`).**
- Convierte las recetas `bonsai2_27b_mtp_q5`, `bonsai2_27b_mtp_q4` y `bonsai2_27b_mtp_q4q5`. El
  comando está en `docs/maintainer/bonsai-ternary-conversion.md`.
- Compáralas contra el Q8 en los 6 prompts: tok/s, **tokens por ronda** y ms por ronda. Si la
  aceptación cae, la variante no sirve aunque la ronda sea más rápida.
- Estimación: ≤0,5 ms por ronda (~4 %).

**8. Simulador `ngram-mod` (no usa la GPU).**
- `pip install tokenizers`
- Luego:

  ```
  python -m tools.spec_sim %USERPROFILE%\.claude\projects\<proyecto>\*.jsonl --claude-code --artifact E:\LLM\qwen3_8_27b.ninfer --ngram-n 8,12,24 --caps 15,32,64 --json profiles\spec_sim.json
  ```

- Reporta los tokens aceptados por ronda y el % de rondas que pasarían de 15 tokens.
- Ojo: esas transcripciones las escribió otro modelo, así que solo miden cuánto se repite el tipo de
  trabajo, no el comportamiento de Qwen3.8 o Bonsai.

---

## 5. Punto seguro para volver atrás

**El último estado del código validado por completo es `73aee4b`.** Ese commit solo agrega
documentación; el código es el mismo de `b14ed06`. En ese estado:
- todos los tests de atención, el de `rk4v4-e8`, el de KV append, el de t5 y la perplejidad dieron PASS;
- la atención de decode `rk4v4-e8` a 128K midió 260 µs, el mejor valor hasta ahora;
- el texto de Bonsai int8 salió con el mismo md5 de siempre.

Todo lo posterior a `73aee4b` falló o está sin validar:
- `0354c53` + `19d7567` dieron **peor** (326 µs);
- `2f4789e`, `531f09b`, `1bde24c`, `ae421af` y `e319660` están **sin compilar ni medir**;
- los commits `docs(...)` posteriores solo registran mediciones, así que se conservan.

**Para tener binarios buenos mientras se depura:** compila `73aee4b` en una carpeta aparte y usa esos
binarios en los `.bat` si el build nuevo falla:

```
git worktree add E:\LLM\ninfer-known-good 73aee4b
cd E:\LLM\ninfer-known-good
:: configurar y compilar igual que en WINDOWS_PORT.md, "Building on Windows"
```

**Para volver atrás en la rama sin perder lo demás:** usa `git revert` sobre el commit que falle. No
hagas `git reset --hard` ni `push --force` en `feat/bonsai-ternary`: se pierden las mediciones y los
otros cambios. Así se revierte cada parte:

| Si falla... | Revertir (en este orden) | Queda como |
|---|---|---|
| La atención de decode no recupera las 2 CTAs por SM o sigue por encima de 260 µs | `git revert 2f4789e 19d7567 0354c53` | kernel de `b14ed06` (260 µs) |
| Ruta de varias lanes (t5 T = 5..32) | `git revert ae421af` | rutas GEMV/GEMM anteriores |
| Kernel de prefill | `git revert e319660` | kernel de prefill anterior, validado |
| Capa MTP Q4/Q5 | `git revert 531f09b` | MTP solo en Q8, que ya funcionaba |
| Simulador `ngram-mod` | `git revert 1bde24c` (no toca el runtime, casi nunca hará falta) | – |

Si `git revert` choca con los commits de documentación, conserva la versión de la documentación y
anota en 9.1 qué se revirtió y por qué.

---

## 6. Qué hacer si algo falla

- **Un error de compilación:** lee el archivo y la línea y arréglalo. Si es de enlace
  (`nvlink ... shared data`), revisa la sección 1 y calcula la memoria compartida de cada
  instanciación.
- **Falla un test de `ae421af` (t5 T = 5..32):**
  - El kernel nuevo es `small_t_kernel<NTiles>` en `src/ops/linear/t5/t5_a8.cuh`. El mapeo de
    fragmentos MMA m16n8k32, la permutación de k y el swizzle solo se revisaron a mano.
  - Si el error aparece en todos los T del rango, sospecha del mapeo de fragmentos. Si solo aparece en
    algunos bordes, sospecha del enrutamiento en `t5_project.cu`.
  - Si no se resuelve rápido: `git revert ae421af` y deja todo lo demás.
- **Falla un test de `e319660` (prefill):**
  - La reescritura es grande. Si no se arregla en poco tiempo, haz `git revert e319660`: el kernel
    anterior está validado.
  - Ojo: antes ya había dos variantes sin cobertura de oráculo, `rk4v4` sin E8 y `rk8v4`.
- **`2f4789e` no recupera las 2 CTAs por SM, o el kernel queda por encima de 260 µs:** revierte
  `2f4789e`, `19d7567` y `0354c53`, en ese orden. Así se vuelve al kernel de `b14ed06`, el mejor medido.
- **Una variante MTP Q4/Q5 baja la aceptación:** Bonsai se queda con Q8. Anota el resultado igual.

---

## 7. Números de referencia medidos (para comparar)

- **Bonsai** (t5, MTP draft 2, `--lm-head-draft`, 60 Hz):
  - Decode medio ~163–166 tok/s en los 6 prompts, ~12,3–13 ms por ronda y ~1,86 tokens por ronda.
  - pp512 3061 tok/s y pp2048 3349 tok/s.
  - Perplejidad quick 5.8549.
  - Eval de calidad 43/45 (Qwen3.8: 44/45).
- **Reparto de la ronda de Bonsai a contexto corto** (sección 9.1, punto 6):
  - GEMV t5: 7,70 ms (62 %).
  - Capa MTP en Q8: 1,32 ms.
  - GDN: 0,94 ms.
  - Head de propuesta: 0,83 ms.
  - Cuantización: 0,70 ms.
  - Atención: 0,46 ms.
- **Qwen3.8:**
  - MTP3: ~124 tok/s y 25,9 ms por ronda (3,23 tokens por ronda).
  - DFlash2 d12: ~33,5 ms por ronda.
  - Los CUDA graphs valen <5 %.
  - Con 3 lanes, 100K de KV y graphs, faltan 1182 MiB de VRAM.
- **Atención `rk4v4-e8` a 128K:** 260 µs por llamada en `b14ed06` (el mejor medido) y 326 µs en
  `19d7567`.

---

## 8. Ideas pendientes, de mejor a peor relación ganancia/esfuerzo

Ninguna está empezada salvo que se indique. Las ganancias son estimaciones.

1. **Monitor en la GPU integrada.** El compositor a 60 Hz le quita ~2,2–2,8 ms a cada ronda de ~15 ms
   (sección 9.1, punto 2). Ganancia estimada: +15–18 %. Es solo una prueba de configuración: conectar
   el monitor a la placa madre y medir los 6 prompts.
2. **Qwen3.8 con MTP3 y CUDA graphs** en vez de DFlash2 d6 con `--no-cuda-graph`. MTP3 midió 124 contra
   112 tok/s, y sus graphs cuestan 86 MiB por lane contra 480 de DFlash2. Hay que confirmar que entra en
   VRAM con los lanes y el KV que usa el usuario.
3. **`--proposal-rows 34816`** en Bonsai para inglés y código: +2,1 % medido, mismo texto, pero pierde
   en otros idiomas.
4. **`ngram-mod` fase 1** (plan en `docs/maintainer/ngram-speculation-plan.md`, 4–6 días): borradores
   del host por la ronda de MTP, hasta 15. Decide con el simulador (paso 8) antes de empezar. Pasar de
   16 cuesta 4–10 días más y solo se justifica si el simulador lo muestra.
5. **Readaptar la MTP de Bonsai con autodestilación.** Es la palanca más grande: +20–40 % estimado.
   - Hoy la MTP de Bonsai es la de Qwen3.8, entrenada sobre el modelo sin cuantizar, y acierta ~56 %
     por posición.
   - Estudio de factibilidad (resumen):
     - Reentrenar solo la capa MTP (~425 M parámetros) contra el top-32 de Bonsai.
     - Datos: estados ocultos finales, después de `final_norm`, exportados desde NInfer (hoy no hay
       herramienta: `score_tokens` solo devuelve logprobs).
     - Entrenamiento en PyTorch en la 4090: ~16–18 GB.
     - Generar 5M tokens: ~8 h. Entrenar: ~1 h.
     - Primero, un piloto de ~200K tokens. Se sigue solo si el acierto sube a ≥65 %.
     - La calidad no cambia, porque la verificación es sin pérdida.
     - La MTP entrenada se puede reinsertar con el conversor (nombres HF `mtp.*`).
   - El usuario todavía no decidió si hacerlo.
6. **Atención de decode a 128K, siguiente paso:** doble buffer, o expandir K directo en registros para
   liberar el tile K de la memoria compartida. Ganancia en la ronda: ~2–4 % a 128K, ~0 a contexto
   corto. Poca prioridad.

Ideas ya evaluadas y descartadas: AirLLM (streaming de capas: no aplica, los modelos caben), NVIDIA
Dynamo (es para varias GPU), rk3v3/rk4v3 (ahorran lo mismo que `rk2v4`) y la A/B limpia de `c1eb910`
(la ganancia es menor que el ruido).

---

## 9. Documentos de referencia

- `AGENTS.md`: reglas del repo, obligatorio.
- `docs/maintainer/bonsai-ternary-design.md`, sección 9.1: estado vivo y mediciones de Bonsai.
- `WINDOWS_PORT.md`: build en Windows, mediciones de Qwen3.8 y presupuesto de VRAM.
- `docs/maintainer/op-development.md`: cómo calificar un kernel (oráculo y rendimiento).
- `docs/maintainer/ngram-speculation-plan.md`: plan de `ngram-mod`.
- `tests/README.md` y `bench/README.md`: comandos de tests y benchmarks.
