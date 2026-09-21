# Compile policy is owned here; each semantic family explicitly contributes sources.
add_library(ninfer_ops STATIC)
ninfer_internal_includes(ninfer_ops)
ninfer_cuda_archive(ninfer_ops)
target_link_libraries(ninfer_ops PUBLIC ninfer_core)

# Warp-specialized SM120 NVFP4 kernels use setmaxnreg to transfer registers from
# producer to consumer warpgroups. RDC makes ptxas discard that contract. These
# self-contained kernels therefore use a non-RDC archive and host launchers.
# Fork-local: the archive exists only for sm_120a. The sm_89 build has no FP4
# tensor cores; its launchers are stubbed below and the family manifests only
# contribute sources to this archive when the target exists.
if(CMAKE_CUDA_ARCHITECTURES STREQUAL "120a")
  add_library(ninfer_nvfp4_non_rdc STATIC)
  ninfer_internal_includes(ninfer_nvfp4_non_rdc)
  ninfer_cuda_non_rdc_archive(ninfer_nvfp4_non_rdc)
  target_link_libraries(ninfer_nvfp4_non_rdc
    PRIVATE ninfer_core ${NINFER_CUDART_TARGET} CUDA::cuda_driver)
  target_link_libraries(ninfer_ops PRIVATE ninfer_nvfp4_non_rdc)
endif()

include(basic_sources.cmake)
include("${CMAKE_CURRENT_LIST_DIR}/attn_input_proj/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/candidate_selector/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/context_kv_materialize/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/dynamic_grouped_conv/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/gdn_gating_proj/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/gdn_input_proj/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/kv_cache/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_add/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_attention/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_pair/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_swiglu/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_topk/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/rmsnorm_rope/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/softmax_attention/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/sparse_moe/sources.cmake")

# Fork-local sm_86/sm_89 build. The family manifests above are upstream's; the
# NVFP4 A4 GEMMs and the NVFP4-valued KV-cache kernels they list need sm_100+
# (cvt.e2m1x2, FP4 tensor cores). Drop those translation units after every
# manifest has contributed, replace their launchers with host stubs that reject
# the routes, and export NINFER_SM86 for the #if guards in the kernels, models
# and tests.
if(CMAKE_CUDA_ARCHITECTURES STREQUAL "86" OR CMAKE_CUDA_ARCHITECTURES STREQUAL "89")
  get_target_property(ninfer_ops_sources ninfer_ops SOURCES)
  list(FILTER ninfer_ops_sources EXCLUDE REGEX "nvfp4/.+_w4a4\.cu$")
  # The NVFP4-valued KV-cache kernels quantize with cvt.e2m1x2 (sm_100+). Their launchers are
  # stubbed in nvfp4_sm86_stubs.cpp and the storage modes are rejected at startup.
  list(FILTER ninfer_ops_sources EXCLUDE REGEX "causal_cache/small_t_(nvfp4|k8v4)\.cu$")
  list(FILTER ninfer_ops_sources EXCLUDE REGEX "kv_cache/append/(nvfp4|k8v4)_launch\.cu$")
  set_property(TARGET ninfer_ops PROPERTY SOURCES ${ninfer_ops_sources})
  target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/nvfp4_sm86_stubs.cpp")
  target_compile_definitions(ninfer_ops PUBLIC NINFER_SM86=1)
endif()
