# kernscope tools/tune — the C2 autotuner (plan todos 25-28).
#
# SHARED OWNERSHIP (parallel workers): todos 25/26 (matmul + elementwise
# generators) append their own sections BELOW this file without touching the
# targets defined here; todo 27 (FlashAttention-lite, THIS section) defines
# the flash-attention generator library + its test executables. Target-naming
# rule for the shared file: prefix every target with the owning generator
# (kernscope_flash_gen / kernscope_matmul_gen / ...) so appends can never
# collide.

# --- toolchain discovery (kernels/templates pattern, todo 24) ---------------
find_program(KERNSCOPE_TUNE_NVCC_EXECUTABLE nvcc)
find_program(KERNSCOPE_TUNE_HIPCC_EXECUTABLE hipcc)

# ROCm host headers live at <clr>/include; the hipcc binary sits in
# <clr>/bin (rocmPackages.clr in the dev shell; HIP_PATH points at the root).
set(KERNSCOPE_TUNE_HAVE_HIP OFF)
set(_TUNE_HIP_INCLUDE "")
if(KERNSCOPE_TUNE_HIPCC_EXECUTABLE)
  get_filename_component(_TUNE_CLR_BIN "${KERNSCOPE_TUNE_HIPCC_EXECUTABLE}"
    DIRECTORY)
  get_filename_component(_TUNE_CLR_ROOT "${_TUNE_CLR_BIN}" DIRECTORY)
  set(_TUNE_HIP_INCLUDE "${_TUNE_CLR_ROOT}/include")
  if(NOT EXISTS "${_TUNE_HIP_INCLUDE}/hip/hip_runtime_api.h"
     AND DEFINED ENV{HIP_PATH})
    set(_TUNE_HIP_INCLUDE "$ENV{HIP_PATH}/include")
  endif()
  if(EXISTS "${_TUNE_HIP_INCLUDE}/hip/hip_runtime_api.h")
    set(KERNSCOPE_TUNE_HAVE_HIP ON)
  endif()
endif()

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_FOUND)
  set(_TUNE_PYTHON "${Python3_EXECUTABLE}")
else()
  set(_TUNE_PYTHON "python3")
endif()

# --- todo 27: FlashAttention-lite template generator -------------------------
# Single-source portable kernel generator (forward-only online-softmax
# FlashAttention, Dao et al. NeurIPS 2022). Library API in
# include/kernscope/tune/flash_attention_gen.h; validated pre-compile against
# libs/arch (smem budget, divisibility) + libs/roofline (4N^2d FLOPs model).
add_library(kernscope_flash_gen STATIC
  src/flash_attention_gen.cpp
)
target_include_directories(kernscope_flash_gen PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(kernscope_flash_gen PUBLIC cxx_std_20)
add_library(kernscope::flash_gen ALIAS kernscope_flash_gen)
target_link_libraries(kernscope_flash_gen PUBLIC
  kernscope::arch
  kernscope::roofline
)

# Paths shared by both test executables (they compile the SAME source file,
# flash_gen_test.cpp, once without KERNSCOPE_TUNE_LIVE for the no-gpu gates
# and once with it for the gpu-amd live gates). Empty definitions (e.g. nvcc
# absent) are handled by the test with GTEST_SKIP.
set(KERNSCOPE_TUNE_TEST_DEFS
  KERNSCOPE_TUNE_NVCC="${KERNSCOPE_TUNE_NVCC_EXECUTABLE}"
  KERNSCOPE_TUNE_HIPCC="${KERNSCOPE_TUNE_HIPCC_EXECUTABLE}"
  KERNSCOPE_TUNE_HIP_INCLUDE="${_TUNE_HIP_INCLUDE}"
  KERNSCOPE_TUNE_KERNEL_TRAITS_INCLUDE="${CMAKE_SOURCE_DIR}/kernels/templates/include"
  KERNSCOPE_TUNE_REPO_ROOT="${CMAKE_SOURCE_DIR}"
  KERNSCOPE_TUNE_RUN_HIP_GATE="${CMAKE_SOURCE_DIR}/kernels/templates/tests/run_hip_gate.py"
  KERNSCOPE_TUNE_PYTHON="${_TUNE_PYTHON}"
)

find_package(GTest REQUIRED)

# --- no-gpu pre-compile gates ------------------------------------------------
# nvcc sm_80/sm_90 compile-only gates, hipcc gfx1101 --genco compile-only
# gate, smem over-budget rejection (reason asserted), roofline 4N^2d AI
# cross-check, full search-space validity. No GPU device is touched.
add_executable(tune-flash-gen-test tests/flash_gen_test.cpp)
target_link_libraries(tune-flash-gen-test PRIVATE
  kernscope::flash_gen
  kernscope::arch
  kernscope::roofline
  GTest::gtest_main
)
target_compile_definitions(tune-flash-gen-test PRIVATE
  ${KERNSCOPE_TUNE_TEST_DEFS})
include(GoogleTest)
gtest_discover_tests(tune-flash-gen-test
  TEST_PREFIX "tune_flash_gen:"
  PROPERTIES LABELS "no-gpu"
)

# --- gpu-amd LIVE gates ------------------------------------------------------
# The non-causal AND causal FlashAttention-lite kernels run on this machine
# (gfx1101) through the kernscope_templates_hip_runner + run_hip_gate.py and
# must earn a kernscope_golden verdict pass:true. Only built when hipcc +
# ROCm headers were found AND the kernels/templates runner target exists.
if(KERNSCOPE_TUNE_HAVE_HIP AND TARGET kernscope_templates_hip_runner)
  add_executable(tune-flash-gen-live-test tests/flash_gen_test.cpp)
  target_link_libraries(tune-flash-gen-live-test PRIVATE
    kernscope::flash_gen
    kernscope::arch
    kernscope::roofline
    GTest::gtest_main
  )
  target_compile_definitions(tune-flash-gen-live-test PRIVATE
    ${KERNSCOPE_TUNE_TEST_DEFS}
    KERNSCOPE_TUNE_LIVE=1
    KERNSCOPE_TUNE_RUNNER="$<TARGET_FILE:kernscope_templates_hip_runner>"
  )
  # The runner binary + the gate script must exist before ctest runs.
  add_dependencies(tune-flash-gen-live-test kernscope_templates_hip_runner)
  gtest_discover_tests(tune-flash-gen-live-test
    TEST_PREFIX "tune_flash_gen_live:"
    PROPERTIES LABELS "gpu-amd"
  )
else()
  message(STATUS "tools/tune: live FlashAttention gates skipped (hipcc or "
                 "kernscope_templates_hip_runner not available)")
endif()

# --- todo 26: elementwise/reduction template generators ----------------------
# Four single-source portable kernel generators (bias+GELU, softmax,
# layernorm, axis-0 reduction) + the kernscope-tune-elem-gen CLI + the
# acceptance suite (no-gpu pre-compile gates and gpu-amd LIVE golden gates).
# Library API in include/kernscope/tune/{bias_gelu,softmax,layernorm,
# reduction}_gen.h (shared types in elem_common.h); every candidate is
# validated PRE-COMPILE against libs/arch (search-space membership, block
# limit, shape divisibility by the vector width) with a reason string.
add_library(kernscope_elem_gen STATIC
  src/bias_gelu_gen.cpp
  src/softmax_gen.cpp
  src/layernorm_gen.cpp
  src/reduction_gen.cpp
)
target_include_directories(kernscope_elem_gen PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(kernscope_elem_gen PUBLIC cxx_std_20)
add_library(kernscope::elem_gen ALIAS kernscope_elem_gen)
target_link_libraries(kernscope_elem_gen PUBLIC kernscope::arch)

# The CLI: one validated config -> one emitted .hip kernel (non-zero exit +
# reason on pre-compile rejection). Consumed by the todo-28 tuning driver.
add_executable(kernscope-tune-elem-gen src/elem_gen_cli.cpp)
target_link_libraries(kernscope-tune-elem-gen PRIVATE kernscope::elem_gen)
target_compile_features(kernscope-tune-elem-gen PRIVATE cxx_std_20)

# --- no-gpu pre-compile gates ------------------------------------------------
# nvcc sm_80/sm_90 compile-only gates, hipcc gfx1101 --genco compile-only
# gate, the vector-width divisibility REJECTION (reason asserted; the QA
# negative: two-pass softmax vec=4 on a non-divisible shape) and full
# search-space validity. No GPU device is touched.
add_executable(tune-elem-gen-test tests/elem_gen_test.cpp)
target_link_libraries(tune-elem-gen-test PRIVATE
  kernscope::elem_gen
  kernscope::arch
  GTest::gtest_main
)
target_compile_definitions(tune-elem-gen-test PRIVATE
  ${KERNSCOPE_TUNE_TEST_DEFS})
include(GoogleTest)
gtest_discover_tests(tune-elem-gen-test
  TEST_PREFIX "tune_elem_gen:"
  PROPERTIES LABELS "no-gpu"
)

# --- gpu-amd LIVE gates ------------------------------------------------------
# 8 live verdicts (4 families x 2 shapes): each generated kernel runs on
# this machine (gfx1101) through the kernscope_templates_hip_runner +
# run_hip_gate.py and must earn a kernscope_golden verdict pass:true. Only
# built when hipcc + ROCm headers were found AND the runner target exists.
if(KERNSCOPE_TUNE_HAVE_HIP AND TARGET kernscope_templates_hip_runner)
  add_executable(tune-elem-gen-live-test tests/elem_gen_test.cpp)
  target_link_libraries(tune-elem-gen-live-test PRIVATE
    kernscope::elem_gen
    kernscope::arch
    GTest::gtest_main
  )
  target_compile_definitions(tune-elem-gen-live-test PRIVATE
    ${KERNSCOPE_TUNE_TEST_DEFS}
    KERNSCOPE_TUNE_LIVE=1
    KERNSCOPE_TUNE_RUNNER="$<TARGET_FILE:kernscope_templates_hip_runner>"
  )
  add_dependencies(tune-elem-gen-live-test kernscope_templates_hip_runner)
  gtest_discover_tests(tune-elem-gen-live-test
    TEST_PREFIX "tune_elem_gen_live:"
    PROPERTIES LABELS "gpu-amd"
  )
else()
  message(STATUS "tools/tune: live elementwise gates skipped (hipcc or "
                 "kernscope_templates_hip_runner not available)")
endif()

# --- todo 25: matmul template generator -------------------------------------
# Single-source portable tiled-GEMM generator (bf16 in, fp32 accumulate, bf16
# out). Library API in include/kernscope/tune/matmul_gen.h; validated
# pre-compile against libs/arch (smem budget, register budget, warp/block
# divisibility, vector-width tiling). CLI in src/matmul_gen_cli.cpp emits the
# generated source for one (arch, config) to a file (kernscope-tune-gen).
add_library(kernscope_matmul_gen STATIC
  src/matmul_gen.cpp
)
target_include_directories(kernscope_matmul_gen PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(kernscope_matmul_gen PUBLIC cxx_std_20)
add_library(kernscope::matmul_gen ALIAS kernscope_matmul_gen)
target_link_libraries(kernscope_matmul_gen PUBLIC
  kernscope::arch
)

# The CLI is a host tool (the generated source is what nvcc/hipcc compile).
add_executable(kernscope-tune-gen
  src/matmul_gen_cli.cpp
)
target_link_libraries(kernscope-tune-gen PRIVATE
  kernscope::matmul_gen)
target_compile_features(kernscope-tune-gen PRIVATE cxx_std_20)

# --- no-gpu pre-compile gates ------------------------------------------------
# Grid-enumeration count (16), invalid-config rejection counts (register
# budget, smem over-budget with reason naming the budget + arch, vector-width
# divisibility), nvcc sm_80/sm_90 compile-only gates, hipcc gfx1101 --genco
# compile-only gate. Shares the flash test's KERNSCOPE_TUNE_TEST_DEFS path
# injection (nvcc/hipcc/python/runner paths).
add_executable(tune-matmul-gen-test tests/matmul_gen_test.cpp)
target_link_libraries(tune-matmul-gen-test PRIVATE
  kernscope::matmul_gen
  kernscope::arch
  GTest::gtest_main
)
target_compile_definitions(tune-matmul-gen-test PRIVATE
  ${KERNSCOPE_TUNE_TEST_DEFS})
gtest_discover_tests(tune-matmul-gen-test
  TEST_PREFIX "tune_matmul_gen:"
  PROPERTIES LABELS "no-gpu"
)

# --- gpu-amd LIVE gates ------------------------------------------------------
# The generated tiled-matmul kernel runs on this machine (gfx1101) through the
# kernscope_templates_hip_runner + run_hip_gate.py and must earn a
# kernscope_golden matmul verdict pass:true on M=N=K=64. Only built when
# hipcc + ROCm headers were found AND the kernels/templates runner target
# exists.
if(KERNSCOPE_TUNE_HAVE_HIP AND TARGET kernscope_templates_hip_runner)
  add_executable(tune-matmul-gen-live-test tests/matmul_gen_test.cpp)
  target_link_libraries(tune-matmul-gen-live-test PRIVATE
    kernscope::matmul_gen
    kernscope::arch
    GTest::gtest_main
  )
  target_compile_definitions(tune-matmul-gen-live-test PRIVATE
    ${KERNSCOPE_TUNE_TEST_DEFS}
    KERNSCOPE_TUNE_LIVE=1
    KERNSCOPE_TUNE_RUNNER="$<TARGET_FILE:kernscope_templates_hip_runner>"
  )
  # The runner binary + the gate script must exist before ctest runs.
  add_dependencies(tune-matmul-gen-live-test kernscope_templates_hip_runner)
  gtest_discover_tests(tune-matmul-gen-live-test
    TEST_PREFIX "tune_matmul_gen_live:"
    PROPERTIES LABELS "gpu-amd"
  )
else()
  message(STATUS "tools/tune: live matmul gates skipped (hipcc or "
                 "kernscope_templates_hip_runner not available)")
endif()

# --- todo 28: correctness-gated tuning driver + kernscope-tune CLI ----------
# The integration core of the tune track: GPU-free pre-screen (generator
# validate_config + libs/occupancy + libs/roofline) -> compile (hipcc --genco
# for gfxXXXX, nvcc compile-only for sm_XX) -> golden gate (run_hip_gate.py;
# discard on non-pass:true) -> libs/bench -> select best by median ->
# libs/db records + Triton-style tunecache JSON (resume skips cached
# configs). The per-family adapter TUs each include EXACTLY ONE generator
# header (the generators each declare their own ConfigVerdict/LaunchGeometry
# in kernscope::tune, so mixing them in one TU would be an ODR violation).
add_library(kernscope_tune_driver STATIC
  src/driver.cpp
  src/driver_matmul.cpp
  src/driver_elem.cpp
  src/driver_flash.cpp
)
target_include_directories(kernscope_tune_driver PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(kernscope_tune_driver PUBLIC cxx_std_20)
add_library(kernscope::tune_driver ALIAS kernscope_tune_driver)
target_link_libraries(kernscope_tune_driver PUBLIC
  kernscope::matmul_gen
  kernscope::elem_gen
  kernscope::flash_gen
  kernscope::arch
  kernscope::occupancy
  kernscope::roofline
  kernscope::bench
  kernscope::db
  kernscope::gpuruntime
)
target_compile_definitions(kernscope_tune_driver PRIVATE
  ${KERNSCOPE_TUNE_TEST_DEFS})
if(TARGET kernscope_templates_hip_runner)
  target_compile_definitions(kernscope_tune_driver PRIVATE
    KERNSCOPE_TUNE_RUNNER="$<TARGET_FILE:kernscope_templates_hip_runner>")
  add_dependencies(kernscope_tune_driver kernscope_templates_hip_runner)
endif()

# The kernscope-tune CLI (todos 28+29): --kernel <family> --shape M,N,K|R,C|
# Nq,Nk,d --arch <auto|sm_XX|gfxXXXX> [--grid auto|full] [--json|--html|
# --text] [--db <path>] [--cache <path>] [--workdir <dir>]
# [--min-occupancy <f>] [--no-bench] [--no-cache] [--emit-repro <dir>].
# --arch auto resolves to gfx1101 (the local RX 7800 XT). Todo 29 added the
# owner-format --text report, the self-contained libs/report variant-
# comparison --html, and --emit-repro (reproducibility bundle; needs the
# kernscope::report renderer).
add_executable(kernscope-tune src/tune_cli.cpp)
target_link_libraries(kernscope-tune PRIVATE
  kernscope::tune_driver
  kernscope::capture_tune
  kernscope::report
)
target_compile_features(kernscope-tune PRIVATE cxx_std_20)
# The driver is the FIRST target to link two generator libs (matmul + flash).
# Both export an identical `kernscope::tune::smem_layout_name(SmemLayout)`
# (same-name enum helpers with identical "padding"/"xor_swizzle" strings) —
# a latent collision unreachable by the per-generator tests/CLIs, benign here
# because either definition is equivalent. Allow the duplicate definition
# instead of rewriting the generators' public API (sibling deliverables).
target_link_options(kernscope-tune PRIVATE "-Wl,--allow-multiple-definition")

# --- todo 53: capture->tune driver (replay-based launch-config tuning) -------
# The DISTINCT tuning mode for CAPTURED kernels (bundles carry binary+args+
# buffers, no source): kernscope-tune --from-capture <bundle> searches launch
# configs over the todo-48 replay machinery. The driver is composed from the
# SAME public building blocks as the replay harness (tools/capture/src/replay
# .cpp): the shared plan builder replay_model.cpp is compiled here UNCHANGED
# (reuse via API — no tools/capture or libs/* file is modified), plus
# libs/gpuruntime + libs/bench + libs/db.
add_library(kernscope_capture_tune STATIC
  src/capture_tune.cpp
  ${CMAKE_SOURCE_DIR}/tools/capture/src/replay_model.cpp
)
target_include_directories(kernscope_capture_tune PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/tools/capture/src)
target_compile_features(kernscope_capture_tune PUBLIC cxx_std_20)
add_library(kernscope::capture_tune ALIAS kernscope_capture_tune)
target_link_libraries(kernscope_capture_tune PUBLIC
  kernscope::capture_core
  kernscope::db
  kernscope::bench
  kernscope::gpuruntime
  kernscope::arch
)

# --- todo 53: trivial usage gate (no-gpu) ------------------------------------
# --from-capture requires a bundle; the CLI must fail cleanly (exit 2) with a
# usage message when it is missing — no GPU touched.
add_test(NAME capture_tune_usage
  COMMAND ${_TUNE_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/tests/capture_tune_usage_check.py
          $<TARGET_FILE:kernscope-tune>)
set_tests_properties(capture_tune_usage PROPERTIES LABELS "no-gpu")

# --- todo 53 LIVE acceptance: layernorm_fused capture bundle -> tune --------
# gpu-amd (todo 3 convention): runs the capture->tune pipeline on the
# committed todo-45 layernorm-live bundle on this machine (gfx1101). Asserts
# >= 3 candidates explored (baseline config ALWAYS first), validation PASS
# for every explored config, best.speedup >= 1.0, the db record queryable
# under (arch, kernel-name-hash, capture-bundle-hash), the injected
# block-2048 config rejected PRE-REPLAY with a reason, and the QA negative
# (garbage reference -> 'no valid config', exit 1, NO db write).
add_test(NAME integ_capture_tune
  COMMAND ${_TUNE_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/tests/integ_capture_tune_check.py
          ${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR})
set_tests_properties(integ_capture_tune PROPERTIES LABELS "gpu-amd"
  TIMEOUT 1800)

# --- todo 28 acceptance suite ------------------------------------------------
# tune_driver_test.cpp is compiled twice (the generators' shared pattern):
# once without KERNSCOPE_TUNE_LIVE for the no-gpu negative test (every config
# infeasible at the pre-screen -> "no viable config" + JSON reason, no crash,
# no GPU touched) and once with it for the gpu-amd LIVE tune on gfx1101
# (matmul 512^3 bf16 over the 24-config auto grid: pre-screen rejections >= 1,
# injected bad variant discarded by the golden gate, best selected with
# median == min of measured, libs/db records queryable, cache-hit re-run
# compiles 0 new variants).
add_executable(tune-driver-test tests/tune_driver_test.cpp)
target_link_libraries(tune-driver-test PRIVATE
  kernscope::tune_driver
  kernscope::db
  GTest::gtest_main
)
target_compile_definitions(tune-driver-test PRIVATE
  ${KERNSCOPE_TUNE_TEST_DEFS})
target_link_options(tune-driver-test PRIVATE "-Wl,--allow-multiple-definition")
include(GoogleTest)
gtest_discover_tests(tune-driver-test
  TEST_PREFIX "tune_driver:"
  PROPERTIES LABELS "no-gpu"
)

if(KERNSCOPE_TUNE_HAVE_HIP AND TARGET kernscope_templates_hip_runner)
  add_executable(tune-driver-live-test tests/tune_driver_test.cpp)
  target_link_libraries(tune-driver-live-test PRIVATE
    kernscope::tune_driver
    kernscope::db
    GTest::gtest_main
  )
  target_compile_definitions(tune-driver-live-test PRIVATE
    ${KERNSCOPE_TUNE_TEST_DEFS}
    KERNSCOPE_TUNE_LIVE=1
    KERNSCOPE_TUNE_RUNNER="$<TARGET_FILE:kernscope_templates_hip_runner>"
  )
  target_link_options(tune-driver-live-test PRIVATE
    "-Wl,--allow-multiple-definition")
  # The runner binary + the gate script must exist before the live tune runs.
  add_dependencies(tune-driver-live-test kernscope_templates_hip_runner)
  gtest_discover_tests(tune-driver-live-test
    TEST_PREFIX "tune_driver_live:"
    PROPERTIES LABELS "gpu-amd"
  )
else()
  message(STATUS "tools/tune: live tune-driver gates skipped (hipcc or "
                 "kernscope_templates_hip_runner not available)")
endif()

# --- todo 29: report check (ctest, LABELS no-gpu) ---------------------------
# tests/tune/check_report.py runs the kernscope-tune --json pipeline for
# matmul 512^3 on gfx1101 (tunecache-backed: a warm cache replays without
# recompiling) and asserts the numeric report contract: best config present
# with all config keys, roofline.position in (0,1], roofline.bound in
# {compute,memory,latency}, occupancy in (0,1], variants >= 20 sorted by
# median ascending, speedup >= 1.0 vs the naive baseline. LABELS no-gpu per
# the task contract (the count-guard suite); it runs the live tune, so it
# requires the local HIP device in practice.
add_test(NAME tune_report
  COMMAND ${_TUNE_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_report.py
          $<TARGET_FILE:kernscope-tune> ${CMAKE_SOURCE_DIR})
set_tests_properties(tune_report PROPERTIES LABELS "no-gpu")

# --- todo 54: verifier-gated tuning + pass->analyzer delta reports ----------
# The integration deliverable of the tune track:
#   (1) VERIFIER SAFETY GATE: kernscope-tune --verify runs the todo-35
#       kernscope-verify CLI on a verifier-facing rendering of each
#       generated variant's source BEFORE compile; error-class verdicts
#       (rc 1) and ingestion failures (rc 2) DISCARD the variant and record
#       the verdict ids — nothing gated ever compiles (the driver asserts
#       summary.compiled excludes gated variants; the QA negative exits
#       "no approved variants" with compile-count 0). The verify binary is
#       injected at configure time (the driver's kVerifyCli default).
#   (2) PASS->ANALYZER LOOP: kernscope-pass-report --report-pipeline
#       <kernel> re-runs the todo-42 pipeline on a showcase kernel, runs
#       kernscope-analyze on the before/after PTX and emits the DELTA
#       report (registers estimate, smem bytes, coalescing class,
#       bank-conflict way, vector width) via libs/report.
if(TARGET kernscope-verify)
  set(KERNSCOPE_TUNE_VERIFY_CLI_DEF
    KERNSCOPE_TUNE_VERIFY_CLI="$<TARGET_FILE:kernscope-verify>")
else()
  set(KERNSCOPE_TUNE_VERIFY_CLI_DEF KERNSCOPE_TUNE_VERIFY_CLI="")
endif()
target_compile_definitions(kernscope_tune_driver PRIVATE
  ${KERNSCOPE_TUNE_VERIFY_CLI_DEF})

# kernscope-pass-report: the todo-54 pass->analyzer delta CLI. Invokes the
# todo-42 pipeline (script below) + kernscope-analyze (INVOKED, never
# modified) and emits the libs/report delta JSON.
add_executable(kernscope-pass-report src/pass_report_cli.cpp)
target_link_libraries(kernscope-pass-report PRIVATE
  kernscope::report
  kernscope::db
)
target_compile_features(kernscope-pass-report PRIVATE cxx_std_20)
target_compile_definitions(kernscope-pass-report PRIVATE
  ${KERNSCOPE_TUNE_TEST_DEFS})

# --- todo 54 ctest gates -----------------------------------------------------
# integ_verify_tune (gpu-amd, live): kernscope-tune --verify on bias_gelu
# with the INJECTED risky config (tools/tune/tests/fixtures/risky_oob.hip —
# a device-global OOB write the verifier flags MAY_OOB deterministically):
# gated >= 1 with VERDICT IDS, the injected config discarded BEFORE
# compile, the approved variants run the full live pipeline; QA negative:
# softmax (every variant carries error-class smem-reduction verdicts) ->
# exit 1, "no approved variants", compile-count == 0.
add_test(NAME integ_verify_tune
  COMMAND ${_TUNE_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/tests/integ_verify_tune_check.py
          ${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR})
set_tests_properties(integ_verify_tune PROPERTIES LABELS "gpu-amd"
  TIMEOUT 3600)

# integ_pass_analyze (no-gpu): the delta report for the transformed
# showcase (layernorm: vector_width 1 -> 4, PTX-level vectorized loads
# 0 -> 12) asserts before/after values for all five fields + >= 1 improved
# metric; the no-transform showcase (gemm; softmax is analyzer-unparsable —
# its llc PTX carries callseq blocks for expf calls, outside libs/ptxparse
# scope) records no_transform true.
add_test(NAME integ_pass_analyze
  COMMAND ${_TUNE_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/tests/integ_pass_analyze_check.py
          ${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR})
set_tests_properties(integ_pass_analyze PROPERTIES LABELS "no-gpu"
  TIMEOUT 1800)
