# kernscope shared-core library tree.
#
# libs/bench: the nvbench/Triton-style benchmarking harness (wave-4 todo 22)
# implementing the validated methodology of
# docs/report/kernscope-technical-report.pdf — time-based warmup/rep budgets
# (Triton do_bench: 25 ms / 100 ms), L2 flush between reps sized from the
# device L2 attribute (nvbench l2flush), HIP/CUDA event timing per rep,
# adaptive trials (nvbench measure_cold: min_time/min_samples/max_noise/
# timeout), median + p20/p80 quantiles. Results feed libs/db records with
# provenance via tools/bench (kernscope-bench CLI).
#
# The harness lib itself is GPU-free host code (kernscope::bench::run talks to
# any GpuRuntime — mock, compile-only or live). The L2-attribute query needs
# the ROCm host headers at COMPILE time (same clr-root resolution + -isystem
# convention as libs/gpuruntime); without hipcc the helper degrades to the
# Triton 256 MiB fallback and everything still builds.
#
# The roofline peak microbenches (libs/bench/kernels/peaks.hip) are compiled
# at BUILD time by `hipcc --genco --offload-arch=gfx1101` into peaks.hsaco
# (the add_kernel.hip pattern) and consumed by the gpu-amd live suite and the
# kernscope-bench CLI. KERNSCOPE_BENCH_HSACO is exported as a CACHE var so
# tools/bench can consume the same module.

find_program(KERNSCOPE_HIPCC_EXECUTABLE hipcc)

set(KERNSCOPE_BENCH_HAVE_HIP OFF)
if(KERNSCOPE_HIPCC_EXECUTABLE)
  get_filename_component(_BENCH_CLR_BIN "${KERNSCOPE_HIPCC_EXECUTABLE}" DIRECTORY)
  get_filename_component(_BENCH_CLR_ROOT "${_BENCH_CLR_BIN}" DIRECTORY)
  set(_BENCH_HIP_INCLUDE "${_BENCH_CLR_ROOT}/include")
  if(NOT EXISTS "${_BENCH_HIP_INCLUDE}/hip/hip_runtime_api.h"
     AND DEFINED ENV{HIP_PATH})
    set(_BENCH_HIP_INCLUDE "$ENV{HIP_PATH}/include")
  endif()
  if(EXISTS "${_BENCH_HIP_INCLUDE}/hip/hip_runtime_api.h")
    set(KERNSCOPE_BENCH_HAVE_HIP ON)
  endif()
endif()

# --- the harness library -----------------------------------------------------
add_library(kernscope_bench STATIC src/bench.cpp)
target_include_directories(kernscope_bench PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(kernscope_bench PUBLIC cxx_std_20)
add_library(kernscope::bench ALIAS kernscope_bench)
# gpuruntime provides the runtime abstraction (and amdhip64 transitively when
# the HIP backend is live); db provides the record store for results.
target_link_libraries(kernscope_bench PUBLIC kernscope::gpuruntime kernscope::db)

if(KERNSCOPE_BENCH_HAVE_HIP)
  # The L2-attribute query (hipDeviceGetAttribute) needs the HIP platform on
  # this TU; headers stay -isystem (ROCm is third-party) so they never trip
  # the global -Wall -Wextra -Werror gate.
  target_compile_definitions(kernscope_bench PRIVATE __HIP_PLATFORM_AMD__=1)
  target_include_directories(kernscope_bench SYSTEM PRIVATE "${_BENCH_HIP_INCLUDE}")
  message(STATUS "bench: HIP L2-attr query enabled (hipcc=${KERNSCOPE_HIPCC_EXECUTABLE})")
else()
  message(STATUS "bench: HIP L2-attr query disabled (no hipcc at configure "
                 "time; harness falls back to the Triton 256 MiB flush buffer)")
endif()

# --- roofline peak microbench module (hipcc --genco, gfx1101) ----------------
if(KERNSCOPE_BENCH_HAVE_HIP)
  set(_BENCH_HSACO "${CMAKE_CURRENT_BINARY_DIR}/kernels/peaks.hsaco")
  add_custom_command(OUTPUT "${_BENCH_HSACO}"
    COMMAND "${KERNSCOPE_HIPCC_EXECUTABLE}" --genco --offload-arch=gfx1101
            "-I${_BENCH_HIP_INCLUDE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/peaks.hip"
            -o "${_BENCH_HSACO}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/peaks.hip"
    COMMENT "hipcc --genco kernels/peaks.hip (gfx1101) -> peaks.hsaco")
  add_custom_target(kernscope_bench_hsaco DEPENDS "${_BENCH_HSACO}")
  # tools/bench and the live suite share the generated module.
  set(KERNSCOPE_BENCH_HSACO "${_BENCH_HSACO}" CACHE STRING
      "Path to the hipcc --genco peaks.hsaco module (consumed by kernscope-bench)")
endif()

# --- unit tests (gtest) -------------------------------------------------------
# ctest hardware-gating convention (todo 3): every registered test carries a
# LABELS property of exactly one of no-gpu | gpu-amd | gpu-nvidia-rental.
find_package(GTest REQUIRED)

# no-gpu harness suite: runs against the MockRuntime (timing structure,
# quantile ordering, L2-flush sizing/fallback, failure-injection no-hang via
# the adaptive-trials short-circuit) — no device needed.
add_executable(bench-test tests/bench_test.cpp)
target_link_libraries(bench-test PRIVATE kernscope::bench GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(bench-test TEST_PREFIX "bench:"
  PROPERTIES LABELS "no-gpu")

# gpu-amd live suite: one peak-bw rep + one peak-flops rep on the local
# gfx1101 (RX 7800 XT), asserting the todo-22 sanity floors (bandwidth > 300
# GB/s, fp32 > 10 TFLOPS) and the L2-flush == device-attr contract.
add_executable(bench-live-test tests/bench_live_test.cpp)
target_link_libraries(bench-live-test PRIVATE kernscope::bench GTest::gtest_main)
if(KERNSCOPE_BENCH_HAVE_HIP)
  target_compile_definitions(bench-live-test PRIVATE
    KERNSCOPE_TEST_HSACO="${_BENCH_HSACO}")
  add_dependencies(bench-live-test kernscope_bench_hsaco)
else()
  # No hipcc at configure time: the live suite SKIPs (KERNSCOPE_NO_LIVE_HIP).
  target_compile_definitions(bench-live-test PRIVATE
    KERNSCOPE_NO_LIVE_HIP=1 KERNSCOPE_TEST_HSACO="")
endif()
include(GoogleTest)
gtest_discover_tests(bench-live-test TEST_PREFIX "bench-live:"
  PROPERTIES LABELS "gpu-amd")
