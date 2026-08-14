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
