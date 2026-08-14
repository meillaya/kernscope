# kernscope-verify: the static GPU-kernel verifier core (todo 30).
#
# Wave-6 verify track. Todo 30 lands the FOUNDATION the check modules
# (todos 31-34) and the CLI (todo 35) build on:
#   * libkernscope_verify — clang CUDA/HIP ingestion (host+device in ONE
#     module; device-only secondary mode), launch-config tiers
#     (Exact/UpperBound/Symbolic per research brief §4.6), the eBPF-style
#     diagnostic engine (severity/class/source-loc/path + per-function
#     budget), and the check-plugin registry.
#   * the check-module subdirectories checks/{oob,barrier,race,ptr} — STUBS
#     now, so todos 31-34 never touch a shared file (they only fill their
#     own subdirectory).
#   * kernscope-verify-core — the todo-30 harness (ingest + --dump-state +
#     exit 2 on ingestion failure). The full CLI (verdict JSON, approved-IR
#     handoff, committed corpus) is todo 35.
#
# INGESTION MODES (documented in tools/verify/README.md):
#   default   host+device merged into ONE module (launch dims readable) —
#             two clang passes (--cuda-host-only / --cuda-device-only) +
#             in-process LLVM link, because clang 21's -emit-llvm runs
#             fatbinary on the device bitcode ("fatbinary elf mismatch",
#             tests/fixtures/README.md + todo-5 evidence).
#   --device-only  degraded secondary mode: device IR only; kernels without
#             nvvm.maxntid degrade to LaunchTier::Symbolic ("symbolic dims").
#
# Linking mirrors libs/amdparse: the monolithic `LLVM` component (the LLVM
# 21 shared library); LLVM include dirs are SYSTEM PUBLIC so consumers of
# the public headers (verifier_state.h uses llvm::ConstantRange) compile
# cleanly; test executables get LLVM_LIBRARY_DIR on BUILD_RPATH for ctest.

# --- the core library ------------------------------------------------------
add_library(kernscope_verify STATIC
  src/budget.cpp
  src/check_plugin.cpp
  src/diagnostic.cpp
  src/ingestion.cpp
  src/launch_config.cpp
  src/verifier_state.cpp
)
target_include_directories(kernscope_verify PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(kernscope_verify SYSTEM PUBLIC
  ${LLVM_INCLUDE_DIRS})
target_compile_features(kernscope_verify PUBLIC cxx_std_20)
target_link_libraries(kernscope_verify PUBLIC LLVM)
add_library(kernscope::verify ALIAS kernscope_verify)

# --- the todo-30 harness (CLI proper is todo 35) ---------------------------
# kernscope-verify-core <src.cu|src.hip> [--device-only] [--dump-state]
#   [--budget N] [--opt O1|O2] [--arch sm_80|gfx1101]
# Exit codes: 0 = ingested + analyzed (no Error-severity findings),
#             1 = Error-severity findings, 2 = ingestion failure (clang
#                 diagnostics forwarded verbatim, never a crash).
add_executable(kernscope-verify-core src/verify_core_main.cpp)
target_link_libraries(kernscope-verify-core PRIVATE kernscope::verify)
set_target_properties(kernscope-verify-core PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_test(NAME verify-version COMMAND kernscope-verify-core --version)
set_tests_properties(verify-version PROPERTIES LABELS "no-gpu")

# --- check-module subdirectories (todos 31-34) -----------------------------
# Stub CMakeLists.txt exist NOW so those todos add their plugin sources in
# their own directory only.
add_subdirectory(checks/oob)
add_subdirectory(checks/barrier)
add_subdirectory(checks/race)
add_subdirectory(checks/ptr)

# --- unit tests (gtest) ----------------------------------------------------
# ctest hardware-gating convention (todo 3): every registered test carries a
# LABELS property of exactly one of no-gpu | gpu-amd | gpu-nvidia-rental.
# The core suite runs clang GPU-less (clang -emit-llvm, no GPU) -> no-gpu;
# the HIP ingestion case is compile-gated (GTEST_SKIP when the HIP
# toolchain is unusable), still no-gpu (compile only, never runs on GPU).
find_package(GTest REQUIRED)

add_executable(verify-core-test tests/verify_core_test.cpp)
target_link_libraries(verify-core-test PRIVATE kernscope::verify
  GTest::gtest_main)
target_compile_definitions(verify-core-test PRIVATE
  VERIFY_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures"
  FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/fixtures")
set_target_properties(verify-core-test PROPERTIES
  BUILD_RPATH "${LLVM_LIBRARY_DIR}")
include(GoogleTest)
gtest_discover_tests(verify-core-test
  TEST_PREFIX "verify_core:"
  PROPERTIES LABELS "no-gpu"
)
