cmake_minimum_required(VERSION 3.20)
project(kernscope CXX)

# C++20 is required project-wide (GPU-kernel tooling).
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Quality gate (review-mandated, re-checked by F2): warnings are errors.
# Applied globally so every target (libs/, tools/, tests/) is clean by
# construction. Do NOT relax these flags in subtrees.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")

# --- LLVM 21 (required) + MLIR 21 (optional), nix-store cmake dirs ----------
# scripts/configure.sh resolves LLVM_DIR/MLIR_DIR from the nix store (pattern:
# /home/mei/projects/polykernel/README.md:74-90). Without LLVM_DIR this
# configure fails loudly, which is the intended negative-path behavior.
# NOTE: nixpkgs patches LLVMConfigVersion.cmake to strict major.minor matching
# (API-compatible only with 21.1.x), so the request must be "21.1", not "21"
# (which would ask for 21.0 and be rejected).
find_package(LLVM 21.1 REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION} (LLVM_DIR=${LLVM_DIR})")

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)

find_package(MLIR CONFIG QUIET)
if(MLIR_FOUND)
  message(STATUS "Found MLIR ${MLIR_VERSION} (MLIR_DIR=${MLIR_DIR})")
  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  include(AddMLIR)
else()
  message(STATUS "MLIR not found (set MLIR_DIR to enable MLIR-backed tools)")
endif()

# ctest hardware-gating convention: every registered test carries a LABELS
# property of exactly one of `no-gpu` | `gpu-amd` | `gpu-nvidia-rental`.
enable_testing()

add_subdirectory(libs/arch)
add_subdirectory(libs/db)
add_subdirectory(libs/report)
# roofline depends on libs/arch (occupancy/roofline engines); registered after
# libs/report and before libs/amdparse so the test suite is part of the
# superbuild. Regression note: todo-10 (f4bcd31) briefly replaced this line
# with libs/amdparse, silently dropping the 14-case roofline suite from ctest.
add_subdirectory(libs/roofline)
add_subdirectory(libs/amdparse)
add_subdirectory(libs/ptxparse)
# occupancy depends on libs/arch (ISA-resolved constants) and is consumed by
# tools/analyze (todo 14) / tools/tune (todo 28) / tools/profile (todo 50).
add_subdirectory(libs/occupancy)
# gpuruntime: thin CUDA/HIP runtime abstraction (todo 21), consumed by the
# capture/profile tracks; registers the no-gpu mock/compile-only suites and
# the gpu-amd live HIP suite.
add_subdirectory(libs/gpuruntime)
# capture-core: shared interception foundation (todo 43) + the CUDA LD_PRELOAD
# shim (libkernscope_intercept.so) + its mock-runtime unit/smoke suites.
# Must stay after libs/gpuruntime (the tests consume the todo-21 mock).
add_subdirectory(libs/capture-core)
# capture-hip: todo-45 LIVE HIP capture tool (rocprofiler-sdk intercept-table
# registration + kernel-dispatch/code-object tracing) + the gpu-amd live test
# against the todo-5 layernorm_fused fixture. Must stay after libs/capture-core
# (consumes kernscope::capture_core + its v0 BundleWriter).
add_subdirectory(libs/capture-hip)
# cupti-sub: todo-44 CUPTI callback/injection capture (NVIDIA PRIMARY) + the
# shared subscription library todo-49 reuses + the dual-mechanism equivalence
# test (ctest -R capture_equiv -L gpu-amd) consuming libs/capture-hip + the
# todo-24 portable kernel. Must stay after libs/capture-core (capture_core)
# and libs/capture-hip (the equivalence test runs the HIP tool). Compiled
# against the cuda_cupti 12.6 headers ($CUPTI_HOME) - compile+unit only.
add_subdirectory(libs/cupti-sub)
# bench: nvbench/Triton-style benchmarking harness + gfx1101 roofline peak
# microbenches (todo 22), consumed by tools/bench and the tune track.
add_subdirectory(libs/bench)
# kernels/templates: portable CUDA/HIP kernel template convention + compile
# gates (todo 24). Consumes libs/gpuruntime (the live runner links
# kernscope::gpuruntime) and tests/golden (live gate verdict); registers
# no-gpu nvcc sm_80/sm_90 + hipcc gfx1101 compile gates and the gpu-amd live
# HIP golden gate. Must stay after libs/gpuruntime.
add_subdirectory(kernels/templates)
add_subdirectory(tools/analyze)
# kernscope-bench: the gfx1101 roofline peak measurement CLI (todo 22).
add_subdirectory(tools/bench)
# tools/tune: the C2 autotuner (todos 25-28). Todo 27 (this commit) adds the
# FlashAttention-lite template generator (kernscope_flash_gen) + its test
# executables; todos 25/26 append the matmul/elementwise generators here.
# Consumes libs/arch + libs/roofline and the kernels/templates live runner.
add_subdirectory(tools/tune)
# tools/: kernscope-verify (todo 30, C3 static verifier core). One-time hook
# for the verify track; future tools (capture/pass/profile) add themselves
# inside tools/CMakeLists.txt.
add_subdirectory(tools)

# lit/FileCheck regression suite (defines the check-kernscope target). Must
# come after tools/ so the kernscope-analyze target it DEPENDS on exists.
add_subdirectory(test)
