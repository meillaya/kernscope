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
add_subdirectory(libs/amdparse)
add_subdirectory(libs/ptxparse)
add_subdirectory(tools/analyze)

# lit/FileCheck regression suite (defines the check-kernscope target). Must
# come after tools/ so the kernscope-analyze target it DEPENDS on exists.
add_subdirectory(test)
