# kernscope shared-core library tree.
#
# libs/amdparse: the AMDGPU ISA *text* parser (todo 10). Drives LLVM 21's MC
# layer (MCAsmParser + the registered AMDGPU target) to turn hipcc/clang -S
# output for gfx9/gfx11 into typed instruction records (VALU/SALU/LDS/global/
# buffer/flat/scratch/branch/barrier/wait/MFMA/WMMA) plus a label-based CFG.
#
# Linking: the top-level CMakeLists runs find_package(LLVM 21.1 CONFIG) and
# exposes LLVM_AVAILABLE_LIBS. We link the monolithic `LLVM` component (the
# LLVM 21 shared library), which carries every registered target including
# AMDGPU (llvm::InitializeAllTargetInfos()/InitializeAllTargetMCs() +
# InitializeAllAsmParsers() happen in amd_parser.cpp at first use). The AMDGPU
# target is guaranteed present: LLVM_TARGETS_TO_BUILD includes AMDGPU in this
# LLVM 21.1.8 (verified in LLVMConfig.cmake during planning).
#
# Runtime RPATH: libLLVM.so lives in the nix store; CMake imported targets
# don't add it to the executable RPATH automatically, so the test executable
# gets LLVM_LIBRARY_DIR on its BUILD_RPATH (needed for ctest to run it).
add_library(kernscope_amdparse STATIC src/amd_parser.cpp)

target_include_directories(kernscope_amdparse PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(kernscope_amdparse SYSTEM PRIVATE
  ${LLVM_INCLUDE_DIRS})
target_compile_features(kernscope_amdparse PUBLIC cxx_std_20)

# Link the `LLVM` component (the LLVM 21 shared library, listed in
# LLVM_AVAILABLE_LIBS from find_package(LLVM 21.1); resolves to the imported
# LLVM target -> libLLVM.so.21.1). PUBLIC so downstream consumers of this
# static lib also get the LLVM symbols.
target_link_libraries(kernscope_amdparse PUBLIC LLVM)

add_library(kernscope::amdparse ALIAS kernscope_amdparse)

# --- unit tests (gtest) -------------------------------------------------------
# ctest hardware-gating convention (todo 3): every registered test carries a
# LABELS property of exactly one of no-gpu | gpu-amd | gpu-nvidia-rental.
# Parsing ISA text is pure host-side (no GPU required) -> no-gpu.
find_package(GTest REQUIRED)
add_executable(amdparse-test tests/amdparse_test.cpp)
target_link_libraries(amdparse-test PRIVATE kernscope::amdparse GTest::gtest_main)
target_compile_definitions(amdparse-test PRIVATE
  TEST_CORPUS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../tests/amdparse/corpus")
set_target_properties(amdparse-test PROPERTIES BUILD_RPATH "${LLVM_LIBRARY_DIR}")
include(GoogleTest)
gtest_discover_tests(amdparse-test PROPERTIES LABELS "no-gpu")
