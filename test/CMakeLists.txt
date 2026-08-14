# lit/FileCheck regression suite for kernscope (defines check-kernscope).
#
# Modeled on the proven sibling wiring (/home/mei/projects/tt-flow/test/
# CMakeLists.txt), itself the upstream out-of-tree standalone pattern
# (mlir/examples/standalone/test). Hardware gating happens via lit
# REQUIRES: no-gpu | gpu-amd | gpu-nvidia-rental features detected in
# lit.cfg.py, mirroring the ctest LABELS convention.

# add_lit_testsuite -> add_lit_target spawns lit as "${Python3_EXECUTABLE}
# <lit>" (AddLLVM.cmake). The installed LLVMConfig.cmake does NOT find
# Python3 for external projects, so find it here or LIT_COMMAND would have an
# empty interpreter and the check target would fail to run lit.
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Variables consumed by lit.site.cfg.py.in. LLVM_TOOLS_BINARY_DIR is exported
# by the installed LLVMConfig.cmake and is the LLVM bin/ holding
# opt/FileCheck/count/not (under nix this is the llvm runtime store path, NOT
# the -dev cmake path).
set(KERNSCOPE_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(KERNSCOPE_BINARY_DIR ${PROJECT_BINARY_DIR})
set(KERNSCOPE_TOOLS_DIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_TOOLS_DIR ${LLVM_TOOLS_BINARY_DIR})

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

# opt/FileCheck/count/not are prebuilt in the installed LLVM (nix store), not
# CMake targets, so DEPENDS only lists the tools we build here.
# ${CMAKE_BINARY_DIR}/tests/verify is the todo-35 kernscope-verify lit suite
# (site cfg configured in tools/verify/CMakeLists.txt) and
# ${CMAKE_BINARY_DIR}/tests/pass is the todo-36 kernscope-pass lit suite
# (site cfg configured in tools/pass/CMakeLists.txt); one lit invocation
# runs all suites, so check-kernscope covers the verifier corpus gate and
# the pass-plugin load-path smokes too. DEPENDS kernscope-pass makes sure
# the plugin .so exists before lit runs.
add_lit_testsuite(check-kernscope "Running the kernscope regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_BINARY_DIR}/tests/verify
  ${CMAKE_BINARY_DIR}/tests/pass
  DEPENDS kernscope-analyze kernscope-verify kernscope-pass
)
set_target_properties(check-kernscope PROPERTIES FOLDER "Tests")
