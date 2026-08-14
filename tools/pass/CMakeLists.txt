# kernscope-pass: the out-of-tree LLVM 21 pass-plugin (todo 36, Wave-7
# foundation). Consumed by todos 37-41 (vectorize/pad/rle/promote/tile).
#
# THE PLUGIN (kernscope-pass):
#   * MODULE shared library dlopen'ed by opt (-load-pass-plugin=) and clang
#     (-fpass-plugin=) — the llvm::PassPluginLibraryInfo contract
#     (research brief §1.2). Like llvm/examples/Bye (in-tree
#     add_llvm_pass_plugin) but out-of-tree: plain add_library(MODULE) +
#     link against the monolithic LLVM shared library, the llvm-tutor
#     pattern ADAPTED for LLVM 21 (llvm-tutor main itself wants >= 22 and
#     is NOT depended on — canonical references are Bye/IRTransforms).
#   * Output lands in ${CMAKE_BINARY_DIR}/lib/kernscope-pass.so (MODULE
#     libraries carry NO "lib" prefix on Linux, like in-tree Bye.so); the
#     lit site cfg (tests/pass/lit.site.cfg.py.in) pins the exact path.
#   * Built with the same settings as the store LLVM 21.1.8
#     (LLVM_ENABLE_RTTI=TRUE -> default RTTI; LLVM_ENABLE_PLUGINS=ON ->
#     dlopen works; verified in LLVMConfig.cmake, Metis-validated).
#
# THE ENTRY (src/plugin.cpp) is generic: llvmGetPassPluginInfo iterates the
# kernscope::pass::PassRegistry static list (register_pass.h —
# REGISTER_PASS_CALLBACK) and invokes every registered callback with the
# PassBuilder. Pass modules register themselves from their OWN TU; the five
# passes/{vectorize,pad,rle,promote,tile} subdirectories are pre-declared
# below as STUBS so todos 37-41 never touch this file.
#
# LIT WIRING: mirrors tools/verify/CMakeLists.txt — configure_lit_site_cfg
# materializes tests/pass/lit.site.cfg.py into ${CMAKE_BINARY_DIR}/tests/
# pass; test/CMakeLists.txt adds that dir to the single check-kernscope
# target (DEPENDS kernscope-pass).

add_library(kernscope-pass MODULE
  src/register_pass.cpp
  src/plugin.cpp
  src/hello_pass.cpp
)
target_include_directories(kernscope-pass PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(kernscope-pass SYSTEM PRIVATE
  ${LLVM_INCLUDE_DIRS})
target_compile_features(kernscope-pass PRIVATE cxx_std_20)
target_link_libraries(kernscope-pass PRIVATE LLVM)
set_target_properties(kernscope-pass PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  # MODULE output without the "lib" prefix, matching the in-tree plugin
  # convention (llvm/examples/Bye -> lib/Bye.so).
  PREFIX "")

# --- lit wiring for tests/pass/ (todo 36) -----------------------------------
# The pass lit suite: opt -load-pass-plugin smoke (hello_pass.test) + the
# clang -fpass-plugin .cu smoke (clang_fpass_plugin.test). One lit
# invocation (check-kernscope) runs the main suite + tests/verify + this
# suite; the site cfg is configured HERE (same split as tools/verify/
# CMakeLists.txt vs test/CMakeLists.txt).
find_package(Python3 COMPONENTS Interpreter REQUIRED)
set(KERNSCOPE_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(KERNSCOPE_BINARY_DIR ${PROJECT_BINARY_DIR})
set(KERNSCOPE_TOOLS_DIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_TOOLS_DIR ${LLVM_TOOLS_BINARY_DIR})
set(KERNSCOPE_PASS_PLUGIN
    ${CMAKE_BINARY_DIR}/lib/kernscope-pass${CMAKE_SHARED_LIBRARY_SUFFIX})
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/tests/pass)
configure_lit_site_cfg(
  ${CMAKE_SOURCE_DIR}/tests/pass/lit.site.cfg.py.in
  ${CMAKE_BINARY_DIR}/tests/pass/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_SOURCE_DIR}/tests/pass/lit.cfg.py
)

# --- pass-module subdirectories (todos 37-41) --------------------------------
# Stub CMakeLists.txt exist NOW so those todos add their plugin sources in
# their own directory only (the todo-36 isolation contract: the parent
# CMakeLists.txt and src/plugin.cpp stay untouched).
add_subdirectory(passes/vectorize)
add_subdirectory(passes/pad)
add_subdirectory(passes/rle)
add_subdirectory(passes/promote)
add_subdirectory(passes/tile)
