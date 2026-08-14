# kernscope-pass module: pad (todo 38).
#
# Self-registers via REGISTER_PASS_CALLBACK (tools/pass/include/kernscope/
# pass/register_pass.h) from its OWN translation unit (pad_pass.cpp) — no
# edits outside this directory (todo-36 isolation contract).
#
# libs/arch reuse: the smem-budget guard calls kernscope::get_arch/has_arch
# from kernscope/arch.h (a header-only INTERFACE lib). The parent target
# links only LLVM, so this module adds the include path itself; nothing in
# libs/ is modified — reuse is header-only, exactly like tools/analyze
# consuming kernscope::arch.

target_sources(kernscope-pass PRIVATE
  pad_pass.cpp
)
target_include_directories(kernscope-pass PRIVATE
  ${CMAKE_SOURCE_DIR}/libs/arch/include
)
