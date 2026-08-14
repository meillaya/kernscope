# kernscope-pass module: promote (todo 40).
#
# The global-to-shared memory promotion pass (cooperative block load).
# Self-registers via REGISTER_PASS_CALLBACK (tools/pass/include/kernscope/
# pass/register_pass.h) from its OWN translation unit — no edits outside
# this directory (todo-36 isolation contract).
#
# libs/arch reuse: the smem-budget guard calls kernscope::get_arch/has_arch
# from kernscope/arch.h (a header-only INTERFACE lib) — the same reuse as
# the todo-38 pad module; nothing in libs/ is modified.
#
# Design notes: include/kernscope/pass/promote.h.
target_sources(kernscope-pass PRIVATE
  src/promote.cpp
)
target_include_directories(kernscope-pass PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/libs/arch/include
)
