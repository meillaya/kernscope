# tools/ — kernscope tool entry points.
#
# One-time addition (todo 30): the verify track. Each tool keeps its own
# CMakeLists.txt; the root superbuild registers this directory so future
# tools (capture/pass/profile) can hook in here instead of touching the
# root CMakeLists.txt again.
add_subdirectory(verify)
# One-time addition (todo 36): the pass track (kernscope-pass, the
# out-of-tree LLVM 21 pass-plugin). Owns its lit wiring for tests/pass/
# inside tools/pass/CMakeLists.txt.
add_subdirectory(pass)
