# kernscope-pass module: rle + addrsimp (todo 39).
#
# Two self-registering function passes, each in its OWN translation unit
# (REGISTER_PASS_CALLBACK — tools/pass/include/kernscope/pass/
# register_pass.h, the todo-36 isolation contract; no edits outside this
# directory):
#   * kernscope-rle      — GPU barrier-aware redundant load elimination
#     (MemorySSA + AA walk; crossing barriers only for globally-invariant
#     locations).
#   * kernscope-addrsimp — SCEV address simplification: mul X,2^k -> shl,
#     GEP folding via InstSimplify, loop-invariant address-chain hoisting.
# Design notes: include/kernscope/pass/{rle,addrsimp}.h.

target_sources(kernscope-pass PRIVATE
  src/rle.cpp
  src/addrsimp.cpp
)
target_include_directories(kernscope-pass PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
