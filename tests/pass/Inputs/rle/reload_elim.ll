; reload_elim.ll — two loads of the same location with no intervening
; def/barrier: within a phase, plain AA + MemorySSA proves %b redundant.
; GOLDEN: %b eliminated, uses rewritten to %a.

define float @reload(ptr %p) {
entry:
  %a = load float, ptr %p
  %b = load float, ptr %p
  %s = fadd float %a, %b
  ret float %s
}
