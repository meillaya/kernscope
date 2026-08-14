; volatile.ll — volatile loads are NEVER candidates and block the walk.
; The IR must come out byte-identical (llvm-diff).

define float @vol(ptr %p) {
entry:
  %a = load volatile float, ptr %p
  %b = load volatile float, ptr %p
  %s = fadd float %a, %b
  ret float %s
}
