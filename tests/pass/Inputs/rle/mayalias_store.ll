; mayalias_store.ll — QA negative: a may-alias store (%p vs %q are plain
; parameters, BasicAA cannot prove them disjoint) between the two loads
; must block elimination. llvm-diff: unchanged at that site.

define float @mayalias(ptr %p, ptr %q) {
entry:
  %a = load float, ptr %p
  store float 0.0, ptr %q
  %b = load float, ptr %p
  %s = fadd float %a, %b
  ret float %s
}
