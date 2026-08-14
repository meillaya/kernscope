; barrier_const.ll — loads of a READ-ONLY constant global separated by a
; barrier: the location is globally invariant (no MemoryDef in the
; function may-modifies it — AA proves the barrier call cannot touch a
; constant object), so cross-barrier elimination IS sound and %b is
; eliminated. GOLDEN.

declare void @llvm.nvvm.barrier0() convergent

@c = internal constant float 1.0

define float @bar_const() {
entry:
  %a = load float, ptr @c
  call void @llvm.nvvm.barrier0()
  %b = load float, ptr @c
  %s = fadd float %a, %b
  ret float %s
}
