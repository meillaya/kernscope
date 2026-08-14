; barrier_global.ll — loads of a (non-constant) global separated by a
; barrier: other threads may write the global between barriers, so %b is
; NOT eliminated. llvm-diff: unchanged.

declare void @llvm.nvvm.barrier0() convergent

define float @bar(ptr %p) {
entry:
  %a = load float, ptr %p
  call void @llvm.nvvm.barrier0()
  %b = load float, ptr %p
  %s = fadd float %a, %b
  ret float %s
}
