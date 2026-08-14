; budget.ll — kernscope-promote budget-exceed fixture (todo 40).
;
; The same loop shape as reduce_loop.ll but over 102400 floats: the tile
; would be 409,600 bytes — far beyond sm_86's 99 KiB per-block budget
; (libs/arch). The pass DECLINES with an annotated note; the IR is
; byte-identical (llvm-diff in promote.test).

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define void @budget_kernel(ptr addrspace(1) %g, ptr addrspace(1) noalias %out) "kernscope.promote.arch"="sm_86" {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %acc = phi float [ 0.000000e+00, %entry ], [ %sum, %loop ]
  %p = getelementptr float, ptr addrspace(1) %g, i64 %i
  %v = load float, ptr addrspace(1) %p, align 4
  %sum = fadd float %acc, %v
  %inc = add i64 %i, 1
  %cmp = icmp ult i64 %inc, 102400
  br i1 %cmp, label %loop, label %exit

exit:
  %r = phi float [ %sum, %loop ]
  store float %r, ptr addrspace(1) %out, align 4
  ret void
}
