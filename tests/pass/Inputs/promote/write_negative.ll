; write_negative.ll — kernscope-promote QA negative (todo 40).
;
; Cross-iteration write to the SAME global between uses: the loop reads
; g[i], then stores g[k] (same underlying object %g), then reads g[i]
; again. The cooperative block load would capture the pre-store value;
; the rewritten smem reads would observe stale data. The AA guard
; (no memory op in the function may-modify the tile range) DECLINES the
; promotion with an annotated note; the IR is byte-identical (llvm-diff
; in promote.test).

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define void @wneg_kernel(ptr addrspace(1) %g, ptr addrspace(1) %out) "kernscope.promote.arch"="sm_86" {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %acc = phi float [ 0.000000e+00, %entry ], [ %sum, %loop ]
  %p = getelementptr float, ptr addrspace(1) %g, i64 %i
  %v = load float, ptr addrspace(1) %p, align 4
  %sum = fadd float %acc, %v
  %k = and i64 %i, 31
  %q = getelementptr float, ptr addrspace(1) %g, i64 %k
  store float %v, ptr addrspace(1) %q, align 4
  %inc = add i64 %i, 1
  %cmp = icmp ult i64 %inc, 128
  br i1 %cmp, label %loop, label %exit

exit:
  %r = phi float [ %sum, %loop ]
  store float %r, ptr addrspace(1) %out, align 4
  ret void
}
