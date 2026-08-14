; remainder.ll — kernscope-tile non-divisible bounds fixture.
; N=100, tile 32 (pinned) -> 3 full tiles + a 4-iteration remainder loop
; with CONSTANT start (96) and trip count — the {s,+,1} window that
; kernscope-promote v1 consumes in the pipeline test.

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define void @remainder_kernel(ptr addrspace(1) %g, ptr addrspace(1) noalias %out) "kernscope.tile.arch"="sm_86" "kernscope.tile.size"="32" {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %acc = phi float [ 0.000000e+00, %entry ], [ %sum, %loop ]
  %p = getelementptr float, ptr addrspace(1) %g, i64 %i
  %v = load float, ptr addrspace(1) %p, align 4
  %sum = fadd float %acc, %v
  %inc = add i64 %i, 1
  %cmp = icmp ult i64 %inc, 100
  br i1 %cmp, label %loop, label %exit

exit:
  %r = phi float [ %sum, %loop ]
  store float %r, ptr addrspace(1) %out, align 4
  ret void
}
