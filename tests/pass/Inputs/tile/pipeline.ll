; pipeline.ll — kernscope-tile end-to-end fixture: tile -> promote ->
; vectorize. N=100, tile 32 -> 3 full tiles + a 4-iteration remainder
; loop; kernscope-promote then consumes the remainder loop's constant-
; start {96,+,1} window (its v1 shape) into a cooperative block load;
; kernscope-vectorize runs clean after (its v1 scope is global loads, the
; promoted reads are addrspace(3) -> untouched).

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define void @pipeline_kernel(ptr addrspace(1) %g, ptr addrspace(1) noalias %out) "kernscope.tile.arch"="sm_86" "kernscope.tile.size"="32" "kernscope.promote.arch"="sm_86" {
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
