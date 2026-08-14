; dependence.ll — kernscope-tile decline fixture (the plan's MUST-NOT):
; a[i] = a[i-1] is a loop-carried (cross-iteration) RAW dependence with
; distance 1. LAI cannot prove the loop memory-safe -> the pass DECLINES,
; IR unchanged.

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define void @dep_kernel(ptr addrspace(1) %a) "kernscope.tile.arch"="sm_86" "kernscope.tile.size"="32" {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %im1 = sub i64 %i, 1
  %pp = getelementptr float, ptr addrspace(1) %a, i64 %im1
  %pv = load float, ptr addrspace(1) %pp, align 4
  %p = getelementptr float, ptr addrspace(1) %a, i64 %i
  store float %pv, ptr addrspace(1) %p, align 4
  %inc = add i64 %i, 1
  %cmp = icmp ult i64 %inc, 64
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}
