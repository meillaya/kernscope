; derived.ll — kernscope-tile arch-derived tile-size fixture.
; NO kernscope.tile.size pin: sm_86 (99 KiB smem, 255 regs) with 4-byte
; elements -> largest power-of-two in {32..1024} fitting both budgets is
; 128. N=300 -> 2 full tiles of 128 + a 44-iteration remainder loop.

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define void @derived_kernel(ptr addrspace(1) %g, ptr addrspace(1) noalias %out) "kernscope.tile.arch"="sm_86" {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %acc = phi float [ 0.000000e+00, %entry ], [ %sum, %loop ]
  %p = getelementptr float, ptr addrspace(1) %g, i64 %i
  %v = load float, ptr addrspace(1) %p, align 4
  %sum = fadd float %acc, %v
  %inc = add i64 %i, 1
  %cmp = icmp ult i64 %inc, 300
  br i1 %cmp, label %loop, label %exit

exit:
  %r = phi float [ %sum, %loop ]
  store float %r, ptr addrspace(1) %out, align 4
  ret void
}
