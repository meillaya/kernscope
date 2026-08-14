; reduce_loop.ll — kernscope-promote happy fixture (todo 40).
;
; A block-wide reduction over g[0..128): EVERY thread of the block loads
; the same 128 floats (the loop bound is uniform). The pass promotes the
; loop reads to a [128 x float] addrspace(3) tile filled by a cooperative
; block load (thread tid loads element tid, strided by blockDim.x — with
; 128 threads and a 128-element tile each thread loads exactly one
; element), followed by llvm.nvvm.barrier0(), and redirects the loop's
; reads to the tile. The store to %out is a different object — the AA
; guard (no may-write to the tile range) passes.
;
; The "kernscope.promote.arch" attribute pins the arch for the smem budget
; guard (sm_86: 99 KiB per block); the target triple selects the NVPTX
; intrinsics (tid.x / ntid.x / barrier0).

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define void @reduce_kernel(ptr addrspace(1) %g, ptr addrspace(1) noalias %out) "kernscope.promote.arch"="sm_86" {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %acc = phi float [ 0.000000e+00, %entry ], [ %sum, %loop ]
  %p = getelementptr float, ptr addrspace(1) %g, i64 %i
  %v = load float, ptr addrspace(1) %p, align 4
  %sum = fadd float %acc, %v
  %inc = add i64 %i, 1
  %cmp = icmp ult i64 %inc, 128
  br i1 %cmp, label %loop, label %exit

exit:
  %r = phi float [ %sum, %loop ]
  store float %r, ptr addrspace(1) %out, align 4
  ret void
}
