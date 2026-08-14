; trivial.ll — kernscope-tile skeleton fixture: a kernel with no loops.
; The pass runs cleanly and changes nothing (llvm-diff).

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define void @no_loop_kernel(ptr addrspace(1) %g) "kernscope.tile.arch"="sm_86" {
entry:
  %v = load float, ptr addrspace(1) %g, align 4
  store float %v, ptr addrspace(1) %g, align 4
  ret void
}
