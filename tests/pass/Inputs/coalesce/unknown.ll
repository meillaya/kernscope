; Fixture: indirect (data-dependent) index — the access a[%j] where %j is a
; LOADED value. The per-lane affine is unresolvable -> unknown-pattern,
; NEVER claimed coalesced (the analyzer's coalescing_unknown honesty rule,
; analyze_coalescing.cpp: "indirect/data-dependent index; coalescing not
; assumed"). The index-gather load %ip itself is affine in tid (stride 8 B
; per lane = 2 x i64 elements per lane -> contiguous-128B).
target triple = "nvptx64-nvidia-cuda"

define void @unknown_kernel(ptr addrspace(1) %a, ptr addrspace(1) %idx) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %t = zext i32 %tid to i64
  %ip = getelementptr i64, ptr addrspace(1) %idx, i64 %t
  %j = load i64, ptr addrspace(1) %ip, align 8
  %p = getelementptr float, ptr addrspace(1) %a, i64 %j
  %v = load float, ptr addrspace(1) %p, align 4
  store float %v, ptr addrspace(1) %p, align 4
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
