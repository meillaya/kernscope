; Fixture: stride-8 float global accesses — per-lane stride 32 B, so one
; warp touches 32 distinct 32B sectors for 128 B of useful data (12.5%
; efficiency) -> the analyzer's strided class
; (tools/analyze/src/analyze_coalescing.cpp: classify_pattern).
target triple = "nvptx64-nvidia-cuda"

define void @strided_kernel(ptr addrspace(1) %a) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %s = mul i32 %tid, 8
  %idx = zext i32 %s to i64
  %p = getelementptr float, ptr addrspace(1) %a, i64 %idx
  %v = load float, ptr addrspace(1) %p, align 4
  store float %v, ptr addrspace(1) %p, align 4
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
