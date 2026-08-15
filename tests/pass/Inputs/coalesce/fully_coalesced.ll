; Fixture: a minimal NVPTX kernel whose global accesses are stride-1 in the
; lane (threadIdx.x) — the analyzer's contiguous-128B case
; (tools/analyze/src/analyze_coalescing.cpp: classify_pattern; brief §3.1
; fully-coalesced = 100% sector utilization). Two float accesses: one load
; a[i], one store a[i]. Skeleton stage: the classifier counts accesses; the
; classification assertion lands with the SCEV classifier commit.
;
;   a[i] where i = tid.x -> per-lane stride 4 B (one float) -> 32 lanes span
;   exactly 128 B = 4 sectors of 32 B -> efficiency 1.0 -> fully coalesced.
target triple = "nvptx64-nvidia-cuda"

define void @coalesce_kernel(ptr addrspace(1) %a) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %idx = zext i32 %tid to i64
  %p = getelementptr float, ptr addrspace(1) %a, i64 %idx
  %v = load float, ptr addrspace(1) %p, align 4
  store float %v, ptr addrspace(1) %p, align 4
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
