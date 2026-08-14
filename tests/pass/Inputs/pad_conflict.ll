; pad_conflict.ll — kernscope-pad happy fixture (alloca form).
;
; A row-major 16x32 float tile accessed column-wise: t[threadIdx.x][0].
; The lane-affine ROW index strides 32 elements = 128 bytes = 32 words per
; lane: every lane lands on the SAME bank -> provable 32-way conflict
; (bank = (word/4) mod 32; way = max multiplicity over the warp; the same
; formula as tools/analyze/src/analyze_banks.cpp compute_banks, todo 16).
; The +1 column pad grows the innermost dim 32 -> 33: the row stride becomes
; 33 words, gcd(33,32)=1 -> conflict-free.
;
; The GEP chain mirrors what clang emits for __shared__ float t[16][32]
; (two-GEP split: outer [16 x [32 x float]] row GEP, inner [32 x float]
; column GEP); the pass must rewrite BOTH source types consistently.
;
; The "kernscope.pad.arch" attribute pins the arch for the smem budget
; guard (sm_86: 99 KiB per block). Without an arch the pass declines rather
; than guess a budget.

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define void @transpose_kernel(ptr addrspace(1) %out) "kernscope.pad.arch"="sm_86" {
entry:
  %tile = alloca [16 x [32 x float]], align 4, addrspace(3)
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %row = zext i32 %tid to i64
  %rowptr = getelementptr inbounds [16 x [32 x float]], ptr addrspace(3) %tile, i64 0, i64 %row
  %colptr = getelementptr inbounds [32 x float], ptr addrspace(3) %rowptr, i64 0, i64 0
  %v = load float, ptr addrspace(3) %colptr, align 4
  store float %v, ptr addrspace(1) %out, align 4
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
