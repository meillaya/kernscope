; pad_conflict_free.ll — kernscope-pad no-change fixture (already
; conflict-free).
;
; Identical to pad_conflict.ll but the tile row is 31 elements: the
; lane-affine row index strides 31 words per lane; gcd(31, 32) = 1, so the
; 32 lanes touch 32 distinct banks -> max way 1 -> the pass must DECLINE and
; leave the IR byte-identical (the QA negative asserts llvm-diff shows no
; diff).

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define void @transpose_clean(ptr addrspace(1) %out) "kernscope.pad.arch"="sm_86" {
entry:
  %tile = alloca [16 x [31 x float]], align 4, addrspace(3)
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %row = zext i32 %tid to i64
  %rowptr = getelementptr inbounds [16 x [31 x float]], ptr addrspace(3) %tile, i64 0, i64 %row
  %colptr = getelementptr inbounds [31 x float], ptr addrspace(3) %rowptr, i64 0, i64 0
  %v = load float, ptr addrspace(3) %colptr, align 4
  store float %v, ptr addrspace(1) %out, align 4
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
