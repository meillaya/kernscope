; pad_global.ll — kernscope-pad happy fixture, GLOBAL form: the shape clang
; actually emits for `__shared__ float t[16][32]` at -O0 (addrspace(3)
; internal global + `ptr addrspacecast` ConstantExpr + the two-GEP chain with
; `inbounds nuw`). The pass must pad the global type, keep the addrspacecast
; (opaque ptr — unchanged), rewrite both GEP source types AND preserve the
; nuw flag.

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@_ZZ9transposePfE1t = internal addrspace(3) global [16 x [32 x float]] zeroinitializer, align 4

define void @transpose(ptr addrspace(1) %out) "kernscope.pad.arch"="sm_86" {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %row = zext i32 %tid to i64
  %rowptr = getelementptr inbounds nuw [16 x [32 x float]], ptr addrspacecast (ptr addrspace(3) @_ZZ9transposePfE1t to ptr), i64 0, i64 %row
  %colptr = getelementptr inbounds [32 x float], ptr %rowptr, i64 0, i64 0
  %v = load float, ptr %colptr, align 4
  store float %v, ptr addrspace(1) %out, align 4
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
