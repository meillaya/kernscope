; pad_budget.ll — kernscope-pad decline fixture: conflict is provable but
; the +1 pad would exceed the arch smem budget.
;
; [257 x 98 x float] row=tid: row stride 98 words, gcd(98,32)=2 -> 2-way
; conflict (provable). Original size 257*98*4 = 100,744 B fits sm_86's
; 99 KiB (101,376 B) per-block cap, but the padded [257 x 99 x float] =
; 101,772 B EXCEEDS it. The budget guard (libs/arch: sm_86
; smem_per_block_max_bytes = 99*1024) must decline and leave the alloca
; untouched, with the deterministic stderr annotation.

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define void @budget_kernel(ptr addrspace(1) %out) "kernscope.pad.arch"="sm_86" {
entry:
  %tile = alloca [257 x [98 x float]], align 4, addrspace(3)
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %row = zext i32 %tid to i64
  %rowptr = getelementptr inbounds [257 x [98 x float]], ptr addrspace(3) %tile, i64 0, i64 %row
  %colptr = getelementptr inbounds [98 x float], ptr addrspace(3) %rowptr, i64 0, i64 0
  %v = load float, ptr addrspace(3) %colptr, align 4
  store float %v, ptr addrspace(1) %out, align 4
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
