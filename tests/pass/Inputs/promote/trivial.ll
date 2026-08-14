; trivial.ll — kernscope-promote smoke fixture (todo 40).
;
; A kernel with no global reads: the pass must run cleanly and change
; nothing (skeleton stage; later stages also leave this untouched — no
; candidate loads -> no rewrite).

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define void @no_reads_kernel(ptr addrspace(1) %out) "kernscope.promote.arch"="sm_86" {
entry:
  store float 1.0, ptr addrspace(1) %out, align 4
  ret void
}
