; addrspace3.ll — negative fixture for the kernscope-vectorize pass
; (todo 37). A fully contiguous + 16-byte-aligned chain of 4 shared-memory
; loads and 4 shared-memory stores (constant offsets, base align 16) — by
; every provability test this WOULD vectorize, except the address space is
; addrspace(3) (NVPTX shared memory) and v1 scope is global (addrspace(1))
; only; shared vectorization is flagged future work. The pass must leave
; the IR byte-identical (asserted via llvm-diff in the lit suite).
target triple = "nvptx64-nvidia-cuda"

define void @shared_chain(float addrspace(3)* align 16 %s, float %v) {
entry:
  %p0 = getelementptr float, float addrspace(3)* %s, i32 0
  %l0 = load float, float addrspace(3)* %p0, align 4
  %p1 = getelementptr float, float addrspace(3)* %s, i32 1
  %l1 = load float, float addrspace(3)* %p1, align 4
  %p2 = getelementptr float, float addrspace(3)* %s, i32 2
  %l2 = load float, float addrspace(3)* %p2, align 4
  %p3 = getelementptr float, float addrspace(3)* %s, i32 3
  %l3 = load float, float addrspace(3)* %p3, align 4
  %s0 = fadd float %l0, %l1
  %s1 = fadd float %s0, %l2
  %s2 = fadd float %s1, %l3
  %q0 = getelementptr float, float addrspace(3)* %s, i32 4
  store float %s2, float addrspace(3)* %q0, align 4
  %q1 = getelementptr float, float addrspace(3)* %s, i32 5
  store float %s2, float addrspace(3)* %q1, align 4
  %q2 = getelementptr float, float addrspace(3)* %s, i32 6
  store float %s2, float addrspace(3)* %q2, align 4
  %q3 = getelementptr float, float addrspace(3)* %s, i32 7
  store float %s2, float addrspace(3)* %q3, align 4
  ret void
}
