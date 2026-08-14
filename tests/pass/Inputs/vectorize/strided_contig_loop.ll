; strided_contig_loop.ll — positive fixture for the kernscope-vectorize pass
; (todo 37). A strided-contiguous loop: each iteration touches 4 consecutive
; floats at a stride of 16 floats (64 bytes) per iteration.
;
;   * contiguity:  the SCEV differences between consecutive addresses fold
;     to the constant element size (a + 16*i + j vs a + 16*i + j+1).
;   * alignment:   the chain head a + 16*i has offset {0,+,16} (min
;     trailing zeros 4) and the base argument is align 16 -> the head is
;     provably 16-byte aligned -> <4 x float> (16 bytes) is legal.
;   * scope:       addrspace(1) global — in v1 scope.
;   * aliasing:    no memory op intervenes inside either chain; %a/%b are
;     noalias (the CUDA __restrict__ idiom).
;
; Expect: one load <4 x float> (replacing %l0..%l3) and one store
; <4 x float> (replacing the four stores to %q0..%q3).
target triple = "nvptx64-nvidia-cuda"

define void @strided_contig(float addrspace(1)* noalias align 16 %a,
                            float addrspace(1)* noalias align 16 %b,
                            float %v, i32 %n) {
entry:
  %cmp0 = icmp sgt i32 %n, 0
  br i1 %cmp0, label %loop, label %exit

loop:                                             ; preds = %entry, %loop
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %acc = phi float [ 0.000000e+00, %entry ], [ %acc.next, %loop ]
  %idx = mul i32 %i, 16
  %p0 = getelementptr float, float addrspace(1)* %a, i32 %idx
  %l0 = load float, float addrspace(1)* %p0, align 4
  %idx1 = add i32 %idx, 1
  %p1 = getelementptr float, float addrspace(1)* %a, i32 %idx1
  %l1 = load float, float addrspace(1)* %p1, align 4
  %idx2 = add i32 %idx, 2
  %p2 = getelementptr float, float addrspace(1)* %a, i32 %idx2
  %l2 = load float, float addrspace(1)* %p2, align 4
  %idx3 = add i32 %idx, 3
  %p3 = getelementptr float, float addrspace(1)* %a, i32 %idx3
  %l3 = load float, float addrspace(1)* %p3, align 4
  %s0 = fadd float %l0, %l1
  %s1 = fadd float %s0, %l2
  %s2 = fadd float %s1, %l3
  %acc.next = fadd float %acc, %s2
  %q0 = getelementptr float, float addrspace(1)* %b, i32 %idx
  store float %v, float addrspace(1)* %q0, align 4
  %q1 = getelementptr float, float addrspace(1)* %b, i32 %idx1
  store float %v, float addrspace(1)* %q1, align 4
  %q2 = getelementptr float, float addrspace(1)* %b, i32 %idx2
  store float %v, float addrspace(1)* %q2, align 4
  %q3 = getelementptr float, float addrspace(1)* %b, i32 %idx3
  store float %v, float addrspace(1)* %q3, align 4
  %i.next = add i32 %i, 1
  %cmp = icmp slt i32 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:                                             ; preds = %entry, %loop
  ret void
}
