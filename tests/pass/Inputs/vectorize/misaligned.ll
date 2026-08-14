; misaligned.ll — negative fixture for the kernscope-vectorize pass
; (todo 37). Contiguous 4-float chains (a[i..i+3], b[i..i+3]) at a stride
; of ONE float per iteration: the chain head a + i has byte offset
; 4*sext({0,+,1}) — only 2 provable trailing zeros -> 4-byte alignment. A
; <4 x float> load needs 16 bytes and a <2 x float> load needs 8, so no
; vector width is legal: the pass must leave the IR byte-identical.
; The lit suite asserts this with llvm-diff.
target triple = "nvptx64-nvidia-cuda"

define void @misaligned(float addrspace(1)* noalias align 16 %a,
                        float addrspace(1)* noalias align 16 %b, float %v,
                        i32 %n) {
entry:
  %cmp0 = icmp sgt i32 %n, 0
  br i1 %cmp0, label %loop, label %exit

loop:                                             ; preds = %entry, %loop
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %acc = phi float [ 0.000000e+00, %entry ], [ %acc.next, %loop ]
  %p0 = getelementptr float, float addrspace(1)* %a, i32 %i
  %l0 = load float, float addrspace(1)* %p0, align 4
  %i1 = add i32 %i, 1
  %p1 = getelementptr float, float addrspace(1)* %a, i32 %i1
  %l1 = load float, float addrspace(1)* %p1, align 4
  %i2 = add i32 %i, 2
  %p2 = getelementptr float, float addrspace(1)* %a, i32 %i2
  %l2 = load float, float addrspace(1)* %p2, align 4
  %i3 = add i32 %i, 3
  %p3 = getelementptr float, float addrspace(1)* %a, i32 %i3
  %l3 = load float, float addrspace(1)* %p3, align 4
  %s0 = fadd float %l0, %l1
  %s1 = fadd float %s0, %l2
  %s2 = fadd float %s1, %l3
  %acc.next = fadd float %acc, %s2
  %q0 = getelementptr float, float addrspace(1)* %b, i32 %i
  store float %v, float addrspace(1)* %q0, align 4
  %q1 = getelementptr float, float addrspace(1)* %b, i32 %i1
  store float %v, float addrspace(1)* %q1, align 4
  %q2 = getelementptr float, float addrspace(1)* %b, i32 %i2
  store float %v, float addrspace(1)* %q2, align 4
  %q3 = getelementptr float, float addrspace(1)* %b, i32 %i3
  store float %v, float addrspace(1)* %q3, align 4
  %i.next = add i32 %i, 1
  %cmp = icmp slt i32 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:                                             ; preds = %entry, %loop
  ret void
}
