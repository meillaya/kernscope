; hoist.ll — the address computation %addr depends only on loop-invariant
; operands (args %base, %idx): its SCEV is loop-invariant, so it is
; hoisted to the preheader. GOLDEN: the GEP moves to entry, the load stays.

define void @hoist(ptr %base, i64 %n, i64 %idx) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %addr = getelementptr float, ptr %base, i64 %idx
  %v = load float, ptr %addr
  %i.next = add i64 %i, 1
  %cmp = icmp ult i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}
