; loop_varying.ll — the GEP index is the induction variable %i (defined
; inside the loop): NOT hoisted. llvm-diff: unchanged.

define void @vary(ptr %base, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %addr = getelementptr float, ptr %base, i64 %i
  %v = load float, ptr %addr
  %i.next = add i64 %i, 1
  %cmp = icmp ult i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}
