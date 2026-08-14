; mul_shl.ll — mul X, 2^k -> shl X, k (SCEV-proven power-of-two constant,
; k == 0 erases the mul; nsw/nuw preserved for k >= 1). GOLDEN.

define i64 @scale(i64 %x) {
entry:
  %a = mul i64 %x, 8
  %b = mul i64 %x, 1
  %c = add i64 %a, %b
  ret i64 %c
}

define i64 @scale_nuw(i64 %x) {
entry:
  %a = mul nuw i64 %x, 16
  ret i64 %a
}

define i64 @scale_nsw(i64 %x) {
entry:
  %a = mul nsw i64 %x, 4
  ret i64 %a
}

define i64 @scale_none(i64 %x) {
entry:
  %a = mul i64 %x, 3
  ret i64 %a
}
