; Smoke test: opt + FileCheck round-trip on a trivial IR module.
; Runs on every machine (no GPU needed) - the no-gpu label.
; REQUIRES: no-gpu
; RUN: opt -S -passes=verify < %s | FileCheck %s

; CHECK-LABEL: define i32 @kernscope_smoke(i32 %x)
; CHECK: %1 = add i32 %x, 1
; CHECK: ret i32 %1
define i32 @kernscope_smoke(i32 %x) {
  %1 = add i32 %x, 1
  ret i32 %1
}
