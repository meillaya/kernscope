; ModuleID = 'build/showcase_tmp/gemm/canon.ll'
source_filename = "/home/mei/projects/llvm.h/tools/pass/showcase/gemm/kernel.cu"
target datalayout = "e-p6:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind memory(argmem: readwrite)
define dso_local ptx_kernel void @gemm_kernel(ptr noalias noundef readonly captures(none) %0, ptr noalias noundef readonly captures(none) %1, ptr noalias noundef writeonly captures(none) %2) local_unnamed_addr #0 {
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = addrspacecast ptr %0 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %8 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
  %9 = mul nuw nsw i32 %7, %8
  %10 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %11 = add nuw nsw i32 %9, %10
  %12 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %13 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %14 = mul i32 %12, %13
  %15 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %16 = add i32 %14, %15
  %17 = mul nuw nsw i32 %11, 100
  br label %23

18:                                               ; preds = %23
  %19 = shl nuw nsw i32 %11, 6
  %20 = add nsw i32 %19, %16
  %21 = sext i32 %20 to i64
  %22 = getelementptr inbounds float, ptr addrspace(1) %6, i64 %21
  store float %36, ptr addrspace(1) %22, align 4, !tbaa !4
  ret void

23:                                               ; preds = %23, %3
  %24 = phi i32 [ 0, %3 ], [ %37, %23 ]
  %25 = phi float [ 0.000000e+00, %3 ], [ %36, %23 ]
  %26 = add nuw nsw i32 %24, %17
  %27 = zext nneg i32 %26 to i64
  %28 = getelementptr inbounds nuw float, ptr addrspace(1) %5, i64 %27
  %29 = load float, ptr addrspace(1) %28, align 4, !tbaa !4
  %30 = shl nuw nsw i32 %24, 6
  %31 = add nsw i32 %30, %16
  %32 = sext i32 %31 to i64
  %33 = getelementptr inbounds float, ptr addrspace(1) %4, i64 %32
  %34 = load float, ptr addrspace(1) %33, align 4, !tbaa !4
  %35 = fmul contract float %29, %34
  %36 = fadd contract float %25, %35
  %37 = add nuw nsw i32 %24, 1
  %38 = icmp eq i32 %37, 100
  br i1 %38, label %18, label %23, !llvm.loop !8
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 65535) i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.y() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind memory(argmem: readwrite) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_80" "target-features"="+ptx85,+sm_80" "uniform-work-group-size"="true" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!nvvm.annotations = !{}
!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{!"clang version 21.1.8"}
!4 = !{!5, !5, i64 0}
!5 = !{!"float", !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C++ TBAA"}
!8 = distinct !{!8, !9, !10}
!9 = !{!"llvm.loop.mustprogress"}
!10 = !{!"llvm.loop.unroll.disable"}
