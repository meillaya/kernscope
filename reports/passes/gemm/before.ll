; ModuleID = '/home/mei/projects/llvm.h/tools/pass/showcase/gemm/kernel.cu'
source_filename = "/home/mei/projects/llvm.h/tools/pass/showcase/gemm/kernel.cu"
target datalayout = "e-p6:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind memory(argmem: readwrite)
define dso_local ptx_kernel void @_Z11gemm_kernelPKfS0_Pf(ptr noalias noundef readonly captures(none) %0, ptr noalias noundef readonly captures(none) %1, ptr noalias noundef writeonly captures(none) %2) local_unnamed_addr #0 {
  %4 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %5 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
  %6 = mul nuw nsw i32 %4, %5
  %7 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %8 = add nuw nsw i32 %6, %7
  %9 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %10 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %11 = mul i32 %9, %10
  %12 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %13 = add i32 %11, %12
  %14 = mul nuw nsw i32 %8, 100
  br label %20

15:                                               ; preds = %20
  %16 = shl nuw nsw i32 %8, 6
  %17 = add nsw i32 %16, %13
  %18 = sext i32 %17 to i64
  %19 = getelementptr inbounds float, ptr %2, i64 %18
  store float %33, ptr %19, align 4, !tbaa !5
  ret void

20:                                               ; preds = %3, %20
  %21 = phi i32 [ 0, %3 ], [ %34, %20 ]
  %22 = phi float [ 0.000000e+00, %3 ], [ %33, %20 ]
  %23 = add nuw nsw i32 %21, %14
  %24 = zext nneg i32 %23 to i64
  %25 = getelementptr inbounds nuw float, ptr %0, i64 %24
  %26 = load float, ptr %25, align 4, !tbaa !5
  %27 = shl nuw nsw i32 %21, 6
  %28 = add nsw i32 %27, %13
  %29 = sext i32 %28 to i64
  %30 = getelementptr inbounds float, ptr %1, i64 %29
  %31 = load float, ptr %30, align 4, !tbaa !5
  %32 = fmul contract float %26, %31
  %33 = fadd contract float %22, %32
  %34 = add nuw nsw i32 %21, 1
  %35 = icmp eq i32 %34, 100
  br i1 %35, label %15, label %20, !llvm.loop !9
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 65535) i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.y() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind memory(argmem: readwrite) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_80" "target-features"="+ptx85,+sm_80" "uniform-work-group-size"="true" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!nvvm.annotations = !{!0}
!llvm.module.flags = !{!1, !2, !3}
!llvm.ident = !{!4}

!0 = !{ptr @_Z11gemm_kernelPKfS0_Pf}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!3 = !{i32 7, !"frame-pointer", i32 2}
!4 = !{!"clang version 21.1.8"}
!5 = !{!6, !6, i64 0}
!6 = !{!"float", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C++ TBAA"}
!9 = distinct !{!9, !10, !11}
!10 = !{!"llvm.loop.mustprogress"}
!11 = !{!"llvm.loop.unroll.disable"}
