; ModuleID = '/home/mei/projects/llvm.h/tools/pass/showcase/softmax/kernel.cu'
source_filename = "/home/mei/projects/llvm.h/tools/pass/showcase/softmax/kernel.cu"
target datalayout = "e-p6:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: convergent mustprogress noinline norecurse nounwind
define dso_local ptx_kernel void @softmax_kernel(ptr noalias noundef readonly captures(none) %0, ptr noalias noundef writeonly captures(none) %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %5 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %6 = mul i32 %4, %5
  %7 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %8 = add i32 %6, %7
  %9 = icmp sgt i32 %2, 0
  br i1 %9, label %10, label %12

10:                                               ; preds = %3
  %11 = mul nsw i32 %2, %8
  br label %17

12:                                               ; preds = %17, %3
  %13 = phi float [ 0xC7EFFFFFE0000000, %3 ], [ %24, %17 ]
  %14 = icmp sgt i32 %2, 0
  br i1 %14, label %15, label %27

15:                                               ; preds = %12
  %16 = mul nsw i32 %2, %8
  br label %32

17:                                               ; preds = %10, %17
  %18 = phi float [ 0xC7EFFFFFE0000000, %10 ], [ %24, %17 ]
  %19 = phi i32 [ 0, %10 ], [ %25, %17 ]
  %20 = add nsw i32 %19, %11
  %21 = sext i32 %20 to i64
  %22 = getelementptr inbounds float, ptr %0, i64 %21
  %23 = load float, ptr %22, align 4, !tbaa !5
  %24 = tail call contract noundef float @__nv_fmaxf(float noundef %18, float noundef %23) #3
  %25 = add nuw nsw i32 %19, 1
  %26 = icmp eq i32 %25, %2
  br i1 %26, label %12, label %17, !llvm.loop !9

27:                                               ; preds = %32, %12
  %28 = phi float [ 0.000000e+00, %12 ], [ %41, %32 ]
  %29 = icmp sgt i32 %2, 0
  br i1 %29, label %30, label %44

30:                                               ; preds = %27
  %31 = mul nsw i32 %2, %8
  br label %45

32:                                               ; preds = %15, %32
  %33 = phi i32 [ 0, %15 ], [ %42, %32 ]
  %34 = phi float [ 0.000000e+00, %15 ], [ %41, %32 ]
  %35 = add nsw i32 %33, %16
  %36 = sext i32 %35 to i64
  %37 = getelementptr inbounds float, ptr %0, i64 %36
  %38 = load float, ptr %37, align 4, !tbaa !5
  %39 = fsub contract float %38, %13
  %40 = tail call contract noundef float @__nv_expf(float noundef %39) #3
  %41 = fadd contract float %34, %40
  %42 = add nuw nsw i32 %33, 1
  %43 = icmp eq i32 %42, %2
  br i1 %43, label %27, label %32, !llvm.loop !12

44:                                               ; preds = %45, %27
  ret void

45:                                               ; preds = %30, %45
  %46 = phi i32 [ 0, %30 ], [ %55, %45 ]
  %47 = add nsw i32 %46, %31
  %48 = sext i32 %47 to i64
  %49 = getelementptr inbounds float, ptr %0, i64 %48
  %50 = load float, ptr %49, align 4, !tbaa !5
  %51 = fsub contract float %50, %13
  %52 = tail call contract noundef float @__nv_expf(float noundef %51) #3
  %53 = fdiv contract float %52, %28
  %54 = getelementptr inbounds float, ptr %1, i64 %48
  store float %53, ptr %54, align 4, !tbaa !5
  %55 = add nuw nsw i32 %46, 1
  %56 = icmp eq i32 %55, %2
  br i1 %56, label %44, label %45, !llvm.loop !13
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: convergent nounwind
declare dso_local float @__nv_fmaxf(float noundef, float noundef) local_unnamed_addr #2

; Function Attrs: convergent nounwind
declare dso_local float @__nv_expf(float noundef) local_unnamed_addr #2

attributes #0 = { convergent mustprogress noinline norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_80" "target-features"="+ptx85,+sm_80" "uniform-work-group-size"="true" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_80" "target-features"="+ptx85,+sm_80" }
attributes #3 = { convergent nounwind }

!nvvm.annotations = !{!0}
!llvm.module.flags = !{!1, !2, !3}
!llvm.ident = !{!4}

!0 = !{ptr @softmax_kernel}
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
!12 = distinct !{!12, !10, !11}
!13 = distinct !{!13, !10, !11}
