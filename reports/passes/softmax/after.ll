; ModuleID = 'build/showcase_tmp/softmax/canon.ll'
source_filename = "/home/mei/projects/llvm.h/tools/pass/showcase/softmax/kernel.cu"
target datalayout = "e-p6:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: convergent mustprogress noinline norecurse nounwind
define dso_local ptx_kernel void @softmax_kernel(ptr noalias noundef readonly captures(none) %0, ptr noalias noundef writeonly captures(none) %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = addrspacecast ptr %0 to ptr addrspace(1)
  %6 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %7 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %8 = mul i32 %6, %7
  %9 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %10 = add i32 %8, %9
  %11 = icmp sgt i32 %2, 0
  br i1 %11, label %12, label %14

12:                                               ; preds = %3
  %13 = mul nsw i32 %2, %10
  br label %19

14:                                               ; preds = %19, %3
  %15 = phi float [ 0xC7EFFFFFE0000000, %3 ], [ %26, %19 ]
  %16 = icmp sgt i32 %2, 0
  br i1 %16, label %17, label %29

17:                                               ; preds = %14
  %18 = mul nsw i32 %2, %10
  br label %34

19:                                               ; preds = %19, %12
  %20 = phi float [ 0xC7EFFFFFE0000000, %12 ], [ %26, %19 ]
  %21 = phi i32 [ 0, %12 ], [ %27, %19 ]
  %22 = add nsw i32 %21, %13
  %23 = sext i32 %22 to i64
  %24 = getelementptr inbounds float, ptr addrspace(1) %5, i64 %23
  %25 = load float, ptr addrspace(1) %24, align 4, !tbaa !4
  %26 = tail call contract noundef float @__nv_fmaxf(float noundef %20, float noundef %25) #3
  %27 = add nuw nsw i32 %21, 1
  %28 = icmp eq i32 %27, %2
  br i1 %28, label %14, label %19, !llvm.loop !8

29:                                               ; preds = %34, %14
  %30 = phi float [ 0.000000e+00, %14 ], [ %43, %34 ]
  %31 = icmp sgt i32 %2, 0
  br i1 %31, label %32, label %46

32:                                               ; preds = %29
  %33 = mul nsw i32 %2, %10
  br label %47

34:                                               ; preds = %34, %17
  %35 = phi i32 [ 0, %17 ], [ %44, %34 ]
  %36 = phi float [ 0.000000e+00, %17 ], [ %43, %34 ]
  %37 = add nsw i32 %35, %18
  %38 = sext i32 %37 to i64
  %39 = getelementptr inbounds float, ptr addrspace(1) %5, i64 %38
  %40 = load float, ptr addrspace(1) %39, align 4, !tbaa !4
  %41 = fsub contract float %40, %15
  %42 = tail call contract noundef float @__nv_expf(float noundef %41) #3
  %43 = fadd contract float %36, %42
  %44 = add nuw nsw i32 %35, 1
  %45 = icmp eq i32 %44, %2
  br i1 %45, label %29, label %34, !llvm.loop !11

46:                                               ; preds = %47, %29
  ret void

47:                                               ; preds = %47, %32
  %48 = phi i32 [ 0, %32 ], [ %57, %47 ]
  %49 = add nsw i32 %48, %33
  %50 = sext i32 %49 to i64
  %51 = getelementptr inbounds float, ptr addrspace(1) %5, i64 %50
  %52 = load float, ptr addrspace(1) %51, align 4, !tbaa !4
  %53 = fsub contract float %52, %15
  %54 = tail call contract noundef float @__nv_expf(float noundef %53) #3
  %55 = fdiv contract float %54, %30
  %56 = getelementptr inbounds float, ptr addrspace(1) %4, i64 %50
  store float %55, ptr addrspace(1) %56, align 4, !tbaa !4
  %57 = add nuw nsw i32 %48, 1
  %58 = icmp eq i32 %57, %2
  br i1 %58, label %46, label %47, !llvm.loop !12
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: convergent nounwind
declare dso_local float @__nv_fmaxf(float noundef, float noundef) local_unnamed_addr #2

; Function Attrs: convergent nounwind
declare dso_local float @__nv_expf(float noundef) local_unnamed_addr #2

attributes #0 = { convergent mustprogress noinline norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_80" "target-features"="+ptx85,+sm_80" "uniform-work-group-size"="true" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_80" "target-features"="+ptx85,+sm_80" }
attributes #3 = { convergent nounwind }

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
!11 = distinct !{!11, !9, !10}
!12 = distinct !{!12, !9, !10}
