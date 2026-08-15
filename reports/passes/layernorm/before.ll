; ModuleID = '/home/mei/projects/llvm.h/tools/pass/showcase/layernorm/kernel.cu'
source_filename = "/home/mei/projects/llvm.h/tools/pass/showcase/layernorm/kernel.cu"
target datalayout = "e-p6:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@LN_IN = dso_local addrspace(1) externally_initialized global [4096 x float] zeroinitializer, align 16
@LN_OUT = dso_local addrspace(1) externally_initialized global [4096 x float] zeroinitializer, align 4
@llvm.compiler.used = appending global [2 x ptr] [ptr addrspacecast (ptr addrspace(1) @LN_IN to ptr), ptr addrspacecast (ptr addrspace(1) @LN_OUT to ptr)], section "llvm.metadata"

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none)
define dso_local ptx_kernel void @_Z16layernorm_kernelv() local_unnamed_addr #0 {
  %1 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %2 = shl nuw nsw i32 %1, 6
  %3 = zext nneg i32 %2 to i64
  %4 = getelementptr inbounds nuw [4096 x float], ptr addrspacecast (ptr addrspace(1) @LN_IN to ptr), i64 0, i64 %3
  br label %7

5:                                                ; preds = %7
  %6 = fmul contract float %58, 1.562500e-02
  br label %66

7:                                                ; preds = %0, %7
  %8 = phi float [ 0.000000e+00, %0 ], [ %58, %7 ]
  %9 = phi i32 [ 0, %0 ], [ %59, %7 ]
  %10 = zext nneg i32 %9 to i64
  %11 = getelementptr inbounds nuw float, ptr %4, i64 %10
  %12 = load float, ptr %11, align 4, !tbaa !5
  %13 = getelementptr inbounds nuw i8, ptr %11, i64 4
  %14 = load float, ptr %13, align 4, !tbaa !5
  %15 = fadd contract float %12, %14
  %16 = getelementptr inbounds nuw i8, ptr %11, i64 8
  %17 = load float, ptr %16, align 4, !tbaa !5
  %18 = fadd contract float %15, %17
  %19 = getelementptr inbounds nuw i8, ptr %11, i64 12
  %20 = load float, ptr %19, align 4, !tbaa !5
  %21 = fadd contract float %18, %20
  %22 = fadd contract float %8, %21
  %23 = getelementptr inbounds nuw i8, ptr %11, i64 16
  %24 = load float, ptr %23, align 4, !tbaa !5
  %25 = getelementptr inbounds nuw i8, ptr %11, i64 20
  %26 = load float, ptr %25, align 4, !tbaa !5
  %27 = fadd contract float %24, %26
  %28 = getelementptr inbounds nuw i8, ptr %11, i64 24
  %29 = load float, ptr %28, align 4, !tbaa !5
  %30 = fadd contract float %27, %29
  %31 = getelementptr inbounds nuw i8, ptr %11, i64 28
  %32 = load float, ptr %31, align 4, !tbaa !5
  %33 = fadd contract float %30, %32
  %34 = fadd contract float %22, %33
  %35 = getelementptr inbounds nuw i8, ptr %11, i64 32
  %36 = load float, ptr %35, align 4, !tbaa !5
  %37 = getelementptr inbounds nuw i8, ptr %11, i64 36
  %38 = load float, ptr %37, align 4, !tbaa !5
  %39 = fadd contract float %36, %38
  %40 = getelementptr inbounds nuw i8, ptr %11, i64 40
  %41 = load float, ptr %40, align 4, !tbaa !5
  %42 = fadd contract float %39, %41
  %43 = getelementptr inbounds nuw i8, ptr %11, i64 44
  %44 = load float, ptr %43, align 4, !tbaa !5
  %45 = fadd contract float %42, %44
  %46 = fadd contract float %34, %45
  %47 = getelementptr inbounds nuw i8, ptr %11, i64 48
  %48 = load float, ptr %47, align 4, !tbaa !5
  %49 = getelementptr inbounds nuw i8, ptr %11, i64 52
  %50 = load float, ptr %49, align 4, !tbaa !5
  %51 = fadd contract float %48, %50
  %52 = getelementptr inbounds nuw i8, ptr %11, i64 56
  %53 = load float, ptr %52, align 4, !tbaa !5
  %54 = fadd contract float %51, %53
  %55 = getelementptr inbounds nuw i8, ptr %11, i64 60
  %56 = load float, ptr %55, align 4, !tbaa !5
  %57 = fadd contract float %54, %56
  %58 = fadd contract float %46, %57
  %59 = add nuw nsw i32 %9, 16
  %60 = icmp samesign ult i32 %9, 48
  br i1 %60, label %7, label %5, !llvm.loop !9

61:                                               ; preds = %66
  %62 = getelementptr inbounds nuw [4096 x float], ptr addrspacecast (ptr addrspace(1) @LN_OUT to ptr), i64 0, i64 %3
  %63 = fmul contract float %149, 1.562500e-02
  %64 = fadd contract float %63, 0x3EE4F8B580000000
  %65 = fdiv contract float 1.000000e+00, %64
  br label %153

66:                                               ; preds = %5, %66
  %67 = phi float [ 0.000000e+00, %5 ], [ %149, %66 ]
  %68 = phi i32 [ 0, %5 ], [ %150, %66 ]
  %69 = zext nneg i32 %68 to i64
  %70 = getelementptr inbounds nuw float, ptr %4, i64 %69
  %71 = load float, ptr %70, align 4, !tbaa !5
  %72 = fsub contract float %71, %6
  %73 = getelementptr inbounds nuw i8, ptr %70, i64 4
  %74 = load float, ptr %73, align 4, !tbaa !5
  %75 = fsub contract float %74, %6
  %76 = getelementptr inbounds nuw i8, ptr %70, i64 8
  %77 = load float, ptr %76, align 4, !tbaa !5
  %78 = fsub contract float %77, %6
  %79 = getelementptr inbounds nuw i8, ptr %70, i64 12
  %80 = load float, ptr %79, align 4, !tbaa !5
  %81 = fsub contract float %80, %6
  %82 = getelementptr inbounds nuw i8, ptr %70, i64 16
  %83 = load float, ptr %82, align 4, !tbaa !5
  %84 = fsub contract float %83, %6
  %85 = getelementptr inbounds nuw i8, ptr %70, i64 20
  %86 = load float, ptr %85, align 4, !tbaa !5
  %87 = fsub contract float %86, %6
  %88 = getelementptr inbounds nuw i8, ptr %70, i64 24
  %89 = load float, ptr %88, align 4, !tbaa !5
  %90 = fsub contract float %89, %6
  %91 = getelementptr inbounds nuw i8, ptr %70, i64 28
  %92 = load float, ptr %91, align 4, !tbaa !5
  %93 = fsub contract float %92, %6
  %94 = getelementptr inbounds nuw i8, ptr %70, i64 32
  %95 = load float, ptr %94, align 4, !tbaa !5
  %96 = fsub contract float %95, %6
  %97 = getelementptr inbounds nuw i8, ptr %70, i64 36
  %98 = load float, ptr %97, align 4, !tbaa !5
  %99 = fsub contract float %98, %6
  %100 = getelementptr inbounds nuw i8, ptr %70, i64 40
  %101 = load float, ptr %100, align 4, !tbaa !5
  %102 = fsub contract float %101, %6
  %103 = getelementptr inbounds nuw i8, ptr %70, i64 44
  %104 = load float, ptr %103, align 4, !tbaa !5
  %105 = fsub contract float %104, %6
  %106 = getelementptr inbounds nuw i8, ptr %70, i64 48
  %107 = load float, ptr %106, align 4, !tbaa !5
  %108 = fsub contract float %107, %6
  %109 = getelementptr inbounds nuw i8, ptr %70, i64 52
  %110 = load float, ptr %109, align 4, !tbaa !5
  %111 = fsub contract float %110, %6
  %112 = getelementptr inbounds nuw i8, ptr %70, i64 56
  %113 = load float, ptr %112, align 4, !tbaa !5
  %114 = fsub contract float %113, %6
  %115 = getelementptr inbounds nuw i8, ptr %70, i64 60
  %116 = load float, ptr %115, align 4, !tbaa !5
  %117 = fsub contract float %116, %6
  %118 = fmul contract float %72, %72
  %119 = fmul contract float %75, %75
  %120 = fadd contract float %118, %119
  %121 = fmul contract float %78, %78
  %122 = fadd contract float %120, %121
  %123 = fmul contract float %81, %81
  %124 = fadd contract float %122, %123
  %125 = fmul contract float %84, %84
  %126 = fadd contract float %124, %125
  %127 = fmul contract float %87, %87
  %128 = fadd contract float %126, %127
  %129 = fmul contract float %90, %90
  %130 = fadd contract float %128, %129
  %131 = fmul contract float %93, %93
  %132 = fadd contract float %130, %131
  %133 = fmul contract float %96, %96
  %134 = fadd contract float %132, %133
  %135 = fmul contract float %99, %99
  %136 = fadd contract float %134, %135
  %137 = fmul contract float %102, %102
  %138 = fadd contract float %136, %137
  %139 = fmul contract float %105, %105
  %140 = fadd contract float %138, %139
  %141 = fmul contract float %108, %108
  %142 = fadd contract float %140, %141
  %143 = fmul contract float %111, %111
  %144 = fadd contract float %142, %143
  %145 = fmul contract float %114, %114
  %146 = fadd contract float %144, %145
  %147 = fmul contract float %117, %117
  %148 = fadd contract float %146, %147
  %149 = fadd contract float %67, %148
  %150 = add nuw nsw i32 %68, 16
  %151 = icmp samesign ult i32 %68, 48
  br i1 %151, label %66, label %61, !llvm.loop !12

152:                                              ; preds = %153
  ret void

153:                                              ; preds = %61, %153
  %154 = phi i32 [ 0, %61 ], [ %266, %153 ]
  %155 = zext nneg i32 %154 to i64
  %156 = getelementptr inbounds nuw float, ptr %4, i64 %155
  %157 = load float, ptr %156, align 4, !tbaa !5
  %158 = fsub contract float %157, %6
  %159 = fmul contract float %65, %158
  %160 = getelementptr inbounds nuw float, ptr %62, i64 %155
  store float %159, ptr %160, align 4, !tbaa !5
  %161 = or disjoint i32 %154, 1
  %162 = zext nneg i32 %161 to i64
  %163 = getelementptr inbounds nuw float, ptr %4, i64 %162
  %164 = load float, ptr %163, align 4, !tbaa !5
  %165 = fsub contract float %164, %6
  %166 = fmul contract float %65, %165
  %167 = getelementptr inbounds nuw float, ptr %62, i64 %162
  store float %166, ptr %167, align 4, !tbaa !5
  %168 = or disjoint i32 %154, 2
  %169 = zext nneg i32 %168 to i64
  %170 = getelementptr inbounds nuw float, ptr %4, i64 %169
  %171 = load float, ptr %170, align 4, !tbaa !5
  %172 = fsub contract float %171, %6
  %173 = fmul contract float %65, %172
  %174 = getelementptr inbounds nuw float, ptr %62, i64 %169
  store float %173, ptr %174, align 4, !tbaa !5
  %175 = or disjoint i32 %154, 3
  %176 = zext nneg i32 %175 to i64
  %177 = getelementptr inbounds nuw float, ptr %4, i64 %176
  %178 = load float, ptr %177, align 4, !tbaa !5
  %179 = fsub contract float %178, %6
  %180 = fmul contract float %65, %179
  %181 = getelementptr inbounds nuw float, ptr %62, i64 %176
  store float %180, ptr %181, align 4, !tbaa !5
  %182 = or disjoint i32 %154, 4
  %183 = zext nneg i32 %182 to i64
  %184 = getelementptr inbounds nuw float, ptr %4, i64 %183
  %185 = load float, ptr %184, align 4, !tbaa !5
  %186 = fsub contract float %185, %6
  %187 = fmul contract float %65, %186
  %188 = getelementptr inbounds nuw float, ptr %62, i64 %183
  store float %187, ptr %188, align 4, !tbaa !5
  %189 = or disjoint i32 %154, 5
  %190 = zext nneg i32 %189 to i64
  %191 = getelementptr inbounds nuw float, ptr %4, i64 %190
  %192 = load float, ptr %191, align 4, !tbaa !5
  %193 = fsub contract float %192, %6
  %194 = fmul contract float %65, %193
  %195 = getelementptr inbounds nuw float, ptr %62, i64 %190
  store float %194, ptr %195, align 4, !tbaa !5
  %196 = or disjoint i32 %154, 6
  %197 = zext nneg i32 %196 to i64
  %198 = getelementptr inbounds nuw float, ptr %4, i64 %197
  %199 = load float, ptr %198, align 4, !tbaa !5
  %200 = fsub contract float %199, %6
  %201 = fmul contract float %65, %200
  %202 = getelementptr inbounds nuw float, ptr %62, i64 %197
  store float %201, ptr %202, align 4, !tbaa !5
  %203 = or disjoint i32 %154, 7
  %204 = zext nneg i32 %203 to i64
  %205 = getelementptr inbounds nuw float, ptr %4, i64 %204
  %206 = load float, ptr %205, align 4, !tbaa !5
  %207 = fsub contract float %206, %6
  %208 = fmul contract float %65, %207
  %209 = getelementptr inbounds nuw float, ptr %62, i64 %204
  store float %208, ptr %209, align 4, !tbaa !5
  %210 = or disjoint i32 %154, 8
  %211 = zext nneg i32 %210 to i64
  %212 = getelementptr inbounds nuw float, ptr %4, i64 %211
  %213 = load float, ptr %212, align 4, !tbaa !5
  %214 = fsub contract float %213, %6
  %215 = fmul contract float %65, %214
  %216 = getelementptr inbounds nuw float, ptr %62, i64 %211
  store float %215, ptr %216, align 4, !tbaa !5
  %217 = or disjoint i32 %154, 9
  %218 = zext nneg i32 %217 to i64
  %219 = getelementptr inbounds nuw float, ptr %4, i64 %218
  %220 = load float, ptr %219, align 4, !tbaa !5
  %221 = fsub contract float %220, %6
  %222 = fmul contract float %65, %221
  %223 = getelementptr inbounds nuw float, ptr %62, i64 %218
  store float %222, ptr %223, align 4, !tbaa !5
  %224 = or disjoint i32 %154, 10
  %225 = zext nneg i32 %224 to i64
  %226 = getelementptr inbounds nuw float, ptr %4, i64 %225
  %227 = load float, ptr %226, align 4, !tbaa !5
  %228 = fsub contract float %227, %6
  %229 = fmul contract float %65, %228
  %230 = getelementptr inbounds nuw float, ptr %62, i64 %225
  store float %229, ptr %230, align 4, !tbaa !5
  %231 = or disjoint i32 %154, 11
  %232 = zext nneg i32 %231 to i64
  %233 = getelementptr inbounds nuw float, ptr %4, i64 %232
  %234 = load float, ptr %233, align 4, !tbaa !5
  %235 = fsub contract float %234, %6
  %236 = fmul contract float %65, %235
  %237 = getelementptr inbounds nuw float, ptr %62, i64 %232
  store float %236, ptr %237, align 4, !tbaa !5
  %238 = or disjoint i32 %154, 12
  %239 = zext nneg i32 %238 to i64
  %240 = getelementptr inbounds nuw float, ptr %4, i64 %239
  %241 = load float, ptr %240, align 4, !tbaa !5
  %242 = fsub contract float %241, %6
  %243 = fmul contract float %65, %242
  %244 = getelementptr inbounds nuw float, ptr %62, i64 %239
  store float %243, ptr %244, align 4, !tbaa !5
  %245 = or disjoint i32 %154, 13
  %246 = zext nneg i32 %245 to i64
  %247 = getelementptr inbounds nuw float, ptr %4, i64 %246
  %248 = load float, ptr %247, align 4, !tbaa !5
  %249 = fsub contract float %248, %6
  %250 = fmul contract float %65, %249
  %251 = getelementptr inbounds nuw float, ptr %62, i64 %246
  store float %250, ptr %251, align 4, !tbaa !5
  %252 = or disjoint i32 %154, 14
  %253 = zext nneg i32 %252 to i64
  %254 = getelementptr inbounds nuw float, ptr %4, i64 %253
  %255 = load float, ptr %254, align 4, !tbaa !5
  %256 = fsub contract float %255, %6
  %257 = fmul contract float %65, %256
  %258 = getelementptr inbounds nuw float, ptr %62, i64 %253
  store float %257, ptr %258, align 4, !tbaa !5
  %259 = or disjoint i32 %154, 15
  %260 = zext nneg i32 %259 to i64
  %261 = getelementptr inbounds nuw float, ptr %4, i64 %260
  %262 = load float, ptr %261, align 4, !tbaa !5
  %263 = fsub contract float %262, %6
  %264 = fmul contract float %65, %263
  %265 = getelementptr inbounds nuw float, ptr %62, i64 %260
  store float %264, ptr %265, align 4, !tbaa !5
  %266 = add nuw nsw i32 %154, 16
  %267 = icmp samesign ult i32 %154, 48
  br i1 %267, label %153, label %152, !llvm.loop !13
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_80" "target-features"="+ptx85,+sm_80" "uniform-work-group-size"="true" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!nvvm.annotations = !{!0}
!llvm.module.flags = !{!1, !2, !3}
!llvm.ident = !{!4}

!0 = !{ptr @_Z16layernorm_kernelv}
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
