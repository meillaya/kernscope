//===- kernel_traits.h - shared types for portable kernel templates -------===//
//
// kernscope kernels/templates convention (todo 24). This header is the ONLY
// platform-dependent part of a template: it selects the HIP or CUDA type
// system with the portable single-source guard
//
//     #if defined(__HIP_PLATFORM_AMD__)   -> HIP (hipcc)
//     #else                               -> CUDA (nvcc -x cu)
//
// (pattern proven in /home/mei/projects/polykernel: one template, nvcc
// sm_80/sm_90 + hipcc gfx1101). Kernel template sources include THIS header
// only; the platform runtime header (hip_runtime.h / cuda_runtime.h) is
// pulled in here, so a template is fully self-contained.
//
// IMPORTANT (hipcc 7.2, verified empirically on this machine): hipcc no
// longer defines __HIP_PLATFORM_AMD__ implicitly (checked via -dM on both
// the host and the --genco device pass). The CMake function
// kernscope_add_hip_kernel passes -D__HIP_PLATFORM_AMD__=1 explicitly, so
// templates must not rely on the compiler providing the macro themselves.
//
// bf16 rounding contract (todo 23): inputs are bf16 round-to-nearest-even,
// all arithmetic happens in fp32, the output is rounded to bf16 exactly once.
//
//===----------------------------------------------------------------------===//

#ifndef KERNSCOPE_KERNEL_TRAITS_H
#define KERNSCOPE_KERNEL_TRAITS_H

#if defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

/// HIP bf16 scalar (ROCm 7.2: hip_bfloat16 struct; round-to-nearest-even
/// float constructor, operator float() zero-extension).
using ks_bf16 = hip_bfloat16;

inline __device__ float ks_bf16_to_float(ks_bf16 v) { return static_cast<float>(v); }
inline __device__ ks_bf16 ks_float_to_bf16(float v) { return ks_bf16(v); }

#else // CUDA (nvcc, forced with -x cu by kernscope_add_cuda_kernel)
#include <cuda_runtime.h>
#include <cuda_bf16.h>

/// CUDA bf16 scalar (__nv_bfloat16; __bfloat162float / __float2bfloat16).
using ks_bf16 = __nv_bfloat16;

inline __device__ float ks_bf16_to_float(ks_bf16 v) { return __bfloat162float(v); }
inline __device__ ks_bf16 ks_float_to_bf16(float v) { return __float2bfloat16(v); }

#endif // __HIP_PLATFORM_AMD__

#endif // KERNSCOPE_KERNEL_TRAITS_H
