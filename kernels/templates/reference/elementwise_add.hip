//===- elementwise_add.hip - portable elementwise-add template -------------===//
//
// kernscope kernels/templates reference template (todo 24). Single-source
// portable convention: compiled under nvcc -arch=sm_80/sm_90 (compile-only,
// GPU-less, via -x cu) AND under hipcc --offload-arch=gfx1101 (compile AND
// run on this machine through the gpuruntime runner, golden-validated).
//
// Kernel contract (documented in kernels/templates/README.md, used by
// tools/hip_template_runner.cpp): (const T* input..., T* output,
// int32 scalar...), device pointers passed by value.
//
// bf16 rounding contract (todo 23): bf16 in, fp32 add, bf16 out.
//
// The live golden gate validates this kernel against the `reduction` family
// with shape 2,N: the golden reference y[j] = sum_i x[i,j] over axis 0 of a
// 2-row tensor IS x[0] + x[1] -- exactly the elementwise add of two vectors
// (see kernels/templates/README.md "Golden mapping").
//
//===----------------------------------------------------------------------===//

#include "kernscope/kernel_traits.h"

extern "C" __global__ void kernscope_templates_elementwise_add(
    const ks_bf16 *a, const ks_bf16 *b, ks_bf16 *out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const float acc = ks_bf16_to_float(a[i]) + ks_bf16_to_float(b[i]);
    out[i] = ks_float_to_bf16(acc);
  }
}
