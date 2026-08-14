// bad_syntax.cu — kernscope-verify QA negative fixture (todo 30).
//
// Deliberately broken source: the verifier must NOT crash; it must forward
// the clang diagnostic and exit 2 (QA negative contract, exercised by
// tools/verify/tests/verify_core_test.cpp + the harness in bash).
//
// The error: unbalanced brace on the kernel's closing line.

#include <cuda_runtime.h>

extern "C" cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim,
                                         size_t sharedMem,
                                         cudaStream_t stream);

__global__ void bad_kernel(int* out, int n)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        out[i] = i
    }
}
