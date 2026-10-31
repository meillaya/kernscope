// ptr_cast_bad.cu — kernscope-verify address-space-cast corpus fixture (todo
// 34), check class 8 (invalid address-space casts). One BAD kernel + one
// LEGAL kernel:
//
//   cast_bad_kernel      `__attribute__((address_space(1)))` cast applied
//                        directly to a __shared__ pointer. clang-21 -O1
//                        emits `addrspacecast ptr addrspace(3) to ptr
//                        addrspace(1)` — a DIRECT shared->global cast with
//                        no generic mediation. Per the verified NVPTX table
//                        (docs/report/kernscope-technical-report.pdf)
//                        only generic(0) <-> {global(1), shared(3),
//                        constant(4), local(5)} are legal CUDA casts; a
//                        direct shared<->global pair has no PTX instruction
//                        (LangRef §addrspacecast: unsupported casts yield
//                        poison) -> INVALID_ADDRSPACE_CAST (error citing the
//                        table).
//   cast_legal_kernel    the GENERIC-MEDIATED chain: shared -> generic
//                        (implicit __shared__ to generic pointer) -> global
//                        (explicit address_space(1) cast of the generic
//                        pointer). Both addrspacecasts are legal
//                        (3->0, 0->1) -> SAFE. This is the QA-negative
//                        pattern: the check must NOT flag it.
//
// Compile-time contract (tools/verify/tests/fixtures/README.md): clang 21
// lowers <<< >>> via cudaConfigureCall unless a CUDA >= 9 SDK is visible,
// so the source declares the nvcc-frontend-internal cudaConfigureCall
// (extern "C") and uses the full four-argument launch form. Compiled by the
// verifier's clang ingestion (-O1 host+device passes, then LLVM link);
// never executed.

#include <cuda_runtime.h>

extern "C" cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim,
                                         size_t sharedMem,
                                         cudaStream_t stream);

__shared__ int s[64];
__device__ int g[64];

__global__ void cast_bad_kernel()
{
    // Direct shared -> global cast: the address_space(1) attribute cast
    // applied straight to the __shared__ pointer compiles to
    // `addrspacecast ptr addrspace(3) @s to ptr addrspace(1)` (verified
    // against clang-21 -O1) — no generic mediation, no PTX instruction.
    __attribute__((address_space(1))) int* gp =
        (__attribute__((address_space(1))) int*)s;
    gp[threadIdx.x] = 1;
}

__global__ void cast_legal_kernel()
{
    // Generic-mediated chain: shared -> generic (implicit) -> global. The
    // empty asm block is a compiler read/write barrier: without it clang's
    // instcombine folds the two casts into one direct 3->1 cast (cast-pair
    // elimination), which would make the LEGAL chain indistinguishable from
    // the illegal pattern. With it, the IR keeps `addrspacecast 3 -> 0`
    // followed by `addrspacecast 0 -> 1` — both legal per the table.
    int* gen = s;  // addrspacecast 3 -> 0 (legal)
    asm volatile("" : "+r"(gen));
    __attribute__((address_space(1))) int* gp =
        (__attribute__((address_space(1))) int*)gen;  // addrspacecast 0 -> 1
    gp[threadIdx.x] = 1;
}

int main()
{
    cast_bad_kernel<<<1, 64, 0, 0>>>();
    cast_legal_kernel<<<1, 64, 0, 0>>>();
    cudaDeviceSynchronize();
    return 0;
}
