// tiny.cu — todo-50 -lineinfo / cubin-only fixture source.
//
// Generated artifacts (committed alongside, regenerable):
//   nvcc -lineinfo -ptx  -arch=sm_86 tiny.cu -o tiny_lineinfo.ptx   (source<->PTX)
//   nvcc -cubin           -arch=sm_86 tiny.cu -o tiny.cubin         (cubin-only: NO
//                                                                     embedded PTX)
__global__ void tiny_kernel(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = in[i] * 2.0f;
        out[i] = v + 1.0f;
    }
}

void launch_tiny(const float* in, float* out, int n) {
    tiny_kernel<<<(n + 255) / 256, 256>>>(in, out, n);
}
