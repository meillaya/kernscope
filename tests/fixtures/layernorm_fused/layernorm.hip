// layernorm.hip — fused layer-norm kernel + host driver (golden fixture).
//
// Fused layernorm: load tile -> mean -> variance -> normalize (bias/scale),
// per the polykernel golden convention:
//   * inputs  : bf16 round-to-nearest-even (raw uint16 bit patterns)
//   * accumulate: fp32
//   * output  : bf16 round-to-nearest-even
//
// Structure: one block (THREADS=128) per output row, N=128 columns.
// The block reduction runs through shared memory (ds_read/ds_write traffic
// for the AMD path; __syncthreads barriers). The variance uses the
// one-pass E[x^2]-mean^2 form; the golden generator applies the identical
// formula so deltas stay at fp32 rounding order (thresholds: cosine >= 0.999,
// max rel err <= 1e-2, PCC >= 0.99).
//
// Host driver: reads a self-describing binary file
//   [int32 M][int32 N][uint16 X: M*N bf16][uint16 gamma: N bf16]
//   [uint16 beta: N bf16]
// and writes [uint16 Y: M*N bf16].
//
// Build: hipcc --offload-arch=gfx1101 layernorm.hip -o layernorm_app
// (or via CMakeLists.txt in this directory).

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

#define THREADS 128  // one warp-wide-tree block per row; N must equal THREADS

// ---------------------------------------------------------------------------
// bf16 <-> fp32 (bit-exact, RNE)
// ---------------------------------------------------------------------------

__device__ __host__ inline float bf16_to_fp32(uint16_t h) {
    union {
        uint32_t u;
        float f;
    } c;
    c.u = static_cast<uint32_t>(h) << 16;
    return c.f;
}

// Round-to-nearest-even fp32 -> bf16, matching ml_dtypes.bfloat16.
__device__ __host__ inline uint16_t fp32_to_bf16_rne(float f) {
    union {
        uint32_t u;
        float f;
    } c;
    c.f = f;
    const uint32_t rounding = 0x7FFFu + ((c.u >> 16) & 1u);
    c.u += rounding;
    return static_cast<uint16_t>(c.u >> 16);
}

// ---------------------------------------------------------------------------
// kernel
// ---------------------------------------------------------------------------

__global__ void layernorm_fused(const uint16_t* __restrict__ X,
                                const uint16_t* __restrict__ gamma,
                                const uint16_t* __restrict__ beta,
                                uint16_t* __restrict__ Y,
                                float eps, int n)
{
    __shared__ float sSum[THREADS];
    __shared__ float sSumSq[THREADS];

    const int row = blockIdx.x;
    const int t   = threadIdx.x;
    if (t >= n) {
        return;
    }

    const float x = bf16_to_fp32(X[row * n + t]);  // fp32 accumulate
    sSum[t]   = x;
    sSumSq[t] = x * x;
    __syncthreads();

    // tree reduction in shared memory
    for (int off = THREADS / 2; off > 0; off >>= 1) {
        if (t < off) {
            sSum[t]   += sSum[t + off];
            sSumSq[t] += sSumSq[t + off];
        }
        __syncthreads();
    }

    const float mean = sSum[0] / static_cast<float>(n);
    const float var  = fmaxf(sSumSq[0] / static_cast<float>(n) - mean * mean,
                             0.0f);
    const float inv  = rsqrtf(var + eps);

    const float g = bf16_to_fp32(gamma[t]);
    const float b = bf16_to_fp32(beta[t]);
    Y[row * n + t] = fp32_to_bf16_rne((x - mean) * inv * g + b);
}

// ---------------------------------------------------------------------------
// host driver
// ---------------------------------------------------------------------------

static void die(const char* msg, hipError_t err) {
    std::fprintf(stderr, "[FAIL] layernorm_fused: %s: %s\n", msg,
                 hipGetErrorString(err));
    std::exit(1);
}

int main(int argc, char** argv) {
    const char* in_path  = (argc > 1) ? argv[1] : "input.bin";
    const char* out_path = (argc > 2) ? argv[2] : "output.bin";

    FILE* in = std::fopen(in_path, "rb");
    if (!in) {
        std::fprintf(stderr, "[FAIL] layernorm_fused: cannot open %s\n",
                     in_path);
        return 1;
    }
    int32_t M = 0, N = 0;
    if (std::fread(&M, sizeof(M), 1, in) != 1 ||
        std::fread(&N, sizeof(N), 1, in) != 1 || M <= 0 || N <= 0) {
        std::fprintf(stderr, "[FAIL] layernorm_fused: bad header in %s\n",
                     in_path);
        return 1;
    }
    std::vector<uint16_t> X(static_cast<size_t>(M) * N);
    std::vector<uint16_t> gamma(N);
    std::vector<uint16_t> beta(N);
    const size_t got =
        std::fread(X.data(), sizeof(uint16_t), X.size(), in);
    const size_t got_g = std::fread(gamma.data(), sizeof(uint16_t), N, in);
    const size_t got_b = std::fread(beta.data(), sizeof(uint16_t), N, in);
    std::fclose(in);
    if (got != X.size() || got_g != static_cast<size_t>(N) ||
        got_b != static_cast<size_t>(N)) {
        std::fprintf(stderr, "[FAIL] layernorm_fused: truncated input %s\n",
                     in_path);
        return 1;
    }

    hipError_t err;
    uint16_t *dX = nullptr, *dG = nullptr, *dB = nullptr, *dY = nullptr;
    if ((err = hipMalloc(&dX, X.size() * sizeof(uint16_t))) != hipSuccess ||
        (err = hipMalloc(&dG, gamma.size() * sizeof(uint16_t))) != hipSuccess ||
        (err = hipMalloc(&dB, beta.size() * sizeof(uint16_t))) != hipSuccess ||
        (err = hipMalloc(&dY, X.size() * sizeof(uint16_t))) != hipSuccess) {
        die("hipMalloc", err);
    }
    if ((err = hipMemcpy(dX, X.data(), X.size() * sizeof(uint16_t),
                         hipMemcpyHostToDevice)) != hipSuccess ||
        (err = hipMemcpy(dG, gamma.data(), gamma.size() * sizeof(uint16_t),
                         hipMemcpyHostToDevice)) != hipSuccess ||
        (err = hipMemcpy(dB, beta.data(), beta.size() * sizeof(uint16_t),
                         hipMemcpyHostToDevice)) != hipSuccess) {
        die("hipMemcpy H2D", err);
    }

    const float eps = 1.0e-5f;
    layernorm_fused<<<M, THREADS, 0, 0>>>(dX, dG, dB, dY, eps, N);
    if ((err = hipGetLastError()) != hipSuccess) {
        die("kernel launch", err);
    }
    if ((err = hipDeviceSynchronize()) != hipSuccess) {
        die("hipDeviceSynchronize", err);
    }

    std::vector<uint16_t> Y(X.size());
    if ((err = hipMemcpy(Y.data(), dY, Y.size() * sizeof(uint16_t),
                         hipMemcpyDeviceToHost)) != hipSuccess) {
        die("hipMemcpy D2H", err);
    }

    FILE* out = std::fopen(out_path, "wb");
    if (!out) {
        std::fprintf(stderr, "[FAIL] layernorm_fused: cannot open %s\n",
                     out_path);
        return 1;
    }
    std::fwrite(Y.data(), sizeof(uint16_t), Y.size(), out);
    std::fclose(out);

    hipFree(dX);
    hipFree(dG);
    hipFree(dB);
    hipFree(dY);
    return 0;
}
