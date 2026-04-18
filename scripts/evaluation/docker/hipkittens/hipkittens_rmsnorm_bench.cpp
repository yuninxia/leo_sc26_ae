// Standalone HipKittens RMSNorm benchmark driver for profiling with HPCToolkit
// BF16 RMSNorm — B=16, N=4096, D=128 on MI300A (gfx942)
// Adapted from HipKittens kernels/rmsnorm/kernel.cpp
// Removes pybind11/PyTorch dependency for standalone profiling.

#include "kittens.cuh"
#include <cstdio>
#include <cstdlib>
using namespace kittens;

constexpr int B = 16;
constexpr int H = 16;
int N = 4096;
constexpr int D = 128;

#define NUM_WORKERS (4)
#define NUM_THREADS (NUM_WORKERS*kittens::WARP_THREADS)

using G = kittens::group<NUM_WORKERS>;

template <int _N> struct rmsnorm_globals {
    using x_gl = gl<bf16, -1, -1, -1, -1>;
    using o_gl = gl<bf16, -1, -1, -1, -1>;
    using gamma_gl = gl<bf16, -1, -1, -1, -1>;

    x_gl x;
    o_gl o;
    gamma_gl gamma;
    float epsilon;

    const int n_per_tile = NUM_WORKERS;
    const int n_tile_size = N / n_per_tile;

    dim3 grid() { return dim3(n_tile_size, B, 1); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return 0; }
};

template<int _D>
__global__ void rmsnorm_hk(
    const rmsnorm_globals<_D> g
) {
    auto warpid = kittens::warpid();
    const int batch = blockIdx.y;
    const int seq_start = blockIdx.x * g.n_per_tile;
    const int seq_idx = seq_start + warpid;

    rv<bf16, _D> x_reg, gamma_reg, x_reg_squared;

    load(x_reg, g.x, {0, batch, seq_idx, 0});
    load(gamma_reg, g.gamma, {0, batch, seq_idx, 0});
    asm volatile("s_waitcnt vmcnt(0)");

    mul(x_reg_squared, x_reg, x_reg);

    bf16 x_var;
    sum(x_var, x_reg_squared);

    float var_f32 = __bfloat162float(x_var) / float(_D);
    float inv_rms_f32 = rsqrtf(var_f32 + g.epsilon);
    bf16 inv_rms = __float2bfloat16(inv_rms_f32);

    mul(x_reg, x_reg, inv_rms);
    mul(x_reg, x_reg, gamma_reg);

    store(g.o, x_reg, {0, batch, seq_idx, 0});
}

template<int _D>
void dispatch_rmsnorm(rmsnorm_globals<_D> g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)rmsnorm_hk<_D>, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    rmsnorm_hk<_D><<<g.grid(), g.block(), mem_size>>>(g);
}

int main(int argc, char** argv) {
    if (argc > 1) N = atoi(argv[1]);
    if (N % NUM_WORKERS != 0) {
        fprintf(stderr, "Error: N=%d must be divisible by NUM_WORKERS=%d\n", N, NUM_WORKERS);
        return 1;
    }

    // Allocate device memory: x, o, gamma
    // Shape: [1, B, N, D] = [1, 16, N, 128]
    size_t tensor_elems = (size_t)B * N * D;
    size_t tensor_bytes = tensor_elems * sizeof(bf16);
    size_t gamma_elems = (size_t)B * N * D;  // gamma is per-position in original code
    size_t gamma_bytes = gamma_elems * sizeof(bf16);

    bf16 *d_x, *d_o, *d_gamma;
    hipMalloc(&d_x, tensor_bytes);
    hipMalloc(&d_o, tensor_bytes);
    hipMalloc(&d_gamma, gamma_bytes);

    hipMemset(d_x, 0, tensor_bytes);
    hipMemset(d_o, 0, tensor_bytes);
    hipMemset(d_gamma, 0, gamma_bytes);

    // Create global layout descriptors
    using x_gl = gl<bf16, -1, -1, -1, -1>;
    x_gl x(d_x, 1, B, N, D);
    x_gl o(d_o, 1, B, N, D);
    x_gl gamma(d_gamma, 1, B, N, D);

    float epsilon = 1e-6f;
    rmsnorm_globals<D> g{x, o, gamma, epsilon};

    for (int i = 0; i < 200; i++) {
        dispatch_rmsnorm(g);
    }
    hipDeviceSynchronize();

    constexpr int NUM_ITERS = 1000;
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start);
    for (int i = 0; i < NUM_ITERS; i++) {
        dispatch_rmsnorm(g);
    }
    hipEventRecord(stop);
    hipDeviceSynchronize();

    float elapsed_ms = 0;
    hipEventElapsedTime(&elapsed_ms, start, stop);
    double avg_ms = elapsed_ms / NUM_ITERS;

    // FLOPs for RMSNorm: ~4*B*N*D (square, sum, mul by inv_rms, mul by gamma)
    double flops = 4.0 * B * N * D;
    double tflops = flops / (avg_ms * 1e9);
    printf("BF16 RMSNorm B=%d N=%d D=%d: %.4f ms avg (%d iters), %.2f TFLOPS\n",
           B, N, D, avg_ms, NUM_ITERS, tflops);

    hipEventDestroy(start);
    hipEventDestroy(stop);
    hipFree(d_x);
    hipFree(d_o);
    hipFree(d_gamma);
    return 0;
}
