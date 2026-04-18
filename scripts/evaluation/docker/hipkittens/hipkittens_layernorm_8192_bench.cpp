// Standalone HipKittens LayerNorm benchmark driver for profiling with HPCToolkit
// BF16 LayerNorm with residual addition — B=16, N=8192, D=2048 on MI300A (gfx942)
// Adapted from HipKittens analysis/layernorm/mi350x/kernel_8192.cpp
// Removes pybind11/PyTorch/rocrand dependency for standalone profiling.

#include "kittens.cuh"
#include <cstdio>
using namespace kittens;

constexpr int B = 16;
constexpr int H = 16;
constexpr int N = 8192;
constexpr int HEAD_D = 128;
constexpr int D = HEAD_D * H; // 2048

#define NUM_WORKERS (4)
#define NUM_THREADS (NUM_WORKERS*kittens::WARP_THREADS)

using G = kittens::group<NUM_WORKERS>;

template<int _d_model> struct norm_globals {
    static constexpr int d_model = _d_model;

    using x_gl            = gl<bf16, -1, -1, -1, -1>;
    using residual_gl     = gl<bf16, -1, -1, -1, -1>;
    using o_gl            = gl<bf16, -1, -1, -1, -1>;
    using o_resid_gl      = gl<bf16, -1, -1, -1, -1>;
    using norm_weight_gl  = gl<bf16, -1, -1, -1, -1>;
    using norm_bias_gl    = gl<bf16, -1, -1, -1, -1>;

    x_gl x;
    residual_gl residual;
    o_gl o;
    o_resid_gl o_resid;
    norm_weight_gl norm_weight;
    norm_bias_gl norm_bias;

    const int n_per_tile = 4;
    const int n_tile_size = N / n_per_tile;

    dim3 grid() { return dim3(n_tile_size, B, 1); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return 0; }
};

template<int _D> __launch_bounds__(NUM_THREADS, 2)
__global__ void layernorm_tk(const norm_globals<_D> g) {

    auto warpid = kittens::warpid();
    const int batch = blockIdx.y;
    const int seq_start = blockIdx.x*g.n_per_tile;

    constexpr int d_model = _D;
    rv<bf16, d_model> residual_s_reg, x_s_reg, norm_weight_s_reg, norm_bias_s_reg;
    load(x_s_reg, g.x, {0, batch, seq_start + warpid, 0});
    asm volatile("s_waitcnt vmcnt(0)");
    load(residual_s_reg, g.residual, {0, batch, seq_start + warpid, 0});
    bf16 mean = __float2bfloat16(0.0f);
    bf16 var  = __float2bfloat16(0.0f);

    // No dropout for profiling (removes rocrand dependency)

    asm volatile("s_waitcnt vmcnt(0)");
    load(norm_weight_s_reg, g.norm_weight, {0,0,0,0});
    load(norm_bias_s_reg, g.norm_bias, {0,0,0,0});
    add(residual_s_reg, residual_s_reg, x_s_reg);
    store(g.o_resid, residual_s_reg, {0, batch, seq_start + warpid, 0});

    // mean and variance
    sum(mean, residual_s_reg);
    constexpr float dim_scale = 1.0f / d_model;
    mean = mean * __float2bfloat16(dim_scale);
    sub(residual_s_reg, residual_s_reg, mean);
    mul(x_s_reg, residual_s_reg, residual_s_reg);
    sum(var, x_s_reg);
    var = var * __float2bfloat16(dim_scale);
    var = __float2bfloat16(sqrt(__bfloat162float(var + __float2bfloat16(1e-05f))));

    // compute norm
    div(residual_s_reg, residual_s_reg, var);
    asm volatile("s_waitcnt vmcnt(0)");
    mul(residual_s_reg, residual_s_reg, norm_weight_s_reg);
    add(residual_s_reg, residual_s_reg, norm_bias_s_reg);
    store(g.o, residual_s_reg, {0, batch, seq_start+warpid, 0});
}

template<int _D>
void dispatch_layernorm(norm_globals<_D> g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)layernorm_tk<_D>, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    layernorm_tk<_D><<<g.grid(), g.block(), mem_size>>>(g);
}

int main() {
    // Allocate device memory: x, residual, o, o_resid, norm_weight, norm_bias
    // Shape: [1, B, N, D] = [1, 16, 8192, 2048]
    size_t tensor_elems = (size_t)B * N * D;
    size_t tensor_bytes = tensor_elems * sizeof(bf16);
    size_t weight_elems = (size_t)D;
    size_t weight_bytes = weight_elems * sizeof(bf16);

    bf16 *d_x, *d_residual, *d_o, *d_o_resid, *d_norm_weight, *d_norm_bias;
    hipMalloc(&d_x, tensor_bytes);
    hipMalloc(&d_residual, tensor_bytes);
    hipMalloc(&d_o, tensor_bytes);
    hipMalloc(&d_o_resid, tensor_bytes);
    hipMalloc(&d_norm_weight, weight_bytes);
    hipMalloc(&d_norm_bias, weight_bytes);

    hipMemset(d_x, 0, tensor_bytes);
    hipMemset(d_residual, 0, tensor_bytes);
    hipMemset(d_o, 0, tensor_bytes);
    hipMemset(d_o_resid, 0, tensor_bytes);
    hipMemset(d_norm_weight, 0, weight_bytes);
    hipMemset(d_norm_bias, 0, weight_bytes);

    // Create global layout descriptors
    using x_gl = gl<bf16, -1, -1, -1, -1>;
    x_gl x(d_x, 1, B, N, D);
    x_gl residual(d_residual, 1, B, N, D);
    x_gl o(d_o, 1, B, N, D);
    x_gl o_resid(d_o_resid, 1, B, N, D);
    x_gl norm_weight(d_norm_weight, 1, 1, 1, D);
    x_gl norm_bias(d_norm_bias, 1, 1, 1, D);

    norm_globals<D> g{x, residual, o, o_resid, norm_weight, norm_bias};

    // Warmup
    for (int i = 0; i < 5; i++) {
        dispatch_layernorm(g);
    }
    hipDeviceSynchronize();

    // Benchmark
    constexpr int NUM_ITERS = 50;
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start);
    for (int i = 0; i < NUM_ITERS; i++) {
        dispatch_layernorm(g);
    }
    hipEventRecord(stop);
    hipDeviceSynchronize();

    float elapsed_ms = 0;
    hipEventElapsedTime(&elapsed_ms, start, stop);
    double avg_ms = elapsed_ms / NUM_ITERS;

    // FLOPs for LayerNorm: ~5*B*N*D (add, sub, mul, div, add operations on D elements)
    double flops = 5.0 * B * N * D;
    double tflops = flops / (avg_ms * 1e9);
    printf("BF16 LayerNorm B=%d N=%d D=%d: %.4f ms avg (%d iters), %.2f TFLOPS\n",
           B, N, D, avg_ms, NUM_ITERS, tflops);

    hipEventDestroy(start);
    hipEventDestroy(stop);
    hipFree(d_x);
    hipFree(d_residual);
    hipFree(d_o);
    hipFree(d_o_resid);
    hipFree(d_norm_weight);
    hipFree(d_norm_bias);
    return 0;
}
