// Optimized HipKittens RMSNorm benchmark driver for profiling with HPCToolkit
// BF16 RMSNorm -- B=16, N=4096, D=128 on MI300A (gfx942)
//
// Optimization: Stage global loads through shared memory (LDS) to replace
// scalar 16-bit global loads (global_load_ushort) with vectorized 128-bit
// loads (global_load_dwordx4).
//
// The original kernel uses rv<bf16, D> with naive layout, which causes
// each lane to issue a scalar global_load_ushort for every BF16 element.
// This produces 55.6% of all stall cycles due to memory latency.
//
// By staging through shared memory:
//   1. Global -> Shared: load(sv, gl, coord) uses float4 transfers
//      (global_load_dwordx4), which is 8x wider per instruction.
//   2. Shared -> Register: load(rv, sv) reads from LDS (ds_read),
//      which has ~30x lower latency than global memory.
//
// Uses NUM_WORKERS=4 (4 warps per block, matching the original kernel).
// Each block processes 4 sequence positions (1 per warp).
// Each warp has its own pair of shared vectors (x_shared, gamma_shared).
// Shared memory: 4 warps * 2 vectors * 256 bytes = 2048 bytes.
//
// The load(sv, gl, coord) function is warp-scoped (uses laneid(), not
// threadIdx.x), so each warp can independently load into its own shared
// vector without interfering with other warps.

#include "kittens.cuh"
#include <cstdio>
using namespace kittens;

constexpr int B = 16;
constexpr int H = 16;
constexpr int N = 4096;
constexpr int D = 128;

// 4 warps per block, matching the original kernel
#define NUM_WORKERS (4)
#define NUM_THREADS (NUM_WORKERS*kittens::WARP_THREADS)

template <int _N> struct rmsnorm_opt_globals {
    using x_gl = gl<bf16, -1, -1, -1, -1>;
    using o_gl = gl<bf16, -1, -1, -1, -1>;
    using gamma_gl = gl<bf16, -1, -1, -1, -1>;

    x_gl x;
    o_gl o;
    gamma_gl gamma;
    float epsilon;

    const int n_per_tile = NUM_WORKERS;  // 4 sequence positions per block
    const int n_tile_size = N / n_per_tile;

    dim3 grid() { return dim3(n_tile_size, B, 1); }
    dim3 block() { return dim3(NUM_THREADS); }

    // Shared memory for 8 shared vectors: 4 warps * 2 vectors (x, gamma)
    // sv<bf16, D> = D * sizeof(bf16) = 128 * 2 = 256 bytes each
    // Total: 8 * 256 = 2048 bytes + alignment padding
    size_t dynamic_shared_memory() {
        return NUM_WORKERS * 2 * D * sizeof(bf16) + 128;  // generous padding for alignment
    }
};

template<int _D>
__global__ void rmsnorm_opt_hk(
    const rmsnorm_opt_globals<_D> g
) {
    // Allocate shared vectors from dynamic shared memory
    // Each warp gets its own pair: x_shared[warpid], gamma_shared[warpid]
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    sv<bf16, _D> (&x_shared)[NUM_WORKERS]     = al.allocate<sv<bf16, _D>, NUM_WORKERS>();
    sv<bf16, _D> (&gamma_shared)[NUM_WORKERS]  = al.allocate<sv<bf16, _D>, NUM_WORKERS>();

    auto warpid = kittens::warpid();
    const int batch = blockIdx.y;
    const int seq_start = blockIdx.x * g.n_per_tile;
    const int seq_idx = seq_start + warpid;

    // ========================================================================
    // Phase 1: Vectorized global loads to shared memory (LDS)
    // load(sv, gl, coord) is warp-scoped (uses laneid()), so each warp
    // independently loads into its own shared vector without conflicts.
    // For sv<bf16, 128>: elem_per_transfer=8, 64 threads -> 1 iteration
    //   Lanes 0-15 each load 8 BF16 elements (16 bytes) = 16 global_load_dwordx4
    //   vs. original: 128 global_load_ushort per warp
    // ========================================================================
    load(x_shared[warpid], g.x, {0, batch, seq_idx, 0});
    load(gamma_shared[warpid], g.gamma, {0, batch, seq_idx, 0});

    // Wait for all global loads to complete and ensure shared memory visibility
    asm volatile("s_waitcnt vmcnt(0)");
    __builtin_amdgcn_s_barrier();

    // ========================================================================
    // Phase 2: Load from shared memory to register vectors
    // load(rv, sv) is warp-scoped (uses laneid()), each warp reads its own sv.
    // Reads from LDS: ds_read_u16 (fast, ~30x lower latency than global)
    // ========================================================================
    rv<bf16, _D> x_reg, gamma_reg, x_reg_squared;

    load(x_reg, x_shared[warpid]);
    load(gamma_reg, gamma_shared[warpid]);

    // Wait for LDS reads to complete
    asm volatile("s_waitcnt lgkmcnt(0)");

    // ========================================================================
    // Phase 3: Compute RMSNorm (identical to original kernel)
    // ========================================================================
    mul(x_reg_squared, x_reg, x_reg);

    bf16 x_var;
    sum(x_var, x_reg_squared);

    float var_f32 = __bfloat162float(x_var) / float(_D);
    float inv_rms_f32 = rsqrtf(var_f32 + g.epsilon);
    bf16 inv_rms = __float2bfloat16(inv_rms_f32);

    mul(x_reg, x_reg, inv_rms);
    mul(x_reg, x_reg, gamma_reg);

    // ========================================================================
    // Phase 4: Store result to global memory (same as original)
    // ========================================================================
    store(g.o, x_reg, {0, batch, seq_idx, 0});
}

template<int _D>
void dispatch_rmsnorm_opt(rmsnorm_opt_globals<_D> g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)rmsnorm_opt_hk<_D>, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    rmsnorm_opt_hk<_D><<<g.grid(), g.block(), mem_size>>>(g);
}

int main() {
    // Allocate device memory: x, o, gamma
    // Shape: [1, B, N, D] = [1, 16, 4096, 128]
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
    rmsnorm_opt_globals<D> g{x, o, gamma, epsilon};

    // Warmup
    for (int i = 0; i < 5; i++) {
        dispatch_rmsnorm_opt(g);
    }
    hipDeviceSynchronize();

    // Benchmark
    constexpr int NUM_ITERS = 50;
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start);
    for (int i = 0; i < NUM_ITERS; i++) {
        dispatch_rmsnorm_opt(g);
    }
    hipEventRecord(stop);
    hipDeviceSynchronize();

    float elapsed_ms = 0;
    hipEventElapsedTime(&elapsed_ms, start, stop);
    double avg_ms = elapsed_ms / NUM_ITERS;

    // FLOPs for RMSNorm: ~4*B*N*D (square, sum, mul by inv_rms, mul by gamma)
    double flops = 4.0 * B * N * D;
    double tflops = flops / (avg_ms * 1e9);
    printf("BF16 RMSNorm (optimized) B=%d N=%d D=%d: %.4f ms avg (%d iters), %.2f TFLOPS\n",
           B, N, D, avg_ms, NUM_ITERS, tflops);

    hipEventDestroy(start);
    hipEventDestroy(stop);
    hipFree(d_x);
    hipFree(d_o);
    hipFree(d_gamma);
    return 0;
}
