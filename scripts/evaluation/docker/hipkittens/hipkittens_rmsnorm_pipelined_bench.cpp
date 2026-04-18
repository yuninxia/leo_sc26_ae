// HipKittens RMSNorm with multi-row software pipelining
//
// Each warp processes ROWS_PER_WARP rows with double-buffered register
// vectors and staggered prefetch, hiding global_load_ushort latency.
//
// Two optimizations combined:
//   1. Split waitcnt: wait for x first, compute on x while gamma loads.
//   2. Multi-row pipeline: prefetch next row while computing current row.
//
// vmcnt values are computed from _D at compile time (not hardcoded):
//   LOADS_PER_RV  = outer_dim of rv<bf16, _D> with naive layout
//   LOADS_PER_ROW = 2 * LOADS_PER_RV  (x + gamma)
//   PIPELINE      = 2 * LOADS_PER_ROW (2 rows in flight)
//
// Pipeline flow (ROWS_PER_WARP=8):
//   Prologue:  load rows 0+1
//   Steady:    process row r, prefetch row r+2 (repeat)
//   Epilogue:  drain last 2 rows

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
#define ROWS_PER_WARP (8)

template <int _N> struct rmsnorm_pipe_globals {
    using x_gl = gl<bf16, -1, -1, -1, -1>;
    using o_gl = gl<bf16, -1, -1, -1, -1>;
    using gamma_gl = gl<bf16, -1, -1, -1, -1>;

    x_gl x;
    o_gl o;
    gamma_gl gamma;
    float epsilon;

    const int n_per_tile = NUM_WORKERS * ROWS_PER_WARP;
    const int n_tile_size = N / n_per_tile;

    dim3 grid() { return dim3(n_tile_size, B, 1); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return 0; }
};

// Compute x-only portion of RMSNorm (before gamma is ready)
// After this, x_reg holds x * inv_rms
#define COMPUTE_X_ONLY(x_reg, x_reg_squared, _D, epsilon) \
    do { \
        mul(x_reg_squared, x_reg, x_reg); \
        bf16 _x_var; \
        sum(_x_var, x_reg_squared); \
        float _var_f32 = __bfloat162float(_x_var) / float(_D); \
        float _inv_rms_f32 = rsqrtf(_var_f32 + epsilon); \
        bf16 _inv_rms = __float2bfloat16(_inv_rms_f32); \
        mul(x_reg, x_reg, _inv_rms); \
    } while(0)

template<int _D>
__global__ void rmsnorm_pipelined_hk(
    const rmsnorm_pipe_globals<_D> g
) {
    static constexpr int LOADS_PER_RV  = rv<bf16, _D>::outer_dim;
    static constexpr int LOADS_PER_ROW = 2 * LOADS_PER_RV;
    static constexpr int PIPELINE      = 2 * LOADS_PER_ROW;
    static constexpr int WAIT_X        = PIPELINE - LOADS_PER_RV;
    static constexpr int WAIT_GAMMA    = PIPELINE - LOADS_PER_ROW;
    static constexpr int WAIT_X_LAST   = LOADS_PER_ROW - LOADS_PER_RV;
    static_assert(PIPELINE <= 63, "pipeline depth exceeds vmcnt range");

    auto warpid = kittens::warpid();
    const int batch = blockIdx.y;
    const int seq_start = blockIdx.x * g.n_per_tile + warpid * ROWS_PER_WARP;

    rv<bf16, _D> x_reg_a, x_reg_b, gamma_reg_a, gamma_reg_b, x_reg_squared;

    // Prologue: load rows 0 and 1 (PIPELINE loads outstanding)
    load(x_reg_a, g.x, {0, batch, seq_start, 0});
    load(gamma_reg_a, g.gamma, {0, batch, seq_start, 0});
    load(x_reg_b, g.x, {0, batch, seq_start + 1, 0});
    load(gamma_reg_b, g.gamma, {0, batch, seq_start + 1, 0});

    // Steady state: process 2 rows per iteration, prefetch next 2
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP - 2; r += 2) {
        asm volatile("s_waitcnt vmcnt(%0)" :: "n"(WAIT_X));
        COMPUTE_X_ONLY(x_reg_a, x_reg_squared, _D, g.epsilon);
        asm volatile("s_waitcnt vmcnt(%0)" :: "n"(WAIT_GAMMA));
        mul(x_reg_a, x_reg_a, gamma_reg_a);
        store(g.o, x_reg_a, {0, batch, seq_start + r, 0});
        load(x_reg_a, g.x, {0, batch, seq_start + r + 2, 0});
        load(gamma_reg_a, g.gamma, {0, batch, seq_start + r + 2, 0});

        asm volatile("s_waitcnt vmcnt(%0)" :: "n"(WAIT_X));
        COMPUTE_X_ONLY(x_reg_b, x_reg_squared, _D, g.epsilon);
        asm volatile("s_waitcnt vmcnt(%0)" :: "n"(WAIT_GAMMA));
        mul(x_reg_b, x_reg_b, gamma_reg_b);
        store(g.o, x_reg_b, {0, batch, seq_start + r + 1, 0});
        load(x_reg_b, g.x, {0, batch, seq_start + r + 3, 0});
        load(gamma_reg_b, g.gamma, {0, batch, seq_start + r + 3, 0});
    }

    // Epilogue: drain last 2 rows (no prefetch)
    asm volatile("s_waitcnt vmcnt(%0)" :: "n"(WAIT_X));
    COMPUTE_X_ONLY(x_reg_a, x_reg_squared, _D, g.epsilon);
    asm volatile("s_waitcnt vmcnt(%0)" :: "n"(WAIT_GAMMA));
    mul(x_reg_a, x_reg_a, gamma_reg_a);
    store(g.o, x_reg_a, {0, batch, seq_start + ROWS_PER_WARP - 2, 0});

    asm volatile("s_waitcnt vmcnt(%0)" :: "n"(WAIT_X_LAST));
    COMPUTE_X_ONLY(x_reg_b, x_reg_squared, _D, g.epsilon);
    asm volatile("s_waitcnt vmcnt(0)");
    mul(x_reg_b, x_reg_b, gamma_reg_b);
    store(g.o, x_reg_b, {0, batch, seq_start + ROWS_PER_WARP - 1, 0});
}

template<int _D>
void dispatch_rmsnorm_pipe(rmsnorm_pipe_globals<_D> g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)rmsnorm_pipelined_hk<_D>, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    rmsnorm_pipelined_hk<_D><<<g.grid(), g.block(), mem_size>>>(g);
}

int main(int argc, char** argv) {
    if (argc > 1) N = atoi(argv[1]);
    int n_per_tile = NUM_WORKERS * ROWS_PER_WARP;
    if (N % n_per_tile != 0) {
        fprintf(stderr, "Error: N=%d must be divisible by NUM_WORKERS*ROWS_PER_WARP=%d\n", N, n_per_tile);
        return 1;
    }

    size_t tensor_elems = (size_t)B * N * D;
    size_t tensor_bytes = tensor_elems * sizeof(bf16);
    size_t gamma_elems = (size_t)B * N * D;
    size_t gamma_bytes = gamma_elems * sizeof(bf16);

    bf16 *d_x, *d_o, *d_gamma;
    hipMalloc(&d_x, tensor_bytes);
    hipMalloc(&d_o, tensor_bytes);
    hipMalloc(&d_gamma, gamma_bytes);

    hipMemset(d_x, 0, tensor_bytes);
    hipMemset(d_o, 0, tensor_bytes);
    hipMemset(d_gamma, 0, gamma_bytes);

    using x_gl = gl<bf16, -1, -1, -1, -1>;
    x_gl x(d_x, 1, B, N, D);
    x_gl o(d_o, 1, B, N, D);
    x_gl gamma(d_gamma, 1, B, N, D);

    float epsilon = 1e-6f;
    rmsnorm_pipe_globals<D> g{x, o, gamma, epsilon};

    for (int i = 0; i < 200; i++) {
        dispatch_rmsnorm_pipe(g);
    }
    hipDeviceSynchronize();

    constexpr int NUM_ITERS = 1000;
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start);
    for (int i = 0; i < NUM_ITERS; i++) {
        dispatch_rmsnorm_pipe(g);
    }
    hipEventRecord(stop);
    hipDeviceSynchronize();

    float elapsed_ms = 0;
    hipEventElapsedTime(&elapsed_ms, start, stop);
    double avg_ms = elapsed_ms / NUM_ITERS;

    double flops = 4.0 * B * N * D;
    double tflops = flops / (avg_ms * 1e9);
    printf("BF16 RMSNorm (pipelined, %d rows/warp) B=%d N=%d D=%d: %.4f ms avg (%d iters), %.2f TFLOPS\n",
           ROWS_PER_WARP, B, N, D, avg_ms, NUM_ITERS, tflops);

    hipEventDestroy(start);
    hipEventDestroy(stop);
    hipFree(d_x);
    hipFree(d_o);
    hipFree(d_gamma);
    return 0;
}
