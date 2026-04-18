// Standalone ThunderKittens BF16 GEMM benchmark for profiling with HPCToolkit
// Runs only the optimized TK kernel (no reference GEMM) for clean PC sampling profiles.
// Default: 4096x4096, accepts N as command-line argument.

#include "kittens.cuh"
#include "prototype.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;

template<int M_BLOCK, int N_BLOCK>
struct matmul_layout {
    using  base_tile      = st_bf<64, 64>;
    using  global_layout  = gl<bf16, 1, 1, -1, -1, base_tile>;
    struct globals        { global_layout A, B, C; };
    struct input_block    { base_tile a[M_BLOCK], b[N_BLOCK]; };
    struct finish_block   { base_tile c[M_BLOCK][N_BLOCK]; };
    struct common_state   { int2 coord; };
    struct consumer_state { rt_fl<16, N_BLOCK*base_tile::cols> accum; };
};

template<int _M_BLOCK=2, int _N_BLOCK=4, int _SUPER_M=12>
struct matmul_template {
    static constexpr int M_BLOCK = _M_BLOCK, N_BLOCK = _N_BLOCK, SUPER_M = _SUPER_M;
    using layout    = matmul_layout<M_BLOCK, N_BLOCK>;
    using wide_tile = st_bf<64, 64*N_BLOCK>;
    static constexpr int NUM_CONSUMER_WARPS=M_BLOCK*4, INPUT_PIPE_STAGES=4, PRODUCER_BARRIER_ARRIVALS=1;
    template<bool PERISISTENT_GRID=true> __host__ static inline dim3 grid(int M, int N, int K) {
        return dim3(PERISISTENT_GRID ? 132 : M*N/(M_BLOCK*N_BLOCK*layout::base_tile::num_elements));
    }
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        int Rblocks = args.globals.C.rows() / (M_BLOCK*64), Cblocks = args.globals.C.cols() / (N_BLOCK*64);
        int super_rows = (Rblocks/SUPER_M)*SUPER_M,
            final_rows = Rblocks - super_rows,
            super_repeat = SUPER_M*Cblocks;
        int task_id = args.task_iter*gridDim.x + blockIdx.x;
        if (task_id < super_rows * Cblocks)
            args.common.coord = { SUPER_M*(task_id/super_repeat) + task_id%SUPER_M,
                           (task_id%super_repeat)/SUPER_M };
        else if (task_id < Rblocks*Cblocks) {
            int remainder_id = task_id - super_rows*Cblocks;
            args.common.coord = { super_rows + (remainder_id%final_rows), remainder_id/final_rows };
        }
        else {
            args.num_iters = -1;
            return;
        }
        args.num_iters = args.globals.A.cols()/64;
        int id = warpgroup::groupid() == NUM_CONSUMER_WARPS/4 ? 0 : warpgroup::groupid();
        args.common.coord = { args.common.coord.x*M_BLOCK + id, args.common.coord.y*N_BLOCK };
    }
    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::decrease_registers<40>();
        }
        __device__ static void load(producer_load_args<layout> args) {
            if (warpgroup::laneid() == 0) {
                tma::expect(args.inputs_arrived, args.input);
                for(int i = 0; i < M_BLOCK; i++)
                    tma::load_async(args.input.a[i], args.globals.A,
                                    {args.common.coord.x+i, args.iter}, args.inputs_arrived);
                for(int i = 0; i < N_BLOCK; i++)
                    tma::load_async(args.input.b[i], args.globals.B,
                                    {args.iter, args.common.coord.y+i}, args.inputs_arrived);
            }
        }
    };
    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::increase_registers<232>();
            kittens::warp::zero(args.state.accum);
        }
        __device__ static void compute(consumer_compute_args<layout> args) {
            warpgroup::mma_AB(
                args.state.accum,
                args.input.a[warpgroup::groupid()],
                reinterpret_cast<wide_tile&>(args.input.b)
            );
            warpgroup::mma_async_wait();
            if (warp::laneid() == 0) arrive(args.inputs_finished);
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            warpgroup::store(reinterpret_cast<wide_tile&>(args.finish.c[warpgroup::groupid()]), args.state.accum);
            warpgroup::sync(warpgroup::groupid()+4);
            if (warpgroup::laneid() == 0) for(int i = 0; i < N_BLOCK; i++) {
                tma::store_async(args.globals.C, args.finish.c[warpgroup::groupid()][i],
                                             {args.common.coord.x, args.common.coord.y+i});
                tma::store_async_read_wait();
            }
            kittens::warp::zero(args.state.accum);
            if (warp::laneid() == 0) arrive(args.finish_finished);
        }
    };
};

#include <iostream>
#include <cstdlib>
#include <cuda_bf16.h>

using mmt = matmul_template<2,4,8>;

template<typename mmt_t>
void launch_gemm(bf16 *d_A, bf16 *d_B, bf16 *d_C, size_t M, size_t N, size_t K) {
    using global_layout = typename mmt_t::layout::global_layout;
    using globals = typename mmt_t::layout::globals;
    global_layout Ag{d_A, nullptr, nullptr, M, K};
    global_layout Bg{d_B, nullptr, nullptr, K, N};
    global_layout Cg{d_C, nullptr, nullptr, M, N};
    globals G{Ag, Bg, Cg};
    prototype::lcf::kernel<mmt_t><<<mmt_t::grid(M, N, K), kittens::prototype::detail::NUM_THREADS_v<mmt_t>, MAX_SHARED_MEMORY-1024>>>(G);
}

// Simple random fill kernel
__global__ void fill_random(bf16 *data, size_t count, uint64_t seed) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        uint64_t x = seed + idx;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        x = x ^ (x >> 31);
        float u = (float)(x >> 40) * (1.0f / 16777216.0f);
        float val = u * 2.0f - 1.0f;
        data[idx] = __float2bfloat16(val);
    }
}

int main(int argc, char **argv) {
    int N = 4096;
    if (argc > 1) N = atoi(argv[1]);

    size_t M = N, K = N;
    printf("ThunderKittens BF16 GEMM %dx%dx%d on H100\n", (int)M, (int)N, (int)K);

    bf16 *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(bf16));
    cudaMalloc(&d_B, K * N * sizeof(bf16));
    cudaMalloc(&d_C, M * N * sizeof(bf16));

    dim3 fblock(256);
    dim3 fgrid_A((M*K + 255) / 256);
    dim3 fgrid_B((K*N + 255) / 256);
    fill_random<<<fgrid_A, fblock>>>(d_A, M*K, 42);
    fill_random<<<fgrid_B, fblock>>>(d_B, K*N, 123);
    cudaMemset(d_C, 0, M * N * sizeof(bf16));
    cudaDeviceSynchronize();

    unsigned long mem_size = MAX_SHARED_MEMORY - 1024;
    cudaFuncSetAttribute(prototype::lcf::kernel<mmt>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

    // Warmup
    for (int i = 0; i < 200; i++) {
        launch_gemm<mmt>(d_A, d_B, d_C, M, N, K);
    }
    cudaDeviceSynchronize();

    // Timed iterations
    constexpr int NUM_ITERS = 1000;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < NUM_ITERS; i++) {
        launch_gemm<mmt>(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float elapsed_ms = 0;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    double avg_us = (elapsed_ms * 1000.0) / NUM_ITERS;
    double flops = 2.0 * M * N * K;
    double tflops = flops / (avg_us * 1e6);

    printf("Block size: %dx%d, Grid: %d (persistent)\n",
           mmt::M_BLOCK*64, mmt::N_BLOCK*64, mmt::grid(M, N, K).x);
    printf("Average kernel time: %.2f us (%d iters, 200 warmup)\n", avg_us, NUM_ITERS);
    printf("Achieved performance: %.2f TFLOPS\n", tflops);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
