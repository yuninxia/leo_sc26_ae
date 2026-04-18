// Standalone ThunderKittens FP8 GEMM benchmark for profiling with HPCToolkit
// Runs only the optimized TK kernel (no reference GEMM) for clean PC sampling profiles.
// Default: 4096x4096, accepts N as command-line argument.
// Operation: C = A * B^T (FP8 E4M3, accumulate in FP32, output FP8)

#include "kittens.cuh"
#include "prototype.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;

struct matmul_layout {
    using  a_tile    = st_fp8e4m3<64, 128>;
    using  b_tile    = st_fp8e4m3<256, 128>;
    using  c_tile    = st_fp8e4m3<64, 256>;
    using  a_layout  = gl<fp8e4m3, 1, 1, -1, -1, a_tile>;
    using  b_layout  = gl<fp8e4m3, 1, 1, -1, -1, b_tile>;
    using  c_layout  = gl<fp8e4m3, 1, 1, -1, -1, c_tile>;
    struct globals        { a_layout A; b_layout B; c_layout C; };
    struct input_block    { a_tile a[2]; b_tile b; };
    struct finish_block   { c_tile c[2]; };
    struct common_state   { int2 coord; };
    struct consumer_state {
        rt_fl<16, c_tile::cols> accum;
        rt_fp8e4m3<16, c_tile::cols> accum_fp8;
    };
};

template<int _SUPER_M=12>
struct matmul_template {
    static constexpr int SUPER_M = _SUPER_M;
    using layout = matmul_layout;
    static constexpr int NUM_CONSUMER_WARPS=8, INPUT_PIPE_STAGES=4, PRODUCER_BARRIER_ARRIVALS=1;
    template<bool PERISISTENT_GRID=true> __host__ static inline dim3 grid(int M, int N, int K) {
        return dim3(PERISISTENT_GRID ? 132 : M*N/(2*layout::c_tile::num_elements));
    }
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        int Rblocks = args.globals.C.rows() / (2*layout::c_tile::rows), Cblocks = args.globals.C.cols() / layout::c_tile::cols;
        int super_rows = (Rblocks/SUPER_M)*SUPER_M,
            final_rows = Rblocks - super_rows,
            super_repeat = SUPER_M*Cblocks;
        int task_id = args.task_iter*gridDim.x + blockIdx.x;
        if (task_id < super_rows * Cblocks)
            args.common.coord = { SUPER_M*(task_id/super_repeat) + task_id%SUPER_M, (task_id%super_repeat)/SUPER_M };
        else if (task_id < Rblocks*Cblocks) {
            int remainder_id = task_id - super_rows*Cblocks;
            args.common.coord = { super_rows + (remainder_id%final_rows), remainder_id/final_rows };
        }
        else {
            args.num_iters = -1;
            return;
        }
        args.num_iters = args.globals.A.cols()/layout::a_tile::cols;
        int id = warpgroup::groupid() == NUM_CONSUMER_WARPS/4 ? 0 : warpgroup::groupid();
        args.common.coord = { args.common.coord.x*2 + id, args.common.coord.y };
    }
    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::decrease_registers<40>();
        }
        __device__ static void load(producer_load_args<layout> args) {
            if(warpgroup::laneid() == 0) {
                tma::expect(args.inputs_arrived, args.input);
                for(int i = 0; i < 2; i++)
                    tma::load_async(args.input.a[i], args.globals.A,
                                    {args.common.coord.x+i, args.iter}, args.inputs_arrived);
                tma::load_async(args.input.b, args.globals.B,
                                {args.common.coord.y, args.iter}, args.inputs_arrived);
            }
        }
    };
    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::increase_registers<232>();
            warp::zero(args.state.accum);
        }
        __device__ static void compute(consumer_compute_args<layout> args) {
            warpgroup::mma_ABt(
                args.state.accum,
                args.input.a[warpgroup::groupid()],
                args.input.b
            );
            warpgroup::mma_async_wait();
            if(warp::laneid() == 0) arrive(args.inputs_finished);
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            kittens::warp::copy(args.state.accum_fp8, args.state.accum);
            warpgroup::store(args.finish.c[warpgroup::groupid()], args.state.accum_fp8);
            warpgroup::sync(warpgroup::groupid()+4);
            if(warpgroup::laneid() == 0) {
                tma::store_async(args.globals.C, args.finish.c[warpgroup::groupid()],
                                 {args.common.coord.x, args.common.coord.y});
                tma::store_async_read_wait();
            }
            warp::zero(args.state.accum);
            if(warp::laneid() == 0) arrive(args.finish_finished);
        }
    };
};

#include <iostream>
#include <cstdlib>
#include <random>
#include <cuda_fp8.h>

using mmt = matmul_template<8>;

template<typename mmt_t>
void launch_gemm(fp8e4m3 *d_A, fp8e4m3 *d_B, fp8e4m3 *d_C, size_t M, size_t N, size_t K) {
    using a_layout = typename mmt_t::layout::a_layout;
    using b_layout = typename mmt_t::layout::b_layout;
    using c_layout = typename mmt_t::layout::c_layout;
    using globals  = typename mmt_t::layout::globals;
    a_layout Ag{d_A, nullptr, nullptr, M, K};
    b_layout Bg{d_B, nullptr, nullptr, N, K};  // B is (N, K) for ABt
    c_layout Cg{d_C, nullptr, nullptr, M, N};
    globals G{Ag, Bg, Cg};
    prototype::lcf::kernel<mmt_t><<<mmt_t::grid(M, N, K), kittens::prototype::detail::NUM_THREADS_v<mmt_t>, MAX_SHARED_MEMORY-1024>>>(G);
}

int main(int argc, char **argv) {
    int N = 4096;
    if (argc > 1) N = atoi(argv[1]);

    size_t M = N, K = N;
    printf("ThunderKittens FP8 GEMM %dx%dx%d on H100\n", (int)M, (int)N, (int)K);

    // Host-side init with small random FP8 values
    std::mt19937 gen(42);
    std::normal_distribution<float> dis(0.0f, 0.1f);

    __nv_fp8_e4m3 *h_A = new __nv_fp8_e4m3[M * K];
    __nv_fp8_e4m3 *h_B = new __nv_fp8_e4m3[N * K];
    for (size_t i = 0; i < M * K; i++) h_A[i] = __nv_fp8_e4m3(dis(gen));
    for (size_t i = 0; i < N * K; i++) h_B[i] = __nv_fp8_e4m3(dis(gen));

    fp8e4m3 *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(fp8e4m3));
    cudaMalloc(&d_B, N * K * sizeof(fp8e4m3));
    cudaMalloc(&d_C, M * N * sizeof(fp8e4m3));

    cudaMemcpy(d_A, h_A, M * K * sizeof(fp8e4m3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * K * sizeof(fp8e4m3), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, M * N * sizeof(fp8e4m3));
    cudaDeviceSynchronize();

    delete[] h_A;
    delete[] h_B;

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

    printf("Grid: %d (persistent)\n", mmt::grid(M, N, K).x);
    printf("Average kernel time: %.2f us (%d iters, 200 warmup)\n", avg_us, NUM_ITERS);
    printf("Achieved performance: %.2f TFLOPS\n", tflops);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
