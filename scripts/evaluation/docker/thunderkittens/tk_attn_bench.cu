// Standalone ThunderKittens MHA (LCF) benchmark for profiling with HPCToolkit
// Forward-only Flash Attention using Load-Compute-Finish pipeline on H100.
// Default: B=4 H=16 N=4096 D=128, accepts N as command-line argument.

#include "kittens.cuh"
#include "prototype.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;

template<int D, int NUM_WORKERS> struct attn_fwd_layout {
    using qo_tile   = st_bf<64, D>;
    using kv_tile   = st_bf<D==64?192:128, D>;
    using qo_global = kittens::gl<bf16, -1, -1, -1, D, qo_tile>;
    using kv_global = kittens::gl<bf16, -1, -1, -1, D, kv_tile>;
    struct globals { qo_global O, Q; kv_global K, V; };
    struct input_block    { kv_tile k, v; };
    struct scratch_block  { qo_tile q[NUM_WORKERS]; };
    struct common_state   { int batch, head, seq; };
    struct consumer_state {
        rt_fl<16, qo_tile::cols> o_reg;
        col_vec<rt_fl<16, kv_tile::rows>> max_vec, norm_vec;
        col_vec<rt_fl<16, kv_tile::rows>> max_vec_last_scaled, max_vec_scaled;
        rt_fl<16, kv_tile::rows> att_block;
        rt_bf<16, kv_tile::rows> att_block_mma;
    };
};

template<int D> struct attn_fwd_template {
    static constexpr int NUM_CONSUMER_WARPS = 12, NUM_WORKERS = NUM_CONSUMER_WARPS/4, INPUT_PIPE_STAGES = 2;
    using layout = attn_fwd_layout<D, NUM_WORKERS>;
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        int task_id = gridDim.x*args.task_iter + blockIdx.x;
        int seq_q = (args.globals.Q.rows() + NUM_WORKERS*layout::qo_tile::rows - 1)/(NUM_WORKERS*layout::qo_tile::rows);
        args.common.batch = task_id / (seq_q*args.globals.K.depth()); task_id -= args.common.batch * seq_q * args.globals.K.depth();
        args.common.head  = task_id / seq_q;                        task_id -= args.common.head  * seq_q;
        args.common.seq   = task_id;
        args.num_iters = args.common.batch < args.globals.Q.batch() ? (args.globals.K.rows() + layout::kv_tile::rows - 1)/(layout::kv_tile::rows) : -1;
    }
    struct producer {
        __device__ static inline void setup(producer_setup_args<layout> args) {
            warpgroup::producer_registers();
        }
        __device__ static inline void load(producer_load_args<layout> args) {
            if(warpgroup::warpid() == 0) {
                warp::tma::expect(args.inputs_arrived, args.input);
                warp::tma::load_async(args.input.k, args.globals.K, {args.common.batch, args.common.head, args.iter, 0}, args.inputs_arrived);
                warp::tma::load_async(args.input.v, args.globals.V, {args.common.batch, args.common.head, args.iter, 0}, args.inputs_arrived);
            }
            else if(laneid() == 0) arrive(args.inputs_arrived);
        }
    };
    struct consumer {
        __device__ static inline void setup(consumer_setup_args<layout> args) {
            warpgroup::consumer_registers<NUM_WORKERS>();
            if((args.common.seq*NUM_WORKERS + warpgroup::groupid())*layout::qo_tile::rows < args.globals.Q.rows())
                warpgroup::load(args.scratch.q[warpgroup::groupid()], args.globals.Q,
                                {args.common.batch, args.common.head, args.common.seq*NUM_WORKERS+warpgroup::groupid(), 0});
            args.state.o_reg = 0.f;
            args.state.norm_vec = 0.f;
            args.state.max_vec = base_types::constants<float>::neg_infty();
            warpgroup::sync(warpgroup::groupid());
        }
        __device__ static inline void compute(consumer_compute_args<layout> args) {
            constexpr float TEMPERATURE_SCALE = (D == 128) ? 0.08838834764f*1.44269504089f : 0.125f*1.44269504089f;
            warpgroup::mm<transpose::N, transpose::T>(args.state.att_block, args.scratch.q[warpgroup::groupid()], args.input.k);
            args.state.max_vec_last_scaled = args.state.max_vec * TEMPERATURE_SCALE;
            warpgroup::mma_async_wait();
            warp::right_fill(args.state.att_block, args.state.att_block, args.globals.K.rows() - args.iter*layout::kv_tile::rows, base_types::constants<float>::neg_infty());
            args.state.max_vec = warp::max<axis::COL>(args.state.att_block, args.state.max_vec);
            args.state.max_vec_scaled = args.state.max_vec * TEMPERATURE_SCALE;
            args.state.att_block = warp::exp2((args.state.att_block*TEMPERATURE_SCALE) - args.state.max_vec_scaled);
            args.state.max_vec_last_scaled = warp::exp2(args.state.max_vec_last_scaled - args.state.max_vec_scaled);
            args.state.norm_vec *= args.state.max_vec_last_scaled;
            args.state.norm_vec = warp::sum<axis::COL>(args.state.att_block, args.state.norm_vec);
            args.state.o_reg *= args.state.max_vec_last_scaled;
            args.state.att_block_mma = args.state.att_block;
            warpgroup::mma<transpose::N, transpose::N>(args.state.o_reg, args.state.att_block_mma, args.input.v);
            warpgroup::mma_async_wait();
            if(laneid() == 0) arrive(args.inputs_finished);
        }
        __device__ static inline void finish(consumer_finish_args<layout> args) {
            if((args.common.seq*NUM_WORKERS+warpgroup::groupid())*64 < args.globals.Q.rows()) {
                args.state.o_reg /= args.state.norm_vec;
                auto &o_smem = reinterpret_cast<typename layout::qo_tile&>(args.scratch.q[warpgroup::groupid()]);
                warpgroup::store(o_smem, args.state.o_reg);
                warpgroup::sync(warpgroup::groupid());
                if(warpgroup::warpid() == 0)
                    warp::tma::store_async(args.globals.O, o_smem, {args.common.batch, args.common.head, args.common.seq*NUM_WORKERS+warpgroup::groupid(), 0});
                warp::tma::store_async_read_wait();
            }
            __syncwarp();
            if(laneid() == 0) arrive(args.finish_finished);
        }
    };
};

#include <iostream>
#include <cstdlib>

constexpr int ATTN_D = 128;
using ker_template = attn_fwd_template<ATTN_D>;

// Simple random fill kernel
__global__ void fill_random(bf16 *data, size_t count, uint64_t seed) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        uint64_t x = seed + idx;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        x = x ^ (x >> 31);
        float u = (float)(x >> 40) * (1.0f / 16777216.0f);
        float val = (u * 2.0f - 1.0f) * 0.1f;  // small values for stable softmax
        data[idx] = __float2bfloat16(val);
    }
}

int main(int argc, char **argv) {
    int ATTN_B = 4, ATTN_H = 16, ATTN_N = 4096;
    if (argc > 1) ATTN_N = atoi(argv[1]);

    size_t total = (size_t)ATTN_B * ATTN_H * ATTN_N * ATTN_D;
    printf("ThunderKittens MHA (LCF) B=%d H=%d N=%d D=%d on H100\n", ATTN_B, ATTN_H, ATTN_N, ATTN_D);

    bf16 *d_q, *d_k, *d_v, *d_o;
    cudaMalloc(&d_q, total * sizeof(bf16));
    cudaMalloc(&d_k, total * sizeof(bf16));
    cudaMalloc(&d_v, total * sizeof(bf16));
    cudaMalloc(&d_o, total * sizeof(bf16));

    dim3 fblock(256);
    dim3 fgrid((total + 255) / 256);
    fill_random<<<fgrid, fblock>>>(d_q, total, 42);
    fill_random<<<fgrid, fblock>>>(d_k, total, 123);
    fill_random<<<fgrid, fblock>>>(d_v, total, 456);
    cudaMemset(d_o, 0, total * sizeof(bf16));
    cudaDeviceSynchronize();

    ker_template::layout::qo_global Qg(d_q, (size_t)ATTN_B, (size_t)ATTN_H, (size_t)ATTN_N, nullptr);
    ker_template::layout::kv_global Kg(d_k, (size_t)ATTN_B, (size_t)ATTN_H, (size_t)ATTN_N, nullptr);
    ker_template::layout::kv_global Vg(d_v, (size_t)ATTN_B, (size_t)ATTN_H, (size_t)ATTN_N, nullptr);
    ker_template::layout::qo_global Og(d_o, (size_t)ATTN_B, (size_t)ATTN_H, (size_t)ATTN_N, nullptr);
    ker_template::layout::globals globals = {Og, Qg, Kg, Vg};

    unsigned long mem_size = MAX_SHARED_MEMORY - 2000;
    cudaFuncSetAttribute(prototype::lcf::kernel<ker_template>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

    constexpr int BLOCK_SIZE = prototype::detail::NUM_THREADS_v<ker_template>;
    dim3 grid(132, 1, 1);

    // Warmup
    for (int i = 0; i < 50; i++)
        prototype::lcf::kernel<ker_template><<<grid, BLOCK_SIZE, mem_size>>>(globals);
    cudaDeviceSynchronize();

    // Timed iterations
    constexpr int NUM_ITERS = 200;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < NUM_ITERS; i++)
        prototype::lcf::kernel<ker_template><<<grid, BLOCK_SIZE, mem_size>>>(globals);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float elapsed_ms = 0;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    double avg_us = (elapsed_ms * 1000.0) / NUM_ITERS;

    // FLOPs: 2*B*H*N*N*D (QK^T) + 2*B*H*N*N*D (attn@V) + 4*B*H*N*N (softmax)
    double flops = 2.0*ATTN_B*ATTN_H*(double)ATTN_N*ATTN_N*ATTN_D * 2 + 4.0*ATTN_B*ATTN_H*(double)ATTN_N*ATTN_N;
    double tflops = flops / (avg_us * 1e6);

    printf("Grid: %d (persistent), Block: %d\n", grid.x, BLOCK_SIZE);
    printf("Average kernel time: %.2f us (%d iters, 50 warmup)\n", avg_us, NUM_ITERS);
    printf("Achieved performance: %.2f TFLOPS\n", tflops);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);
    return 0;
}
