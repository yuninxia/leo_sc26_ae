// Standalone ThunderKittens Rotary (RoPE) benchmark for profiling with HPCToolkit
// Uses LCSF (Load-Compute-Store-Finish) pipeline on H100.
// Default: B=4 H=32 N=4096 D=128, accepts N as command-line argument.

#include "kittens.cuh"
#include "prototype.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcsf;

template<int _headdim, int _warps> struct rotary_layout {
    static constexpr int headdim = _headdim, warps = _warps;
    using seq_tile    = st_bf<16, headdim>;
    using seq_global  = gl<bf16, -1, -1, -1, headdim, seq_tile>;
    using rope_global = gl<bf16,  1,  1, -1, headdim/2>;
    struct globals {
        seq_global o, x;
        rope_global sin, cos;
        int batches;
    };
    struct input_block    { seq_tile x[warps]; };
    struct output_block   { seq_tile o[warps]; };
    struct producer_state { int active_warps;  };
    struct consumer_state { rt_fl<16, headdim/2> sin, cos; };
};

template<int _headdim> struct rotary_template {
    static constexpr int headdim=_headdim, NUM_CONSUMER_WARPS=8, NUM_BLOCKS=1, OUTPUT_PIPE_STAGES=3, INPUT_PIPE_STAGES=3;
    using layout = rotary_layout<headdim, NUM_CONSUMER_WARPS>;
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        if(args.task_iter == 0) {
            args.num_iters = min(args.globals.batches, (int)(args.globals.x.batch()-blockIdx.y*args.globals.batches)) * args.globals.x.depth();
        }
        else args.num_iters = -1;
    }
    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::producer_registers();
            args.state.active_warps = min((int)NUM_CONSUMER_WARPS,
                                          (int)(args.globals.x.rows()/16 - blockIdx.x*NUM_CONSUMER_WARPS));
        }
        __device__ static void load(producer_load_args<layout> args) {
            if(warpgroup::warpid() == args.iter%4) {
                kittens::coord idx = { blockIdx.y*args.globals.batches+args.iter/args.globals.x.depth(),
                                       args.iter%args.globals.x.depth(),
                                       blockIdx.x*NUM_CONSUMER_WARPS, 0 };
                warp::tma::expect_bytes(args.inputs_arrived, sizeof(layout::seq_tile)*args.state.active_warps);
                for(int i = 0; i < args.state.active_warps; i++)
                    warp::tma::load_async(args.input.x[i], args.globals.x, {idx.b,idx.d,idx.r+i,idx.c}, args.inputs_arrived);
                if(laneid() == 0) arrive(args.inputs_arrived, 3);
                __syncwarp();
            }
        }
        __device__ static void store(producer_store_args<layout> args) {
            if(warpgroup::warpid() == args.iter%4) {
                kittens::coord idx = { blockIdx.y*args.globals.batches+args.iter/args.globals.x.depth(),
                                       args.iter%args.globals.x.depth(),
                                       blockIdx.x*NUM_CONSUMER_WARPS, 0 };
                for(int i = 0; i < args.state.active_warps; i++)
                    warp::tma::store_async(args.globals.o, args.output.o[i], {idx.b,idx.d,idx.r+i,idx.c});
                warp::tma::store_async_read_wait();
                if(laneid() == 0) arrive(args.outputs_finished, 4);
                __syncwarp();
            }
        }
    };
    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::consumer_registers<NUM_CONSUMER_WARPS/4>();
            kittens::coord idx = { blockIdx.x*NUM_CONSUMER_WARPS + warpid(), 0 };
            warp::load(args.state.sin, args.globals.sin, idx);
            warp::load(args.state.cos, args.globals.cos, idx);
        }
        __device__ static void compute(consumer_compute_args<layout> args) {
            rt_fl<16, headdim> x;
            rt_fl<16, headdim/2> x1, x2, temp1, temp2;
            warp::load(x, args.input.x[warpid()]);
            if(laneid() == 0) arrive(args.inputs_finished);
            __syncwarp();
            for(int i = 0; i < headdim/32; i++) {
                #pragma unroll
                for(int j = 0; j < 4; j++) {
                    x1.tiles[0][i].data[j] = x.tiles[0][i].data[j];
                    x2.tiles[0][i].data[j] = x.tiles[0][i+headdim/32].data[j];
                }
            }
            warp::mul(temp1, x1, args.state.cos);
            warp::mul(temp2, x2, args.state.cos);
            warp::mul(x2, x2, -1.f);
            warp::mul(x1, x1, args.state.sin);
            warp::mul(x2, x2, args.state.sin);
            warp::add(temp1, temp1, x2);
            warp::add(temp2, temp2, x1);
            for(int i = 0; i < headdim/32; i++) {
                #pragma unroll
                for(int j = 0; j < 4; j++) {
                    x.tiles[0][i].data[j]            = temp1.tiles[0][i].data[j];
                    x.tiles[0][i+headdim/32].data[j] = temp2.tiles[0][i].data[j];
                }
            }
            warp::store(args.output.o[warpid()], x);
            __syncwarp();
            if(laneid() == 0) arrive(args.outputs_arrived);
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            if(laneid() == 0) arrive(args.finish_finished);
        }
    };
};

#include <iostream>
#include <cstdlib>

constexpr int ATTN_D = 128;
using rope_t = rotary_template<ATTN_D>;

__global__ void fill_random(bf16 *data, size_t count, uint64_t seed) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        uint64_t x = seed + idx;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        x = x ^ (x >> 31);
        float u = (float)(x >> 40) * (1.0f / 16777216.0f);
        data[idx] = __float2bfloat16(u * 2.0f - 1.0f);
    }
}

int main(int argc, char **argv) {
    int ATTN_B = 4, ATTN_H = 32, ATTN_N = 4096;
    if (argc > 1) ATTN_N = atoi(argv[1]);

    constexpr int BATCHES_PER_BLOCK = 4;

    printf("ThunderKittens Rotary (RoPE) B=%d H=%d N=%d D=%d on H100\n", ATTN_B, ATTN_H, ATTN_N, ATTN_D);

    size_t total_seq = (size_t)ATTN_B * ATTN_H * ATTN_N * ATTN_D;
    size_t total_rope = (size_t)ATTN_N * (ATTN_D / 2);

    bf16 *d_x, *d_o, *d_sin, *d_cos;
    cudaMalloc(&d_x, total_seq * sizeof(bf16));
    cudaMalloc(&d_o, total_seq * sizeof(bf16));
    cudaMalloc(&d_sin, total_rope * sizeof(bf16));
    cudaMalloc(&d_cos, total_rope * sizeof(bf16));

    dim3 fblock(256);
    fill_random<<<dim3((total_seq + 255) / 256), fblock>>>(d_x, total_seq, 42);
    fill_random<<<dim3((total_rope + 255) / 256), fblock>>>(d_sin, total_rope, 123);
    fill_random<<<dim3((total_rope + 255) / 256), fblock>>>(d_cos, total_rope, 456);
    cudaMemset(d_o, 0, total_seq * sizeof(bf16));
    cudaDeviceSynchronize();

    using seq_globals  = typename rope_t::layout::seq_global;
    using rope_globals = typename rope_t::layout::rope_global;
    using globals = typename rope_t::layout::globals;

    seq_globals  Og{d_o, (size_t)ATTN_B, (size_t)ATTN_H, (size_t)ATTN_N, nullptr};
    seq_globals  Xg{d_x, (size_t)ATTN_B, (size_t)ATTN_H, (size_t)ATTN_N, nullptr};
    rope_globals SINg{d_sin, nullptr, nullptr, (size_t)ATTN_N, nullptr};
    rope_globals COSg{d_cos, nullptr, nullptr, (size_t)ATTN_N, nullptr};
    globals g{Og, Xg, SINg, COSg, BATCHES_PER_BLOCK};

    unsigned long mem_size = MAX_SHARED_MEMORY - 2048;
    cudaFuncSetAttribute(prototype::lcsf::kernel<rope_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

    constexpr int ROWS_PER_BLOCK = rope_t::NUM_CONSUMER_WARPS * rope_t::layout::seq_tile::rows;
    dim3 grid((ATTN_N + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, (ATTN_B + BATCHES_PER_BLOCK - 1) / BATCHES_PER_BLOCK);
    dim3 block(kittens::prototype::detail::NUM_THREADS_v<rope_t>);

    // Warmup
    for (int i = 0; i < 200; i++)
        prototype::lcsf::kernel<rope_t><<<grid, block, mem_size>>>(g);
    cudaDeviceSynchronize();

    // Timed iterations
    constexpr int NUM_ITERS = 1000;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < NUM_ITERS; i++)
        prototype::lcsf::kernel<rope_t><<<grid, block, mem_size>>>(g);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float elapsed_ms = 0;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    double avg_us = (elapsed_ms * 1000.0) / NUM_ITERS;
    double bytes = (double)(total_seq + total_seq + 2*total_rope) * sizeof(bf16);
    double bw_gbps = bytes / (avg_us * 1e3);

    printf("Grid: (%d, %d), Block: %d\n", grid.x, grid.y, block.x);
    printf("Average kernel time: %.2f us (%d iters, 200 warmup)\n", avg_us, NUM_ITERS);
    printf("Effective bandwidth: %.2f GB/s\n", bw_gbps);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_x);
    cudaFree(d_o);
    cudaFree(d_sin);
    cudaFree(d_cos);
    return 0;
}
