// Standalone ThunderKittens Mamba2 SSM benchmark for profiling with HPCToolkit
// Uses LCSF (Load-Compute-Store-Finish) pipeline on H100.
// Default: B=4 H=16 N=4096 D=64, accepts N as command-line argument.

#include "kittens.cuh"
#include "prototype.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcsf;

struct mamba2_fwd_layout {
    using q_tile   = st_bf<64, 64>;
    using k_tile   = st_bf<64, 64>;
    using v_tile   = st_bf<64, 64>;
    using o_tile   = st_bf<64, 64>;
    using a_vec    = sv_fl<64>;
    using q_global = kittens::gl<bf16, -1, -1, -1, 64, q_tile>;
    using k_global = kittens::gl<bf16, -1, -1, -1, 64, k_tile>;
    using v_global = kittens::gl<bf16, -1, -1, -1, 64, v_tile>;
    using o_global = kittens::gl<bf16, -1, -1, -1, 64, o_tile>;
    using a_global = kittens::gl<float, -1, -1, 1, -1, a_vec>;
    struct globals { q_global Q; k_global K; v_global V; o_global O; a_global A; };
    struct input_block    {
        q_tile q;
        k_tile k;
        v_tile v[2];
        a_vec  a[2];
        a_vec  padding[6];
    };
    struct output_block {
        o_tile o[2];
    };
    struct scratch_block  {
        st_bf<64, 64> kv[2], k[2];
        a_vec         a_cumsum[2];
        a_vec         padding[6];
    };
    struct common_state {
        int batch, head;
    };
    struct consumer_state {
        rt_fl<16, 64> o_reg;
        rt_fl<16, 64> att_block;
        rt_bf<16, 64> att_block_mma;
        rt_fl<16, 64> local_decay;
        rt_bf<16, 64> q_reg, k_reg;
        rt_fl<16, 64> kv;
    };
};

struct mamba2_fwd_template {
    static constexpr int NUM_CONSUMER_WARPS=8, OUTPUT_PIPE_STAGES=2, INPUT_PIPE_STAGES=2, PRODUCER_BARRIER_ARRIVALS=1, CONSUMER_BARRIER_ARRIVALS=NUM_CONSUMER_WARPS/4;
    using layout = mamba2_fwd_layout;
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        int task_id = args.task_iter * gridDim.x + blockIdx.x;
        args.common.batch = task_id / (args.globals.V.depth()/(NUM_CONSUMER_WARPS/4));
        task_id -= args.common.batch*(args.globals.V.depth()/(NUM_CONSUMER_WARPS/4));
        args.common.head = task_id*2;
        args.num_iters = args.common.batch < args.globals.Q.batch() ? args.globals.K.rows()/layout::k_tile::rows : -1;
    }
    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::producer_registers();
        }
        __device__ static void load(producer_load_args<layout> args) {
            if(warpgroup::warpid() == args.iter%4) {
                warp::tma::expect(args.inputs_arrived, args.input.q, args.input.k, args.input.v[0], args.input.a[0], args.input.v[1], args.input.a[1]);
                warp::tma::load_async(args.input.q, args.globals.Q, {args.common.batch, 0, args.iter, 0}, args.inputs_arrived);
                warp::tma::load_async(args.input.k, args.globals.K, {args.common.batch, 0, args.iter, 0}, args.inputs_arrived);
                #pragma unroll
                for(int i = 0; i < NUM_CONSUMER_WARPS/4; i++) {
                    warp::tma::load_async(args.input.v[i], args.globals.V, {args.common.batch,  args.common.head+i, args.iter, 0}, args.inputs_arrived);
                    warp::tma::load_async(args.input.a[i], args.globals.A, {args.common.batch,  args.common.head+i, 0, args.iter}, args.inputs_arrived);
                }
                __syncwarp();
            }
        }
        __device__ static void store(producer_store_args<layout> args) {
            if(warpgroup::warpid() == args.iter%4) {
                #pragma unroll
                for(int i = 0; i < NUM_CONSUMER_WARPS/4; i++)
                    warp::tma::store_async(args.globals.O, args.output.o[i], {args.common.batch, args.common.head+i, args.iter, 0});
                warp::tma::store_async_read_wait();
                __syncwarp();
                if(laneid() == 0) arrive(args.outputs_finished);
                __syncwarp();
            }
        }
    };
    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::consumer_registers<NUM_CONSUMER_WARPS/WARPGROUP_WARPS>();
            warp::zero(args.state.kv);
        }
        __device__ static void compute(consumer_compute_args<layout> args) {
            int warpgroupid = warpgroup::groupid();
            warpgroup::sync(warpgroupid);
            warpgroup::copy(args.scratch.a_cumsum[warpgroupid], args.input.a[warpgroupid]);
            warpgroup::sync(warpgroupid);
            if(warpgroup::warpid() <= 1) {
                int tid = warpgroup::laneid();
                for (int offset = 1; offset < 64; offset *= 2) {
                    float temp = (tid >= offset) ? args.scratch.a_cumsum[warpgroupid][tid - offset] : 0.0f;
                    group<2>::sync(warpgroupid+2);
                    args.scratch.a_cumsum[warpgroupid][tid] += temp;
                    group<2>::sync(warpgroupid+2);
                }
            }
            warpgroup::sync(warpgroupid);
            #pragma unroll
            for(int i = 0; i < 4; i++) {
                int base_row = warpgroup::warpid()*16 + laneid()/4;
                int base_col = i*16 + (laneid()%4)*2;
                args.state.local_decay.tiles[0][i].data[0].x = args.scratch.a_cumsum[warpgroupid][base_row + 0] - args.scratch.a_cumsum[warpgroupid][base_col + 0];
                args.state.local_decay.tiles[0][i].data[0].y = args.scratch.a_cumsum[warpgroupid][base_row + 0] - args.scratch.a_cumsum[warpgroupid][base_col + 1];
                args.state.local_decay.tiles[0][i].data[1].x = args.scratch.a_cumsum[warpgroupid][base_row + 8] - args.scratch.a_cumsum[warpgroupid][base_col + 0];
                args.state.local_decay.tiles[0][i].data[1].y = args.scratch.a_cumsum[warpgroupid][base_row + 8] - args.scratch.a_cumsum[warpgroupid][base_col + 1];
                args.state.local_decay.tiles[0][i].data[2].x = args.scratch.a_cumsum[warpgroupid][base_row + 0] - args.scratch.a_cumsum[warpgroupid][base_col + 8];
                args.state.local_decay.tiles[0][i].data[2].y = args.scratch.a_cumsum[warpgroupid][base_row + 0] - args.scratch.a_cumsum[warpgroupid][base_col + 9];
                args.state.local_decay.tiles[0][i].data[3].x = args.scratch.a_cumsum[warpgroupid][base_row + 8] - args.scratch.a_cumsum[warpgroupid][base_col + 8];
                args.state.local_decay.tiles[0][i].data[3].y = args.scratch.a_cumsum[warpgroupid][base_row + 8] - args.scratch.a_cumsum[warpgroupid][base_col + 9];
            }
            warp::exp(args.state.local_decay, args.state.local_decay);
            #pragma unroll
            for(int i = 0; i < 4; i++) {
                auto &decay_subtile = reinterpret_cast<rt_fl<16,16>&>(args.state.local_decay.tiles[0][i]);
                if      (i >  warpgroup::warpid()) { warp::zero       (decay_subtile); }
                else if (i == warpgroup::warpid()) { warp::make_causal(decay_subtile, decay_subtile, kittens::base_types::constants<float>::zero()); }
            }
            warpgroup::load(args.state.q_reg, args.input.q);
            warpgroup::mm_ABt(args.state.att_block, args.state.q_reg, args.input.k);
            warpgroup::mma_async_wait();
            warp::mul(args.state.att_block, args.state.att_block, args.state.local_decay);
            warp::copy(args.state.att_block_mma, args.state.att_block);
            warpgroup::mm_AB(args.state.o_reg, args.state.att_block_mma, args.input.v[warpgroupid]);
            warpgroup::mma_async_wait();
            {
                int base_row = warpgroup::warpid()*16 + laneid()/4;
                bf16 top = __float2bfloat16(expf(args.scratch.a_cumsum[warpgroupid][base_row + 0]));
                bf16 bottom = __float2bfloat16(expf(args.scratch.a_cumsum[warpgroupid][base_row +8]));
                #pragma unroll
                for(int i = 0; i < 4; i++) {
                    args.state.q_reg.tiles[0][i].data[0].x *= top;
                    args.state.q_reg.tiles[0][i].data[0].y *= top;
                    args.state.q_reg.tiles[0][i].data[1].x *= bottom;
                    args.state.q_reg.tiles[0][i].data[1].y *= bottom;
                    args.state.q_reg.tiles[0][i].data[2].x *= top;
                    args.state.q_reg.tiles[0][i].data[2].y *= top;
                    args.state.q_reg.tiles[0][i].data[3].x *= bottom;
                    args.state.q_reg.tiles[0][i].data[3].y *= bottom;
                }
            }
            warpgroup::store(args.scratch.kv[warpgroupid], args.state.kv);
            warpgroup::sync(warpgroupid);
            warpgroup::mma_AB(args.state.o_reg, args.state.q_reg, args.scratch.kv[warpgroupid]);
            warpgroup::mma_async_wait();
            warpgroup::store(args.output.o[warpgroupid], args.state.o_reg);
            warpgroup::sync(warpgroupid);
            float last_decay = args.scratch.a_cumsum[warpgroupid][args.scratch.a_cumsum[warpgroupid].length-1];
            float total_decay = expf(last_decay);
            warp::mul(args.state.kv, args.state.kv, total_decay);
            warpgroup::load(args.state.k_reg, args.input.k);
            {
                int base_row = warpgroup::warpid()*16 + laneid()/4;
                bf16 top = __float2bfloat16(expf(last_decay - args.scratch.a_cumsum[warpgroupid][base_row + 0]));
                bf16 bottom = __float2bfloat16(expf(last_decay - args.scratch.a_cumsum[warpgroupid][base_row +8]));
                #pragma unroll
                for(int i = 0; i < 4; i++) {
                    args.state.k_reg.tiles[0][i].data[0].x *= top;
                    args.state.k_reg.tiles[0][i].data[0].y *= top;
                    args.state.k_reg.tiles[0][i].data[1].x *= bottom;
                    args.state.k_reg.tiles[0][i].data[1].y *= bottom;
                    args.state.k_reg.tiles[0][i].data[2].x *= top;
                    args.state.k_reg.tiles[0][i].data[2].y *= top;
                    args.state.k_reg.tiles[0][i].data[3].x *= bottom;
                    args.state.k_reg.tiles[0][i].data[3].y *= bottom;
                }
            }
            warpgroup::store(args.scratch.k[warpgroupid], args.state.k_reg);
            warpgroup::sync(warpgroupid);
            warpgroup::mma_AtB(args.state.kv, args.scratch.k[warpgroupid], args.input.v[warpgroupid]);
            warpgroup::mma_async_wait();
            if(warpgroup::laneid() == 0) {
                arrive(args.outputs_arrived);
                arrive(args.inputs_finished);
            }
            __syncwarp();
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            if(warpgroup::laneid() == 0) arrive(args.finish_finished);
            __syncwarp();
        }
    };
};

#include <iostream>
#include <cstdlib>

__global__ void fill_random_bf16(bf16 *data, size_t count, uint64_t seed) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        uint64_t x = seed + idx;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        x = x ^ (x >> 31);
        float u = (float)(x >> 40) * (1.0f / 16777216.0f);
        data[idx] = __float2bfloat16((u * 2.0f - 1.0f) * 0.1f);
    }
}

__global__ void fill_decay(float *data, size_t count, uint64_t seed) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        // Small negative values for decay factors (log-space)
        data[idx] = -0.01f;
    }
}

int main(int argc, char **argv) {
    int B = 4, H = 16, N = 4096;
    constexpr int D = 64;
    if (argc > 1) N = atoi(argv[1]);

    printf("ThunderKittens Mamba2 SSM B=%d H=%d N=%d D=%d on H100\n", B, H, N, D);

    // Q, K: (B, 1, N, D) - shared across heads
    size_t qk_total = (size_t)B * 1 * N * D;
    // V, O: (B, H, N, D)
    size_t vo_total = (size_t)B * H * N * D;
    // A: (B, H, 1, N) - decay factors in log-space
    size_t a_total = (size_t)B * H * N;

    bf16 *d_q, *d_k, *d_v, *d_o;
    float *d_a;
    cudaMalloc(&d_q, qk_total * sizeof(bf16));
    cudaMalloc(&d_k, qk_total * sizeof(bf16));
    cudaMalloc(&d_v, vo_total * sizeof(bf16));
    cudaMalloc(&d_o, vo_total * sizeof(bf16));
    cudaMalloc(&d_a, a_total * sizeof(float));

    dim3 fblock(256);
    fill_random_bf16<<<dim3((qk_total + 255) / 256), fblock>>>(d_q, qk_total, 42);
    fill_random_bf16<<<dim3((qk_total + 255) / 256), fblock>>>(d_k, qk_total, 123);
    fill_random_bf16<<<dim3((vo_total + 255) / 256), fblock>>>(d_v, vo_total, 456);
    fill_decay<<<dim3((a_total + 255) / 256), fblock>>>(d_a, a_total, 789);
    cudaMemset(d_o, 0, vo_total * sizeof(bf16));
    cudaDeviceSynchronize();

    mamba2_fwd_template::layout::q_global Qg(d_q, (size_t)B, 1ul, (size_t)N, nullptr);
    mamba2_fwd_template::layout::k_global Kg(d_k, (size_t)B, 1ul, (size_t)N, nullptr);
    mamba2_fwd_template::layout::a_global Ag(d_a, (size_t)B, (size_t)H, nullptr, (size_t)N);
    mamba2_fwd_template::layout::v_global Vg(d_v, (size_t)B, (size_t)H, (size_t)N, nullptr);
    mamba2_fwd_template::layout::o_global Og(d_o, (size_t)B, (size_t)H, (size_t)N, nullptr);
    mamba2_fwd_template::layout::globals globals = {Qg, Kg, Vg, Og, Ag};

    unsigned long mem_size = kittens::prototype::detail::MAX_SHARED_MEMORY_v<mamba2_fwd_template>;
    cudaFuncSetAttribute(prototype::lcsf::kernel<mamba2_fwd_template>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

    dim3 grid(132, 1, 1);
    constexpr int BLOCK_SIZE = prototype::detail::NUM_THREADS_v<mamba2_fwd_template>;

    // Warmup
    for (int i = 0; i < 50; i++)
        prototype::lcsf::kernel<mamba2_fwd_template><<<grid, BLOCK_SIZE, mem_size>>>(globals);
    cudaDeviceSynchronize();

    // Timed iterations
    constexpr int NUM_ITERS = 500;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < NUM_ITERS; i++)
        prototype::lcsf::kernel<mamba2_fwd_template><<<grid, BLOCK_SIZE, mem_size>>>(globals);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float elapsed_ms = 0;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    double avg_us = (elapsed_ms * 1000.0) / NUM_ITERS;

    printf("Grid: %d (persistent), Block: %d\n", grid.x, BLOCK_SIZE);
    printf("Average kernel time: %.2f us (%d iters, 50 warmup)\n", avg_us, NUM_ITERS);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);
    cudaFree(d_a);
    return 0;
}
