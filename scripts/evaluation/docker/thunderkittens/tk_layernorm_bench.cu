// Standalone ThunderKittens LayerNorm benchmark for profiling with HPCToolkit
// Fused residual + LayerNorm kernel. Default: B=16 N=4096 D=1024.
// Accepts N as command-line argument.

#include "kittens.cuh"
#include <cooperative_groups.h>
#include <curand_kernel.h>
#include <cuda/semaphore>
#include <cuda/pipeline>
#include <tuple>

static constexpr int NUM_WORKERS = (2);
static constexpr int NUM_THREADS = (NUM_WORKERS*kittens::WARP_THREADS);

using namespace kittens;

template<kittens::ducks::rv::all T>
__device__ void dropout_mask(T &dst, float keep_prob) {
    unsigned long long seed = 0;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, 0, &state);
    #pragma unroll
    for ( int i = 0 ; i < dst.outer_dim ; i ++ ) {
        #pragma unroll
        for(int j = 0; j < dst.inner_dim; j++) {
            float rand = curand_uniform(&state);
            if (rand < keep_prob) {
                dst[i][j].x = base_types::constants<bf16>::zero();
                dst[i][j].y = base_types::constants<bf16>::zero();
            }
        }
    }
    mul(dst, dst, __float2bfloat16(1/(1-keep_prob)));
}

template<kittens::ducks::sv::all T>
__device__ void dropout_mask(T &dst, float keep_prob) {
    unsigned long long seed = 0;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, 0, &state);
    #pragma unroll
    for(int cur = laneid(); cur < T::length; cur+=WARP_THREADS) {
        float rand = curand_uniform(&state);
        if (rand < keep_prob) {
            dst[cur] = base_types::constants<bf16>::zero();
        }
    }
    warp::mul(dst, dst, __float2bfloat16(1/(1-keep_prob)));
}

template<int _d_model> struct norm_globals {
    static constexpr int d_model = _d_model;
    static constexpr int dropout_p = 0.0;
    using vec_smem_1xD  = sv_bf<d_model>;
    using tile_smem_1xD = st<bf16, 1, d_model>;
    using tile_reg_1xD  = rt_bf<1, d_model>;
    using x_gl            = gl<bf16, -1, -1, -1, -1, vec_smem_1xD>;
    using residual_gl     = gl<bf16, -1, -1, -1, -1, vec_smem_1xD>;
    using o_gl            = gl<bf16, -1, -1, -1, -1, vec_smem_1xD>;
    using o_resid_gl      = gl<bf16, -1, -1, -1, -1, vec_smem_1xD>;
    using norm_weight_gl  = gl<bf16, -1, -1, -1, -1, vec_smem_1xD>;
    using norm_bias_gl    = gl<bf16, -1, -1, -1, -1, vec_smem_1xD>;
    x_gl x;
    residual_gl residual;
    o_gl o;
    o_resid_gl o_resid;
    norm_weight_gl norm_weight;
    norm_bias_gl norm_bias;
    const int n_tile_size;
    const int n_per_tile;
};

template<int D>
__global__ __launch_bounds__(NUM_THREADS, 1)
void layernorm_tk(const __grid_constant__ norm_globals<D> g, int n_per_tile) {
    auto warpid = kittens::warpid();
    int batch = blockIdx.y;
    int seq_start = blockIdx.x*2;
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    static constexpr int d_model = D;
    using vec_smem_1xD = sv_bf<d_model>;
    vec_smem_1xD (&x_s)           [2][NUM_WORKERS] = al.allocate<vec_smem_1xD,2,NUM_WORKERS>();
    vec_smem_1xD (&residual_s)    [2][NUM_WORKERS] = al.allocate<vec_smem_1xD,2,NUM_WORKERS>();
    vec_smem_1xD (&norm_weight_s) = al.allocate<vec_smem_1xD>();
    vec_smem_1xD (&norm_bias_s  ) = al.allocate<vec_smem_1xD>();
    int tic = 0, toc = 1;
    if (warpid == 0) {
        warp::load(norm_bias_s, g.norm_bias, {0,0,0,0});
        warp::load(norm_weight_s, g.norm_weight, {0,0,0,0});
    }
    bf16 mean = __float2bfloat16(0.0f);
    bf16 var  = __float2bfloat16(0.0f);
    warp::load_async(       x_s[warpid][tic], g.x,        {batch, 0, seq_start+warpid, 0});
    warp::load_async(residual_s[warpid][tic], g.residual, {batch, 0, seq_start+warpid, 0});
    __syncthreads();
    int n_blocks = g.n_per_tile/NUM_WORKERS;
    for (int block = 0; block < n_blocks; block ++, tic ^=1, toc ^=1) {
        auto cur_idx  = (block + 0)*NUM_WORKERS + warpid;
        auto next_idx = (block + 1)*NUM_WORKERS + warpid;
        if( block < n_blocks - 1 ) {
            warp::load_async(       x_s[warpid][toc], g.x,        {batch, 0, seq_start+next_idx, 0});
            warp::load_async(residual_s[warpid][toc], g.residual, {batch, 0, seq_start+next_idx, 0});
        }
        load_async_wait();
        __syncwarp();
        dropout_mask(x_s[warpid][tic], g.dropout_p);
        warp::add(residual_s[warpid][tic], residual_s[warpid][tic], x_s[warpid][tic]);
        warp::store(g.o_resid, residual_s[warpid][tic], {batch, 0, seq_start+cur_idx, 0});
        __syncwarp();
        warp::sum(mean, residual_s[warpid][tic]);
        mean = mean / __float2bfloat16(d_model);
        warp::sub(residual_s[warpid][tic], residual_s[warpid][tic], mean);
        warp::mul(x_s[warpid][tic], residual_s[warpid][tic], residual_s[warpid][tic]);
        warp::sum(var, x_s[warpid][tic]);
        var = var / __float2bfloat16(d_model);
        var = __float2bfloat16(sqrt(__bfloat162float(var + __float2bfloat16(1e-05f))));
        warp::div(residual_s[warpid][tic], residual_s[warpid][tic], var);
        warp::mul(residual_s[warpid][tic], residual_s[warpid][tic], norm_weight_s);
        warp::add(residual_s[warpid][tic], residual_s[warpid][tic], norm_bias_s);
        __syncwarp();
        warp::store(g.o, residual_s[warpid][tic], {batch, 0, seq_start+cur_idx, 0});
    }
}

#include <iostream>
#include <cstdlib>

// Simple random fill kernel
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
    int B = 16, N = 4096;
    constexpr int D = 1024;
    if (argc > 1) N = atoi(argv[1]);

    printf("ThunderKittens LayerNorm B=%d N=%d D=%d on H100\n", B, N, D);

    size_t total = (size_t)B * N * D;
    bf16 *d_x, *d_residual, *d_o, *d_o_resid, *d_norm_weight, *d_norm_bias;
    cudaMalloc(&d_x, total * sizeof(bf16));
    cudaMalloc(&d_residual, total * sizeof(bf16));
    cudaMalloc(&d_o, total * sizeof(bf16));
    cudaMalloc(&d_o_resid, total * sizeof(bf16));
    cudaMalloc(&d_norm_weight, D * sizeof(bf16));
    cudaMalloc(&d_norm_bias, D * sizeof(bf16));

    dim3 fblock(256);
    fill_random<<<dim3((total + 255) / 256), fblock>>>(d_x, total, 42);
    fill_random<<<dim3((total + 255) / 256), fblock>>>(d_residual, total, 123);
    fill_random<<<dim3((D + 255) / 256), fblock>>>(d_norm_weight, D, 456);
    fill_random<<<dim3((D + 255) / 256), fblock>>>(d_norm_bias, D, 789);
    cudaDeviceSynchronize();

    // Set up dispatch (mirrors dispatch_layernorm from upstream)
    using vec_smem_1xD  = sv_bf<static_cast<size_t>(D)>;
    using x_gl           = gl<bf16, -1, -1, -1, -1, vec_smem_1xD>;
    using residual_gl    = gl<bf16, -1, -1, -1, -1, vec_smem_1xD>;
    using o_gl           = gl<bf16, -1, -1, -1, -1, vec_smem_1xD>;
    using o_resid_gl     = gl<bf16, -1, -1, -1, -1, vec_smem_1xD>;
    using norm_weight_gl = gl<bf16, -1, -1, -1, -1, vec_smem_1xD>;
    using norm_bias_gl   = gl<bf16, -1, -1, -1, -1, vec_smem_1xD>;

    x_gl           x_arg{d_x, (size_t)B, 1ul, (size_t)N, (size_t)D};
    residual_gl    residual_arg{d_residual, (size_t)B, 1ul, (size_t)N, (size_t)D};
    o_gl           o_arg{d_o, (size_t)B, 1ul, (size_t)N, (size_t)D};
    o_resid_gl     o_resid_arg{d_o_resid, (size_t)B, 1ul, (size_t)N, (size_t)D};
    norm_weight_gl norm_weight_arg{d_norm_weight, 1ul, 1ul, 1ul, (size_t)D};
    norm_bias_gl   norm_bias_arg{d_norm_bias, 1ul, 1ul, 1ul, (size_t)D};

    const int n_tile_size = N / 2;
    const int n_per_tile = 2;
    norm_globals<D> g{x_arg, residual_arg, o_arg, o_resid_arg, norm_weight_arg, norm_bias_arg,
                      n_tile_size, n_per_tile};

    unsigned long mem_size = 25480;
    cudaFuncSetAttribute(layernorm_tk<D>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

    dim3 grid(n_tile_size, B, 1);

    // Warmup
    for (int i = 0; i < 200; i++)
        layernorm_tk<D><<<grid, NUM_THREADS, mem_size>>>(g, n_per_tile);
    cudaDeviceSynchronize();

    // Timed iterations
    constexpr int NUM_ITERS = 1000;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < NUM_ITERS; i++)
        layernorm_tk<D><<<grid, NUM_THREADS, mem_size>>>(g, n_per_tile);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float elapsed_ms = 0;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    double avg_us = (elapsed_ms * 1000.0) / NUM_ITERS;

    // Bandwidth: read x + residual + weights + bias, write o + o_resid
    double bytes = (double)(2*total + 2*D + 2*total) * sizeof(bf16);
    double bw_gbps = bytes / (avg_us * 1e3);

    printf("Grid: (%d, %d)\n", n_tile_size, B);
    printf("Average kernel time: %.2f us (%d iters, 200 warmup)\n", avg_us, NUM_ITERS);
    printf("Effective bandwidth: %.2f GB/s\n", bw_gbps);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_x);
    cudaFree(d_residual);
    cudaFree(d_o);
    cudaFree(d_o_resid);
    cudaFree(d_norm_weight);
    cudaFree(d_norm_bias);
    return 0;
}
