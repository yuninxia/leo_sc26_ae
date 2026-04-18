// Profile driver for Astra-generated CUDA kernels.
// Compiled together with the kernel .cu file, linked against libtorch.
//
// Build:
//   nvcc astra_profile_driver.cu <kernel.cu> \
//     $(python3.11 -c "import torch; print(torch.utils.cmake_prefix_path)")/../../include \
//     ... (see build script below)
//
// Profile:
//   hpcrun -e gpu=cuda,pc ./kernel_profile 256 8192 200

#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// Forward declaration — implemented in the kernel .cu file
// All Astra RMSNorm/SiLU/MergeState kernels export this signature pattern.
void sgl_fused_add_rmsnorm(torch::Tensor input, torch::Tensor residual,
                           torch::Tensor weight, double eps, bool enable_pdl);

int main(int argc, char* argv[]) {
    int B     = argc > 1 ? atoi(argv[1]) : 256;
    int D     = argc > 2 ? atoi(argv[2]) : 8192;
    int iters = argc > 3 ? atoi(argv[3]) : 200;

    auto opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    auto x = torch::randn({B, D}, opts);
    auto r = torch::randn({B, D}, opts);
    auto w = torch::randn({D}, opts);

    // Warmup
    for (int i = 0; i < 20; i++)
        sgl_fused_add_rmsnorm(x.clone(), r.clone(), w.clone(), 1e-5, false);
    cudaDeviceSynchronize();

    // Profiling iterations
    for (int i = 0; i < iters; i++)
        sgl_fused_add_rmsnorm(x.clone(), r.clone(), w.clone(), 1e-5, false);
    cudaDeviceSynchronize();

    printf("Done: %d iterations of %dx%d\n", iters, B, D);
    return 0;
}
