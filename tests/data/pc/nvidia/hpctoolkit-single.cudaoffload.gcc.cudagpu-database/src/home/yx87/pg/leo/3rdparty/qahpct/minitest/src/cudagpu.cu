// SPDX-FileCopyrightText: Contributors to the HPCToolkit Project
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>
#include "minitest.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#ifdef USE_MPI
#include <mpi.h>
#endif

// Manage multiple GPUs in the system.
std::vector<int> gpu_devices;

int checkxfers();

__global__ void
xcompute(const double *d_l1, const double *d_r1, double *d_p1, int nelements, int kernmax )
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < nelements) {

#include "compute.h"

  }
}

void
twork( int iter, int threadnum)
{
  double *l1 = lptr[threadnum];
  double *r1 = rptr[threadnum];
  double *p1 = pptr[threadnum];
  int kernmax = kkmax;

  cudaError_t err = cudaSuccess;

  hrtime_t starttime = gethrtime();

  // Distribute work across GPU devices using round-robin assignment
  // If MPI is used, consider both thread number and MPI rank; otherwise, use only thread number
  int rank = 0;
#ifdef USE_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
#if 0
  fprintf(stdout, "[%d]    T %d, I %d, calling cudaSetDevice()\n",
      thispid, threadnum, iter );
#endif
  int gpu_index = (threadnum + rank) % gpu_devices.size();
  cudaSetDevice(gpu_devices[gpu_index]);

#if 0
  fprintf(stdout, "[%d]    T %d, I %d, start cuda twork\n",
      thispid, threadnum, iter );
#endif
  //allocate device memory for copying in l1 and r1, copying out p1
  size_t size = nn * sizeof(double);
  double *d_l1 = NULL;
  double *d_r1 = NULL;
  double *d_p1 = NULL;

  err = cudaMalloc((void **)&d_l1, size);
  if (err != cudaSuccess) {
    fflush(stdout);
    fprintf(stderr, "[%d]    T %d, I %d, Failed to allocate device vector d_l1 (error code %s)!\n",
      thispid, threadnum, iter, cudaGetErrorString(err));
    exit(-1);
#if 0
  } else {
    fprintf(stdout, "[%d]    T %d, I %d, Alloc device vector d_l1 (%p)\n",
      thispid, threadnum, iter, d_l1 );
#endif
  }
  err = cudaMalloc((void **)&d_r1, size);
  if (err != cudaSuccess) {
    fflush(stdout);
    fprintf(stderr, "[%d]    T %d, I %d, Failed to allocate device vector d_r1 (error code %s)!\n",
      thispid, threadnum, iter, cudaGetErrorString(err));
    exit(-1);
#if 0
  } else {
    fprintf(stdout, "[%d]    T %d, I %d, Alloc device vector d_r1 (%p)\n",
      thispid, threadnum, iter, d_r1 );
#endif
  }
  err = cudaMalloc((void **)&d_p1, size);
  if (err != cudaSuccess) {
    fflush(stdout);
    fprintf(stderr, "[%d]    T %d, I %d, Failed to allocate device vector d_p1 (error code %s)!\n",
      thispid, threadnum, iter, cudaGetErrorString(err));
    exit(-1);
#if 0
  } else {
    fprintf(stdout, "[%d]    T %d, I %d, Alloc device vector d_p1 (%p)\n",
      thispid, threadnum, iter, d_p1 );
#endif
  }

  // Copy l1,r1 and p1 to the device
  err = cudaMemcpy(d_l1, l1, size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fflush(stdout);
    fprintf(stderr, "[%d]    T %d, I %d, Failed to copy device l1 to d_l1 (error code %s)!\n",
      thispid, threadnum, iter, cudaGetErrorString(err));
    exit(-1);
#if 0
  } else {
    fprintf(stdout, "[%d]    T %d, I %d, Copied l1 to device vector d_l1\n",
      thispid, threadnum, iter );
#endif
  }
  err = cudaMemcpy(d_r1, r1, size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fflush(stdout);
    fprintf(stderr, "[%d]    T %d, I %d, Failed to copy device r1 to d_r1 (error code %s)!\n",
      thispid, threadnum, iter, cudaGetErrorString(err));
    exit(-1);
#if 0
  } else {
    fprintf(stdout, "[%d]    T %d, I %d, Copied r1 to device vector d_r1\n",
      thispid, threadnum, iter );
#endif
  }
  err = cudaMemcpy(d_p1, p1, size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fflush(stdout);
    fprintf(stderr, "[%d]    T %d, I %d, Failed to copy device p1 to d_p1 (error code %s)!\n",
      thispid, threadnum, iter, cudaGetErrorString(err));
    exit(-1);
#if 0
  } else {
    fprintf(stdout, "[%d]    T %d, I %d, Copied p1 to device vector d_p1\n",
      thispid, threadnum, iter );
#endif
  }

  // Set up and launch the CUDA kernel
  int threadsPerBlock = 256;
  int blocksPerGrid = ( nn + threadsPerBlock -1 ) / threadsPerBlock;

#if 1
  fprintf(stdout, "      [%d]  t %d, i%d, threadsPerBlock =  %d;  blocksPerGrid = %d\n",
    thispid, threadnum, iter, threadsPerBlock, blocksPerGrid );
#endif

  xcompute<<<blocksPerGrid, threadsPerBlock>>>(d_l1, d_r1, d_p1, nn, kernmax);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fflush(stdout);
    fprintf(stderr, "[%d]    T %d, I %d, Failed to launch compute kernel (error code %s)!\n",
      thispid, threadnum, iter, cudaGetErrorString(err));
    exit(-1);
#if 0
  } else {
    fprintf(stdout, "[%d]    T %d, I %d, Ran GPU kernel for xcompute\n",
      thispid, threadnum, iter );
#endif
  }

  // Copy p1 back to the host
  err = cudaMemcpy(p1, d_p1, size, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fflush(stdout);
    fprintf(stderr, "[%d]    T %d, I %d, Failed to copy p1 from device (error code %s)!\n",
      thispid, threadnum, iter, cudaGetErrorString(err));
    exit(-1);
#if 0
  } else {
    fprintf(stdout, "[%d]    T %d, I %d, Copied d_p1 back to host\n",
      thispid, threadnum, iter );
#endif
  }

  // Free the device memory
  err = cudaFree(d_l1);
  if (err != cudaSuccess) {
    fflush(stdout);
    fprintf(stderr, "[%d]    T %d, I %d, Failed to free d_l1 from device (error code %s)!\n",
      thispid, threadnum, iter, cudaGetErrorString(err));
    exit(-1);
#if 0
  } else {
    fprintf(stdout, "[%d]    T %d, I %d, Freed device vector d_l1 (%p)\n",
      thispid, threadnum, iter, d_l1 );
#endif
  }
  err = cudaFree(d_r1);
  if (err != cudaSuccess) {
    fflush(stdout);
    fprintf(stderr, "[%d]    T %d, I %d, Failed to free d_r1 from device (error code %s)!\n",
      thispid, threadnum, iter, cudaGetErrorString(err));
    exit(-1);
#if 0
  } else {
    fprintf(stdout, "[%d]    T %d, I %d, Freed device vector d_r1 (%p)\n",
      thispid, threadnum, iter, d_r1 );
#endif
  }
  err = cudaFree(d_p1);
  if (err != cudaSuccess) {
    fflush(stdout);
    fprintf(stderr, "[%d]    T %d, I %d, Failed to free d_p1 from device (error code %s)!\n",
      thispid, threadnum, iter, cudaGetErrorString(err));
    exit(-1);
#if 0
  } else {
    fprintf(stdout, "[%d]    T %d, I %d, Freed device vector d_p1 (%p)\n",
      thispid, threadnum, iter, d_p1 );
#endif
  }

  hrtime_t endtime = gethrtime();
  double  tempus =  (double) (endtime - starttime) / (double)1000000000.;
#if 1
  fprintf(stdout, "    [%d] Thread %d: completed iteration %d on GPU %d in %.9f s.\n",
    thispid, threadnum, iter, gpu_devices[gpu_index], tempus);
#endif
  spacer (50, true);
}

void
initgpu()
{
  hrtime_t initstart = gethrtime();
  double  tempus =  (double) (initstart - starttime) / (double)1000000000.;
#if 1
  fprintf(stdout, "    [%d] Started initgpu() at timestamp %.9f s.\n",
    thispid, tempus);
#endif

  int deviceCount;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);
  if (err != cudaSuccess) {
    fflush(stdout);
    fprintf(stderr, "[%d] Failed to get device count (error code %s)!\n", thispid, cudaGetErrorString(err));
    exit(-1);
  }
  fprintf(stdout, "    [%d] Machine has %d CUDA-capable GPU device%s\n", thispid, deviceCount, (deviceCount==1 ? "" : "s"));

  for (int i = 0; i < deviceCount; i++) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, i);
    fprintf(stdout, "    [%d] GPU %d: %s\n", thispid, i, deviceProp.name);
    gpu_devices.push_back(i);
  }

#if 0
  fprintf(stdout, "    [%d] CUDA GPU initialization complete\n", thispid);
#endif

  hrtime_t initdone = gethrtime();
  tempus =  (double) (initdone - starttime) / (double)1000000000.;
#if 1
  fprintf(stdout, "    [%d] Leaving initgpu() at timestamp %.9f s.\n",
    thispid, tempus);
#endif
}

int
checkxfers()
{
// This is only needed for the OpenMP offload version
  return 0;
}

