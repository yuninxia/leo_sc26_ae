// SPDX-FileCopyrightText: Contributors to the HPCToolkit Project
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>
#include "minitest.h"
#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>
#include <hip/device_functions.h>

#ifdef USE_MPI
#include <mpi.h>
#endif

// Manage multiple GPUs in the system.
std::vector<hipDevice_t> gpu_devices;

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

  hipError_t err = hipSuccess;

  hrtime_t starttime = gethrtime();

  // Distribute work across GPU devices using round-robin assignment.
  // If MPI is used, consider both thread number and MPI rank; otherwise, use only thread number.
  int rank = 0;
#ifdef USE_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
  int gpu_index = (threadnum + rank) % gpu_devices.size();
  
  err = hipSetDevice(gpu_index);
  if (err != hipSuccess) {
    fflush (stdout);
    fprintf(stderr, "[%d] Failed to set device %d (error code %s)!\n", thispid, gpu_index, hipGetErrorString(err));
    exit(-1);
  }

  //allocate device memory for copying in l1 and r1, copying out p1
  size_t size = nn * sizeof(double);
  double *d_l1 = NULL;
  double *d_r1 = NULL;
  double *d_p1 = NULL;
  err = hipMalloc((void **)&d_l1, size);
  if (err != hipSuccess) {
    fflush (stdout);
    fprintf(stderr, "[%d] Failed to allocate device vector d_l1 (error code %s)!\n", thispid, hipGetErrorString(err));
    exit(-1);
#if 0
  } else {
    fprintf(stdout, "[%d] Allocated device vector d_l1\n", thispid );
#endif
  }
  err = hipMalloc((void **)&d_r1, size);
  if (err != hipSuccess) {
    fflush (stdout);
    fprintf(stderr, "[%d] Failed to allocate device vector d_r1 (error code %s)!\n", thispid, hipGetErrorString(err));
    exit(-1);
#if 0
  } else {
    fprintf(stdout, "[%d] Allocated device vector d_r1\n", thispid );
#endif
  }
  err = hipMalloc((void **)&d_p1, size);
  if (err != hipSuccess) {
    fflush (stdout);
    fprintf(stderr, "[%d] Failed to allocate device vector d_p1 (error code %s)!\n", thispid, hipGetErrorString(err));
    exit(-1);
#if 0
  } else {
    fprintf(stdout, "[%d] Allocated device vector d_p1\n", thispid );
#endif
  }

  // Copy l1,r1 and p1 to the device
  err = hipMemcpy(d_l1, l1, size, hipMemcpyHostToDevice);
  if (err != hipSuccess) {
    fflush (stdout);
    fprintf(stderr, "[%d] Failed to copy device l1 to d_l1 (error code %s)!\n", thispid, hipGetErrorString(err));
    exit(-1);
#if 0
  } else {
    fprintf(stdout, "[%d] Copied l1 to device vector d_l1\n", thispid );
#endif
  }
  err = hipMemcpy(d_r1, r1, size, hipMemcpyHostToDevice);
  if (err != hipSuccess) {
    fflush (stdout);
    fprintf(stderr, "[%d] Failed to copy device r1 to d_r1 (error code %s)!\n", thispid, hipGetErrorString(err));
    exit(-1);
#if 0
  } else {
    fprintf(stdout, "[%d] Copied r1 to device vector d_r1\n", thispid );
#endif
  }
  err = hipMemcpy(d_p1, p1, size, hipMemcpyHostToDevice);
  if (err != hipSuccess) {
    fflush (stdout);
    fprintf(stderr, "[%d] Failed to copy device p1 to d_p1 (error code %s)!\n", thispid, hipGetErrorString(err));
    exit(-1);
#if 0
  } else {
    fprintf(stdout, "[%d] Copied p1 to device vector d_p1\n", thispid );
#endif
  }

  // Set up and launch the HIP kernel
  int threadsPerBlock = 256;
  int blocksPerGrid = ( nn + threadsPerBlock -1 ) / threadsPerBlock;

#if 0
  fprintf(stdout, "      [%d] threadsPerBlock =  %d;  blocksPerGrid = %d\n", thispid, threadsPerBlock, blocksPerGrid );
#endif

  xcompute<<<blocksPerGrid, threadsPerBlock>>>(d_l1, d_r1, d_p1, nn, kernmax);
  err = hipGetLastError();
  if (err != hipSuccess) {
    fflush (stdout);
    fprintf(stderr, "[%d] Failed to launch compute kernel (error code %s)!\n", thispid, hipGetErrorString(err));
    exit(-1);
#if 0
  } else {
    fprintf(stdout, "[%d] Ran GPU kernel for xcompute\n", thispid );
#endif
  }

  // Copy p1 back to the host
  err = hipMemcpy(p1, d_p1, size, hipMemcpyDeviceToHost);
  if (err != hipSuccess) {
    fflush (stdout);
    fprintf(stderr, "[%d] Failed to copy p1 from device (error code %s)!\n", thispid, hipGetErrorString(err));
    exit(-1);
#if 0
  } else {
    fprintf(stdout, "[%d] Copied d_p1 back to host\n", thispid );
#endif
  }

  // Free the device memory
  err = hipFree(d_l1);
  if (err != hipSuccess) {
    fflush (stdout);
    fprintf(stderr, "[%d] Failed to free d_l1 from device (error code %s)!\n", thispid, hipGetErrorString(err));
    exit(-1);
#if 0
  } else {
    fprintf(stdout, "[%d] Freed device vector d_l1\n", thispid );
#endif
  }
  err = hipFree(d_r1);
  if (err != hipSuccess) {
    fflush (stdout);
    fprintf(stderr, "[%d] Failed to free d_r1 from device (error code %s)!\n", thispid, hipGetErrorString(err));
    exit(-1);
#if 0
  } else {
    fprintf(stdout, "[%d] Freed device vector d_r1\n", thispid );
#endif
  }
  err = hipFree(d_p1);
  if (err != hipSuccess) {
    fflush (stdout);
    fprintf(stderr, "[%d] Failed to free d_p1 from device (error code %s)!\n", thispid, hipGetErrorString(err));
    exit(-1);
#if 0
  } else {
    fprintf(stdout, "[%d] Freed device vector d_p1\n", thispid );
#endif
  }

  hrtime_t endtime = gethrtime();
  double  tempus =  (double) (endtime - starttime) / (double)1000000000.;
#if 1
  fprintf(stdout, "    [%d] Thread %d: completed iteration %d on GPU %d in %.9f s.\n",
    thispid, threadnum, iter, gpu_index, tempus);
#endif
  spacer(50, true);
}

void
initgpu()
{
  hrtime_t initstart = gethrtime();
  double  tempus =  (double) (initstart - starttime) / (double)1000000000.;
#if 1
  fprintf(stdout, "    [%d] Started HIP initgpu() at timestamp %.9f s.\n",
    thispid, tempus);
#endif
  fflush(stdout);

#if 1
  hipError_t err;

  // Get the number of devices
  int numdev;
  err = hipGetDeviceCount(&numdev);
  if (err != hipSuccess) {
    fprintf(stderr, "[%d] HIP Failed to get device count (error code %s)!\n", thispid, hipGetErrorString(err));
    exit(-1);
  }
  fprintf(stdout, "    [%d] Machine has %d HIP GPU device%s\n", thispid, numdev, (numdev==1 ? "" : "s"));
  fflush(stdout);

  // Initialize all the available devices
  for (int i = 0; i < numdev; i++) {
    hipDevice_t device;
    err = hipDeviceGet(&device, i);
    if (err != hipSuccess) {
      fprintf(stderr, "[%d] HIP Failed to get device %d (error code %s)!\n", thispid, i, hipGetErrorString(err));
      exit(-1);
    }

    gpu_devices.push_back(device);

    hipDeviceProp_t deviceProp;
    err = hipGetDeviceProperties(&deviceProp, i);
    if (err != hipSuccess) {
      fprintf(stderr, "[%d] HIP Failed to get device properties for device %d (error code %s)!\n", thispid, i, hipGetErrorString(err));
      exit(-1);
    }

    fprintf(stdout, "    [%d] HIP GPU %d: %s\n", thispid, i, deviceProp.name);
    fflush(stdout);
  }

  fprintf(stdout, "    [%d] HIP GPU initialization complete\n", thispid);
#endif

#if 0
  /* determine number of GPU's */
  int numdev = omp_get_num_devices();
  fprintf (stdout, "    [%d] Machine has %d GPU device%s\n", thispid, numdev, (numdev==1 ? "" : "s") );

  /* Test if GPU is available */
  int	idev = omp_is_initial_device();

  int runningOnGPU = -1;
  #pragma omp target map(from:runningOnGPU)
  {
    runningOnGPU = omp_is_initial_device();
  }

  /* If still running on CPU, GPU must not be available */
  if (runningOnGPU != 0) {
#ifndef IGNORE_BAD_INITIAL_DEVICE
    fprintf(stderr, "[%d] ERROR unable to use the GPU! idev = %d, runningOnGpU -- omp_is_initial_device() = %d; exiting\n",
      thispid, idev, runningOnGPU);
    exit(1);
#else
    fprintf(stdout, "[%d] ignore error unable to use gpu! idev = %d, runningOnGpU -- omp_is_initial_device() = %d; trying anyway\n",
      thispid, idev, runningOnGPU);
#endif
  } else {
    fprintfstdout, "    [%d] gputest is able to use the GPU! idev = %d, runningOnGpU -- omp_is_initial_device()\n", thispid, idev );
  }

  int ret = checkxfers();
  if (ret != 0 ) {
    fprintf(stdout, "[%d] Return from checkxfers = %d\n", thispid, ret);
  }
#endif

  hrtime_t initdone = gethrtime();
  tempus =  (double) (initdone - starttime) / (double)1000000000.;
#if 1
  fprintf(stdout, "    [%d] Leaving HIP initgpu() at timestamp %.9f s.\n",
    thispid, tempus);
#endif
  fflush(stdout);
}

int
checkxfers()
{
// This is only needed for the OpenMP version
  return 0;
}
