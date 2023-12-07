/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include "kernel.cu"
#include "support.h"
#include <stdio.h>
#include <time.h>

#define ITER 20000
#define THREADS_PER_BLOCK 256

// prototypes
void allocateDeviceSpace(float **d_A, float **d_B, float **d_C, int n);

int main(int argc, char **argv) {

  Timer timer;
  cudaError_t cuda_ret;
  time_t t;
  float timeRegKern, timeStrmKern;

  // Initialize host variables ----------------------------------------------

  printf("\nSetting up the problem...");
  fflush(stdout);
  startTime(&timer);

  unsigned int n;
  if (argc == 1) {
    n = 1000000;
  } else if (argc == 2) {
    n = atoi(argv[1]);
  } else {
    printf("\n    Invalid input parameters!"
           "\n    Usage: ./vecadd               # Vector of size 10,000 is used"
           "\n    Usage: ./vecadd <m>           # Vector of size m is used"
           "\n");
    exit(0);
  }

  /* Intializes random number generator */
  srand((unsigned)time(&t));

  // float* A_h = (float*) malloc( sizeof(float)*n );
  float *A_h;
  cudaHostAlloc((void **)&A_h, sizeof(float) * n,
                cudaHostAllocDefault); // Allocate Pinned memory
  for (unsigned int i = 0; i < n; i++) {
    A_h[i] = (rand() % 100) / 100.00;
  }

  // float* B_h = (float*) malloc( sizeof(float)*n );
  float *B_h;
  cudaHostAlloc((void **)&B_h, sizeof(float) * n,
                cudaHostAllocDefault); // Allocate Pinned memory
  for (unsigned int i = 0; i < n; i++) {
    B_h[i] = (rand() % 100) / 100.00;
  }

  // float* C_h = (float*) malloc( sizeof(float)*n );
  float *C_h;
  cudaHostAlloc((void **)&C_h, sizeof(float) * n,
                cudaHostAllocDefault); // Allocate Pinned memory

  stopTime(&timer);
  printf("%f s\n", elapsedTime(timer));
  printf("    Vector size = %u\n", n);

  // -------------------------- Setup a regular Kernel launch
  // ------------------------------------------
  printf("\n[Setting up regular Kernel configurations]\n");

  // Allocate device variables ----------------------------------------------
  printf("Allocating device variables...");
  fflush(stdout);
  startTime(&timer);

  float *A_d;
  cuda_ret = cudaMalloc((void **)&A_d, sizeof(float) * n);
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to allocate device memory");

  float *B_d;
  cuda_ret = cudaMalloc((void **)&B_d, sizeof(float) * n);
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to allocate device memory");

  float *C_d;
  cuda_ret = cudaMalloc((void **)&C_d, sizeof(float) * n);
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to allocate device memory");

  cudaDeviceSynchronize();
  stopTime(&timer);
  printf("%f s\n", elapsedTime(timer));

  // Compute thread grid size
  const unsigned int numBlocks = (n - 1) / THREADS_PER_BLOCK + 1;
  dim3 gridDim(numBlocks, 1, 1), blockDim(THREADS_PER_BLOCK, 1, 1);

  // Invoke multiple iterations of regular communication and computation
  printf("Performing %d iterations of regular Kernel launch...\n", ITER);
  startTime(&timer);
  for (int i = 0; i < ITER; i++) {
    // Copy host vectors to device
    cuda_ret = cudaMemcpy(A_d, A_h, sizeof(float) * n, cudaMemcpyHostToDevice);
    if (cuda_ret != cudaSuccess)
      FATAL("Unable to copy to device memory");

    cuda_ret = cudaMemcpy(B_d, B_h, sizeof(float) * n, cudaMemcpyHostToDevice);
    if (cuda_ret != cudaSuccess)
      FATAL("Unable to copy to device memory");

    // Launch kernel
    vecAddKernel<<<gridDim, blockDim>>>(A_d, B_d, C_d, n);
    cuda_ret = cudaGetLastError();
    if (cuda_ret != cudaSuccess)
      FATAL("Unable to launch kernel");

    // Copy array back to host
    cuda_ret = cudaMemcpy(C_h, C_d, sizeof(float) * n, cudaMemcpyDeviceToHost);
    if (cuda_ret != cudaSuccess)
      FATAL("Unable to copy to host memory");
  }
  stopTime(&timer);
  timeRegKern = elapsedTime(timer);
  printf("Total time in %d iterations of regular Kernel launch: %f s\n", ITER,
         timeRegKern);

  // Verify correctness -----------------------------------------------------
  printf("Verifying results...");
  fflush(stdout);
  verify(A_h, B_h, C_h, n);

  // ------------------------ Reset before Stream Kernel setup
  // --------------------------------------
  printf("[Cleaning up before Stream Kernel setup]\n");
  printf("Freeing up device memory...\n");
  cuda_ret = cudaFree(A_d);
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to free CUDA memory");
  cuda_ret = cudaFree(B_d);
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to free CUDA memory");
  cuda_ret = cudaFree(C_d);
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to free CUDA memory");
  printf("Clearing result vector...\n");
  memset(C_h, 0, n * sizeof(float)); // clear results

  // --------------------------- Setup Stream Kernel launches
  // ---------------------------
  printf("\n[Setting up stream based Kernel configurations]\n");
  cudaStream_t stream0, stream1;
  cudaStreamCreate(&stream0);
  cudaStreamCreate(&stream1);

  // No. of elements per-stream
  int n0 = n / 2;
  int n1 = n - n0;

  // Allocate device memory for stream data
  printf("Allocating device variables...\n");
  fflush(stdout);
  float *A0_d, *B0_d, *C0_d;
  float *A1_d, *B1_d, *C1_d;
  allocateDeviceSpace(&A0_d, &B0_d, &C0_d, n0);
  allocateDeviceSpace(&A1_d, &B1_d, &C1_d, n1);

  // calculate thread grid for streams
  int blockDimX = THREADS_PER_BLOCK;
  int gridDimX0 = n0 / THREADS_PER_BLOCK;
  if (n0 % THREADS_PER_BLOCK)
    gridDimX0++;
  int gridDimX1 = n1 / THREADS_PER_BLOCK;
  if (n1 % THREADS_PER_BLOCK)
    gridDimX1++;

  // Invoke multiple iterations of regular communication and computation
  // -----------------
  printf("Performing %d iterations of stream Kernel launch...\n", ITER);
  startTime(&timer);
  for (int i = 0; i < ITER; i++) {
    // Copy host vectors to device for stream0
    cuda_ret = cudaMemcpyAsync(A0_d, A_h, sizeof(float) * n0,
                               cudaMemcpyHostToDevice, stream0);
    if (cuda_ret != cudaSuccess)
      FATAL("Unable to copy to device memory (stream0)");

    cuda_ret = cudaMemcpyAsync(B0_d, B_h, sizeof(float) * n0,
                               cudaMemcpyHostToDevice, stream0);
    if (cuda_ret != cudaSuccess)
      FATAL("Unable to copy to device memory (stream0)");

    // Launch kernel for stream0
    vecAddKernel<<<gridDimX0, blockDimX, 0, stream0>>>(A0_d, B0_d, C0_d, n0);
    cuda_ret = cudaGetLastError();
    if (cuda_ret != cudaSuccess)
      FATAL("Unable to launch kernel (stream0)");

    // Copy host vectors to device for stream1
    cuda_ret = cudaMemcpyAsync(A1_d, A_h + n0, sizeof(float) * n1,
                               cudaMemcpyHostToDevice, stream1);
    if (cuda_ret != cudaSuccess)
      FATAL("Unable to copy to device memory (stream1)");

    cuda_ret = cudaMemcpyAsync(B1_d, B_h + n0, sizeof(float) * n1,
                               cudaMemcpyHostToDevice, stream1);
    if (cuda_ret != cudaSuccess)
      FATAL("Unable to copy to device memory (stream1)");

    // Launch kernel for stream1
    vecAddKernel<<<gridDimX1, blockDimX, 0, stream1>>>(A1_d, B1_d, C1_d, n1);
    cuda_ret = cudaGetLastError();
    if (cuda_ret != cudaSuccess)
      FATAL("Unable to launch kernel (stream1)");

    // Copy array back to host
    cuda_ret = cudaMemcpyAsync(C_h, C0_d, sizeof(float) * n0,
                               cudaMemcpyDeviceToHost, stream0);
    if (cuda_ret != cudaSuccess)
      FATAL("Unable to copy to host memory (stream0)");

    cuda_ret = cudaMemcpyAsync(C_h + n0, C1_d, sizeof(float) * n1,
                               cudaMemcpyDeviceToHost, stream1);
    if (cuda_ret != cudaSuccess)
      FATAL("Unable to copy to host memory (stream1)");
  }
  cudaDeviceSynchronize();
  stopTime(&timer);
  timeStrmKern = elapsedTime(timer);
  printf("Total time in %d iterations of stream Kernel launch: %f s\n", ITER,
         timeStrmKern);

  // Verify correctness
  printf("Verifying results...");
  fflush(stdout);
  verify(A_h, B_h, C_h, n);

  // Calculate Speedup
  printf("Speedup (regular/stream): %.3f\n\n", timeRegKern / timeStrmKern);

  // Free memory ------------------------------------------------------------
  // Free page-locked memory
  cuda_ret = cudaFreeHost(A_h);
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to free CUDA memory");
  cuda_ret = cudaFreeHost(B_h);
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to free CUDA memory");
  cuda_ret = cudaFreeHost(C_h);
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to free CUDA memory");

  // Free device memory
  cuda_ret = cudaFree(A0_d);
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to free CUDA memory");
  cuda_ret = cudaFree(B0_d);
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to free CUDA memory");
  cuda_ret = cudaFree(C0_d);
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to free CUDA memory");

  cuda_ret = cudaFree(A1_d);
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to free CUDA memory");
  cuda_ret = cudaFree(B1_d);
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to free CUDA memory");
  cuda_ret = cudaFree(C1_d);
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to free CUDA memory");

  return 0;
}

// Helper functions --------------------------------------------------------

// allocates device variable, given pointer to float arrays and length (n)
void allocateDeviceSpace(float **d_A, float **d_B, float **d_C, int n) {
  cudaError_t cuda_ret;

  cuda_ret = cudaMalloc((void **)d_A, sizeof(float) * n);
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to allocate device memory");
  cuda_ret = cudaMalloc((void **)d_B, sizeof(float) * n);
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to allocate device memory");
  cuda_ret = cudaMalloc((void **)d_C, sizeof(float) * n);
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to allocate device memory");
}
