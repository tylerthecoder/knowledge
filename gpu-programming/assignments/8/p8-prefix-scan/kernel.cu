/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <cstddef>
#include <cstdio>
#define BLOCK_SIZE 512

// Define your kernels in this file you may use more than one kernel if you
// need to

__global__ void preScanKernel(float *inout, unsigned size, float *sum) {
  __shared__ float temp[BLOCK_SIZE * 2];

  int t = threadIdx.x;
  int n = size;
  int offset = 1;

  // load input into shared memory
  temp[2 * t] = inout[2 * t];
  temp[2 * t + 1] = inout[2 * t + 1];

  __syncthreads();

  // Up-Sweep
  for (int d = n >> 1; d > 0; d >>= 1) {
    __syncthreads();
    if (t < d) {
      int ai = offset * (2 * t + 1) - 1;
      int bi = offset * (2 * t + 2) - 1;
      temp[bi] += temp[ai];
    }
    offset *= 2;
  }

  // clear the last element
  if (t == 0) {
    temp[n - 1] = 0;
  }

  __syncthreads();

  // traverse down tree & build scan
  for (int d = 1; d < n; d *= 2) {
    offset /= 2;
    __syncthreads();
    if (t < d) {
      int ai = offset * (2 * t + 1) - 1;
      int bi = offset * (2 * t + 2) - 1;
      float val = temp[ai];
      if (bi < BLOCK_SIZE * 2 && ai < BLOCK_SIZE * 2) {
        temp[ai] = temp[bi];
      }
      if (bi < BLOCK_SIZE * 2) {
        temp[bi] += val;
      }
    }
  }

  __syncthreads();

  inout[2 * t] = temp[2 * t];
  inout[2 * t + 1] = temp[2 * t + 1];
}

__global__ void addKernel(float *inout, float *sum, unsigned size) {
  unsigned int t = threadIdx.x;
  unsigned int start = 2 * blockIdx.x * BLOCK_SIZE;
  unsigned int globalIndex = start + t;

  if (blockIdx.x > 0) {
    if (globalIndex < size) {
      inout[globalIndex] += sum[blockIdx.x - 1];
    }
    if (globalIndex + BLOCK_SIZE < size) {
      inout[globalIndex + BLOCK_SIZE] += sum[blockIdx.x - 1];
    }
  }
}

unsigned int nextPow2(unsigned int x) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

/******************************************************************************
Setup and invoke your kernel(s) in this function. You may also allocate more
GPU memory if you need to
*******************************************************************************/
void preScan(float *inout, unsigned in_size) {
  float *sum;
  float *paddedInout;
  unsigned num_blocks;
  cudaError_t cuda_ret;
  dim3 dim_grid, dim_block;

  num_blocks = in_size / (BLOCK_SIZE * 2);
  if (in_size % (BLOCK_SIZE * 2) != 0)
    num_blocks++;

  dim_block.x = BLOCK_SIZE;
  dim_block.y = 1;
  dim_block.z = 1;
  dim_grid.x = num_blocks;
  dim_grid.y = 1;
  dim_grid.z = 1;

  unsigned paddedSize = nextPow2(in_size);

  float *real_inout = inout;
  unsigned real_size = in_size;

  // Check if padding is needed
  if (paddedSize != in_size) {
    printf("Padding needed\n");

    cuda_ret = cudaMalloc((void **)&paddedInout, paddedSize * sizeof(float));
    if (cuda_ret != cudaSuccess)
      FATAL("Unable to allocate device memory for padded inout");

    cuda_ret = cudaMemcpy(paddedInout, inout, in_size * sizeof(float),
                          cudaMemcpyHostToDevice);
    if (cuda_ret != cudaSuccess)
      FATAL("Unable to copy to padded inout");

    // Initialize the padded elements to 0
    cuda_ret = cudaMemset(paddedInout + in_size, 0,
                          (paddedSize - in_size) * sizeof(float));
    if (cuda_ret != cudaSuccess)
      FATAL("Unable to initialize padded elements");

    real_inout = paddedInout; // Use the padded array
    real_size = paddedSize;
  }

  // print the padded size
  printf("Using size: %d\n", in_size);

  if (num_blocks > 1) {
    cuda_ret = cudaMalloc((void **)&sum, num_blocks * sizeof(float));
    if (cuda_ret != cudaSuccess)
      FATAL("Unable to allocate device memory");

    preScanKernel<<<dim_grid, dim_block>>>(inout, real_size, sum);
    // preScan(sum, num_blocks);
    // addKernel<<<dim_grid, dim_block>>>(inout, sum, in_size);

    cudaFree(sum);
  } else {
    preScanKernel<<<dim_grid, dim_block>>>(real_inout, real_size, NULL);
  }

  if (paddedSize != in_size) {
    printf("Copying back\n");
    // copy data from padded input to original input
    cuda_ret = cudaMemcpy(inout, paddedInout, in_size * sizeof(float),
                          cudaMemcpyDeviceToDevice);

    if (cuda_ret != cudaSuccess) {
      fprintf(stderr, "GPUassert: %s\n", cudaGetErrorString(cuda_ret));
      FATAL("Unable to copy from padded inout");
    }

    cuda_ret = cudaFree(paddedInout);
    if (cuda_ret != cudaSuccess)
      FATAL("Unable to free padded inout");
  }
}
