/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/
#define BLOCK_SIZE 512

__global__ void reduction(float *out, float *in, unsigned size) {
  // Declare shared memory
  __shared__ float sdata[BLOCK_SIZE << 1];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (BLOCK_SIZE << 1) + threadIdx.x;

  // Load a segment of the input vector into shared memory
  sdata[tid] = (i < size) ? in[i] : 0;
  sdata[tid + BLOCK_SIZE] = (i + BLOCK_SIZE) < size ? in[i + BLOCK_SIZE] : 0;

  // Synchronize to make sure data is loaded into shared memory
  __syncthreads();

  // Traverse the reduction tree
  for (unsigned int s = BLOCK_SIZE; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    // Synchronize to make sure all threads see the same shared memory
    __syncthreads();
  }

  // Write the computed sum to the output vector at the correct index
  if (tid == 0) {
    out[blockIdx.x] = sdata[0];
  }
}
