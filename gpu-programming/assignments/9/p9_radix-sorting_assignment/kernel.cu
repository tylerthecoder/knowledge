/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#define BLOCK_SIZE 1024

// run per block inclusive scan and save the partial sums.
__global__ void ScanKernel(unsigned int *array, unsigned int size,
                           unsigned int *index0, unsigned int *index1,
                           unsigned int *sum0, unsigned int *sum1,
                           unsigned int bitPos) {
  __shared__ unsigned int temp0[BLOCK_SIZE];
  __shared__ unsigned int temp1[BLOCK_SIZE];
  int thid = threadIdx.x;

  // Collaborative loading the value of the bit at location bitPos from array.
  // The value needs to be inverted and saved into shared memory. This is for
  // calculating index0, the positions of those elements in array whose bit
  // value at location bitPos is 0.
  int ai = thid;
  int global_ai = blockIdx.x * blockDim.x + ai;
  temp0[ai] = (global_ai < size) ? !(array[global_ai] >> bitPos & 1) : 0;

  // Run inclusive scan on index0 using naive approach
  for (int d = 1; d < blockDim.x; d *= 2) {
    __syncthreads();
    int t = (ai >= d) ? temp0[ai - d] : 0;
    __syncthreads();
    temp0[ai] += t;
  }
  __syncthreads();

  // Write back sum and scan result for index0
  if (global_ai < size) {
    index0[global_ai] = temp0[ai];
  }
  if (ai == blockDim.x - 1) {
    sum0[blockIdx.x] = temp0[ai];
  }

  // Collaborative loading the value of the bit at location bitPos from array.
  // This is for calculating index1, the positions of those elements in array
  // whose bit value at location bitPos is 1.
  temp1[ai] = (global_ai < size) ? (array[global_ai] >> bitPos & 1) : 0;

  // Run inclusive reverse scan on index1 using naive approach
  for (int d = 1; d < blockDim.x; d *= 2) {
    __syncthreads();
    int t = (ai < blockDim.x - d) ? temp1[ai + d] : 0;
    __syncthreads();
    temp1[ai] += t;
  }
  __syncthreads();

  // Write back sum and scan result for index1
  if (global_ai < size) {
    index1[global_ai] = temp1[ai];
  }
  if (ai == 0) {
    sum1[blockIdx.x] = temp1[ai];
  }
}

// convert local scan to global scan; for index0, further convert inclusive scan
// to exclusive scan
__global__ void indexKernel(unsigned int *index0, unsigned int *sum0,
                            unsigned int *index1, unsigned int *sum1,
                            unsigned int size) {
  unsigned int gtx = blockIdx.x * BLOCK_SIZE + threadIdx.x;

  // use the scan of partial sums to update block items of index0
  if (gtx < size && blockIdx.x > 0) {
    index0[gtx] += sum0[blockIdx.x - 1];
  }

  // use the scan of partial sums to update block items of index1
  if (gtx < size && blockIdx.x < blockDim.x - 1) {
    index1[gtx] += sum1[blockIdx.x + 1];
  }

  __syncthreads();

  // subtract 1 from index0 to get actual indices, i.e., convert inclusive scan
  // to exclusive scan
  if (gtx < size) {
    index0[gtx] -= 1;
  }

  // subtract index1 from size to get actual indices
  if (gtx < size) {
    index1[gtx] = size - index1[gtx];
  }
}

// global permuation
__global__ void permuteKernel(unsigned int *array, unsigned int *out,
                              unsigned int size, unsigned int *index0,
                              unsigned int *index1, unsigned int bitPos) {

  unsigned int gtx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if (gtx < size) {
    unsigned int bit = (array[gtx] >> bitPos) & 1;
    unsigned int index = bit ? index1[gtx] : index0[gtx];
    out[index] = array[gtx];
  }
}

/******************************************************************************
Setup and invoke your kernel(s) in this function. You may also allocate more
GPU memory if you need to
*******************************************************************************/
void radixSort(unsigned int *array, unsigned int num_elements) {

  unsigned int num_blocks;
  cudaError_t cuda_ret;
  dim3 dim_grid, dim_block;

  // Allocate more variables for the algorithm
  unsigned int *buf_d, *temp_ptr, *buf_h; // for double buffering array

  cuda_ret = cudaMalloc((void **)&buf_d, num_elements * sizeof(unsigned int));
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to allocate device memory");

  unsigned int *sum0_h, *sum1_h;
  unsigned int *index0_d, *index1_d, *sum0_d, *sum1_d, *index0_h, *index1_h;

  cuda_ret =
      cudaMalloc((void **)&index0_d, num_elements * sizeof(unsigned int));
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to allocate device memory");
  cuda_ret =
      cudaMalloc((void **)&index1_d, num_elements * sizeof(unsigned int));
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to allocate device memory");

  // Compute thread grid
  num_blocks = num_elements / BLOCK_SIZE;
  if (num_elements % BLOCK_SIZE != 0)
    num_blocks++;

  dim_block.x = BLOCK_SIZE;
  dim_block.y = 1;
  dim_block.z = 1;
  dim_grid.x = num_blocks;
  dim_grid.y = 1;
  dim_grid.z = 1;

  // allocate space for partial sums
  sum0_h = (unsigned int *)malloc(num_blocks * sizeof(unsigned int));
  sum1_h = (unsigned int *)malloc(num_blocks * sizeof(unsigned int));

  index0_h = (unsigned int *)malloc(num_elements * sizeof(unsigned int));
  index1_h = (unsigned int *)malloc(num_elements * sizeof(unsigned int));

  buf_h = (unsigned int *)malloc(num_elements * sizeof(unsigned int));

  cuda_ret = cudaMalloc((void **)&sum0_d, num_blocks * sizeof(unsigned int));
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to allocate device memory");
  cuda_ret = cudaMalloc((void **)&sum1_d, num_blocks * sizeof(unsigned int));
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to allocate device memory");

  // perform radix-sort for bitPos 0 to 31
  for (int bitPos = 0; bitPos < 32; bitPos++) {
    // Launch Scan kernel
    ScanKernel<<<dim_grid, dim_block>>>(array, num_elements, index0_d, index1_d,
                                        sum0_d, sum1_d, bitPos);
    cuda_ret = cudaDeviceSynchronize();
    if (cuda_ret != cudaSuccess)
      FATAL("Unable to launch kernel");

    // copy sums for host-side scan
    cuda_ret = cudaMemcpy(sum0_h, sum0_d, num_blocks * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost);
    if (cuda_ret != cudaSuccess)
      FATAL("Unable to copy memory to host");

    cuda_ret = cudaMemcpy(sum1_h, sum1_d, num_blocks * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost);
    if (cuda_ret != cudaSuccess)
      FATAL("Unable to copy memory to host");

    cuda_ret = cudaDeviceSynchronize();
    if (cuda_ret != cudaSuccess)
      FATAL("Unable to copy sums from device to host");

    // copy index0 and index1 to host
    cuda_ret =
        cudaMemcpy(index0_h, index0_d, num_elements * sizeof(unsigned int),
                   cudaMemcpyDeviceToHost);
    if (cuda_ret != cudaSuccess)
      FATAL("Unable to copy memory to host");

    cuda_ret =
        cudaMemcpy(index1_h, index1_d, num_elements * sizeof(unsigned int),
                   cudaMemcpyDeviceToHost);
    if (cuda_ret != cudaSuccess)
      FATAL("Unable to copy memory to host");

    // Print values
    // printf("\n\n Sums from GPU:");
    // for (unsigned int i = 0; i < num_blocks; i++) {
    //   printf("\n 0: %u | 1: %u", sum0_h[i], sum1_h[i]);
    // }

    // Perform scan on sums and copy the result to GPU
    for (unsigned int i = 1; i < num_blocks; i++)
      sum0_h[i] += sum0_h[i - 1];
    for (unsigned int i = num_blocks - 1; i > 0; i--)
      sum1_h[i - 1] += sum1_h[i];

    // Values after scan
    // printf("\n\n Sums from GPU after scan:");
    // for (unsigned int i = 0; i < num_blocks; i++) {
    //   printf("\n 0: %u | 1: %u", sum0_h[i], sum1_h[i]);
    // }

    // Print Indices
    // if (bitPos < 3) {
    //   printf("\n\n Indices before calc:");
    //   for (unsigned int i = 0; i < num_elements; i++) {
    //     printf("\n 0: %u | 1: %u", index0_h[i], index1_h[i]);
    //   }
    // }

    cuda_ret = cudaMemcpy(sum0_d, sum0_h, num_blocks * sizeof(unsigned int),
                          cudaMemcpyHostToDevice);
    if (cuda_ret != cudaSuccess)
      FATAL("Unable to copy memory to host");

    cuda_ret = cudaMemcpy(sum1_d, sum1_h, num_blocks * sizeof(unsigned int),
                          cudaMemcpyHostToDevice);
    if (cuda_ret != cudaSuccess)
      FATAL("Unable to copy memory to host");

    cuda_ret = cudaDeviceSynchronize();
    if (cuda_ret != cudaSuccess)
      FATAL("Unable to copy sums from host to device");

    // Launch index computation kernel
    indexKernel<<<dim_grid, dim_block>>>(index0_d, sum0_d, index1_d, sum1_d,
                                         num_elements);
    cuda_ret = cudaDeviceSynchronize();
    if (cuda_ret != cudaSuccess)
      FATAL("Unable to launch kernel");

    // Copy the data from the index0_d and index1_d to the array
    cuda_ret =
        cudaMemcpy(index0_h, index0_d, num_elements * sizeof(unsigned int),
                   cudaMemcpyDeviceToHost);
    if (cuda_ret != cudaSuccess)
      FATAL("Unable to copy memory to host");

    cuda_ret =
        cudaMemcpy(index1_h, index1_d, num_elements * sizeof(unsigned int),
                   cudaMemcpyDeviceToHost);
    if (cuda_ret != cudaSuccess)
      FATAL("Unable to copy memory to host");

    // Print the values
    // if (bitPos < 3) {
    //   printf("\n\n Values from GPU after index computation:");
    //   for (unsigned int i = 0; i < num_elements; i++) {
    //     printf("\n 0: %u | 1: %u", index0_h[i], index1_h[i]);
    //   }
    // }

    // Launch permutation kernel
    permuteKernel<<<dim_grid, dim_block>>>(array, buf_d, num_elements, index0_d,
                                           index1_d, bitPos);
    cuda_ret = cudaDeviceSynchronize();
    if (cuda_ret != cudaSuccess)
      FATAL("Unable to launch kernel");

    // Copy the data from the buf_d to the buf_h
    cuda_ret = cudaMemcpy(buf_h, buf_d, num_elements * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost);
    if (cuda_ret != cudaSuccess)
      FATAL("Unable to copy memory to host");

    // Print the values
    // if (bitPos < 3) {
    //   printf("\n\n Values from GPU after permutation:");
    //   for (unsigned int i = 0; i < num_elements; i++) {
    //     printf("\n %u", buf_h[i]);
    //   }
    // }

    // swap pointers for next iteration
    temp_ptr = array;
    array = buf_d;
    buf_d = temp_ptr;
  }

  // release device memory
  cuda_ret = cudaFree(buf_d);
  if (cuda_ret != cudaSuccess)
    FATAL("Unable free CUDA memory buf_d");

  cuda_ret = cudaFree(index0_d);
  if (cuda_ret != cudaSuccess)
    FATAL("Unable free CUDA memory index0_d");
  cuda_ret = cudaFree(index1_d);
  if (cuda_ret != cudaSuccess)
    FATAL("Unable free CUDA memory index1_d");

  cuda_ret = cudaFree(sum0_d);
  if (cuda_ret != cudaSuccess)
    FATAL("Unable free CUDA memory sum0_d");
  cuda_ret = cudaFree(sum1_d);
  if (cuda_ret != cudaSuccess)
    FATAL("Unable free CUDA memory sum1_d");

  // realease host memory
  free(sum0_h);
  free(sum1_h);
}
