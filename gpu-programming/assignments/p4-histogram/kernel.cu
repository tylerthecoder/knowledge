/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

// Define your kernels in this file you may use more than one kernel if you
// need to

__global__ void histogram_kernel(unsigned int* input, unsigned int* bins,
    unsigned int num_elements, unsigned int num_bins) {

        
    // Shared memory for local histogram
    extern __shared__ unsigned int local_bins[];

    // Initialize local histogram
    for (unsigned int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        local_bins[i] = 0;
    }
    __syncthreads();

    // Compute local histogram
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    while (idx < num_elements) {
        unsigned int binIndex = input[idx];
        if (binIndex < num_bins) {
            atomicAdd(&(local_bins[binIndex]), 1);
        }
        idx += blockDim.x * gridDim.x;  // Stride over grid
    }
    __syncthreads();

    // Merge local histogram into global histogram
    for (unsigned int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        atomicAdd(&(bins[i]), local_bins[i]);
    }
}

__global__ void convert_kernel(unsigned int *bins32, uint8_t *bins8, unsigned int num_bins) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < num_bins) {
        bins8[idx] = min(bins32[idx], 255U);  // Ensuring saturation at 255
    }
}


/******************************************************************************
Setup and invoke your kernel(s) in this function. You may also allocate more
GPU memory if you need to
*******************************************************************************/
void histogram(unsigned int* input, uint8_t* bins, unsigned int num_elements,
        unsigned int num_bins) {

    cudaError_t cuda_ret;

    // Create 32 bit bins
    unsigned int *bins32;
    cuda_ret = cudaMalloc((void**)&bins32, num_bins * sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) {
		printf("Unable to allocate device memory");
		exit(-1);
	}

    cuda_ret = cudaMemset(bins32, 0, num_bins * sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) {
		printf("Unable to set device memory");
		exit(-1);
	}


    // Launch histogram kernel using 32-bit bins
    dim3 dim_grid, dim_block;
    dim_block.x = 512; dim_block.y = dim_block.z = 1;
    dim_grid.x = 30; dim_grid.y = dim_grid.z = 1;
    histogram_kernel<<<dim_grid, dim_block, num_bins*sizeof(unsigned int)>>>
        (input, bins32, num_elements, num_bins);

    // Convert 32-bit bins into 8-bit bins
    dim_block.x = 512;
    dim_grid.x = (num_bins - 1)/dim_block.x + 1;
    convert_kernel<<<dim_grid, dim_block>>>(bins32, bins, num_bins);

    // Free allocated device memory
    cuda_ret = cudaFree(bins32);
	if(cuda_ret != cudaSuccess) {
		printf("Unable free CUDA memory");
		exit(-1);
	}

}


