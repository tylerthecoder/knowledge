/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>
#include <stdint.h>
#include <time.h>


#include "support.h"
#include "kernel.cu"

int main(int argc, char* argv[])
{
    Timer timer;
    time_t t;

    /* Intializes random number generator */
    srand((unsigned) time(&t));

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

    unsigned int *in_h;
    uint8_t* bins_h;
    unsigned int *in_d;
    uint8_t* bins_d;
    unsigned int num_elements, num_bins;
    cudaError_t cuda_ret;

    if(argc == 1) {
        num_elements = 1000000;
        num_bins = 4096;
    } else if(argc == 2) {
        num_elements = atoi(argv[1]);
        num_bins = 4096;
    } else if(argc == 3) {
        num_elements = atoi(argv[1]);
        num_bins = atoi(argv[2]);
    } else {
        printf("\n    Invalid input parameters!"
           "\n    Usage: ./histogram            # Input: 1,000,000, Bins: 4,096"
           "\n    Usage: ./histogram <m>        # Input: m, Bins: 4,096"
           "\n    Usage: ./histogram <m> <n>    # Input: m, Bins: n"
           "\n");
        exit(0);
    }
    initVector(&in_h, num_elements, num_bins);
    bins_h = (uint8_t*) malloc(num_bins*sizeof(uint8_t));

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    Input size = %u\n    Number of bins = %u\n", num_elements,
        num_bins);

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

    cuda_ret = cudaMalloc((void**)&in_d, num_elements * sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) {
		printf("Unable to allocate device memory");
		exit(-1);
	}
    cuda_ret = cudaMalloc((void**)&bins_d, num_bins * sizeof(uint8_t));
    if(cuda_ret != cudaSuccess) {
		printf("Unable to allocate device memory");
		exit(-1);
	}

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device ------------------------------------------

    printf("Copying data from host to device..."); fflush(stdout);
    startTime(&timer);

    cuda_ret = cudaMemcpy(in_d, in_h, num_elements * sizeof(unsigned int), cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) {
		printf("Unable to copy memory to the device");
		exit(-1);
	}

    cuda_ret = cudaMemset(bins_d, 0, num_bins * sizeof(uint8_t));
    if(cuda_ret != cudaSuccess) {
		printf("Unable to set device memory");
		exit(-1);
	}

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Launch kernel ----------------------------------------------------------
    printf("Launching kernel..."); fflush(stdout);
    startTime(&timer);

    histogram(in_d, bins_d, num_elements, num_bins);
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) {
		printf("Unable to launch/execute kernel");
		exit(-1);
	}

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy device variables from host ----------------------------------------

    printf("Copying data from device to host..."); fflush(stdout);
    startTime(&timer);

    cuda_ret = cudaMemcpy(bins_h, bins_d, num_bins * sizeof(uint8_t),
        cudaMemcpyDeviceToHost);
	if(cuda_ret != cudaSuccess) {
		printf("Unable to copy memory to host");
		exit(-1);
	}

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Verify correctness -----------------------------------------------------

    printf("Verifying results..."); fflush(stdout);

    verify(in_h, bins_h, num_elements, num_bins);

    // Free memory ------------------------------------------------------------

    cuda_ret = cudaFree(in_d);
	if(cuda_ret != cudaSuccess) {
		printf("Unable free CUDA memory");
		exit(-1);
	}

    cuda_ret = cudaFree(bins_d);
	if(cuda_ret != cudaSuccess) {
		printf("Unable free CUDA memory");
		exit(-1);
	}

    free(in_h); free(bins_h);

    return 0;
}

