/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>

#include "support.h"
#include "kernel.cu"

#include <time.h>

int main(int argc, char* argv[])
{
    Timer timer;

    time_t t;

    /* Intializes random number generator */
    srand((unsigned) time(&t));

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

	unsigned int *in_h, *out_h;
	unsigned int *inout_d;
	unsigned int num_elements;
	cudaError_t cuda_ret;

	/* Allocate and initialize input vector */
    if(argc == 1) {
        num_elements = 1000000;
    } else if(argc == 2) {
        num_elements = atoi(argv[1]);
    } else {
        printf("\n    Invalid input parameters!"
           "\n    Usage: ./radix-sort        # Input of size 1,000,000 is used"
           "\n    Usage: ./radix-sort <m>    # Input of size m is used"
           "\n");
        exit(0);
    }
    initVector(&in_h, num_elements);


	// Allocate and initialize output vector
    out_h = (unsigned int*)malloc(num_elements*sizeof(unsigned int));
	if(out_h == NULL) FATAL("Unable to allocate host");


    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    Input size = %u\n", num_elements);


    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

	cuda_ret = cudaMalloc((void**)&inout_d, num_elements*sizeof(unsigned int));
	if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device ------------------------------------------

    printf("Copying data from host to device..."); fflush(stdout);
    startTime(&timer);

    cuda_ret = cudaMemcpy(inout_d, in_h, num_elements*sizeof(unsigned int),
        cudaMemcpyHostToDevice);
	if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");


    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Launch kernel ----------------------------------------------------------
    printf("Launching kernel..."); fflush(stdout);
    startTime(&timer);

    radixSort(inout_d, num_elements);

	cuda_ret = cudaDeviceSynchronize();
	if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel");

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));


    // Copy device variables to host ----------------------------------------

    printf("Copying data from device to host..."); fflush(stdout);
    startTime(&timer);

    cuda_ret = cudaMemcpy(out_h, inout_d, num_elements*sizeof(unsigned int),
        cudaMemcpyDeviceToHost);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to host");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));


    // Verify correctness -----------------------------------------------------

    printf("Verifying results..."); fflush(stdout);

    verify(in_h, out_h, num_elements);



    // Free memory ------------------------------------------------------------

    cuda_ret = cudaFree(inout_d);
	if(cuda_ret != cudaSuccess) FATAL("Unable free CUDA memory");


	free(in_h); free(out_h);

	return 0;
}

