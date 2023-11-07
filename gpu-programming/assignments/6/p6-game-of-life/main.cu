#include <stdio.h>
#include <iostream>
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

    cudaError_t cuda_ret;
    int  height, width;
    dim3 dim_grid, dim_block;

	/* Read image dimensions */
    if (argc == 1) {
        height = 1400;
	    width =1400;
    } else if (argc == 2) {
        height= atoi(argv[1]);
	width= atoi(argv[1]);
    } else if (argc == 3) {
        height = atoi(argv[1]);
        width = atoi(argv[2]);
    }else {
        printf("\n    Invalid input parameters!"
           "\n    Usage: ./gameoflife          # Matrix is 1400 x 1400"
           "\n    Usage: ./gameoflife <m>      # Matrix is m x m"
	   "\n    Usage: ./gameoflife <m> <n>  # Matrix is m x n"
           "\n");
        exit(0);
    }



	/* Allocate host memory */
	int *grid=new int [height*width*2];
	int *Ggrid_result=new int [height*width*2];
	/* Initialize Matrix */
	InitialGrid(grid,height,width);
	GiveLife(0,height*width/2,grid,height,width);


    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    printf("\nThe size of the universe is %d x %d.\n\n", height, width);

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);
    int *GPUgrid;

    long long int size=sizeof(int)*2*width*height;
    cuda_ret = (cudaMalloc((void**) &GPUgrid, size));
    if(cuda_ret != cudaSuccess) {
		printf("Unable to allocate GPU global memory");
		exit(-1);
	}

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device ------------------------------------------

    printf("Copying data from host to device..."); fflush(stdout);
    startTime(&timer);

    cuda_ret =(cudaMemcpy(GPUgrid,grid,size,cudaMemcpyHostToDevice));
    if(cuda_ret != cudaSuccess) {
		printf("Unable to copy to GPU memory");
		exit(-1);
	}

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Launch kernel ----------------------------------------------------------
    printf("Launching kernel..."); fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE

    // Just before launching the kernel
    dim_block = dim3(16, 16);  // For example, blocks of 16x16 threads
    dim_grid = dim3((width + dim_block.x - 1)/dim_block.x, (height + dim_block.y - 1)/dim_block.y);

    // INSERT CODE ABOVE
	int select =0;
	for(int m=0;m<ITERATION;m++){
        	GameofLife<<<dim_grid, dim_block>>>(GPUgrid,select,width,height);
         	select=1-select;
        }
	cuda_ret = cudaDeviceSynchronize();
	if(cuda_ret != cudaSuccess) {
		printf("Unable to launch/execute kernel");
		exit(-1);
	}

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy device variables from host ----------------------------------------

    printf("Copying data from device to host...\n"); fflush(stdout);
    startTime(&timer);

    cudaMemcpy(Ggrid_result,GPUgrid,size,cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

   //CPU -----------------------------------------------------------------------

	int nowGrid=0;
        for(int n=0;n<ITERATION;n++)
        {
		GameofLife_CPU( grid, width, height,nowGrid);
		nowGrid=1-nowGrid;

      	}

// Verify correctness -----------------------------------------------------
	printf("Verifying..."); fflush(stdout);
	verify(Ggrid_result,grid,height,width);

// Free memory ------------------------------------------------------------

    cuda_ret = cudaFree(GPUgrid);
    if(cuda_ret != cudaSuccess) {
		printf("Unable free cuda memory");
		exit(-1);
	}

	 delete [] grid;
	 delete [] Ggrid_result;
	 return 0;
}
