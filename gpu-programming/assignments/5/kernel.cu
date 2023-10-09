/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#define EXT_BLOCK_SIZE (BLOCK_SIZE + FILTER_SIZE - 1)

__constant__ float M_c[FILTER_SIZE][FILTER_SIZE];

__global__ void convolution(Matrix N, Matrix P)
{
    /********************************************************************
	Determine input and output indexes of each thread
	Load a tile of the input image to shared memory
	Apply the filter on the input image tile
	Write the compute values to the output image at the correct indexes
	********************************************************************/

        // Adjusted shared memory declaration to include halo
    __shared__ float N_s[EXT_BLOCK_SIZE][EXT_BLOCK_SIZE];

    // Global indices
    int globalRow = blockIdx.y * blockDim.y + threadIdx.y;
    int globalCol = blockIdx.x * blockDim.x + threadIdx.x;

    // Local indices (within the tile including halo)
    int localRow = threadIdx.y + FILTER_SIZE / 2;
    int localCol = threadIdx.x + FILTER_SIZE / 2;

    // Load tile into shared memory including halo cells
    for (int i = 0; i < FILTER_SIZE; ++i) {
        for (int j = 0; j < FILTER_SIZE; ++j) {
            int rowOffset = localRow + i - FILTER_SIZE / 2;
            int colOffset = localCol + j - FILTER_SIZE / 2;
            if (rowOffset >= 0 && rowOffset < (BLOCK_SIZE + FILTER_SIZE - 1) && colOffset >= 0 && colOffset < (BLOCK_SIZE + FILTER_SIZE - 1)) {
                int globalRowOffset = globalRow + i - FILTER_SIZE / 2;
                int globalColOffset = globalCol + j - FILTER_SIZE / 2;
                if (globalRowOffset >= 0 && globalRowOffset < N.height && globalColOffset >= 0 && globalColOffset < N.width) {
                    N_s[rowOffset][colOffset] = N.elements[globalRowOffset * N.pitch + globalColOffset];
                } else {
                    N_s[rowOffset][colOffset] = 0.0f;
                }
            }
        }
    }
    __syncthreads();  // Ensure the tile is loaded

    // Apply convolution filter
    float output = 0.0f;
    if(globalRow < P.height && globalCol < P.width) {
        for(int i = 0; i < FILTER_SIZE; ++i) {
            for(int j = 0; j < FILTER_SIZE; ++j) {
                int rowOffset = threadIdx.y + i - FILTER_SIZE / 2 + FILTER_SIZE / 2;
                int colOffset = threadIdx.x + j - FILTER_SIZE / 2 + FILTER_SIZE / 2;
                output += N_s[rowOffset][colOffset] * M_c[i][j];
            }
        }
        P.elements[globalRow * P.pitch + globalCol] = output;
    }
}






