/* select- 
*         0: input in the first half, output in the second half
*         1: input in the second half, output in the first half
*/
 
__global__ void GameofLife(int *GPUgrid,  int select, int width, int height){
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int blockDimX = blockDim.x; // Block dimension, assume it's the same in the y-direction
    
    // Calculate global indices
    int xIndex = bx * blockDimX + tx;
    int yIndex = by * blockDimX + ty;

    // Linearized index for global and shared memory
    int globalIdx = yIndex * width + xIndex;

    // Dynamically allocate shared memory
    extern __shared__ int shared_mem[];
    
    // Read input data into shared memory
    if(xIndex < width && yIndex < height) {
        shared_mem[ty * blockDimX + tx] = GPUgrid[globalIdx + select * width * height];
    }
    __syncthreads(); // Make sure all threads have loaded their data before proceeding
    
    int live_neighbors = 0;
    
    // Compute the number of live neighbors
    for(int dx = -1; dx <= 1; dx++) {
        for(int dy = -1; dy <= 1; dy++) {
            int neighborX = tx + dx;
            int neighborY = ty + dy;
            if(neighborX >= 0 && neighborX < blockDimX && neighborY >= 0 && neighborY < blockDimX) {
                live_neighbors += shared_mem[neighborY * blockDimX + neighborX];
            }
        }
    }
    
    // Subtract the current cell value; it was counted as a live neighbor
    live_neighbors -= shared_mem[ty * blockDimX + tx];
    
    int output;
    if(shared_mem[ty * blockDimX + tx] == 1) { // If the cell is alive
        if(live_neighbors < 2 || live_neighbors > 3) {
            output = 0;
        } else {
            output = 1;
        }
    } else { // If the cell is dead
        if(live_neighbors == 3) {
            output = 1;
        } else {
            output = 0;
        }
    }
    
    // Write output back to global memory
    if(xIndex < width && yIndex < height) {
        GPUgrid[globalIdx + (1 - select) * width * height] = output;
    }
}

