# GPU Programming

Cuda is a set of apis

Empty apple basket > no apple basket

CPU & GPU memory bandwidth matters

CPU 
- Multiple ALUs
- Control logic
- Big Cache


GPU 
- Many simple cores
- Simple control logic
- No or small cache 

GPU has much more registers (32 bit words)
MIPS has 32, Fermi GPU has 32K 

GPUs register file is big for context switching

different stream processor has different precision (32, 64)

GPUs don't halt tasks like CPUs

Each thread has its own core it runs on

We want threads to do different things so they don't repeat work.

Matrix multiplication is a good example of a problem that can be parallelized.



## Lecture 2 (CUDA Basics)

Nvidia invented GPU around 2000

G80 -> GT200 -> Fermi -> Kepler -> Maxwell -> Pascal -> Volta

The hello world for GPUs is matrix multiplication

$P = M x N$

The approach with c is
```c
for (i = 0; i < WIDTH; i++) {
    for (j = 0; j < HEIGHT; j++) {
        P[i][j] += A[i][k] * B[k][j];
    }
}
```


The approach with CUDA is straight forward

- one thread calculates one element of P
- Issue M x N threads simultaneously

CPU is connected to GPU through PCI Bus

CPU copies data to GPU memory

CPU can be called host, 
GPU can be called device

There is a thread grid of thread blocks

You can't schedule threads on a GPU, you schedule a thread block

Each thread block has an id

All threads in a thread block run the same code, just with different data

Each thread has a thread id

$blockIdx.x, blockIdx.y, blockIdx.z$ are the block ids
$threadIdx.x, threadIdx.y, threadIdx.z$ are the thread ids



CPU memory is different than GPU memeory. 

cuda has its own malloc `cudaMalloc`. Need to free memory as well

When scheduling threads, you shoudl keep the thread blocks smaller. This way, you can run the most threads at a time. 


## Matrix Multiplication

Rows need to match the column number



We have matrix P [ M x N ]

Each element in M is going to be read N times

It is better to "share" inside of the thread block. So you compute the product of the matrix blocks at a time. Each block reads the entire row




Types of variables

local: each thread has its own copy 
- Normally in a register
shared: each block has its own copy
- Shared memory is tiny compared to local memory
- typically at the kilobyte


There is no communication between thread blocks. 


