/* 
Author: Rudra Goel, Jackie Mac Hale
Class: ECE4122 A, ECE6122 A
Last Date Modified: 12/2/2025

Description: 

This program performs matrix multiplication using three different methods on very
large matrices to time the latencies for the specified method. The three possible
methods are using the CPU, doing naive matrix multiplication with CUDA, and doing
efficient tiling with CUDA using the shared memory buffer. For the matrix dimensions
of 1024x1024 square matrices, using the CPU is the slowest method, doing naive
matrix multiplication with CUDA is the second fastest, and the efficient tiling
method with CUDA is the fastest.

*/

#include <stdio.h>
#include "util.h"

//these two need to match
#define A_WIDTH     1024
#define B_HEIGHT    1024

#define A_HEIGHT    1024
#define B_WIDTH     1024


#define C_WIDTH B_WIDTH
#define C_HEIGHT A_HEIGHT

#define BLOCK_SIZE 8
#define NUM_SUBS (A_WIDTH / BLOCK_SIZE)


// device-side arrays
__device__ float dA[A_HEIGHT][A_WIDTH];
__device__ float dB[B_HEIGHT][B_WIDTH];
__device__ float dC[C_HEIGHT][C_WIDTH];

// host-side arrays
float hA[A_HEIGHT][A_WIDTH];
float hB[B_HEIGHT][B_WIDTH];
float hC[C_HEIGHT][C_WIDTH];
float hCRef[C_HEIGHT][C_WIDTH];

int matrixMulTest(float C[C_HEIGHT][C_WIDTH], float Cref[C_HEIGHT][C_WIDTH]);

/*
    This is the CPU-based matrix multiply.
    It calculates output matrix C, from the input matrices A and B.
*/
void matrixMulCPU(float A[A_HEIGHT][A_WIDTH], float B[C_HEIGHT][C_WIDTH], float C[C_HEIGHT][C_WIDTH])
{
    int x, y, k;
    for (y = 0; y < C_HEIGHT; y++)
    {
        for (x = 0; x < C_WIDTH; x++)
        {
            C[y][x] = 0;
            for (k = 0; k < A_WIDTH; k++)
            {
                C[y][x] += A[y][k] * B[k][x];
            }
        }
    }

}


/*
    This is a GPU-based matrix multiply.
    It calculates output matrix dC, from the input matrices dA and dB.
*/
__global__ void matrixMulCUDA()
{
    // TODO implement simple CUDA matrix multiply here
    // inputs: dA, dB (global variables)
    // output: dC (global variable)
    // do not use shared memory
    // note the launch parameters: this kernel is called for each
    //         cell in the output matrix

    //figure out the thread index in x and y directions
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    float total = 0;

    //loop through the B_HEIGHT or A_WIDTH (either works since they have to match for matrices to multiply)
    for(int i = 0; i < B_HEIGHT; i++)
    {
        total += dA[y][i] * dB[i][x];
    }
    dC[y][x] = total;

    //if this block is on the ending edge of the grid
    //need the threads in this block to compute the output matrix value for the cells
    if (blockIdx.x == gridDim.x - 1)
    {
        //increment the current thread index's X position by the size of the block dim
        int colNew = x + blockDim.x;
        //check to make sure the new x index is still within bounds
        if (colNew < C_WIDTH)
        {
            total = 0;
            for (int i = 0; i < B_HEIGHT; i++)
            {
                total += dA[y][i] * dB[i][colNew];
            }
            dC[y][colNew] = total;
        }
    }
    if (blockIdx.y == gridDim.y - 1)
    {
        
        int rowNew = y + blockDim.y;
        
        if (rowNew < C_HEIGHT)
        {
            total = 0;
            for(int i = 0; i < B_HEIGHT; i++)
            {
                total += dA[rowNew][i] * dB[i][x];
            }
            dC[rowNew][x] = total;
        }
    }
    
    //force the corner block to handle the cells not yet covered in the corner
    if (blockIdx.x == gridDim.x - 1 && blockIdx.y == gridDim.y - 1)
    {
        int colNew = x + blockDim.x;
        int rowNew = y + blockDim.y;

        if (rowNew < C_HEIGHT && colNew < C_WIDTH)
        {
            for (int i = 0; i < B_HEIGHT; i++)
            {
                dC[rowNew][colNew] += dA[rowNew][i] * dB[i][colNew];
            }
        }

    }

}

/*
    This is a GPU-based matrix multiply.
    It calculates output matrix dC, from the input matrices dA and dB.
    It uses shared memory.
*/
__global__ void matrixMulCUDATiled()
{
    // TODO implement tiled CUDA matrix multiply here
    // inputs: dA, dB (global variables)
    // output: dC (global variable)
    // use tiled shared memory as described in the assignment
    // note the launch parameters: this kernel is called for each
    //         cell in the output matrix

    //allocate shared memory for the chunks of A and B statically
    //size of each is BLOCK_SIZE * BLOCK_SIZE * 4 bytes
    __shared__ float chunkABuffer[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float chunkBBuffer[BLOCK_SIZE][BLOCK_SIZE];

    //this is a buffer for copying values from the edges of matricesd A and B
    //that are not encompassed by another block due to insufficuent blocks being launched by the kernel
    __shared__ float edgeCellsABuffer[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float edgeCellsBBuffer[BLOCK_SIZE][BLOCK_SIZE];

    //figure out what thread I currently am in the overall matrix C
    int globalX = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int globalY = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    int globalEdgeX = globalX + BLOCK_SIZE;
    int globalEdgeY = globalY + BLOCK_SIZE;

    bool remainingCellsX = C_WIDTH % BLOCK_SIZE != 0;
    bool remainingCellsY = C_HEIGHT % BLOCK_SIZE != 0;


    /**
        Main Idea
            need to copy in chunks (tiles) of A and B into shared memory
            then need to perform the partial sum of products
            we loop through this for A_WIDTH / BLOCK_SIZE amount of times or B_HEIGHT / BLOCK_SIZE times

    */

    float total = 0;

    float totalXOverflow = 0;
    float totalYOverflow = 0;
    float totalCornerOverflow = 0;


    //guard against case where the A_WIDTH does not divide evenly with BLOCK_SIZE
    //so must go A_width + 1 / block_size and guard against threads that are outside of the dims
    for (int tileIteration = 0; tileIteration < (A_WIDTH / BLOCK_SIZE) + 1; tileIteration++)
    {

        //these two vars are used for copying the right data from inputs A and B into shmem
        //based on the loop iteration, offset the x dim of the thread 
        int xTiledOffset = tileIteration * BLOCK_SIZE + threadIdx.x;
        //based on the loop iteration, offset the y dim of the thread 
        int yTiledOffset = tileIteration * BLOCK_SIZE + threadIdx.y;
        

        int bufferEnd = tileIteration == A_WIDTH / BLOCK_SIZE ? A_WIDTH % BLOCK_SIZE : BLOCK_SIZE;

        if (xTiledOffset < A_WIDTH) {
            //have the thread within the currentn block copy data into the A and B shared mem buffer
            chunkABuffer[threadIdx.y][threadIdx.x] = dA[globalY][xTiledOffset];
        }

        if (yTiledOffset < B_HEIGHT) {
            chunkBBuffer[threadIdx.y][threadIdx.x] = dB[yTiledOffset][globalX];
        }

        //sync all the threads in a block on each tile iteration (for writing to shmem)
        //after synced, guarunteed that shmem buffer is complete for A and B tiles
        __syncthreads();
        //compute partial sum of products
        for (int i = 0; i < bufferEnd; i++)
        {
            total += chunkABuffer[threadIdx.y][i] * chunkBBuffer[i][threadIdx.x];
        }
        //wait till all threads in the block finish with the partial sum
        __syncthreads();

        /**
            If the current thread belongs to a block that is on the edge of the grid
            --> responsible for computing the output at index one block greater than it in the edge direction

            since the kernel launch roudns down, if the grid size is not evenly divisible by the block size
            then the kernel is launched without enough blocks in grid to cover output C
        */

        // first check if we are on an edge block in the x dimension
        if (remainingCellsX && blockIdx.x == gridDim.x - 1)
        {
            
            /**
                Check 2 things 
                    1. check to see if incrementing the x position of the thread pushed it out of bounds
                    2. check to see if the row corresponding to the current tile iteration
                        is within the height of B since the tile size are not guarunteed to be evenly divisible by te hieght of B
            */
            if (globalEdgeX < C_WIDTH && yTiledOffset < B_HEIGHT)
            {
                //the shared memory buffer that was copied on this tile iteration for matrix A is fine 
                //we need to copy the partial tile of matrix b into shared memory buffer
                edgeCellsBBuffer[threadIdx.y][threadIdx.x] = dB[yTiledOffset][globalEdgeX];
            }
            
            //safe to call this since all threads in this block are guarunteed to come to this section
            //wait for the threads that copy memory into the shmem buffer for B finish 
            __syncthreads();
            
            //if the new x index is within C_WIDTH
            if (globalEdgeX < C_WIDTH)
            {
                //compute the partial sum of products for the edge cell of C
                for (int i = 0; i < bufferEnd; i++)
                {
                    totalXOverflow += chunkABuffer[threadIdx.y][i] * edgeCellsBBuffer[i][threadIdx.x];
                }
            }
            
            //wait for some threads to finish computing the partial sum of those edge cells
            __syncthreads();
        }
        
        
        //Then check if the current thread is in a block that is on the y edge
        if (remainingCellsY && blockIdx.y == gridDim.y - 1)
        {
            
            /**
            Check 2 things 
                1. check to see if incrementing the y position of the thread pushed it out of bounds
                2. check to see if the row corresponding to the current tile iteration
                    is within the height of B since the tile size are not guarunteed to be evenly divisible by te hieght of B
            */
            if (globalEdgeY < C_HEIGHT && xTiledOffset < A_WIDTH)
            {
                //the shared memory buffer that was copied on this tile iteration for matrix A is fine 
                //we need to copy the partial tile of matrix b into shared memory buffer
                edgeCellsABuffer[threadIdx.y][threadIdx.x] = dA[globalEdgeY][xTiledOffset];
            }
            
            //safe to call this since all threads in this block are guarunteed to come to this section
            //wait for the threads that copy memory into the shmem buffer for B finish 
            __syncthreads();
            
            //if the new y index is within C_HEIGHT
            if (globalEdgeY < C_HEIGHT)
            {
                // compute the partial sum now
                for (int i = 0; i < bufferEnd; i++)
                {
                    totalYOverflow += edgeCellsABuffer[threadIdx.y][i] * chunkBBuffer[i][threadIdx.x];
                }
            }

            //wait for some threads to finish computing the partial sum of those edge cells
            __syncthreads();
        }

        //last edge case - corner block that is edge in x and y dimension
        if (remainingCellsX && remainingCellsY && 
            blockIdx.x == gridDim.x - 1 && blockIdx.y == gridDim.y - 1)
        {

            //check to see if the new threads made by offset are within the bounds
            if (globalEdgeX < C_WIDTH && globalEdgeY < C_HEIGHT)
            {
                //compute partial sum for each tile
                for (int i = 0; i < bufferEnd; i++)
                {
                    totalCornerOverflow += edgeCellsABuffer[threadIdx.y][i] * edgeCellsBBuffer[i][threadIdx.x];
                }
            }
        }   
    }

    dC[globalY][globalX] = total;

    if (remainingCellsX && blockIdx.x == gridDim.x - 1 && globalEdgeX < C_WIDTH)
    {
        dC[globalY][globalEdgeX] = totalXOverflow;
    }
    if (remainingCellsY && blockIdx.y == gridDim.y - 1 && globalEdgeY < C_HEIGHT)
    {
        dC[globalEdgeY][globalX] = totalYOverflow;
    }
    if (remainingCellsX && remainingCellsY &&
        blockIdx.x == gridDim.x - 1 && blockIdx.y == gridDim.y - 1 &&
        globalEdgeX < C_WIDTH && globalEdgeY < C_HEIGHT)
    {
        dC[globalEdgeY][globalEdgeX] = totalCornerOverflow;
    }
    



}

int mainMatrix(int modeArg)
{
    unsigned int memSizeA, memSizeB, memSizeC;
    unsigned int x, y;
    float msec;
    cudaEvent_t start, stop;
    int mode = modeArg;

    if (A_WIDTH != B_HEIGHT)
    {
        printf("Error: A_WIDTH and B_HEIGHT do not match\n");
        return 1;
    }

    memSizeA = sizeof(float) * A_WIDTH * A_HEIGHT;
    memSizeB = sizeof(float) * B_WIDTH * B_HEIGHT;
    memSizeC = sizeof(float) * C_WIDTH * C_HEIGHT;

    // Initialise A
    for (y = 0; y < A_HEIGHT; y++)
    {
        for (x = 0; x < A_WIDTH; x++)
        {
            hA[y][x] = (float)rand() / RAND_MAX;
        }
    }
    // Initialise B
    for (y = 0; y < B_HEIGHT; y++)
    {
        for (x = 0; x < B_WIDTH; x++)
        {
            hB[y][x] = (float)rand() / RAND_MAX;
        }
    }

    // copy host memory to device
    if (mode > 0)
    {
        CHECK_ERROR(cudaMemcpyToSymbol(dA, hA, memSizeA));
        CHECK_ERROR(cudaMemcpyToSymbol(dB, hB, memSizeB));
    }

    // Start timing
    CHECK_ERROR(cudaEventCreate(&start));
    CHECK_ERROR(cudaEventCreate(&stop));
    CHECK_ERROR(cudaEventRecord(start));

    // Setup execution parameters
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(C_WIDTH / BLOCK_SIZE, C_HEIGHT / BLOCK_SIZE);
    
    switch (mode)
    {
        case 0: printf("Running CPU version\n");
            matrixMulCPU(hA, hB, hC);
            break;
        case 1: printf("Running naive GPU version\n");
            matrixMulCUDA<<<grid, threads>>>();
            check_launch("matrixMulCUDA");
            break;
        case 2: printf("Running tiled GPU version\n");
            matrixMulCUDATiled<<<grid, threads>>>();
            check_launch("matrixMulCUDATiled");
            break;
        default: printf("Unknown mode %d\n",mode);
            break;
    }
        
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&msec, start, stop);

    // Copy result from device to host
    if (mode > 0)
    {
        CHECK_ERROR(cudaMemcpyFromSymbol(hC, dC, memSizeC));
    }

    // compare the GPU results against the CPU results
    if (mode > 0)
    {
        // Compute reference CPU version
        matrixMulCPU(hA, hB, hCRef);

        // Check for errors
        matrixMulTest(hC, hCRef);
    }

    printf("Completed in %f msec\n", msec);
    return 0;
}

static const int maxUlps = 1000;

int matrixMulTest(float C[C_HEIGHT][C_WIDTH], float Cref[C_HEIGHT][C_WIDTH])
{
    int errors = 0;
    int y, x;

    for (y = 0; y < C_HEIGHT; y++)
    {
        for (x = 0; x < C_WIDTH; x++)
        {
            if (!AlmostEqual2sComplement(C[y][x], Cref[y][x], maxUlps))
            {
                errors++;
                printf("Device item c[%d][%d] = %f does not match host result %f Error %d\n", y, x, C[y][x], Cref[y][x], errors);
                if (errors > 5)
                {
                    printf("Too many errors, aborting comparison\n");
                    return errors;
                }
            }
        }
    }
    if (errors)
    {
        printf("%d errors found\n", errors);
    }
    else
    {
        printf("Test passed successfully\n");
    }
    return errors;
}
