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
__device__ float d_A[A_HEIGHT][A_WIDTH];
__device__ float d_B[B_HEIGHT][B_WIDTH];
__device__ float d_C[C_HEIGHT][C_WIDTH];

// host-side arrays
float h_A[A_HEIGHT][A_WIDTH];
float h_B[B_HEIGHT][B_WIDTH];
float h_C[C_HEIGHT][C_WIDTH];
float h_C_ref[C_HEIGHT][C_WIDTH];

int matrixMulTest(float C[C_HEIGHT][C_WIDTH], float Cref[C_HEIGHT][C_WIDTH]);

/*
    This is the CPU-based matrix multiply.
    It calculates output matrix C, from the input matrices A and B.
*/
void matrixMulCPU(float A[A_HEIGHT][A_WIDTH], float B[C_HEIGHT][C_WIDTH], float C[C_HEIGHT][C_WIDTH]) {
    int x, y, k;
    for (y = 0; y < C_HEIGHT; y++){
        for (x = 0; x < C_WIDTH; x++){
            C[y][x] = 0;
            for (k = 0; k < A_WIDTH; k++){
                C[y][x] += A[y][k] * B[k][x];
            }
        }
    }

}


/*
    This is a GPU-based matrix multiply.
    It calculates output matrix d_C, from the input matrices d_A and d_B.
*/
__global__ void matrixMulCUDA() {
    // TODO implement simple CUDA matrix multiply here
    // inputs: d_A, d_B (global variables)
    // output: d_C (global variable)
    // do not use shared memory
    // note the launch parameters: this kernel is called for each
    //         cell in the output matrix

    //figure out the thread index in x and y directions
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    float total = 0;

    //loop through the B_HEIGHT or A_WIDTH (either works since they have to match for matrices to multiply)
    for(int i = 0; i < B_HEIGHT; i++) {
        total += d_A[y][i] * d_B[i][x];
    }
    d_C[y][x] = total;

    //if this block is on the ending edge of the grid
    //need the threads in this block to compute the output matrix value for the cells
    if (blockIdx.x == gridDim.x - 1) {
        //increment the current thread index's X position by the size of the block dim
        int col_new = x + blockDim.x;
        //check to make sure the new x index is still within bounds
        if (col_new < C_WIDTH) {
            total = 0;
            for (int i = 0; i < B_HEIGHT; i++) {
                total += d_A[y][i] * d_B[i][col_new];
            }
            d_C[y][col_new] = total;
        }
    }
    if (blockIdx.y == gridDim.y - 1) {
        
        int row_new = y + blockDim.y;
        
        if (row_new < C_HEIGHT) {
            total = 0;
            for(int i = 0; i < B_HEIGHT; i++) {
                total += d_A[row_new][i] * d_B[i][x];
            }
            d_C[row_new][x] = total;
        }
    }
    
    //force the corner block to handle the cells not yet covered in the corner
    if (blockIdx.x == gridDim.x - 1 && blockIdx.y == gridDim.y - 1) {
        int col_new = x + blockDim.x;
        int row_new = y + blockDim.y;

        if (row_new < C_HEIGHT && col_new < C_WIDTH) {
            for (int i = 0; i < B_HEIGHT; i++) {
                d_C[row_new][col_new] += d_A[row_new][i] * d_B[i][col_new];
            }
        }

    }

}

/*
    This is a GPU-based matrix multiply.
    It calculates output matrix d_C, from the input matrices d_A and d_B.
    It uses shared memory.
*/
__global__ void matrixMulCUDATiled() {
    // TODO implement tiled CUDA matrix multiply here
    // inputs: d_A, d_B (global variables)
    // output: d_C (global variable)
    // use tiled shared memory as described in the assignment
    // note the launch parameters: this kernel is called for each
    //         cell in the output matrix

    //allocate shared memory for the chunks of A and B statically
    //size of each is BLOCK_SIZE * BLOCK_SIZE * 4 bytes
    __shared__ float chunk_A_buffer[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float chunk_B_buffer[BLOCK_SIZE][BLOCK_SIZE];

    //this is a buffer for copying values from the edges of matricesd A and B
    //that are not encompassed by another block due to insufficuent blocks being launched by the kernel
    __shared__ float edge_cells_A_buffer[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float edge_cells_B_buffer[BLOCK_SIZE][BLOCK_SIZE];

    //figure out what thread I currently am in the overall matrix C
    int global_x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int global_y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    int global_edge_x = global_x + BLOCK_SIZE;
    int global_edge_y = global_y + BLOCK_SIZE;

    bool remaining_cells_x = C_WIDTH % BLOCK_SIZE != 0;
    bool remaining_cells_y = C_HEIGHT % BLOCK_SIZE != 0;


    /**
        Main Idea
            need to copy in chunks (tiles) of A and B into shared memory
            then need to perform the partial sum of products
            we loop through this for A_WIDTH / BLOCK_SIZE amount of times or B_HEIGHT / BLOCK_SIZE times

    */

    float total = 0;

    float total_x_overflow = 0;
    float total_y_overflow = 0;
    float total_corner_overflow = 0;


    //guard against case where the A_WIDTH does not divide evenly with BLOCK_SIZE
    //so must go A_width + 1 / block_size and guard against threads that are outside of the dims
    for (int tile_iteration = 0; tile_iteration < (A_WIDTH / BLOCK_SIZE) + 1; tile_iteration++) {

        //these two vars are used for copying the right data from inputs A and B into shmem
        //based on the loop iteration, offset the x dim of the thread 
        int x_tiled_offset = tile_iteration * BLOCK_SIZE + threadIdx.x;
        //based on the loop iteration, offset the y dim of the thread 
        int y_tiled_offset = tile_iteration * BLOCK_SIZE + threadIdx.y;
        

        int buffer_end = tile_iteration == A_WIDTH / BLOCK_SIZE ? A_WIDTH % BLOCK_SIZE : BLOCK_SIZE;

        if (x_tiled_offset < A_WIDTH) {
            //have the thread within the currentn block copy data into the A and B shared mem buffer
            chunk_A_buffer[threadIdx.y][threadIdx.x] = d_A[global_y][x_tiled_offset];
        }

        if (y_tiled_offset < B_HEIGHT) {
            chunk_B_buffer[threadIdx.y][threadIdx.x] = d_B[y_tiled_offset][global_x];
        }

        //sync all the threads in a block on each tile iteration (for writing to shmem)
        //after synced, guarunteed that shmem buffer is complete for A and B tiles
        __syncthreads();
        //compute partial sum of products
        for (int i = 0; i < buffer_end; i++) {
            total += chunk_A_buffer[threadIdx.y][i] * chunk_B_buffer[i][threadIdx.x];
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
        if (remaining_cells_x && blockIdx.x == gridDim.x - 1) {
            
            /**
                Check 2 things 
                    1. check to see if incrementing the x position of the thread pushed it out of bounds
                    2. check to see if the row corresponding to the current tile iteration
                        is within the height of B since the tile size are not guarunteed to be evenly divisible by te hieght of B
            */
            if (global_edge_x < C_WIDTH && y_tiled_offset < B_HEIGHT) {
                //the shared memory buffer that was copied on this tile iteration for matrix A is fine 
                //we need to copy the partial tile of matrix b into shared memory buffer
                edge_cells_B_buffer[threadIdx.y][threadIdx.x] = d_B[y_tiled_offset][global_edge_x];
            }
            
            //safe to call this since all threads in this block are guarunteed to come to this section
            //wait for the threads that copy memory into the shmem buffer for B finish 
            __syncthreads();
            
            //if the new x index is within C_WIDTH
            if (global_edge_x < C_WIDTH) {
                //compute the partial sum of products for the edge cell of C
                for (int i = 0; i < buffer_end; i++) {
                    total_x_overflow += chunk_A_buffer[threadIdx.y][i] * edge_cells_B_buffer[i][threadIdx.x];
                }
            }
            
            //wait for some threads to finish computing the partial sum of those edge cells
            __syncthreads();
        }
        
        
        //Then check if the current thread is in a block that is on the y edge
        if (remaining_cells_y && blockIdx.y == gridDim.y - 1) {
            
            /**
            Check 2 things 
                1. check to see if incrementing the y position of the thread pushed it out of bounds
                2. check to see if the row corresponding to the current tile iteration
                    is within the height of B since the tile size are not guarunteed to be evenly divisible by te hieght of B
            */
            if (global_edge_y < C_HEIGHT && x_tiled_offset < A_WIDTH) {
                //the shared memory buffer that was copied on this tile iteration for matrix A is fine 
                //we need to copy the partial tile of matrix b into shared memory buffer
                edge_cells_A_buffer[threadIdx.y][threadIdx.x] = d_A[global_edge_y][x_tiled_offset];
            }
            
            //safe to call this since all threads in this block are guarunteed to come to this section
            //wait for the threads that copy memory into the shmem buffer for B finish 
            __syncthreads();
            
            //if the new y index is within C_HEIGHT
            if (global_edge_y < C_HEIGHT) {
                // compute the partial sum now
                for (int i = 0; i < buffer_end; i++) {
                    total_y_overflow += edge_cells_A_buffer[threadIdx.y][i] * chunk_B_buffer[i][threadIdx.x];
                }
            }

            //wait for some threads to finish computing the partial sum of those edge cells
            __syncthreads();
        }

        //last edge case - corner block that is edge in x and y dimension
        if (remaining_cells_x && remaining_cells_y && 
            blockIdx.x == gridDim.x - 1 && blockIdx.y == gridDim.y - 1) {

            //check to see if the new threads made by offset are within the bounds
            if (global_edge_x < C_WIDTH && global_edge_y < C_HEIGHT) {
                //compute partial sum for each tile
                for (int i = 0; i < buffer_end; i++) {
                    total_corner_overflow += edge_cells_A_buffer[threadIdx.y][i] * edge_cells_B_buffer[i][threadIdx.x];
                }
            }
        }   
    }

    d_C[global_y][global_x] = total;

    if (remaining_cells_x && blockIdx.x == gridDim.x - 1 && global_edge_x < C_WIDTH) {
        d_C[global_y][global_edge_x] = total_x_overflow;
    }
    if (remaining_cells_y && blockIdx.y == gridDim.y - 1 && global_edge_y < C_HEIGHT) {
        d_C[global_edge_y][global_x] = total_y_overflow;
    }
    if (remaining_cells_x && remaining_cells_y &&
        blockIdx.x == gridDim.x - 1 && blockIdx.y == gridDim.y - 1 &&
        global_edge_x < C_WIDTH && global_edge_y < C_HEIGHT) {

        d_C[global_edge_y][global_edge_x] = total_corner_overflow;
    }
    



}

// int main(int argc, char **argv) {
int mainMatrix(int modeArg) {
    unsigned int mem_size_A, mem_size_B, mem_size_C;
    unsigned int x, y;
    float msec;
    cudaEvent_t start, stop;
    int mode = modeArg;

    if (A_WIDTH != B_HEIGHT){
        printf("Error: A_WIDTH and B_HEIGHT do not match\n");
        return 1;
    }

    mem_size_A = sizeof(float) * A_WIDTH * A_HEIGHT;
    mem_size_B = sizeof(float) * B_WIDTH * B_HEIGHT;
    mem_size_C = sizeof(float) * C_WIDTH * C_HEIGHT;

    // Initialise A
    for (y = 0; y < A_HEIGHT; y++)
        for (x = 0; x <A_WIDTH; x++)
            h_A[y][x] = (float)rand() / RAND_MAX;
    // Initialise B
    for (y = 0; y < B_HEIGHT; y++)
        for (x = 0; x <B_WIDTH; x++)
            h_B[y][x] = (float)rand() / RAND_MAX;

    // copy host memory to device
    if (mode > 0) {
        CHECK_ERROR(cudaMemcpyToSymbol(d_A, h_A, mem_size_A));
        CHECK_ERROR(cudaMemcpyToSymbol(d_B, h_B, mem_size_B));
    }

    // Start timing
    CHECK_ERROR(cudaEventCreate(&start));
    CHECK_ERROR(cudaEventCreate(&stop));
    CHECK_ERROR(cudaEventRecord(start));

    // Setup execution parameters
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(C_WIDTH / BLOCK_SIZE, C_HEIGHT / BLOCK_SIZE);
    
    switch (mode) {
        case 0: printf("Running CPU version\n");
            matrixMulCPU(h_A, h_B, h_C);
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
        CHECK_ERROR(cudaMemcpyFromSymbol(h_C, d_C, mem_size_C));

    // compare the GPU results against the CPU results
    if (mode > 0) {
        // Compute reference CPU version
        matrixMulCPU(h_A, h_B, h_C_ref);

        // Check for errors
        matrixMulTest(h_C, h_C_ref);
    }

    printf("Completed in %f msec\n", msec);
    return 0;
}

static const int maxUlps = 1000;

int matrixMulTest(float C[C_HEIGHT][C_WIDTH], float Cref[C_HEIGHT][C_WIDTH]) {
    int errors = 0;
    int y, x;

    for (y = 0; y < C_HEIGHT; y++){
        for (x = 0; x < C_WIDTH; x++){
            if (!AlmostEqual2sComplement(C[y][x], Cref[y][x], maxUlps)) {
                errors++;
                printf("Device item c[%d][%d] = %f does not match host result %f Error %d\n", y, x, C[y][x], Cref[y][x], errors);
                if (errors > 5) {
                    printf("Too many errors, aborting comparison\n");
                    return errors;
                }
            }
        }
    }
    if (errors)
        printf("%d errors found\n", errors);
    else
        printf("Test passed successfully\n");
    return errors;
}
