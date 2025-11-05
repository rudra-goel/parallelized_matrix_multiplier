#include <stdlib.h>
#include <stdio.h>
#include <ctime>

#include "util.h"

#define BLOCK_SIZE 1
#define BLUR_SIZE 33

#define PRINT_ACCEDD_BY_INDEX 0

// pointers to host & device arrays
int host_shmem_accesses[BLOCK_SIZE][BLOCK_SIZE];
int host_gdram_accesses[BLOCK_SIZE][BLOCK_SIZE];

__device__ int device_shmem_accesses[BLOCK_SIZE][BLOCK_SIZE];
__device__ int device_gdram_accesses[BLOCK_SIZE][BLOCK_SIZE];

__global__ void compute_mem_accesses() {
    int total_in_blur_area = BLUR_SIZE * BLUR_SIZE;
    int half = (BLUR_SIZE - 1) / 2;
    int shmem_accesses = 0;
    int gdram_accesses = 0;

    int sub_dim_width = 0;
    int sub_dim_height = 0;

    int num_cells_to_left = blockDim.y - threadIdx.y - 1;
    int num_cells_to_right = threadIdx.x;

    int num_cells_to_top = blockDim.y - threadIdx.y - 1;
    int num_cells_to_bottom = threadIdx.y;

    if (num_cells_to_left >= half)
        sub_dim_width += half;
    else
        sub_dim_width += num_cells_to_left;

    if (num_cells_to_right >= half)
        sub_dim_width += half;
    else
        sub_dim_width += num_cells_to_right;

    if (num_cells_to_bottom >= half)
        sub_dim_height += half;
    else 
        sub_dim_height += num_cells_to_bottom;
    
    if (num_cells_to_top >= half)
        sub_dim_height += half;
    else
        sub_dim_height += num_cells_to_top;

    //for the current pixel, add a dimension
    sub_dim_width++;
    sub_dim_height++;

    shmem_accesses = sub_dim_height * sub_dim_width;
    gdram_accesses = total_in_blur_area - shmem_accesses;

    device_shmem_accesses[threadIdx.y][threadIdx.x] = shmem_accesses;
    device_gdram_accesses[threadIdx.y][threadIdx.x] = gdram_accesses;
}


int main(void) {

    int num_elements = BLOCK_SIZE * BLOCK_SIZE;

    // compute the size of the arrays in bytes
    int num_bytes = num_elements * sizeof(int);

    //set hots falvs to zero
    for(int i=0; i < BLOCK_SIZE; i++) {
        for(int j = 0; j < BLOCK_SIZE; j++) {
            host_shmem_accesses[i][j] = 0;
            host_gdram_accesses[i][j] = 0;
        }
    }


    // cudaMalloc device arrays
    CHECK_ERROR(cudaMemcpyToSymbol(device_shmem_accesses, host_shmem_accesses, num_bytes));
    CHECK_ERROR(cudaMemcpyToSymbol(device_gdram_accesses, host_gdram_accesses, num_bytes));

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(1, 1);

    // launch kernel
    compute_mem_accesses<<<grid,threads>>>();
    check_launch("Compute Mem Accesses ");

    CHECK_ERROR(cudaMemcpyFromSymbol(host_shmem_accesses, device_shmem_accesses, num_bytes));
    CHECK_ERROR(cudaMemcpyFromSymbol(host_gdram_accesses, device_gdram_accesses, num_bytes));



    printf("\n\n");

    int total_shmem = 0;
    int total_gdram = 0;

    for(int i=0; i < BLOCK_SIZE; i++) {
        for(int j = 0; j < BLOCK_SIZE; j++) {
            if (PRINT_ACCEDD_BY_INDEX)
                printf("%d,%d\t", host_shmem_accesses[i][j], host_gdram_accesses[i][j]);
            total_shmem+=host_shmem_accesses[i][j];
            total_gdram+=host_gdram_accesses[i][j];
        }
        if (PRINT_ACCEDD_BY_INDEX) printf("\n");
    }

    
    printf("\n");
    printf("Total Shared Mem Accesses: %d ", total_shmem);
    printf("\n");
    printf("Total Global Mem Accesses: %d ", total_gdram);
    printf("\n");
    
    printf("Ratio: %.3f ", total_shmem / (1.0*total_gdram));
    printf("\n");
    
}

