/* 
 * CUDA audio blur
 */

#include "audioblur.cuh"

#include <cstdio>
#include <cuda_runtime.h>

//changes the const float* gpu_blur_v to be float*
//this is shared memory variable
__device__ void cuda_blur_kernel_convolution(uint thread_index, const float* gpu_raw_data,
                                  float* gpu_blur_v, float* gpu_out_data,
                                  const unsigned int n_frames,
                                  const unsigned int blur_v_size) {
    // TODO: Implement the necessary convolution function that should be
    //       completed for each thread_index. Use the CPU implementation in
    //       blur.cpp as a reference.
    
    // if the current thread is smaller than the gaussian filter
    // we compute the convolution differently as we only multiply and add from
    // idx 0 of the filter to the current thread idx
    // in other words, the entire convolution kernel (gauss distr) has not overlapped
    // with the input signal --> bounds of convolution integration happen from 0 (start of input signal)
    // to whatever thread we are at
    
    if (thread_index < blur_v_size) {
        for (int j = 0; j <= thread_index; j++) {
            gpu_out_data[thread_index] += gpu_blur_v[j] * gpu_raw_data[thread_index - j];   
        }
    } else {
        //now that the current thread is in a position over the input 
        // to where the convolutiuon kernel is completely overlapped
        // iterate through the entire conv kernel and to the MAC ops
        for (int j = 0; j < blur_v_size; j++) {
            // gpu_out_data[thread_index] += gpu_blur_v[j] * gpu_raw_data[thread_index - j];
            gpu_out_data[thread_index] += gpu_blur_v[j] * gpu_raw_data[thread_index - j];
        }
    }

}

__global__ void cuda_blur_kernel(const float *gpu_raw_data, const float *gpu_blur_v,
                      float *gpu_out_data, int n_frames, int blur_v_size) {
    // TODO: Compute the current thread index.
    uint thread_index;

    //allocate the create the shared memory variable STATICALLY
    __shared__ float convolution_kernel[GAUSSIAN_SIZE];

    //this is the initial index of the current thread
    //this thread index will increment by blockDim.x + gridDim.x
    //gridDim.x tells us how many blocks were spawned during the kernel launch
    //blockDim.x tells us how many threads are in 1 block
    thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    //populate the shard mem buffer with gaussian kernel
    //have all 32 threads in the block do this population rather than one thread per block
    int idx = threadIdx.x;
    while (idx < GAUSSIAN_SIZE) {
        convolution_kernel[idx] = gpu_blur_v[idx];
        idx += blockDim.x;
    }
    //sync the threads
    __syncthreads();

    // TODO: Update the while loop to handle all indices for this thread.
    //       Remember to advance the index as necessary.

    // ensure when we update the thread index that it is still within the bounds of the signal
    //this is the equential part of the program

    while (thread_index < n_frames) {
        // Do computation for this thread index
        cuda_blur_kernel_convolution(thread_index, gpu_raw_data,
                                     convolution_kernel, gpu_out_data,
                                     n_frames, blur_v_size);
        
        // TODO: Update the thread index
        thread_index += gridDim.x * blockDim.x;
    }
}

float cuda_call_blur_kernel(const unsigned int blocks,
                            const unsigned int threads_per_block,
                            const float *raw_data,
                            const float *blur_v,
                            float *out_data,
                            const unsigned int n_frames,
                            const unsigned int blur_v_size) {
    // Use the CUDA machinery for recording time
    cudaEvent_t start_gpu, stop_gpu;
    float time_milli = -1;
    CHECK_ERROR(cudaEventCreate(&start_gpu));
    CHECK_ERROR(cudaEventCreate(&stop_gpu));
    CHECK_ERROR(cudaEventRecord(start_gpu));

    // TODO: Allocate GPU memory for the raw input data (either audio file
    //       data or randomly generated data). The data is of type float and
    //       has n_frames elements. Then copy the data in raw_data into the
    //       GPU memory you allocated.
    float* gpu_raw_data;
    //dunacially allocate GPU memory --> enough for the input buffer
    CHECK_ERROR(cudaMalloc(&gpu_raw_data, sizeof(float) * n_frames));
    //transfer data to GPU global memory
    CHECK_ERROR(cudaMemcpy(gpu_raw_data, raw_data, n_frames * sizeof(float), cudaMemcpyHostToDevice));
    
    // TODO: Allocate GPU memory for the impulse signal. The data is of type
    //       float and has blur_v_size elements. Then copy the data in blur_v
    //       into the GPU memory you allocated.
    float* gpu_blur_v;
    CHECK_ERROR(cudaMalloc(&gpu_blur_v, sizeof(float) * blur_v_size));
    CHECK_ERROR(cudaMemcpy(gpu_blur_v, blur_v, blur_v_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // TODO: Allocate GPU memory to store the output audio signal after the
    //       convolution. The data is of type float and has n_frames elements.
    //       Initialize the data as necessary.
    float* gpu_out_data;
    //dunacially allocate GPU memory --> enough for the input buffer
    CHECK_ERROR(cudaMalloc(&gpu_out_data, sizeof(float) * n_frames));
    
    // TODO: Appropriately call the kernel function, specifying
    //       block size and thereads per block
    //kernel launch --> tell the grid size (number of blocks per grid) and the 
    // block size (numnber of threads per block)
    cuda_blur_kernel<<<blocks, threads_per_block>>>(gpu_raw_data, gpu_blur_v, gpu_out_data, n_frames, blur_v_size);


    // Check for errors on kernel call
    CHECK_ERROR(cudaDeviceSynchronize());

    // TODO: Now that kernel calls have finished, copy the output signal
    //       back from the GPU to host memory. (We store this channel's result
    //       in out_data on the host.)
    CHECK_ERROR(cudaMemcpy(out_data, gpu_out_data, n_frames * sizeof(float), cudaMemcpyDeviceToHost));

    // TODO: Now that we have finished our computations on the GPU, free the
    //       GPU resources.
    CHECK_ERROR(cudaFree(gpu_raw_data));
    CHECK_ERROR(cudaFree(gpu_blur_v));
    CHECK_ERROR(cudaFree(gpu_out_data));

    // Stop the recording timer and return the computation time
    CHECK_ERROR(cudaEventRecord(stop_gpu));
    CHECK_ERROR(cudaEventSynchronize(stop_gpu));
    CHECK_ERROR(cudaEventElapsedTime(&time_milli, start_gpu, stop_gpu));
    return time_milli;
}
