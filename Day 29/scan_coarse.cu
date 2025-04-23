#include <iostream>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 64
#define SECTION_SIZE 1024  

__global__ void coarse_scan(float *input, float *output, int N) 
{
    __shared__ float shared[SECTION_SIZE];
    __shared__ float last_elements[THREADS_PER_BLOCK];

    unsigned int tid = threadIdx.x;
    unsigned int block_offset = blockIdx.x * SECTION_SIZE;
    unsigned int subsection_size = SECTION_SIZE / THREADS_PER_BLOCK;

    float sum = 0.0f;

    // Phase 1: local scan
    for (unsigned int i = 0; i < subsection_size; i++) 
    {
        unsigned int index = block_offset + tid * subsection_size + i;
        if (index < N) 
        {
            shared[tid * subsection_size + i] = input[index];
            sum += shared[tid * subsection_size + i];
            shared[tid * subsection_size + i] = sum;
        } 
        else 
        {
            shared[tid * subsection_size + i] = 0.0f;
        }
    }

    last_elements[tid] = sum;
    __syncthreads();

    // Phase 2: scan of last elements (Brent Kung)
    for (unsigned int stride = 1; stride < THREADS_PER_BLOCK; stride *= 2) 
    {
        float temp = 0.0f;
        if (tid >= stride) 
        {
            temp = last_elements[tid] + last_elements[tid - stride];
        }
        __syncthreads();
        if (tid >= stride) 
        {
            last_elements[tid] = temp;
        }
        __syncthreads();
    }

    // Phase 3: apply offsets to shared results
    if (tid > 0) 
    {
        float prefix_sum = last_elements[tid - 1];
        for (unsigned int i = 0; i < subsection_size; ++i) 
        {
            unsigned int idx = tid * subsection_size + i;
            if (block_offset + idx < N) 
            {
                output[block_offset + idx] = shared[idx] + prefix_sum;
            }
        }
    } 
    else 
    {
        for (unsigned int i = 0; i < subsection_size; ++i) 
        {
            unsigned int idx = tid * subsection_size + i;
            if (block_offset + idx < N) 
            {
                output[block_offset + idx] = shared[idx];
            }
        }
    }
}




}

int main() 
{
    const int N = 1024;
    const int threadsPerBlock = SECTION_SIZE / 2;
    const int numBlocks = (N + SECTION_SIZE - 1) / SECTION_SIZE;

    float *input, *output;
    cudaMallocManaged(&input, N * sizeof(float));
    cudaMallocManaged(&output, N * sizeof(float));

    for (int i = 0; i < N; ++i)
        input[i] = 1.0f;

    coarse_scan<<<numBlocks, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();

    std::cout << "Last element: " << output[N - 1] << std::endl;

    cudaFree(input);
    cudaFree(output);

    return 0;
}
