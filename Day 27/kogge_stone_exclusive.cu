#include <iostream>
#include <cuda_runtime.h>

#define SECTION_SIZE 512

__global__ void kogge_stone_block_scan(float *input, float *output, float *block_sums, int N) {
    __shared__ float temp[SECTION_SIZE];
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < N && threadIdx.x != 0)
        temp[threadIdx.x] = input[i-1];
    else
        temp[threadIdx.x] = 0.0f;

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        float val = 0;
        if (threadIdx.x >= stride)
            val = temp[threadIdx.x] + temp[threadIdx.x - stride];
        else
            val = temp[threadIdx.x];
        __syncthreads();
        temp[threadIdx.x] = val;
    }

    if (i < N)
        output[i] = temp[threadIdx.x];

    if (threadIdx.x == blockDim.x - 1 && block_sums != nullptr)
    block_sums[blockIdx.x] = temp[threadIdx.x] + input[i]; 
}

__global__ void add_block_sums(float *output, float *block_sums, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (blockIdx.x == 0 || i >= N) return;

    float offset = 0.0f;
    for (int j = 0; j < blockIdx.x; j++)
        offset += block_sums[j];

    output[i] += offset;
}

int main() {
    const int N = 1024;
    const int threadsPerBlock = SECTION_SIZE;
    const int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    float *input, *output, *block_sums;
    cudaMallocManaged(&input, N * sizeof(float));
    cudaMallocManaged(&output, N * sizeof(float));
    cudaMallocManaged(&block_sums, numBlocks * sizeof(float));

    for (int i = 0; i < N; ++i)
        input[i] = 1.0f;

    kogge_stone_block_scan<<<numBlocks, threadsPerBlock>>>(input, output, block_sums, N);
    cudaDeviceSynchronize();

    add_block_sums<<<numBlocks, threadsPerBlock>>>(output, block_sums, N);
    cudaDeviceSynchronize();

    std::cout << "Last element will be -  " << output[N - 1] << std::endl;

    // float expected = static_cast<float>(N);
    // float error = fabs(output[N - 1] - expected);
    // std::cout << "Error: " << error << std::endl;

    cudaFree(input);
    cudaFree(output);
    cudaFree(block_sums);

    return 0;
}
