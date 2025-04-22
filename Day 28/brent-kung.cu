#include <iostream>
#include <cuda_runtime.h>

#define SECTION_SIZE 512

__global__ void brent_kung_scan(float *input, float *output, int N) {
    __shared__ float shared[SECTION_SIZE];

    int tIdx = threadIdx.x;
    int i = 2 * blockIdx.x * blockDim.x + tIdx;

    if (i < N)
        shared[tIdx] = input[i];
    else
        shared[tIdx] = 0.0f;
        
    if (i + blockDim.x < N)
        shared[tIdx + blockDim.x] = input[i + blockDim.x];
    else
        shared[tIdx + blockDim.x] = 0.0f;

    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        __syncthreads();
        int index = (tIdx + 1) * 2 * stride - 1;
        if (index < SECTION_SIZE) {
            shared[index] += shared[index - stride];
        }
    }

    for (unsigned int stride = SECTION_SIZE / 4; stride > 0; stride /= 2) {
        __syncthreads();
        int index = (tIdx + 1) * 2 * stride - 1;
        if (index + stride < SECTION_SIZE) {
            shared[index + stride] += shared[index];
        }
    }

    __syncthreads();

    if (i < N)
        output[i] = shared[tIdx];
    if (i + blockDim.x < N)
        output[i + blockDim.x] = shared[tIdx + blockDim.x];
}

int main() {
    const int N = 1024;
    const int threadsPerBlock = SECTION_SIZE / 2;
    const int numBlocks = (N + SECTION_SIZE - 1) / SECTION_SIZE;

    float *input, *output;
    cudaMallocManaged(&input, N * sizeof(float));
    cudaMallocManaged(&output, N * sizeof(float));

    for (int i = 0; i < N; ++i)
        input[i] = 1.0f;

    brent_kung_scan<<<numBlocks, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();

    std::cout << "Last element: " << output[N - 1] << std::endl;

    cudaFree(input);
    cudaFree(output);

    return 0;
}
