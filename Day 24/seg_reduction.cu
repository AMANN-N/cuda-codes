#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

#define THREADS_PER_BLOCK 256

__global__ void seg_reduction(float *inp, float *out, int N) {
    __shared__ float shared[THREADS_PER_BLOCK];

    int segmentStart = 2 * blockDim.x * blockIdx.x;
    int i = segmentStart + threadIdx.x;
    int t = threadIdx.x;

    float a = 0.0f, b = 0.0f;
    if (i < N) a = inp[i];
    if (i + blockDim.x < N) b = inp[i + blockDim.x];

    shared[t] = a + b;

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        if (t < stride) {
            shared[t] += shared[t + stride];
        }
    }
    if (t == 0) {
        atomicAdd(out, shared[0]);
    }
}

int main()
{
    int threadsPerBlock = THREADS_PER_BLOCK;
    int blocksPerGrid = 4;
    int N = 2 * threadsPerBlock * blocksPerGrid; 
    float *input, *output;

    cudaMallocManaged(&input, N * sizeof(float));
    cudaMallocManaged(&output, sizeof(float));
    
    for (int i = 0; i < N; i++) {
        input[i] = 1.0f;
    }
    *output = 0.0f;

    seg_reduction<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();

    std::cout << "Sum is: " << *output << std::endl;

    float expected = static_cast<float>(N);
    float error = fabs(*output - expected);
    std::cout << "Error: " << error << std::endl;

    cudaFree(input);
    cudaFree(output);

    return 0;
}
