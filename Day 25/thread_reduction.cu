#include <iostream>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256
#define COARSENING_FACTOR 2

__global__ void thread_reduction(float *inp, float *out) {
    __shared__ float smem[THREADS_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    unsigned int start = COARSENING_FACTOR * blockDim.x * blockIdx.x + tid;

    float sum = 0.0f;
    for (int i = 0; i < COARSENING_FACTOR; ++i) {
        sum += inp[start + i * blockDim.x];
    }

    smem[tid] = sum;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out, smem[0]);
    }
}

int main()
{
    const int threadsPerBlock = THREADS_PER_BLOCK;
    const int coarseningFactor = COARSENING_FACTOR;
    const int blocks = 1;
    const int N = threadsPerBlock * coarseningFactor * blocks;

    float *input, *output;
    cudaMallocManaged(&input, N * sizeof(float));
    cudaMallocManaged(&output, sizeof(float));
    *output = 0.0f;

    for (int i = 0; i < N; i++) {
        input[i] = 1.0f;
    }

    thread_reduction<<<blocks, threadsPerBlock>>>(input, output);
    cudaDeviceSynchronize();

    std::cout << "Sum is: " << *output << std::endl;

    float expected = static_cast<float>(N);
    float error = fabs(*output - expected);
    std::cout << "Error: " << error << std::endl;

    cudaFree(input);
    cudaFree(output);

    return 0;
}
