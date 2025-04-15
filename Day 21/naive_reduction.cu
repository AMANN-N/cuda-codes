#include <iostream>
#include <cuda_runtime.h>

__global__ void simpleReductionSumKernel(float *inp, float *out) {
    int i = 2 * threadIdx.x;
    for (int stride = 1; stride <= blockDim.x; stride *= 2) {
        if (threadIdx.x % stride == 0) {
            inp[i] += inp[i + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *out = inp[0];
    }
}


int main()
{
    int threadsPerBlock = 256;
    int N = 2 * threadsPerBlock; 
    float *input, *output;

    cudaMallocManaged(&input, N * sizeof(float));
    cudaMallocManaged(&output, sizeof(float));

    for (int i = 0; i < N; i++)
    {
        input[i] = 1.0f; 
    }
    simpleReductionSumKernel<<<1, threadsPerBlock>>>(input, output);
    cudaDeviceSynchronize();

    std::cout << "Sum is: " << *output << std::endl;

    float expected = static_cast<float>(N);
    float error = fabs(*output - expected);
    std::cout << "Error: " << error << std::endl;

    cudaFree(input);
    cudaFree(output);

    return 0;
}
