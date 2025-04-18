#include <iostream>
#include <cuda_runtime.h>

__global__ void shared_reduction(float *inp, float *out) 
{
    extern __shared__ float sdata[];  

    int tid = threadIdx.x;
    int global_idx = tid + blockDim.x;

    sdata[tid] = inp[tid] + inp[global_idx];
    __syncthreads();

    for(int stride = blockDim.x / 2; stride >= 1; stride /= 2)
    {
        if(tid < stride)
        {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if(tid == 0)
    {
        *out = sdata[0];
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

    shared_reduction<<<1, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(input, output);
    cudaDeviceSynchronize();

    std::cout << "Sum is: " << *output << std::endl;

    float expected = static_cast<float>(N);
    float error = fabs(*output - expected);
    std::cout << "Error: " << error << std::endl;

    cudaFree(input);
    cudaFree(output);

    return 0;
}
