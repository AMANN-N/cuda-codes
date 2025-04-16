#include <iostream>
#include <cuda_runtime.h>

__global__ void control_div(float *inp, float *out) 
{
    int i = threadIdx.x;
    for(int stride = blockDim.x; stride >= 1; stride = stride/2)
    {
        if(threadIdx.x < stride)
        {
            inp[i] = inp[i] + inp[i+stride];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0)
    {
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
    control_div<<<1, threadsPerBlock>>>(input, output);
    cudaDeviceSynchronize();

    std::cout << "Sum is: " << *output << std::endl;

    float expected = static_cast<float>(N);
    float error = fabs(*output - expected);
    std::cout << "Error: " << error << std::endl;

    cudaFree(input);
    cudaFree(output);

    return 0;
}
