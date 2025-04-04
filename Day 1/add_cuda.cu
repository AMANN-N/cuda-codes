#include <iostream>
#include <math.h>
#include <cuda_runtime.h>

__global__
void add(int n, float *x, float *y)
{
    int i = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (i < n)
    {
        y[i] = x[i] + y[i];
    }
}

int main()
{
    int N = 1 << 20;  // 1 million elements


    float *x, *y;
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    for (int i = 0; i < N; i++)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    add<<<numBlocks, blockSize>>>(N, x, y);
    
    cudaDeviceSynchronize();
    for (int i = 0; i < 10; i++)
    {
        std::cout << y[i] << std::endl;
    }

    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
    {
        maxError += fabs(y[i] - 3.0f);
    }
    std::cout << "Max error: " << maxError << std::endl;

    cudaFree(x);
    cudaFree(y);
    
    return 0;
}
