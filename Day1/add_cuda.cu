#include <iostream>  
#include <math.h> 


__global__  // This is a CUDA kernel function
void add(int N, float *x, float *y)
{
    for (int i = 0; i < N; i++)
    {
        y[i] = x[i] + y[i]; 
    }
}

int main()
{
    int N = 1<<20;    

    float *x, *y;
    cudaMallocManaged(&x , N*sizeof(float));
    cudaMallocManaged(&y , N*sizeof(float));

    for (int i = 0; i < N; i++)
    {
        x[i] = 1.0f;    
        y[i] = 2.0f;    
    }

    add<<<1, 1>>>(N, x, y);     //Launches one GPU thread to run add function
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
    {
        maxError += fabs(y[i] - 3.0f);    
        std::cout << y[i] << std::endl;    
    }
    
    std::cout << "Max error: " << maxError << std::endl;    

    cudaFree(x);
    cudaFree(y);
    return 0;
}