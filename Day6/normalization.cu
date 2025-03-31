#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include <iostream>

#define N 5 

__global__ 
void layer_norm(const float *A, float *C, int n)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;  

    if(row < n)
    {
        extern __shared__ float shared[];  
        float *row_data = shared;  


        for(int col = 0; col < n; col++)
        {
            row_data[col] = A[row * n + col];  
        }
        __syncthreads(); 

        float mean = 0.0f;
        for(int col = 0; col < n; col++)
        {
            mean += row_data[col];
        }
        mean /= n;

        float variance = 0.0f;
        for(int col = 0; col < n; col++)
        {
            variance += (row_data[col] - mean) * (row_data[col] - mean);
        }
        variance /= n;
        float stddev = sqrtf(variance + 1e-5f); 

        for(int col = 0; col < n; col++)
        {
            C[row * n + col] = (row_data[col] - mean) / stddev;  
        }
    }
}

int main() {
    float *h_A, *h_C;   
    float *d_A, *d_C;   

    size_t size = N * N * sizeof(float);

    h_A = (float*)malloc(size);
    h_C = (float*)malloc(size);


    for (int i = 0; i < N; i++) 
    {
        for(int j = 0; j < N; j++)
        {
            h_A[i * N + j] = static_cast<float>(i * N + j); 
            h_C[i * N + j] = 0.0f; 
        }
    }


    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_C, size);


    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);


    int blockSize = N;   
    int gridSize = 1;
    size_t sharedMemSize = N * sizeof(float);  

    layer_norm<<<gridSize, blockSize, sharedMemSize>>>(d_A, d_C, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);


    printf("Original Matrix:\n");
    for(int i = 0; i < N; i++)      
    {
        for(int j = 0; j < N; j++)
        {
            printf("%4.1f ", h_A[i * N + j]);
        }
        printf("\n");
    }


    printf("\nNormalized Matrix:\n");
    for(int i = 0; i < N; i++)      
    {
        for(int j = 0; j < N; j++)
        {
            printf("%6.2f ", h_C[i * N + j]);
        }
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_C);
    free(h_A);
    free(h_C);  

    return 0;
}
