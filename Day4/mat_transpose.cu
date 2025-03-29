#include <stdio.h>
#include <cuda_runtime.h>

#define N 5  // Reduced for visibility

__global__ 
void mat_transpose(float *A, float *C, int width)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;  // Column 
    int j = threadIdx.y + blockIdx.y * blockDim.y;  // Row 

    if(i < width && j < width)
    {
        C[i * width + j] = A[j * width + i];  // Transpose operation
    }
}

int main() {
    float *h_A, *h_C;
    float *d_A, *d_C;    
    size_t size = N * N * sizeof(float);

    h_A = (float*)malloc(size);
    h_C = (float*)malloc(size);

    // Initialize with values that make transposition visible
    for (int i = 0; i < N; i++) 
    {
        for(int j = 0; j < N; j++)
        {
            h_A[i * N + j] = i * N + j;  // Row-major order initialization
            h_C[i * N + j] = 0.0f; 
        }
    }

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_C, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    mat_transpose<<<grid, block>>>(d_A, d_C, N);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("Original Matrix:\n");
    for(int i = 0; i < N; i++)      
    {
        for(int j = 0; j < N; j++)
        {
            printf("%4.0f ", h_A[i * N + j]);
        }
        printf("\n");
    }

    printf("\nTransposed Matrix:\n");
    for(int i = 0; i < N; i++)      
    {
        for(int j = 0; j < N; j++)
        {
            printf("%4.0f ", h_C[i * N + j]);
        }
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_C);
    free(h_A);
    free(h_C);  

    return 0;
}