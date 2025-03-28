#include <stdio.h>
#include <cuda_runtime.h>

#define N 1000  // Vector size

// CUDA Kernel function to perform vector addition on the GPU
// A[i][j] * B[j][k] = C[i][k]      
__global__ 
void matMul(float *A, float *B, float *C, int width)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;  //col
    int j = threadIdx.y + blockIdx.y * blockDim.y;  //row

    if(i < width && j < width)
    {
        float Pvalue = 0;
        // C[i*n + j] = A[i*n + j] + B[i*n + j];
        for(int k=0; k<width; k++)
        {
            Pvalue = Pvalue + (A[j*width + k] * B[width*k + i]);
        }
        // C[j*width + i] = Pvalue
        C[j*width + i] = Pvalue;
    }           
}


int main() {
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;    
    size_t size = N * N * sizeof(float);

    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    for (int i = 0; i < N; i++) 
    {
        for(int j=0; j< N; j++)
        {
            h_A[i*N + j] = i * 1.0f;
            h_B[i*N + j] = i * 2.0f;
            h_C[i*N + j] = 0.0f;
        }
    }


    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 block(16 , 16);
    dim3 grid((N + block.x - 1)/block.x, (N + block.y - 1)/block.y);

    matMul<<<grid, block>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    for(int i=0; i< 5; i++)
    {
        for(int j=0; j< 5; j++)
        {
            printf("h_C[%d][%d] = %f\n", i, j, h_C[i*N + j]);
        }
    }
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);  

    return 0;
}
