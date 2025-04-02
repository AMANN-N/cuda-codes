#include <stdio.h>
#include <cuda_runtime.h>

#define N 8   
#define M 3   
#define STRIDE 1  

__global__ 
void conv1d(float *A, float *B, float *C, int input_size, int kernel_size, int stride, int output_size) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < output_size)  
    {
        float sum = 0.0f;
        for (int k = 0; k < kernel_size; k++) 
        {
            sum += A[idx * stride + k] * B[k];
        }
        C[idx] = sum;
    }
}

int main() {


    float *h_A, *h_B, *h_C;
    int output_size = ((N - M) / STRIDE) + 1;
    
    size_t size_A = N * sizeof(float);
    size_t size_B = M * sizeof(float);
    size_t size_C = output_size * sizeof(float);

    h_A = (float*)malloc(size_A);
    h_B = (float*)malloc(size_B);
    h_C = (float*)malloc(size_C);

    for (int i = 0; i < N; i++) {
        h_A[i] = (float)(i + 1);  
    }
    
    for (int i = 0; i < M; i++) {
        h_B[i] = 1.0f;  
    }


    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);


    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);


    int threadsPerBlock = 16;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;
    conv1d<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N, M, STRIDE, output_size);


    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);


    printf("Output after 1D Convolution:\n");
    for (int i = 0; i < output_size; i++) {
        printf("C[%d] = %f\n", i, h_C[i]);
    }
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
