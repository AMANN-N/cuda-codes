#include <stdio.h>
#include <cuda_runtime.h>

#define N 5   // Input matrix size (NxN)
#define M 3   // Kernel size (MxM)
#define STRIDE 1  // Stride

__global__ 
void conv2d(float *A, float *B, float *C, int input_size, int kernel_size, int stride, int output_size) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < output_size && col < output_size)  
    {
        float sum = 0.0f;
        for (int i = 0; i < kernel_size; i++) 
        {
            for (int j = 0; j < kernel_size; j++) 
            {
                sum += A[(row * stride + i) * input_size + (col * stride + j)] * B[i * kernel_size + j];
            }
        }
        C[row * output_size + col] = sum;
    }
}

int main() {
    float *h_A, *h_B, *h_C;
    int output_size = ((N - M) / STRIDE) + 1;
    
    size_t size_A = N * N * sizeof(float);
    size_t size_B = M * M * sizeof(float);
    size_t size_C = output_size * output_size * sizeof(float);

    h_A = (float*)malloc(size_A);
    h_B = (float*)malloc(size_B);
    h_C = (float*)malloc(size_C);

    for (int i = 0; i < N * N; i++) {
        h_A[i] = (float)(i + 1);
    }
    
    for (int i = 0; i < M * M; i++) {
        h_B[i] = 1.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((output_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                        (output_size + threadsPerBlock.y - 1) / threadsPerBlock.y);
    conv2d<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N, M, STRIDE, output_size);

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    printf("Output after 2D Convolution:\n");
    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < output_size; j++) {
            printf("%f ", h_C[i * output_size + j]);
        }
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
