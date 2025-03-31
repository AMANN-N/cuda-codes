#include <stdio.h>
#include <cuda_runtime.h>

#define N 1000

__global__ void calcsum(float *x, float *out, int n) {
    extern __shared__ float sdata[];
    int a = threadIdx.x;
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[a] = (b < N) ? x[b] : 0.0f;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) 
    {
        if (a < s) sdata[a] += sdata[a + s];
        __syncthreads();
    }
    
    if (a == 0) out[blockIdx.x] = sdata[0];
}

int main() 
{
    size_t size = N * sizeof(float);
    float *h_A = (float*)malloc(size);
    float *h_output = (float*)malloc(sizeof(float));

    for (int i = 0; i < N; i++)
    {
        h_A[i] = 1.0f;
    } 
    
    float *d_A, *d_output;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_output, ((N + 255) / 256) * sizeof(float));
    
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    
    int threads = 256;
    int blocks = (N + threads - 1) / threads;


    calcsum<<<blocks, threads, threads * sizeof(float)>>>(d_A, d_output, N);


    cudaDeviceSynchronize();
    
    float *h_partialSums = (float*)malloc(blocks * sizeof(float));
    cudaMemcpy(h_partialSums, d_output, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    
    float sum = 0.0f;
    for (int i = 0; i < blocks; i++) sum += h_partialSums[i];
    
    printf("CUDA Parallel Sum = %.1f (expected 1000.0)\n", sum);
    
    cudaFree(d_A); cudaFree(d_output);
    free(h_A); free(h_output); free(h_partialSums);
    return 0;
}
