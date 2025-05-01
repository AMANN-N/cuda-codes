#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define N 16  

__global__ void softmax_exp_kernel(float* input, float* exp_output, float* sum_exp, int n) {
    __shared__ float partial_sum[256];  

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    float val = 0.0f;
    if (i < n) {
        val = expf(input[i]);
        exp_output[i] = val;
    }

    partial_sum[tid] = (i < n) ? val : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_sum[tid] += partial_sum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(sum_exp, partial_sum[0]);
    }
}


__global__ void softmax_normalize_kernel(float* exp_output, float* softmax_output, float* sum_exp, int n) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        softmax_output[i] = exp_output[i] / (*sum_exp);
    }
}

int main() {
    float *h_input, *h_softmax;
    float *d_input, *d_exp, *d_softmax, *d_sum;

    size_t size = N * sizeof(float);

    h_input = (float*)malloc(size);
    h_softmax = (float*)malloc(size);

    for (int i = 0; i < N; i++) 
    {
        h_input[i] = (float)(i - 8);  
    }

    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_exp, size);
    cudaMalloc((void**)&d_softmax, size);
    cudaMalloc((void**)&d_sum, sizeof(float));

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    cudaMemset(d_sum, 0, sizeof(float));

    int threadsPerBlock = 16;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    softmax_exp_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_exp, d_sum, N);
    cudaDeviceSynchronize();

    softmax_normalize_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_exp, d_softmax, d_sum, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_softmax, d_softmax, size, cudaMemcpyDeviceToHost);

    printf("Softmax Output:\n");
    for (int i = 0; i < N; i++) {
        printf("softmax(%f) = %f\n", h_input[i], h_softmax[i]);
    }

    cudaFree(d_input);
    cudaFree(d_exp);
    cudaFree(d_softmax);
    cudaFree(d_sum);
    free(h_input);
    free(h_softmax);

    return 0;
}
