#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define N 16 


__global__ void sigmoid_kernel(float* input, float* output, int n) 
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < n)
    {
        output[index] = 1.0f / (1.0f + exp(-input[index])); 
    }

}

int main() {
    float *h_input, *h_output;
    float *d_input, *d_output;

    size_t size = N * sizeof(float);

    h_input = (float*)malloc(size);
    h_output = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(i - 8); 
    }

    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 16;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    sigmoid_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);

    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    printf("Sigmoid Output:\n");
    for (int i = 0; i < N; i++) {
        printf("sigmoid(%f) = %f\n", h_input[i], h_output[i]);
    }

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
