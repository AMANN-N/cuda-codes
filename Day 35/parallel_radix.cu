#include <iostream>
#include <cuda_runtime.h>
using namespace std;

#define RADIX 10

__global__ void countsort(int *d_input, int *hist, int exp, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        int digit = (d_input[index] / exp) % RADIX;
        atomicAdd(&hist[digit], 1);
    }
}

__global__ void prefix_sum_kernel(int *hist, int *prefix_sum) {
    __shared__ int temp[RADIX];
    int index = threadIdx.x;

    if (index < RADIX)
        temp[index] = hist[index];
    __syncthreads();

    for (int offset = 1; offset < RADIX; offset *= 2) {
        int val = 0;
        if (index >= offset)
            val = temp[index - offset];
        __syncthreads();
        if (index >= offset)
            temp[index] += val;
        __syncthreads();
    }

    if (index < RADIX)
        prefix_sum[index] = temp[index] - hist[index]; 
}

__global__ void reorder(int *d_input, int *d_output, int *prefix_sum, int exp, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        int digit = (d_input[index] / exp) % RADIX;
        int pos = atomicAdd(&prefix_sum[digit], 1);
        d_output[pos] = d_input[index];
    }
}

void radix_sort(int *d_A, int *d_C, int n) {
    int *hist, *prefix_sum, *d_input, *d_output;

    cudaMalloc(&hist, RADIX * sizeof(int));
    cudaMalloc(&prefix_sum, RADIX * sizeof(int));
    cudaMalloc(&d_input, n * sizeof(int));
    cudaMalloc(&d_output, n * sizeof(int));

    cudaMemcpy(d_input, d_A, n * sizeof(int), cudaMemcpyDeviceToDevice);

    for (int exp = 1; exp <= 1000000; exp *= RADIX) {
        cudaMemset(hist, 0, RADIX * sizeof(int));

        countsort<<<(n + 255) / 256, 256>>>(d_input, hist, exp, n);
        prefix_sum_kernel<<<1, RADIX>>>(hist, prefix_sum);
        reorder<<<(n + 255) / 256, 256>>>(d_input, d_output, prefix_sum, exp, n);

        int *temp = d_input;
        d_input = d_output;
        d_output = temp;
    }

    cudaMemcpy(d_C, d_input, n * sizeof(int), cudaMemcpyDeviceToDevice);

    cudaFree(hist);
    cudaFree(prefix_sum);
    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    int A[] = {1, 3, 5, 7, 2, 4, 6, 8};
    int C[8];

    int n = 8;
    int *d_A, *d_C;

    cudaMalloc(&d_A, n * sizeof(int));
    cudaMalloc(&d_C, n * sizeof(int));
    cudaMemcpy(d_A, A, n * sizeof(int), cudaMemcpyHostToDevice);

    radix_sort(d_A, d_C, n);

    cudaMemcpy(C, d_C, n * sizeof(int), cudaMemcpyDeviceToHost);

    cout << "Sorted Array: ";
    for (int i = 0; i < n; i++) cout << C[i] << " ";
    cout << endl;

    cudaFree(d_A);
    cudaFree(d_C);

    return 0;
}
