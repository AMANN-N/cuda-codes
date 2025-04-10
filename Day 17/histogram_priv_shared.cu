#include <iostream>
#include <cuda_runtime.h>
#include <cstring>

#define NUM_BINS 7
#define BLOCK_SIZE 256

__global__
void histogram_shared(char *input, int length, int *global_hist) {
    __shared__ int private_hist[NUM_BINS];

    for (int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        private_hist[bin] = 0;
    }
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length) {
        int idx = input[i] - 'a';
        if (idx >= 0 && idx < 26) {
            atomicAdd(&private_hist[idx / 4], 1);
        }
    }
    __syncthreads();

    for (int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        int binVal = private_hist[bin];
        if (binVal > 0) {
            atomicAdd(&global_hist[bin], binVal);
        }
    }
}

int main() {
    const char *text = "cuda challenge day number sixteen less go";
    int length = strlen(text);

    char *h_input = new char[length];
    int *h_hist = new int[NUM_BINS]();

    memcpy(h_input, text, length);

    char *d_input;
    int *d_hist;
    cudaMalloc(&d_input, length * sizeof(char));
    cudaMalloc(&d_hist, NUM_BINS * sizeof(int));

    cudaMemcpy(d_input, h_input, length * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemset(d_hist, 0, NUM_BINS * sizeof(int));

    int grid = (length + BLOCK_SIZE - 1) / BLOCK_SIZE;
    histogram_shared<<<grid, BLOCK_SIZE>>>(d_input, length, d_hist);

    cudaMemcpy(h_hist, d_hist, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Histogram (4-letter bins):\n");
    for (int i = 0; i < NUM_BINS; i++) {
        char start = 'a' + i * 4;
        char end = (i == NUM_BINS - 1) ? 'z' : start + 3;
        printf("Bin %d (%c-%c): %d\n", i, start, end, h_hist[i]);
    }

    cudaFree(d_input);
    cudaFree(d_hist);
    delete[] h_input;
    delete[] h_hist;

    return 0;
}
