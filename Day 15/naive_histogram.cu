#include <iostream>
#include <cuda_runtime.h>

#define NUM_BINS 7  // 26 letters / 4 letters per bin (ceil)
#define BLOCK_SIZE 256

__global__
void histogram_kernel(char *input, int length, int *hist) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length) {
        int idx = input[i] - 'a';
        if (idx >= 0 && idx < 26) {
            atomicAdd(&hist[idx / 4], 1);
        }
    }
}

int main() {
    const char *text = "cuda challenge day number fiftten less go";
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
    histogram_kernel<<<grid, BLOCK_SIZE>>>(d_input, length, d_hist);

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
