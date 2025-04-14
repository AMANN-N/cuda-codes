#include <iostream>
#include <cuda_runtime.h>
#include <cstring>

#define NUM_BINS 7
#define THREADS 64
#define CFACTOR 3 

__global__
void histogram_coarsened_interleaved_aggregated(char *data, int length, int *histo, int blocks) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int total_threads = gridDim.x * blockDim.x;

    int prevBin = -1;
    int accumulator = 0;

    for (int i = gid; i < length; i += total_threads) {
        for (int j = 0; j < CFACTOR && i + j * total_threads < length; ++j) {
            int idx = i + j * total_threads;
            int pos = data[idx] - 'a';
            if (pos >= 0 && pos < 26) {
                int bin = pos / 4;
                if (bin == prevBin) {
                    ++accumulator;
                } else {
                    if (accumulator > 0) {
                        atomicAdd(&histo[blockIdx.x * NUM_BINS + prevBin], accumulator);
                    }
                    accumulator = 1;
                    prevBin = bin;
                }
            }
        }
    }

    if (accumulator > 0 && prevBin >= 0) {
        atomicAdd(&histo[blockIdx.x * NUM_BINS + prevBin], accumulator);
    }

    __syncthreads();

    if (blockIdx.x == 0) {
        for (int b = 1; b < blocks; b++) {
            for (int bin = tid; bin < NUM_BINS; bin += blockDim.x) {
                int val = histo[b * NUM_BINS + bin];
                if (val > 0) {
                    atomicAdd(&histo[bin], val);
                }
            }
        }
    }
}

int main() {
    const char *input = "cuda challenge day number nineteen less go";
    int length = strlen(input);

    int *h_histo = new int[NUM_BINS]();
    char *h_data = new char[length];
    memcpy(h_data, input, length);

    int total_threads = (length + CFACTOR - 1) / CFACTOR;
    int BLOCKS = (total_threads + THREADS - 1) / THREADS;

    size_t histo_bytes = NUM_BINS * BLOCKS * sizeof(int);
    size_t data_bytes = length * sizeof(char);

    char *d_data;
    int *d_histo;
    cudaMalloc(&d_data, data_bytes);
    cudaMalloc(&d_histo, histo_bytes);
    cudaMemcpy(d_data, h_data, data_bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_histo, 0, histo_bytes);

    histogram_coarsened_interleaved_aggregated<<<BLOCKS, THREADS>>>(d_data, length, d_histo, BLOCKS);
    cudaDeviceSynchronize();

    cudaMemcpy(h_histo, d_histo, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < NUM_BINS; i++) {
        char start = 'a' + i * 4;
        char end = (i == NUM_BINS - 1) ? 'z' : start + 3;
        std::cout << "Bin " << i << " (" << start << "-" << end << "): " << h_histo[i] << std::endl;
    }

    cudaFree(d_data);
    cudaFree(d_histo);
    delete[] h_data;
    delete[] h_histo;

    return 0;
}
