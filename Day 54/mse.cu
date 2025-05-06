#include <cuda_runtime.h>
#include <cstdio>

#define NUM_VALS 1000

__global__ void mse_reduce_kernel(const float* preds, const float* labels, float* partial_sums, int n) {
    extern __shared__ float local_data[];
    int tid = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    float diff_squared = 0.0f;
    if (global_id < n) {
        float err = preds[global_id] - labels[global_id];
        diff_squared = err * err;
    }

    local_data[tid] = diff_squared;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            local_data[tid] += local_data[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = local_data[0];
    }
}

int main() {
    size_t data_size = NUM_VALS * sizeof(float);
    float *host_preds = (float*)malloc(data_size);
    float *host_labels = (float*)malloc(data_size);

    for (int i = 0; i < NUM_VALS; ++i) {
        host_preds[i] = 1.0f;
        host_labels[i] = 2.0f;
    }

    float *dev_preds, *dev_labels, *dev_partials;
    int threads = 256;
    int blocks = (NUM_VALS + threads - 1) / threads;

    cudaMalloc(&dev_preds, data_size);
    cudaMalloc(&dev_labels, data_size);
    cudaMalloc(&dev_partials, blocks * sizeof(float));

    cudaMemcpy(dev_preds, host_preds, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_labels, host_labels, data_size, cudaMemcpyHostToDevice);


    mse_reduce_kernel<<<blocks, threads, threads * sizeof(float)>>>(dev_preds, dev_labels, dev_partials, NUM_VALS);
    cudaDeviceSynchronize();

    float* host_partials = (float*)malloc(blocks * sizeof(float));
    cudaMemcpy(host_partials, dev_partials, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    float sum = 0.0f;
    for (int i = 0; i < blocks; ++i) {
        sum += host_partials[i];
    }

    float mse = sum / NUM_VALS;
    printf("Mean Squared Error (CUDA) = %.2f\n", mse);

    cudaFree(dev_preds);
    cudaFree(dev_labels);
    cudaFree(dev_partials);
    free(host_preds);
    free(host_labels);
    free(host_partials);

    return 0;
}
