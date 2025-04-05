#include <stdio.h>
#include <cuda_runtime.h>

#define N 5  

__global__ 
void stencil_3d(float *input, float *output, int width)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= 1 && x < width - 1 &&
        y >= 1 && y < width - 1 &&
        z >= 1 && z < width - 1)
    {
        int idx = z * width * width + y * width + x;

        float center = input[idx];
        float xp = input[z * width * width + y * width + (x + 1)];
        float xm = input[z * width * width + y * width + (x - 1)];
        float yp = input[z * width * width + (y + 1) * width + x];
        float ym = input[z * width * width + (y - 1) * width + x];
        float zp = input[(z + 1) * width * width + y * width + x];
        float zm = input[(z - 1) * width * width + y * width + x];

        output[idx] = center + xp + xm + yp + ym + zp + zm;
    }
}

int main() {
    float *h_input, *h_output;
    float *d_input, *d_output;
    size_t size = N * N * N * sizeof(float);

    h_input = (float*)malloc(size);
    h_output = (float*)malloc(size);

    for (int z = 0; z < N; z++) {
        for (int y = 0; y < N; y++) {
            for (int x = 0; x < N; x++) {
                int idx = z * N * N + y * N + x;
                h_input[idx] = idx; 
                h_output[idx] = 0.0f;
            }
        }
    }

    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    dim3 block(8, 8, 8);
    dim3 grid((N + block.x - 1) / block.x,
              (N + block.y - 1) / block.y,
              (N + block.z - 1) / block.z);

    stencil_3d<<<grid, block>>>(d_input, d_output, N);

    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    printf("Sample Output (center points only):\n");
    for (int z = 1; z < N - 1; z++) {
        for (int y = 1; y < N - 1; y++) {
            for (int x = 1; x < N - 1; x++) {
                int idx = z * N * N + y * N + x;
                printf("output[%d][%d][%d] = %.1f\n", z, y, x, h_output[idx]);
            }
        }
    }

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
