#include <iostream>
#include <cuda_runtime.h>

#define N 6
#define TILE_X 4
#define TILE_Y 4
#define TILE_Z 4
#define RAD 1

__global__
void tiled_stencil_3d_coarsened(float *input, float *output, int n) {
    __shared__ float in0[TILE_Y + 2 * RAD][TILE_X + 2 * RAD];
    __shared__ float in1[TILE_Y + 2 * RAD][TILE_X + 2 * RAD];
    __shared__ float in2[TILE_Y + 2 * RAD][TILE_X + 2 * RAD];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * TILE_X + tx;
    int y = blockIdx.y * TILE_Y + ty;

    if (x >= n || y >= n) return;

    for (int zBase = 1; zBase < n - 1; zBase += TILE_Z) {
        // Load in0, in1, in2
        int z0 = zBase - 1;
        int z1 = zBase;
        int z2 = zBase + 1;

        if (z0 < n) {
            in0[ty + RAD][tx + RAD] = input[z0 * n * n + y * n + x];
        }
        if (z1 < n) {
            in1[ty + RAD][tx + RAD] = input[z1 * n * n + y * n + x];
        }
        if (z2 < n) {
            in2[ty + RAD][tx + RAD] = input[z2 * n * n + y * n + x];
        }

        __syncthreads();

        for (int dz = 0; dz < TILE_Z; dz++) {
            int z = zBase + dz;
            if (z <= 0 || z >= n - 1) continue;

            float center = in1[ty + RAD][tx + RAD];
            float left   = (tx > 0)              ? in1[ty + RAD][tx + RAD - 1] : 0.0f;
            float right  = (tx < TILE_X - 1)     ? in1[ty + RAD][tx + RAD + 1] : 0.0f;
            float up     = (ty > 0)              ? in1[ty + RAD - 1][tx + RAD] : 0.0f;
            float down   = (ty < TILE_Y - 1)     ? in1[ty + RAD + 1][tx + RAD] : 0.0f;
            float front  = in0[ty + RAD][tx + RAD];
            float back   = in2[ty + RAD][tx + RAD];

            float result = 0.4f * center +
                           0.1f * (left + right + up + down + front + back);

            output[z * n * n + y * n + x] = result;

            // Rotate planes
            __syncthreads();
            if (dz < TILE_Z - 1 && z + 2 < n) {
                // Rotate: in0 <- in1, in1 <- in2, in2 <- new
                in0[ty + RAD][tx + RAD] = in1[ty + RAD][tx + RAD];
                in1[ty + RAD][tx + RAD] = in2[ty + RAD][tx + RAD];
                in2[ty + RAD][tx + RAD] = input[(z + 2) * n * n + y * n + x];
            }
            __syncthreads();
        }
    }
}

int main() {
    size_t size = N * N * N * sizeof(float);
    float *h_input = new float[N * N * N];
    float *h_output = new float[N * N * N];

    for (int z = 0; z < N; z++)
        for (int y = 0; y < N; y++)
            for (int x = 0; x < N; x++)
                h_input[z * N * N + y * N + x] = static_cast<float>(z * N * N + y * N + x);

    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    dim3 block(TILE_X, TILE_Y);
    dim3 grid((N + TILE_X - 1) / TILE_X, (N + TILE_Y - 1) / TILE_Y);
    tiled_stencil_3d_coarsened<<<grid, block>>>(d_input, d_output, N);

    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    printf("Stencil output (center region):\n");
    for (int z = 1; z < 4; ++z)
        for (int y = 1; y < 4; ++y)
            for (int x = 1; x < 4; ++x)
                printf("output[%d][%d][%d] = %.2f\n", z, y, x, h_output[z * N * N + y * N + x]);

    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_output;
    return 0;
}
