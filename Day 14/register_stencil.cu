#include <iostream>
#include <cuda_runtime.h>

#define N 6
#define TILE_X 4
#define TILE_Y 4
#define TILE_Z 4
#define RAD 1

__global__
void register_tiled_stencil_3d(float *input, float *output, int n) {
    __shared__ float inCurr[TILE_Y + 2 * RAD][TILE_X + 2 * RAD];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * TILE_X + tx;
    int y = blockIdx.y * TILE_Y + ty;

    if (x >= n || y >= n) return;
    float inPrev, inNext, inCenter;

    for (int zBase = RAD; zBase < n - RAD; zBase += TILE_Z) {
        for (int dz = 0; dz < TILE_Z + 2; dz++) {
            int z = zBase + dz - 1;
            if (z >= 0 && z < n) {
                int idx = z * n * n + y * n + x;
                float val = input[idx];

                if (dz == 0)
                    inPrev = val;
                else if (dz == 1) {
                    inCenter = val;
                    inCurr[ty + RAD][tx + RAD] = val;
                } else
                    inNext = val;
            }
        }

        __syncthreads();

        for (int dz = 0; dz < TILE_Z; dz++) {
            int z = zBase + dz;
            if (z < RAD || z >= n - RAD) continue;

            float center = inCenter;
            float left   = (tx > 0) ? inCurr[ty + RAD][tx + RAD - 1] : 0.0f;
            float right  = (tx < TILE_X - 1) ? inCurr[ty + RAD][tx + RAD + 1] : 0.0f;
            float up     = (ty > 0) ? inCurr[ty + RAD - 1][tx + RAD] : 0.0f;
            float down   = (ty < TILE_Y - 1) ? inCurr[ty + RAD + 1][tx + RAD] : 0.0f;
            float front  = inPrev;
            float back   = inNext;

            float result = 0.4f * center +
                           0.1f * (left + right + up + down + front + back);

            int outIdx = z * n * n + y * n + x;
            output[outIdx] = result;
            inPrev = inCenter;
            inCenter = inNext;
            int nextZ = z + 2;
            if (nextZ < n) {
                int nextIdx = nextZ * n * n + y * n + x;
                inNext = input[nextIdx];
            }
            inCurr[ty + RAD][tx + RAD] = inCenter;

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
    register_tiled_stencil_3d<<<grid, block>>>(d_input, d_output, N);
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
