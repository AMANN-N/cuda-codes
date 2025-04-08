#include <iostream>
#include <cuda_runtime.h>

#define N 6
#define TILE_X 4
#define TILE_Y 4
#define TILE_Z 4
#define RAD 1

__global__
void register_tiled_stencil_3d(float *input, float *output, int n) {
    __shared__ float smem[TILE_Y + 2*RAD][TILE_X + 2*RAD];
    
    float reg_prev, reg_curr, reg_next;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * TILE_X + tx;
    int y = blockIdx.y * TILE_Y + ty;
    
    if (x >= n || y >= n) return;
    
    // Initial load
    if (0 < n) reg_prev = input[0 * n * n + y * n + x];
    if (1 < n) reg_curr = input[1 * n * n + y * n + x];
    
    for (int zBase = 1; zBase < n - 1; zBase += 1) {  // Changed to process one z at a time
        // Load next plane
        int zNext = zBase + 1;
        if (zNext < n) {
            reg_next = input[zNext * n * n + y * n + x];
        }
        
        // Load current plane into shared memory
        smem[ty + RAD][tx + RAD] = reg_curr;
        
        // Load halos
        if (tx < RAD) {
            // Left halo
            smem[ty + RAD][tx] = (x - RAD >= 0) ? input[zBase * n * n + y * n + (x - RAD)] : 0.0f;
            // Right halo
            smem[ty + RAD][tx + TILE_X + RAD] = (x + TILE_X < n) ? input[zBase * n * n + y * n + (x + TILE_X)] : 0.0f;
        }
        
        if (ty < RAD) {
            // Top halo
            smem[ty][tx + RAD] = (y - RAD >= 0) ? input[zBase * n * n + (y - RAD) * n + x] : 0.0f;
            // Bottom halo
            smem[ty + TILE_Y + RAD][tx + RAD] = (y + TILE_Y < n) ? input[zBase * n * n + (y + TILE_Y) * n + x] : 0.0f;
        }
        
        __syncthreads();
        
        if (zBase >= 1 && zBase < n - 1) {
            float center = smem[ty + RAD][tx + RAD];
            float left = smem[ty + RAD][tx + RAD - 1];
            float right = smem[ty + RAD][tx + RAD + 1];
            float up = smem[ty + RAD - 1][tx + RAD];
            float down = smem[ty + RAD + 1][tx + RAD];
            float front = reg_prev;
            float back = reg_next;
            
            float result = 0.4f * center + 0.1f * (left + right + up + down + front + back);
            output[zBase * n * n + y * n + x] = result;
        }
        
        // Rotate registers
        reg_prev = reg_curr;
        reg_curr = reg_next;
        
        __syncthreads();
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