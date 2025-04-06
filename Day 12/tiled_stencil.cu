#include <stdio.h>
#include <cuda_runtime.h>

#define N 5                      
#define TILE_SIZE 4            
#define RADIUS 1                
#define BLOCK_SIZE (TILE_SIZE + 2 * RADIUS) 

#define C0 1.0f
#define C1 0.1f
#define C2 0.1f
#define C3 0.1f
#define C4 0.1f
#define C5 0.1f
#define C6 0.1f

__global__ void tiled_stencil_3d(float *input, float *output, int width) 
{
    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    int x = blockIdx.x * TILE_SIZE + tx - RADIUS;
    int y = blockIdx.y * TILE_SIZE + ty - RADIUS;
    int z = blockIdx.z * TILE_SIZE + tz - RADIUS;

    int index = z * width * width + y * width + x;


    if (x >= 0 && x < width && y >= 0 && y < width && z >= 0 && z < width) {
        tile[tz][ty][tx] = input[index];
    } else {
        tile[tz][ty][tx] = 0.0f; 
    }

    __syncthreads();

    if (tx >= RADIUS && tx < BLOCK_SIZE - RADIUS &&
        ty >= RADIUS && ty < BLOCK_SIZE - RADIUS &&
        tz >= RADIUS && tz < BLOCK_SIZE - RADIUS &&
        x >= 1 && x < width - 1 &&
        y >= 1 && y < width - 1 &&
        z >= 1 && z < width - 1) {

        float result = C0 * tile[tz][ty][tx] +
                       C1 * tile[tz][ty][tx - 1] +
                       C2 * tile[tz][ty][tx + 1] +
                       C3 * tile[tz][ty - 1][tx] +
                       C4 * tile[tz][ty + 1][tx] +
                       C5 * tile[tz - 1][ty][tx] +
                       C6 * tile[tz + 1][ty][tx];

        output[index] = result;
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
                h_input[idx] = (float)idx;
                h_output[idx] = 0.0f;
            }
        }
    }

    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (N + TILE_SIZE - 1) / TILE_SIZE,
              (N + TILE_SIZE - 1) / TILE_SIZE);

    tiled_stencil_3d<<<grid, block>>>(d_input, d_output, N);

    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);


    printf("Stencil output (center region):\n");
    for (int z = 1; z < N - 1; z++) {
        for (int y = 1; y < N - 1; y++) {
            for (int x = 1; x < N - 1; x++) {
                int idx = z * N * N + y * N + x;
                printf("output[%d][%d][%d] = %.2f\n", z, y, x, h_output[idx]);
            }
        }
    }

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
