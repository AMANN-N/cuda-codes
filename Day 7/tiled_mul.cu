#include <stdio.h>
#include <cuda_runtime.h>

#define N 1000  // Matrix size
#define TILE_SIZE 16 // Tile size for shared memory tiling

__global__ 
void matMulTiled(float *A, float *B, float *C, int width) 
{
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float Pvalue = 0;
    
    for (int t = 0; t < (width + TILE_SIZE - 1) / TILE_SIZE; ++t) 
    {
        int loadRow = row;
        int loadCol = t * TILE_SIZE + threadIdx.x;
        if (loadRow < width && loadCol < width)
            tileA[threadIdx.y][threadIdx.x] = A[loadRow * width + loadCol];
        else
            tileA[threadIdx.y][threadIdx.x] = 0;
        
        loadRow = t * TILE_SIZE + threadIdx.y;
        loadCol = col;
        if (loadRow < width && loadCol < width)
            tileB[threadIdx.y][threadIdx.x] = B[loadRow * width + loadCol];
        else
            tileB[threadIdx.y][threadIdx.x] = 0;
        
        __syncthreads();
        
        for (int k = 0; k < TILE_SIZE; ++k)
            Pvalue += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        
        __syncthreads();
    }
    
    if (row < width && col < width)
        C[row * width + col] = Pvalue;
}

int main() {
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;    
    size_t size = N * N * sizeof(float);

    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    for (int i = 0; i < N; i++) 
    {
        for (int j = 0; j < N; j++) 
        {
            h_A[i * N + j] = i * 1.0f;
            h_B[i * N + j] = i * 2.0f;
            h_C[i * N + j] = 0.0f;
        }
    }

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    matMulTiled<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            printf("h_C[%d][%d] = %f\n", i, j, h_C[i * N + j]);
        }
    }
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);  

    return 0;
}
