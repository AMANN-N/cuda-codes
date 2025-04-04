#include <stdio.h>
#include <cuda_runtime.h>

#define N 1000


__global__ 
void blurImg(float *A, float *C, int width)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;  // Column 
    int j = threadIdx.y + blockIdx.y * blockDim.y;  // Row 

    if(i < width && j < width)
    {
        float sum = 0.0f;
        int count = 0;

        for(int di = -2; di <= 2; di++)
        {
            for(int dj = -2; dj <= 2; dj++)
            {
                int ni = i + di;
                int nj = j + dj;


                if(ni >= 0 && ni < width && nj >= 0 && nj < width)
                {
                    sum += A[nj * width + ni];
                    count++;
                }
            }
        }
        
        C[j * width + i] = sum / count;  
    }           
}

int main() {
    float *h_A, *h_C;
    float *d_A, *d_C;    
    size_t size = N * N * sizeof(float);

    h_A = (float*)malloc(size);
    h_C = (float*)malloc(size);


    for (int i = 0; i < N; i++) 
    {
        for(int j = 0; j < N; j++)
        {
            h_A[i * N + j] = (float)(i + j); 
            h_C[i * N + j] = 0.0f; 
        }
    }

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_C, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);


    blurImg<<<grid, block>>>(d_A, d_C, N);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("Blurred Image (First 5x5 pixels):\n");
    for(int i = 0; i < 5; i++)
    {
        for(int j = 0; j < 5; j++)
        {
            printf("%0.2f ", h_C[i * N + j]);
        }
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_C);
    free(h_A);
    free(h_C);  

    return 0;
}
