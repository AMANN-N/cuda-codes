#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__device__ int co_rank(int k, const int* A, int m, const int* B, int n) {
    int low = max(0, k - n);
    int high = min(k, m);
    int i, j;

    while (low < high) {
        i = (low + high) / 2;
        j = k - i;

        if (i > 0 && j < n && A[i - 1] > B[j]) {
            high = i;
        } else if (j > 0 && i < m && B[j - 1] > A[i]) {
            low = i + 1;
        } else {
            return i;
        }
    }
    return low;
}

__device__ void merge_sequential_circular(const int* A_buf, int m, const int* B_buf, int n, int* C, int start_idx, int tile_size) {
    int idxA = 0, idxB = 0, idxC = start_idx;
    while (idxA < m && idxB < n) {
        int A_idx = idxA % tile_size;
        int B_idx = idxB % tile_size;
        if (A_buf[A_idx] <= B_buf[B_idx]) {
            C[idxC++] = A_buf[A_idx];
            idxA++;
        } else {
            C[idxC++] = B_buf[B_idx];
            idxB++;
        }
    }
    while (idxA < m) {
        int A_idx = idxA % tile_size;
        C[idxC++] = A_buf[A_idx];
        idxA++;
    }
    while (idxB < n) {
        int B_idx = idxB % tile_size;
        C[idxC++] = B_buf[B_idx];
        idxB++;
    }
}

__global__ void merge_kernel(const int* A, int m, const int* B, int n, int* C) {
    extern __shared__ int shared_mem[];

    int* A_buf = shared_mem;
    int* B_buf = &shared_mem[m];  

    int tid = threadIdx.x;
    int totalElems = m + n;

    if (tid < m) A_buf[tid] = A[tid];
    if (tid < n) B_buf[tid] = B[tid];
    __syncthreads();

    if (tid == 0) {
        merge_sequential_circular(A_buf, m, B_buf, n, C, 0, max(m, n));
    }
}

int main() {
    int A[] = {1, 3, 5, 7, 9};
    int B[] = {2, 4, 6, 8};
    int m = sizeof(A) / sizeof(A[0]);
    int n = sizeof(B) / sizeof(B[0]);
    int total = m + n;

    int* C = new int[total];

    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m * sizeof(int));
    cudaMalloc(&d_B, n * sizeof(int));
    cudaMalloc(&d_C, total * sizeof(int));

    cudaMemcpy(d_A, A, m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 128; 
    int gridSize = 1;    

    int sharedMemSize = (m + n) * sizeof(int);  

    merge_kernel<<<gridSize, blockSize, sharedMemSize>>>(d_A, m, d_B, n, d_C);
    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, total * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cout << "Merged output: ";
    for (int i = 0; i < total; i++) cout << C[i] << " ";
    cout << endl;

    delete[] C;
    return 0;
}
