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

__device__ void merge_sequential(const int* A, int m, const int* B, int n, int* C) {
    int idxA = 0, idxB = 0, idxC = 0;
    while (idxA < m && idxB < n) {
        C[idxC++] = (A[idxA] <= B[idxB]) ? A[idxA++] : B[idxB++];
    }
    while (idxA < m) C[idxC++] = A[idxA++];
    while (idxB < n) C[idxC++] = B[idxB++];
}

__global__ void merge_kernel(const int* A, int m, const int* B, int n, int* C) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElems = m + n;
    int totalThreads = gridDim.x * blockDim.x;
    int elemsPerThread = (totalElems + totalThreads - 1) / totalThreads;

    int k_start = tid * elemsPerThread;
    int k_end = min(k_start + elemsPerThread, totalElems);

    int i_start = co_rank(k_start, A, m, B, n);
    int i_end = co_rank(k_end, A, m, B, n);
    int j_start = k_start - i_start;
    int j_end = k_end - i_end;

    merge_sequential(&A[i_start], i_end - i_start, &B[j_start], j_end - j_start, &C[k_start]);
}

int main() {
    int A[] = {1, 3, 5, 7, 9};
    int B[] = {2, 4, 6, 8};
    int m = sizeof(A) / sizeof(A[0]);
    int n = sizeof(B) / sizeof(B[0]);
    int total = m + n;
    int C[total];

    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m * sizeof(int));
    cudaMalloc(&d_B, n * sizeof(int));
    cudaMalloc(&d_C, total * sizeof(int));

    cudaMemcpy(d_A, A, m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 128;
    int gridSize = (total + blockSize - 1) / blockSize;

    merge_kernel<<<gridSize, blockSize>>>(d_A, m, d_B, n, d_C);
    cudaMemcpy(C, d_C, total * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cout << "Merged output: ";
    for (int i = 0; i < total; i++) cout << C[i] << " ";
    cout << endl;

    return 0;
}
