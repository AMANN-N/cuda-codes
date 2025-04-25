#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__device__ int co_rank(int k, const int* A, int m, const int* B, int n) {
    int i_low = max(0, k - n);
    int i_high = min(k, m);
    int i, j;

    while (i_low < i_high) {
        i = (i_low + i_high) / 2;
        j = k - i;

        if (i > 0 && j < n && A[i - 1] > B[j]) {
            i_high = i;  
        } else if (j > 0 && i < m && B[j - 1] > A[i]) {
            i_low = i + 1;  
        } else {
            return i;
        }
    }
    return i_low;
}

__device__ void merge_seq(const int* A, int m, const int* B, int n, int* C) {
    int i = 0, j = 0, k = 0;
    while (i < m && j < n) {
        C[k++] = (A[i] <= B[j]) ? A[i++] : B[j++];
    }
    while (i < m) C[k++] = A[i++];
    while (j < n) C[k++] = B[j++];
}

__global__ void merge_kernel(const int* A, int m, const int* B, int n, int* C) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = m + n;
    int threads = gridDim.x * blockDim.x;
    int elemsPerThread = (total + threads - 1) / threads;

    int k_start = tid * elemsPerThread;
    int k_end = min(k_start + elemsPerThread, total);

    int i_start = co_rank(k_start, A, m, B, n);
    int i_end = co_rank(k_end, A, m, B, n);
    int j_start = k_start - i_start;
    int j_end = k_end - i_end;

    merge_seq(&A[i_start], i_end - i_start, &B[j_start], j_end - j_start, &C[k_start]);
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
    int numBlocks = (total + blockSize - 1) / blockSize;

    merge_kernel<<<numBlocks, blockSize>>>(d_A, m, d_B, n, d_C);
    cudaMemcpy(C, d_C, total * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cout << "Parallel Merged Array: ";
    for (int i = 0; i < total; i++) cout << C[i] << " ";
    cout << endl;

    return 0;
}
