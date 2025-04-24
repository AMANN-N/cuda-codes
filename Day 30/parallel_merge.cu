#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__device__ int co_rank(int k, int *A, int m, int *B, int n) {
    int i = min(k, m), j = k - i;
    int i_low = max(0, k - n), j_low = max(0, k - m);
    int delta;
    while (true) {
        if (i > 0 && j < n && A[i - 1] > B[j]) {
            delta = (i - i_low + 1) / 2;
            j_low = j;
            j += delta;
            i -= delta;
        } else if (j > 0 && i < m && B[j - 1] > A[i]) {
            delta = (j - j_low + 1) / 2;
            i_low = i;
            i += delta;
            j -= delta;
        } else {
            break;
        }
    }
    return i;
}

__device__ void merge_seq(int *A, int m, int *B, int n, int *C) {
    int i = 0, j = 0, k = 0;
    while (i < m && j < n) {
        C[k++] = (A[i] < B[j]) ? A[i++] : B[j++];
    }
    while (i < m) C[k++] = A[i++];
    while (j < n) C[k++] = B[j++];
}

__global__ void merge_kernel(int *A, int m, int *B, int n, int *C) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = m + n;
    int elemsPerThread = (total + gridDim.x * blockDim.x - 1) / (gridDim.x * blockDim.x);

    int k_start = tid * elemsPerThread;
    int k_end = min(k_start + elemsPerThread, total);

    int i_start = co_rank(k_start, A, m, B, n);
    int i_end = co_rank(k_end, A, m, B, n);
    int j_start = k_start - i_start;
    int j_end = k_end - i_end;

    merge_seq(&A[i_start], i_end - i_start, &B[j_start], j_end - j_start, &C[k_start]);
}

int main() {
    int A[] = {1, 3, 5, 7};
    int B[] = {2, 4, 6, 8};
    int C[8];

    int m = 4;
    int n = 4;

    int *d_A, *d_B, *d_C;
    size_t size_A = m * sizeof(int);
    size_t size_B = n * sizeof(int);
    size_t size_C = (m + n) * sizeof(int);

    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);

    int blockSize = 128;
    int numBlocks = (m + n + blockSize - 1) / blockSize;

    merge_kernel<<<numBlocks, blockSize>>>(d_A, m, d_B, n, d_C);

    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cout << "Parallel Merged Array: ";
    for (int i = 0; i < 8; i++) cout << C[i] << " ";
    cout << endl;

    return 0;
}
