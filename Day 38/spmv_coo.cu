#include <iostream>
#include <cuda_runtime.h>

using namespace std;

struct COOMatrix {
    unsigned int* rowIdx;
    unsigned int* colIdx;
    float* values;
    unsigned int numNonZeros;
    unsigned int numRows;
    unsigned int numCols;
};

__global__ void spmv_coo_kernel(const unsigned int* rowIdx, const unsigned int* colIdx,
                                const float* values, const float* x, float* y, int nnz) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < nnz) {
        int row = rowIdx[tid];
        int col = colIdx[tid];
        float val = values[tid];
        atomicAdd(&y[row], val * x[col]);
    }
}

int main() {
    unsigned int h_rowIdx[] = {0, 0, 1};
    unsigned int h_colIdx[] = {0, 1, 1};
    float h_values[] = {1.0f, 2.0f, 3.0f};
    float h_x[] = {1.0f, 2.0f};
    float h_y[2] = {0.0f, 0.0f};

    COOMatrix A;
    A.rowIdx = h_rowIdx;
    A.colIdx = h_colIdx;
    A.values = h_values;
    A.numNonZeros = 3;
    A.numRows = 2;
    A.numCols = 2;

    unsigned int *d_rowIdx, *d_colIdx;
    float *d_values, *d_x, *d_y;

    cudaMalloc(&d_rowIdx, A.numNonZeros * sizeof(unsigned int));
    cudaMalloc(&d_colIdx, A.numNonZeros * sizeof(unsigned int));
    cudaMalloc(&d_values, A.numNonZeros * sizeof(float));
    cudaMalloc(&d_x, A.numCols * sizeof(float));
    cudaMalloc(&d_y, A.numRows * sizeof(float));

    cudaMemcpy(d_rowIdx, A.rowIdx, A.numNonZeros * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx, A.colIdx, A.numNonZeros * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, A.values, A.numNonZeros * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, A.numCols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, A.numRows * sizeof(float));

    int threadsPerBlock = 256;
    int blocks = (A.numNonZeros + threadsPerBlock - 1) / threadsPerBlock;
    spmv_coo_kernel<<<blocks, threadsPerBlock>>>(d_rowIdx, d_colIdx, d_values, d_x, d_y, A.numNonZeros);
    cudaDeviceSynchronize();

    cudaMemcpy(h_y, d_y, A.numRows * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_rowIdx);
    cudaFree(d_colIdx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);

    for (int i = 0; i < A.numRows; ++i) {
        cout << h_y[i] << " ";
    }
    cout << endl;

    return 0;
}
