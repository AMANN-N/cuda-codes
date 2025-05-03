#include <iostream>
#include <cuda_runtime.h>

using namespace std;

struct JDSMatrix {
    float* values;               
    unsigned int* colIdx;       
    unsigned int* jdPtr;        
    unsigned int* perm;        
    unsigned int* nnzPerRow;   
    unsigned int numRows;
    unsigned int numCols;
};

__global__ void spmv_jds_kernel(const float* values, const unsigned int* colIdx,
                                const unsigned int* jdPtr, const unsigned int* perm,
                                const unsigned int* nnzPerRow, const float* x,
                                float* y, int numRows) {
    int logicalRow = threadIdx.x + blockIdx.x * blockDim.x;
    if (logicalRow < numRows) {
        float accum = 0.0f;
        int actualRow = perm[logicalRow];
        for (int j = 0; j < nnzPerRow[logicalRow]; ++j) {
            int idx = jdPtr[j] + logicalRow;
            accum += values[idx] * x[colIdx[idx]];
        }
        y[actualRow] = accum;
    }
}

int main() {
    // [5 8 0]
    // [0 0 3]
    // [6 0 0]

    // Row order (perm): [0, 2, 1]
    // nnz per row: [2, 1, 1]
    // jagged diagonals: [5 6 3 8]

    float h_values[] = {5.0f, 6.0f, 3.0f, 8.0f};
    unsigned int h_colIdx[] = {0, 0, 2, 1};
    unsigned int h_jdPtr[] = {0, 3};    
    unsigned int h_perm[] = {0, 2, 1};     
    unsigned int h_nnzPerRow[] = {2, 1, 1};

    float h_x[] = {1.0f, 2.0f, 3.0f};
    float h_y[3] = {0.0f, 0.0f, 0.0f};

    JDSMatrix A;
    A.values = h_values;
    A.colIdx = h_colIdx;
    A.jdPtr = h_jdPtr;
    A.perm = h_perm;
    A.nnzPerRow = h_nnzPerRow;
    A.numRows = 3;
    A.numCols = 3;

    float *d_values, *d_x, *d_y;
    unsigned int *d_colIdx, *d_jdPtr, *d_perm, *d_nnzPerRow;

    cudaMalloc(&d_values, 4 * sizeof(float));
    cudaMalloc(&d_colIdx, 4 * sizeof(unsigned int));
    cudaMalloc(&d_jdPtr, 2 * sizeof(unsigned int));
    cudaMalloc(&d_perm, 3 * sizeof(unsigned int));
    cudaMalloc(&d_nnzPerRow, 3 * sizeof(unsigned int));
    cudaMalloc(&d_x, 3 * sizeof(float));
    cudaMalloc(&d_y, 3 * sizeof(float));

    cudaMemcpy(d_values, A.values, 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx, A.colIdx, 4 * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_jdPtr, A.jdPtr, 2 * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_perm, A.perm, 3 * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nnzPerRow, A.nnzPerRow, 3 * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, 3 * sizeof(float));

    int threadsPerBlock = 256;
    int numBlocks = (A.numRows + threadsPerBlock - 1) / threadsPerBlock;

    spmv_jds_kernel<<<numBlocks, threadsPerBlock>>>(d_values, d_colIdx, d_jdPtr,
                                                    d_perm, d_nnzPerRow, d_x, d_y, A.numRows);
    cudaDeviceSynchronize();

    cudaMemcpy(h_y, d_y, 3 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_values);
    cudaFree(d_colIdx);
    cudaFree(d_jdPtr);
    cudaFree(d_perm);
    cudaFree(d_nnzPerRow);
    cudaFree(d_x);
    cudaFree(d_y);

    for (int i = 0; i < A.numRows; ++i) {
        cout << h_y[i] << " ";
    }
    cout << endl;

    return 0;
}
