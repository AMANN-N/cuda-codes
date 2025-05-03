#include <iostream>
#include <cuda_runtime.h>

using namespace std;

struct EllMatrix {
    unsigned int* colIdx;         
    float* values;                
    unsigned int* nnzPerRow;     
    unsigned int numRows;
    unsigned int numCols;
    unsigned int maxNnzPerRow;    
};

__global__ void spmv_ell_kernel(const unsigned int* colIdx, const float* values,
                                const unsigned int* nnzPerRow, const float* x,
                                float* y, int numRows, int maxNnzPerRow) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRows) {
        float temp = 0.0f;
        for (int i = 0; i < nnzPerRow[row]; ++i) {
            int idx = i * numRows + row;
            temp += values[idx] * x[colIdx[idx]];
        }
        y[row] = temp;
    }
}

int main() {
    // ELLPACK version of:
    // [1 2]
    // [0 3]
    // Padded with zero in second row first column
    unsigned int h_colIdx[] = {0, 1,   // row 0
                               1, 0};  // row 1 (padded)
    float h_values[] = {1.0f, 2.0f,
                        3.0f, 0.0f};
    unsigned int h_nnzPerRow[] = {2, 1};  

    float h_x[] = {1.0f, 2.0f};
    float h_y[2] = {0.0f, 0.0f};

    EllMatrix A;
    A.colIdx = h_colIdx;
    A.values = h_values;
    A.nnzPerRow = h_nnzPerRow;
    A.numRows = 2;
    A.numCols = 2;
    A.maxNnzPerRow = 2;

    unsigned int *d_colIdx, *d_nnzPerRow;
    float *d_values, *d_x, *d_y;

    cudaMalloc(&d_colIdx, A.numRows * A.maxNnzPerRow * sizeof(unsigned int));
    cudaMalloc(&d_values, A.numRows * A.maxNnzPerRow * sizeof(float));
    cudaMalloc(&d_nnzPerRow, A.numRows * sizeof(unsigned int));
    cudaMalloc(&d_x, A.numCols * sizeof(float));
    cudaMalloc(&d_y, A.numRows * sizeof(float));

    cudaMemcpy(d_colIdx, A.colIdx, A.numRows * A.maxNnzPerRow * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, A.values, A.numRows * A.maxNnzPerRow * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nnzPerRow, A.nnzPerRow, A.numRows * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, A.numCols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, A.numRows * sizeof(float));

    int threadsPerBlock = 256;
    int numBlocks = (A.numRows + threadsPerBlock - 1) / threadsPerBlock;
    spmv_ell_kernel<<<numBlocks, threadsPerBlock>>>(
        d_colIdx, d_values, d_nnzPerRow, d_x, d_y, A.numRows, A.maxNnzPerRow
    );
    cudaDeviceSynchronize();

    cudaMemcpy(h_y, d_y, A.numRows * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_colIdx);
    cudaFree(d_values);
    cudaFree(d_nnzPerRow);
    cudaFree(d_x);
    cudaFree(d_y);

    for (int i = 0; i < A.numRows; ++i) {
        cout << h_y[i] << " ";
    }
    cout << endl;

    return 0;
}
