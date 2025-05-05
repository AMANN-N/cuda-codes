#include <iostream>
#include <climits>
#include <cuda_runtime.h>
using namespace std;

struct CSRGraph {
    unsigned int numVerts;
    unsigned int* rowPtrs;
    unsigned int* colInd;
};

__global__ void bfs_csr_kernel_shared(CSRGraph g, unsigned int* levels, unsigned int* prevFrontier, unsigned int* currFrontier, unsigned int* numPrevFrontier, unsigned int* numCurrFrontier, unsigned int currLevel) {
    const unsigned int LOCAL_FRONTIER_CAPACITY = 1024;
    __shared__ unsigned int sharedFrontier[LOCAL_FRONTIER_CAPACITY];
    __shared__ unsigned int sharedCount;

    if (threadIdx.x == 0) sharedCount = 0;
    __syncthreads();

    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < *numPrevFrontier) {
        unsigned int u = prevFrontier[i];
        for (unsigned int e = g.rowPtrs[u]; e < g.rowPtrs[u + 1]; ++e) {
            unsigned int v = g.colInd[e];
            if (atomicCAS(&levels[v], UINT_MAX, currLevel) == UINT_MAX) {
                unsigned int localIdx = atomicAdd(&sharedCount, 1);
                if (localIdx < LOCAL_FRONTIER_CAPACITY) {
                    sharedFrontier[localIdx] = v;
                } else {
                    unsigned int globalIdx = atomicAdd(numCurrFrontier, 1);
                    currFrontier[globalIdx] = v;
                }
            }
        }
    }

    __syncthreads();

    __shared__ unsigned int startIdx;
    if (threadIdx.x == 0) {
        startIdx = atomicAdd(numCurrFrontier, sharedCount);
    }
    __syncthreads();

    for (unsigned int j = threadIdx.x; j < sharedCount; j += blockDim.x) {
        currFrontier[startIdx + j] = sharedFrontier[j];
    }
}

int main() {
    unsigned int numVerts = 4;
    unsigned int rowPtrs_h[] = {0, 1, 2, 3, 4}; 
    unsigned int colInd_h[] = {1, 2, 3, 0};   

    unsigned int start = 0;
    unsigned int levels_h[4];

    unsigned int *d_rowPtrs, *d_colInd, *d_levels, *d_prevFrontier, *d_currFrontier, *d_numPrev, *d_numCurr;

    cudaMalloc(&d_rowPtrs, (numVerts + 1) * sizeof(unsigned int));
    cudaMalloc(&d_colInd, rowPtrs_h[numVerts] * sizeof(unsigned int));
    cudaMalloc(&d_levels, numVerts * sizeof(unsigned int));
    cudaMalloc(&d_prevFrontier, numVerts * sizeof(unsigned int));
    cudaMalloc(&d_currFrontier, numVerts * sizeof(unsigned int));
    cudaMalloc(&d_numPrev, sizeof(unsigned int));
    cudaMalloc(&d_numCurr, sizeof(unsigned int));

    cudaMemcpy(d_rowPtrs, rowPtrs_h, (numVerts + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colInd, colInd_h, rowPtrs_h[numVerts] * sizeof(unsigned int), cudaMemcpyHostToDevice);

    for (unsigned int i = 0; i < numVerts; ++i) {
        levels_h[i] = (i == start) ? 0 : UINT_MAX;
    }
    cudaMemcpy(d_levels, levels_h, numVerts * sizeof(unsigned int), cudaMemcpyHostToDevice);

    CSRGraph g_d;
    g_d.numVerts = numVerts;
    g_d.rowPtrs = d_rowPtrs;
    g_d.colInd = d_colInd;

    CSRGraph* d_graph_ptr;
    cudaMalloc(&d_graph_ptr, sizeof(CSRGraph));
    cudaMemcpy(d_graph_ptr, &g_d, sizeof(CSRGraph), cudaMemcpyHostToDevice);

    unsigned int h_numPrev = 1;
    unsigned int h_numCurr = 0;
    unsigned int h_prevFrontier[] = {start};

    cudaMemcpy(d_prevFrontier, h_prevFrontier, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_numPrev, &h_numPrev, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_numCurr, &h_numCurr, sizeof(unsigned int), cudaMemcpyHostToDevice);

    unsigned int currLevel = 1;

    while (h_numPrev > 0) {
        h_numCurr = 0;
        cudaMemcpy(d_numCurr, &h_numCurr, sizeof(unsigned int), cudaMemcpyHostToDevice);

        int threadsPerBlock = 256;
        int blocksPerGrid = (h_numPrev + threadsPerBlock - 1) / threadsPerBlock;

        bfs_csr_kernel_shared<<<blocksPerGrid, threadsPerBlock>>>(*d_graph_ptr, d_levels, d_prevFrontier, d_currFrontier, d_numPrev, d_numCurr, currLevel);
        cudaDeviceSynchronize();

        unsigned int* temp = d_prevFrontier;
        d_prevFrontier = d_currFrontier;
        d_currFrontier = temp;

        cudaMemcpy(&h_numPrev, d_numCurr, sizeof(unsigned int), cudaMemcpyDeviceToHost);

        currLevel++;
    }

    cudaMemcpy(levels_h, d_levels, numVerts * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    for (unsigned int i = 0; i < numVerts; ++i) {
        cout << "Vertex " << i << ": Level " << levels_h[i] << endl;
    }

    cudaFree(d_rowPtrs);
    cudaFree(d_colInd);
    cudaFree(d_levels);
    cudaFree(d_prevFrontier);
    cudaFree(d_currFrontier);
    cudaFree(d_numPrev);
    cudaFree(d_numCurr);
    cudaFree(d_graph_ptr);

    return 0;
}
