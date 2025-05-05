#include <iostream>
#include <climits>
#include <cuda_runtime.h>
using namespace std;

struct CSRGraph {
    unsigned int numVerts;
    unsigned int* rowPtrs;   
    unsigned int* colInd;   
};

__global__ void bfs_csr_kernel(CSRGraph g, unsigned int* levels, unsigned int* prevFrontier, unsigned int* currFrontier, unsigned int* numPrevFrontier, unsigned int* numCurrFrontier, unsigned int currLevel) 
{
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < *numPrevFrontier) 
    {
        unsigned int u = prevFrontier[i];
        for (unsigned int edge = g.rowPtrs[u]; edge < g.rowPtrs[u + 1]; ++edge) 
        {
            unsigned int v = g.colInd[edge];
            if (atomicCAS(&levels[v], UINT_MAX, currLevel) == UINT_MAX) 
            {
                unsigned int idx = atomicAdd(numCurrFrontier, 1);
                currFrontier[idx] = v;
            }
        }
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

    unsigned int h_numPrev = 1, h_numCurr = 0;
    unsigned int h_prevFrontier[] = {start};
    cudaMemcpy(d_prevFrontier, h_prevFrontier, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_numPrev, &h_numPrev, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_numCurr, &h_numCurr, sizeof(unsigned int), cudaMemcpyHostToDevice);

    unsigned int currLevel = 1;

    while (h_numPrev > 0) {
        int threadsPerBlock = 256;
        int blocksPerGrid = (h_numPrev + threadsPerBlock - 1) / threadsPerBlock;

        bfs_csr_kernel<<<blocksPerGrid, threadsPerBlock>>>(*d_graph_ptr, d_levels, d_prevFrontier, d_currFrontier, d_numPrev, d_numCurr, currLevel);
        cudaDeviceSynchronize();

        unsigned int* temp = d_prevFrontier;
        d_prevFrontier = d_currFrontier;
        d_currFrontier = temp;
        cudaMemcpy(&h_numPrev, d_numCurr, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        h_numCurr = 0;
        cudaMemcpy(d_numCurr, &h_numCurr, sizeof(unsigned int), cudaMemcpyHostToDevice);

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
