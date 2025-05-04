#include <iostream>
#include <cuda_runtime.h>
#include <climits>

using namespace std;

struct CSRGraph {
    unsigned int numVerts;
    unsigned int* rowPtrs;  
    unsigned int* colInds;  
};

__global__ void bfs_kernel(CSRGraph g, unsigned int* levels, unsigned int* visitedFlag, unsigned int currLvl) 
{
    unsigned int v = threadIdx.x + blockIdx.x * blockDim.x;
    if (v < g.numVerts && levels[v] == currLvl - 1) 
    {
        for (unsigned int i = g.rowPtrs[v]; i < g.rowPtrs[v + 1]; ++i) 
        {
            unsigned int neighbor = g.colInds[i];
            if (levels[neighbor] == UINT_MAX) 
            {
                levels[neighbor] = currLvl;
                *visitedFlag = 1;
            }
        }
    }
}

int main() {
    unsigned int numVerts = 4;
    unsigned int h_rowPtrs[] = {0, 1, 2, 3, 4};
    unsigned int h_colInds[] = {1, 2, 3, 0};
    unsigned int start = 0;

    unsigned int* d_rowPtrs;
    unsigned int* d_colInds;
    unsigned int* d_levels;
    unsigned int* d_flag;

    cudaMalloc(&d_rowPtrs, (numVerts + 1) * sizeof(unsigned int));
    cudaMalloc(&d_colInds, h_rowPtrs[numVerts] * sizeof(unsigned int));
    cudaMalloc(&d_levels, numVerts * sizeof(unsigned int));
    cudaMalloc(&d_flag, sizeof(unsigned int));

    cudaMemcpy(d_rowPtrs, h_rowPtrs, (numVerts + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colInds, h_colInds, h_rowPtrs[numVerts] * sizeof(unsigned int), cudaMemcpyHostToDevice);

    unsigned int h_levels[4];
    for (unsigned int i = 0; i < numVerts; ++i) {
        h_levels[i] = (i == start) ? 0 : UINT_MAX;
    }

    cudaMemcpy(d_levels, h_levels, numVerts * sizeof(unsigned int), cudaMemcpyHostToDevice);

    CSRGraph d_graph;
    d_graph.numVerts = numVerts;
    d_graph.rowPtrs = d_rowPtrs;
    d_graph.colInds = d_colInds;

    CSRGraph* d_graph_ptr;
    cudaMalloc(&d_graph_ptr, sizeof(CSRGraph));
    cudaMemcpy(d_graph_ptr, &d_graph, sizeof(CSRGraph), cudaMemcpyHostToDevice);

    unsigned int currLvl = 1;
    unsigned int h_flag = 1;

    while (h_flag) {
        cudaMemset(d_flag, 0, sizeof(unsigned int));

        int threadsPerBlock = 256;
        int blocksPerGrid = (numVerts + threadsPerBlock - 1) / threadsPerBlock;

        bfs_kernel<<<blocksPerGrid, threadsPerBlock>>>(*d_graph_ptr, d_levels, d_flag, currLvl);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_flag, d_flag, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        currLvl++;
    }

    cudaMemcpy(h_levels, d_levels, numVerts * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    for (unsigned int i = 0; i < numVerts; ++i) {
        cout << "Vertex " << i << ": Level " << h_levels[i] << endl;
    }

    cudaFree(d_rowPtrs);
    cudaFree(d_colInds);
    cudaFree(d_levels);
    cudaFree(d_flag);
    cudaFree(d_graph_ptr);

    return 0;
}
