#include <iostream>
#include <climits>
#include <cuda_runtime.h>
using namespace std;

struct CSCGraph {
    unsigned int numVerts;
    unsigned int* colPtrs; 
    unsigned int* rowInds; 
};

__global__ void bfs_pull_kernel(CSCGraph g, unsigned int* levels, unsigned int* changed, unsigned int currLvl) {
    unsigned int v = threadIdx.x + blockIdx.x * blockDim.x;
    if (v < g.numVerts && levels[v] == UINT_MAX) {
        for (unsigned int i = g.colPtrs[v]; i < g.colPtrs[v + 1]; ++i) {
            unsigned int neighbor = g.rowInds[i];
            if (levels[neighbor] == currLvl - 1) {
                levels[v] = currLvl;
                atomicExch(changed, 1);
                break;
            }
        }
    }
}

int main() {

    unsigned int numVerts = 4;
    unsigned int h_colPtrs[] = {0, 0, 1, 2, 3};  
    unsigned int h_rowInds[] = {0, 1, 2};       

    unsigned int start = 0;
    unsigned int h_levels[4];

    unsigned int *d_colPtrs, *d_rowInds, *d_levels, *d_changed;

    cudaMalloc(&d_colPtrs, (numVerts + 1) * sizeof(unsigned int));
    cudaMalloc(&d_rowInds, h_colPtrs[numVerts] * sizeof(unsigned int));
    cudaMalloc(&d_levels, numVerts * sizeof(unsigned int));
    cudaMalloc(&d_changed, sizeof(unsigned int));

    cudaMemcpy(d_colPtrs, h_colPtrs, (numVerts + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rowInds, h_rowInds, h_colPtrs[numVerts] * sizeof(unsigned int), cudaMemcpyHostToDevice);

    for (unsigned int i = 0; i < numVerts; i++) {
        h_levels[i] = (i == start) ? 0 : UINT_MAX;
    }

    cudaMemcpy(d_levels, h_levels, numVerts * sizeof(unsigned int), cudaMemcpyHostToDevice);

    CSCGraph d_graph;
    d_graph.numVerts = numVerts;
    d_graph.colPtrs = d_colPtrs;
    d_graph.rowInds = d_rowInds;

    CSCGraph* d_graph_ptr;
    cudaMalloc(&d_graph_ptr, sizeof(CSCGraph));
    cudaMemcpy(d_graph_ptr, &d_graph, sizeof(CSCGraph), cudaMemcpyHostToDevice);

    unsigned int currLvl = 1;
    unsigned int h_changed = 1;

    while (h_changed) {
        cudaMemset(d_changed, 0, sizeof(unsigned int));

        int threadsPerBlock = 256;
        int blocksPerGrid = (numVerts + threadsPerBlock - 1) / threadsPerBlock;

        bfs_pull_kernel<<<blocksPerGrid, threadsPerBlock>>>(*d_graph_ptr, d_levels, d_changed, currLvl);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_changed, d_changed, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        currLvl++;
    }

    cudaMemcpy(h_levels, d_levels, numVerts * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    for (unsigned int i = 0; i < numVerts; i++) {
        cout << "Vertex " << i << ": Level " << h_levels[i] << endl;
    }

    cudaFree(d_colPtrs);
    cudaFree(d_rowInds);
    cudaFree(d_levels);
    cudaFree(d_changed);
    cudaFree(d_graph_ptr);

    return 0;
}
