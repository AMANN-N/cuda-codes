#include <iostream>
#include <climits>
#include <cuda_runtime.h>
using namespace std;

struct COOGraph {
    unsigned int numEdges;
    unsigned int numVerts;
    unsigned int* srcVerts;
    unsigned int* dstVerts;
};

__global__ void bfs_edge_kernel(COOGraph g, unsigned int* levels, unsigned int* visitedFlag, unsigned int currLvl) {
    unsigned int e = threadIdx.x + blockIdx.x * blockDim.x;
    if (e < g.numEdges) {
        unsigned int u = g.srcVerts[e];
        unsigned int v = g.dstVerts[e];
        if (levels[u] == currLvl - 1 && levels[v] == UINT_MAX) {
            levels[v] = currLvl;
            *visitedFlag = 1;
        }
    }
}

int main() {

    unsigned int numVerts = 4;
    unsigned int numEdges = 3;
    unsigned int h_src[] = {0, 1, 2};
    unsigned int h_dst[] = {1, 2, 3};
    unsigned int start = 0;
    unsigned int h_levels[4];

    unsigned int *d_src, *d_dst, *d_levels, *d_flag;

    cudaMalloc(&d_src, numEdges * sizeof(unsigned int));
    cudaMalloc(&d_dst, numEdges * sizeof(unsigned int));
    cudaMalloc(&d_levels, numVerts * sizeof(unsigned int));
    cudaMalloc(&d_flag, sizeof(unsigned int));

    cudaMemcpy(d_src, h_src, numEdges * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst, h_dst, numEdges * sizeof(unsigned int), cudaMemcpyHostToDevice);

    for (unsigned int i = 0; i < numVerts; ++i) {
        h_levels[i] = (i == start) ? 0 : UINT_MAX;
    }
    cudaMemcpy(d_levels, h_levels, numVerts * sizeof(unsigned int), cudaMemcpyHostToDevice);

    COOGraph d_graph;
    d_graph.numEdges = numEdges;
    d_graph.numVerts = numVerts;
    d_graph.srcVerts = d_src;
    d_graph.dstVerts = d_dst;

    COOGraph* d_graph_ptr;
    cudaMalloc(&d_graph_ptr, sizeof(COOGraph));
    cudaMemcpy(d_graph_ptr, &d_graph, sizeof(COOGraph), cudaMemcpyHostToDevice);

    unsigned int currLvl = 1;
    unsigned int h_flag = 1;

    while (h_flag) {
        cudaMemset(d_flag, 0, sizeof(unsigned int));

        int threadsPerBlock = 256;
        int blocksPerGrid = (numEdges + threadsPerBlock - 1) / threadsPerBlock;

        bfs_edge_kernel<<<blocksPerGrid, threadsPerBlock>>>(*d_graph_ptr, d_levels, d_flag, currLvl);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_flag, d_flag, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        currLvl++;
    }

    cudaMemcpy(h_levels, d_levels, numVerts * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    for (unsigned int i = 0; i < numVerts; ++i) {
        cout << "Vertex " << i << ": Level " << h_levels[i] << endl;
    }

    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_levels);
    cudaFree(d_flag);
    cudaFree(d_graph_ptr);

    return 0;
}
