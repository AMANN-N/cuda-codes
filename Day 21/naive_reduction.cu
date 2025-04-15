__global__ void naive_reduction(float *input , float *output)
{
    unsigned int i = threadIdx.x * 2;

    for(unsigned int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if(threadIdx.x % stride == 0)
        {
            input[i] += input[i + stride];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0)
    {
        *output = input[0];
    }
}
