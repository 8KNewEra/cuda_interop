struct HistData {
    unsigned int hist_r[256];
    unsigned int hist_g[256];
    unsigned int hist_b[256];
    unsigned int max_r;
    unsigned int max_g;
    unsigned int max_b;
};

extern "C"
__global__ void histogram_shared_kernel(
    HistData* Histdata,
    cudaTextureObject_t texObj,
    int width, int height)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    __shared__ unsigned int block_hist_r[256];
    __shared__ unsigned int block_hist_g[256];
    __shared__ unsigned int block_hist_b[256];

    __shared__ unsigned int local_max_r[256];
    __shared__ unsigned int local_max_g[256];
    __shared__ unsigned int local_max_b[256];

    if (tid < 256) {
        block_hist_r[tid] = 0;
        block_hist_g[tid] = 0;
        block_hist_b[tid] = 0;
    }
    __syncthreads();

    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;

    if (x < width && y < height) {
        uchar4 p = tex2D<uchar4>(texObj, x + 0.5f, y + 0.5f);
        atomicAdd(&block_hist_r[p.x], 1u);
        atomicAdd(&block_hist_g[p.y], 1u);
        atomicAdd(&block_hist_b[p.z], 1u);
    }
    __syncthreads();

    if (tid < 256) {
        unsigned int r = block_hist_r[tid];
        unsigned int g = block_hist_g[tid];
        unsigned int b = block_hist_b[tid];

        if (r) atomicAdd(&Histdata->hist_r[tid], r);
        if (g) atomicAdd(&Histdata->hist_g[tid], g);
        if (b) atomicAdd(&Histdata->hist_b[tid], b);

        local_max_r[tid] = Histdata->hist_r[tid];
        local_max_g[tid] = Histdata->hist_g[tid];
        local_max_b[tid] = Histdata->hist_b[tid];
    }
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            local_max_r[tid] = max(local_max_r[tid], local_max_r[tid + stride]);
            local_max_g[tid] = max(local_max_g[tid], local_max_g[tid + stride]);
            local_max_b[tid] = max(local_max_b[tid], local_max_b[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax(&Histdata->max_r, local_max_r[0]);
        atomicMax(&Histdata->max_g, local_max_g[0]);
        atomicMax(&Histdata->max_b, local_max_b[0]);
    }
}




