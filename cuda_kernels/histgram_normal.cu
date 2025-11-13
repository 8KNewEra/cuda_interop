struct HistData {
    unsigned int hist_r[256];
    unsigned int hist_g[256];
    unsigned int hist_b[256];
    unsigned int max_r;
    unsigned int max_g;
    unsigned int max_b;
};

extern "C"
__global__ void histogram_normal_kernel(
    HistData* Histdata,
    cudaTextureObject_t texObj,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    uchar4 p = tex2D<uchar4>(texObj, (float)x + 0.5f, (float)y + 0.5f);

    atomicAdd(&Histdata->hist_r[p.x], 1u);
    atomicAdd(&Histdata->hist_g[p.y], 1u);
    atomicAdd(&Histdata->hist_b[p.z], 1u);

    atomicMax(&Histdata->max_r, Histdata->hist_r[p.x]);
    atomicMax(&Histdata->max_g, Histdata->hist_g[p.x]);
    atomicMax(&Histdata->max_b, Histdata->hist_b[p.x]);
}
