extern "C"
__global__ void histogram_calc_and_draw_kernel(
    uint8_t* rgba, size_t rgba_step, int width, int height,
    unsigned int* hist_r, unsigned int* hist_g, unsigned int* hist_b,
    int graph_h)
{
    //calc histgram
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const uchar4* row = (const uchar4*)(rgba + y * rgba_step);
    uchar4 pixel = row[x];

    atomicAdd(&hist_r[pixel.x], 1);
    atomicAdd(&hist_g[pixel.y], 1);
    atomicAdd(&hist_b[pixel.z], 1);

    __syncthreads();

    //histgram
    __shared__ unsigned int max_r, max_g, max_b;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        max_r = max_g = max_b = 0;
        for (int i = 0; i < 256; i++) {
            if (hist_r[i] > max_r) max_r = hist_r[i];
            if (hist_g[i] > max_g) max_g = hist_g[i];
            if (hist_b[i] > max_b) max_b = hist_b[i];
        }
    }
    __syncthreads();

    int bin = x;
    if (bin >= 256) return;

    float scale_r = max_r > 0 ? (float)graph_h / max_r : 0.0f;
    float scale_g = max_g > 0 ? (float)graph_h / max_g : 0.0f;
    float scale_b = max_b > 0 ? (float)graph_h / max_b : 0.0f;

    int draw_x = width - 256 + bin;
    for (int i = 0; i < graph_h; i++) {
        int draw_y = i;
        uchar4* out_pixel = (uchar4*)(rgba + draw_y * rgba_step) + draw_x;
        out_pixel->x = (i < hist_r[bin]*scale_r) ? 255 : 0;
        out_pixel->y = (i < hist_g[bin]*scale_g) ? 255 : 0;
        out_pixel->z = (i < hist_b[bin]*scale_b) ? 255 : 0;
        out_pixel->w = 255;
    }
}







