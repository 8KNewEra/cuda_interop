extern "C"
__global__ void vbo_hist_kernel(
    float* vbo,
    int num_bins,
    const unsigned int* hist_r,
    const unsigned int* hist_g,
    const unsigned int* hist_b,
    unsigned int* max_r,
    unsigned int* max_g,
    unsigned int* max_b
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_bins) return;

    float inv_r = 1.0f / max(1u, *max_r);
    float inv_g = 1.0f / max(1u, *max_g);
    float inv_b = 1.0f / max(1u, *max_b);

    float x = (float)i / (num_bins - 1);

    int r_offset = 0;
    int g_offset = num_bins * 3;
    int b_offset = num_bins * 3 * 2;

    vbo[r_offset + i*3 + 0] = x;
    vbo[r_offset + i*3 + 1] = hist_r[i] * inv_r;
    vbo[r_offset + i*3 + 2] = 0.0f;

    vbo[g_offset + i*3 + 0] = x;
    vbo[g_offset + i*3 + 1] = hist_g[i] * inv_g;
    vbo[g_offset + i*3 + 2] = 0.0f;

    vbo[b_offset + i*3 + 0] = x;
    vbo[b_offset + i*3 + 1] = hist_b[i] * inv_b;
    vbo[b_offset + i*3 + 2] = 0.0f;
}







