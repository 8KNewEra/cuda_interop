struct HistData {
    unsigned int hist_r[256];
    unsigned int hist_g[256];
    unsigned int hist_b[256];
    unsigned int max_r;
    unsigned int max_g;
    unsigned int max_b;
};

struct HistStats {
    int min_r, max_r;
    int min_g, max_g;
    int min_b, max_b;
    double avg_r, avg_g, avg_b;
    int max_y_axis;
};

extern "C"
__global__ void histgram_normalize_kernel(
    float* vbo,
    int num_bins,
    HistData* Histdata,
    HistStats* input_stats)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= num_bins) return;

    float inv = 1.0f / max(1u, input_stats->max_y_axis);

    int r_offset = 0;
    int g_offset = num_bins * 3;
    int b_offset = num_bins * 3 * 2;

    float i = (float)x / (num_bins - 1);

    vbo[r_offset + x*3 + 0] = i;
    vbo[r_offset + x*3 + 1] = Histdata->hist_r[x] * inv;
    vbo[r_offset + x*3 + 2] = 0.0f;

    vbo[g_offset + x*3 + 0] = i;
    vbo[g_offset + x*3 + 1] = Histdata->hist_g[x] * inv;
    vbo[g_offset + x*3 + 2] = 0.0f;

    vbo[b_offset + x*3 + 0] = i;
    vbo[b_offset + x*3 + 1] = Histdata->hist_b[x] * inv;
    vbo[b_offset + x*3 + 2] = 0.0f;
}
