extern "C" __global__
void draw_histogram_kernel(
    cudaSurfaceObject_t surface,
    int width,
    int height,
    const unsigned int* hist_r,
    const unsigned int* hist_g,
    const unsigned int* hist_b)
{
    const int graph_h = 1600;
    const int graph_w = 2048;
    const int margin = 1;

    int start_x = width - graph_w;
    int start_y = height - graph_h;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= graph_w || y >= graph_h) return;

    uchar4 color = make_uchar4(0,0,0,255);

    if (x < margin || x >= graph_w - margin || y < margin || y >= graph_h - margin) {
        color = make_uchar4(255,255,255,255);
        surf2Dwrite(color, surface, (start_x + x) * sizeof(uchar4), start_y + y);
        return;
    }

    if (x % (graph_w / 16) == 0 || y % (graph_h / 4) == 0) {
        color = make_uchar4(128,128,128,255);
        surf2Dwrite(color, surface, (start_x + x) * sizeof(uchar4), start_y + y);
        return;
    }

    __shared__ unsigned int max_r, max_g, max_b;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        max_r = max_g = max_b = 1;
        for (int i = 0; i < 256; i++) {
            max_r = max(max_r, hist_r[i]);
            max_g = max(max_g, hist_g[i]);
            max_b = max(max_b, hist_b[i]);
        }
    }
    __syncthreads();

    const float thickness = 1.0f;
    float fy = (float)y;

    for (int dx = -1; dx <= 1; dx++) {
        int px0 = x + dx;
        int px1 = px0 + 1;
        if (px0 < 0 || px1 >= graph_w) continue;

        int bin0 = px0 * 256 / graph_w;
        int bin1 = min(255, px1 * 256 / graph_w);

        float y_r0 = (float)hist_r[bin0] * graph_h / max_r;
        float y_r1 = (float)hist_r[bin1] * graph_h / max_r;
        float y_g0 = (float)hist_g[bin0] * graph_h / max_g;
        float y_g1 = (float)hist_g[bin1] * graph_h / max_g;
        float y_b0 = (float)hist_b[bin0] * graph_h / max_b;
        float y_b1 = (float)hist_b[bin1] * graph_h / max_b;

        if (fy >= fminf(y_r0, y_r1) - thickness && fy <= fmaxf(y_r0, y_r1) + thickness)
            color.x = 255;
        if (fy >= fminf(y_g0, y_g1) - thickness && fy <= fmaxf(y_g0, y_g1) + thickness)
            color.y = 255;
        if (fy >= fminf(y_b0, y_b1) - thickness && fy <= fmaxf(y_b0, y_b1) + thickness)
            color.z = 255;
    }

    surf2Dwrite(color, surface, (start_x + x) * sizeof(uchar4), start_y + y);
}
