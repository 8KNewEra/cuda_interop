extern "C"
__global__ void yuv420p_to_rgba_10bit_kernel(
    uint8_t* rgba, size_t rgba_pitch,
    const uint8_t* y_plane, size_t y_pitch,
    const uint8_t* u_plane, size_t u_pitch,
    const uint8_t* v_plane, size_t v_pitch,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    const uint16_t* y16 = (const uint16_t*)y_plane;
    const uint16_t* u16 = (const uint16_t*)u_plane;
    const uint16_t* v16 = (const uint16_t*)v_plane;

    int y_stride = y_pitch >> 1;
    int u_stride = u_pitch >> 1;
    int v_stride = v_pitch >> 1;

    uint16_t Y10 = y16[y * y_stride + x];
    uint16_t U10 = u16[(y >> 1) * u_stride + (x >> 1)];
    uint16_t V10 = v16[(y >> 1) * v_stride + (x >> 1)];

    int Y = (Y10 & 0x03FF) >> 2;
    int U = (U10 & 0x03FF) >> 2;
    int V = (V10 & 0x03FF) >> 2;

    int C = Y - 16;
    int D = U - 128;
    int E = V - 128;

    int R = (298 * C + 409 * E + 128) >> 8;
    int G = (298 * C - 100 * D - 208 * E + 128) >> 8;
    int B = (298 * C + 516 * D + 128) >> 8;

    R = min(max(R, 0), 255);
    G = min(max(G, 0), 255);
    B = min(max(B, 0), 255);

    uint8_t* out = (uint8_t*)((uint8_t*)rgba + y * rgba_pitch + x * 4);
    out[0] = (uint8_t)R;
    out[1] = (uint8_t)G;
    out[2] = (uint8_t)B;
    out[3] = 255;
}

