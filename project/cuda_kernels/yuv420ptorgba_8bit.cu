extern "C"
__global__ void yuv420p_to_rgba_8bit_kernel(
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

    uint8_t Y = *(uint8_t*)((uint8_t*)y_plane + y * y_pitch + x);

    int uv_x = x >> 1;
    int uv_y = y >> 1;

    uint8_t U = *(uint8_t*)((uint8_t*)u_plane + uv_y * u_pitch + uv_x);
    uint8_t V = *(uint8_t*)((uint8_t*)v_plane + uv_y * v_pitch + uv_x);

    int C = (int)Y - 16;
    int D = (int)U - 128;
    int E = (int)V - 128;

    int R = (298 * C + 409 * E + 128) >> 8;
    int G = (298 * C - 100 * D - 208 * E + 128) >> 8;
    int B = (298 * C + 516 * D + 128) >> 8;

    R = min(max(R, 0), 255);
    G = min(max(G, 0), 255);
    B = min(max(B, 0), 255);

    uint8_t* out = (uint8_t*)((uint8_t*)rgba + y * rgba_pitch + x * 4);
    out[0] = R;
    out[1] = G;
    out[2] = B;
    out[3] = 255;
}



