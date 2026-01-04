extern "C"
__global__ void yuv422p_to_rgba_8bit_kernel(
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

    int Y = (int)y_plane[y * y_pitch + x];
    int U = (int)u_plane[y * u_pitch + (x >> 1)];
    int V = (int)v_plane[y * v_pitch + (x >> 1)];

    int C = Y;
    int D = U - 128;
    int E = V - 128;

    int R = C + ((1436 * E) >> 10);
    int G = C - ((352 * D + 731 * E) >> 10);
    int B = C + ((1814 * D) >> 10);

    R = min(max(R, 0), 255);
    G = min(max(G, 0), 255);
    B = min(max(B, 0), 255);

    uchar4* dst = (uchar4*)((uint8_t*)rgba + y * rgba_pitch);
    dst[x] = make_uchar4(R, G, B, 255);
}





