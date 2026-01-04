extern "C"
__global__ void rgb_to_rgba_10bit_kernel(
    uint8_t* rgba, size_t rgba_pitch,

    const uint8_t* r_plane, size_t r_pitch,
    const uint8_t* g_plane, size_t g_pitch,
    const uint8_t* b_plane, size_t b_pitch,

    int width, int height
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    const uint16_t* Rp = (const uint16_t*)(r_plane + y * r_pitch);
    const uint16_t* Gp = (const uint16_t*)(g_plane + y * g_pitch);
    const uint16_t* Bp = (const uint16_t*)(b_plane + y * b_pitch);

    uint16_t R10 = Rp[x];
    uint16_t G10 = Gp[x];
    uint16_t B10 = Bp[x];

    uint8_t R8 = (uint8_t)((R10 + 2) >> 2);
    uint8_t G8 = (uint8_t)((G10 + 2) >> 2);
    uint8_t B8 = (uint8_t)((B10 + 2) >> 2);

    uint8_t* out = rgba + y * rgba_pitch + x * 4;
    out[0] = R8;
    out[1] = G8;
    out[2] = B8;
    out[3] = 255;
}

