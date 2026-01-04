extern "C"
__global__ void rgb_to_rgba_10bit_packed_kernel(
    uint8_t* rgba, size_t rgba_pitch,
    const uint8_t* src, size_t src_pitch,
    int width, int height
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    const uint32_t* in =
        (const uint32_t*)(src + y * src_pitch);

    uint32_t p = in[x];

    uint16_t R10 = (p >>  0) & 0x3FF;
    uint16_t G10 = (p >> 10) & 0x3FF;
    uint16_t B10 = (p >> 20) & 0x3FF;

    uint8_t R8 = (uint8_t)((R10 + 2) >> 2);
    uint8_t G8 = (uint8_t)((G10 + 2) >> 2);
    uint8_t B8 = (uint8_t)((B10 + 2) >> 2);

    uint8_t* out = rgba + y * rgba_pitch + x * 4;
    out[0] = R8;
    out[1] = G8;
    out[2] = B8;
    out[3] = 255;
}


