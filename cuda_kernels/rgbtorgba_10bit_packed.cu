__constant__ int c_idx[6][3] = {
    {0,1,2}, {2,1,0}, {1,0,2},
    {2,0,1}, {0,2,1}, {1,2,0}
};

extern "C"
__global__ void rgb_to_rgba_10bit_packed_kernel(
    uint8_t* rgba, size_t rgba_pitch,
    const uint8_t* src, size_t src_pitch,
    int width, int height,
    int mode
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const uint32_t* in = (const uint32_t*)(src + y * src_pitch) + x;

    uint32_t v = *in;

    uint16_t c[3];
    c[0] =  v        & 0x3FF;
    c[1] = (v >> 10) & 0x3FF;
    c[2] = (v >> 20) & 0x3FF;

    uint8_t* out = rgba + y * rgba_pitch + x * 4;
    out[0] = c[c_idx[mode][0]] >> 2;
    out[1] = c[c_idx[mode][1]] >> 2;
    out[2] = c[c_idx[mode][2]] >> 2;
    out[3] = 255;
}



