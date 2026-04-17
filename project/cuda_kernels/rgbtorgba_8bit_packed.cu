__constant__ int c_idx[6][3] = {
    {0,1,2},
    {2,1,0},
    {2,0,1},
    {1,2,0},
    {0,2,1},
    {1,0,2},
};

extern "C"
__global__ void rgb_to_rgba_8bit_packed_kernel(
    uint8_t* rgba, size_t rgba_pitch,
    const uint8_t* src, size_t src_pitch,
    int width, int height,
    int mode
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const uint8_t* in = src + y * src_pitch + x * 3;
    uint8_t* out = rgba + y * rgba_pitch + x * 4;

    out[0] = in[c_idx[mode][0]];
    out[1] = in[c_idx[mode][1]];
    out[2] = in[c_idx[mode][2]];
    out[3] = 255;
}


