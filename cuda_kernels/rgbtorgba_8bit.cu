extern "C"
__global__ void rgb_to_rgba_8bit_kernel(
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

    uint8_t R = r_plane[y * r_pitch + x];
    uint8_t G = g_plane[y * g_pitch + x];
    uint8_t B = b_plane[y * b_pitch + x];

    uint8_t* out = rgba + y * rgba_pitch + x * 4;
    out[0] = R;
    out[1] = G;
    out[2] = B;
    out[3] = 255;
}
