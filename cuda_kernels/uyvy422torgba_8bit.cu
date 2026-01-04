extern "C"
__global__ void uyvy422_to_rgba_8bit_kernel(
    uint8_t* rgba, size_t rgba_pitch,
    const uint8_t* yuv, size_t yuv_pitch,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const uint8_t* row = yuv + y * yuv_pitch;
    int pair = (x >> 1) * 4;

    uint8_t U = row[pair + 0];
    uint8_t Y = row[pair + 1 + ((x & 1) << 1)];
    uint8_t V = row[pair + 2];

    int C = int(Y) - 16;
    int D = int(U) - 128;
    int E = int(V) - 128;

    int R = (298 * C + 409 * E + 128) >> 8;
    int G = (298 * C - 100 * D - 208 * E + 128) >> 8;
    int B = (298 * C + 516 * D + 128) >> 8;

    R = min(max(R, 0), 255);
    G = min(max(G, 0), 255);
    B = min(max(B, 0), 255);

    uint8_t* out = rgba + y * rgba_pitch + x * 4;
    out[0] = R;
    out[1] = G;
    out[2] = B;
    out[3] = 255;
}

