extern "C"
__global__ void nv12_to_rgba_8bit_kernel(
    uint8_t* rgba, size_t rgba_step,
    const uint8_t* y_plane, size_t y_step,
    const uint8_t* uv_plane, size_t uv_step,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    uint8_t Y = y_plane[y * y_step + x];

    int uv_index = (y / 2) * uv_step + (x / 2) * 2;
    uint8_t U = uv_plane[uv_index];
    uint8_t V = uv_plane[uv_index + 1];

    float fY = (float)Y - 16.0f;
    float fU = (float)U - 128.0f;
    float fV = (float)V - 128.0f;

    float R = 1.164f * fY + 1.596f * fV;
    float G = 1.164f * fY - 0.392f * fU - 0.813f * fV;
    float B = 1.164f * fY + 2.017f * fU;

    uint8_t r = (uint8_t)fminf(fmaxf(R, 0.0f), 255.0f);
    uint8_t g = (uint8_t)fminf(fmaxf(G, 0.0f), 255.0f);
    uint8_t b = (uint8_t)fminf(fmaxf(B, 0.0f), 255.0f);

    uchar4 pixel;
    pixel.x = r;
    pixel.y = g;
    pixel.z = b;
    pixel.w = 255;

    uchar4* row = (uchar4*)((uint8_t*)rgba + y * rgba_step);
    row[x] = pixel;
}



