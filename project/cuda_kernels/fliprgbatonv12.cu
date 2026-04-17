extern "C"
__global__ void flip_rgba_to_nv12_kernel(
    uint8_t* y_plane, size_t y_step,
    uint8_t* uv_plane, size_t uv_step,
    const uint8_t* rgba, size_t rgba_step,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int flipped_y = height - 1 - y;

    const uchar4* row = (const uchar4*)((const uint8_t*)rgba + flipped_y * rgba_step);
    uchar4 pixel = row[x];

    float R = pixel.x;
    float G = pixel.y;
    float B = pixel.z;

    float yf = 0.257f * R + 0.504f * G + 0.098f * B + 16.0f;
    float uf = -0.148f * R - 0.291f * G + 0.439f * B + 128.0f;
    float vf = 0.439f * R - 0.368f * G - 0.071f * B + 128.0f;

    uint8_t Y = (uint8_t)fminf(fmaxf(yf, 0.0f), 255.0f);
    uint8_t U = (uint8_t)fminf(fmaxf(uf, 0.0f), 255.0f);
    uint8_t V = (uint8_t)fminf(fmaxf(vf, 0.0f), 255.0f);

    y_plane[y * y_step + x] = Y;

    if ((x % 2 == 0) && (y % 2 == 0)) {
        int uv_index = (y / 2) * uv_step + (x / 2) * 2;
        uv_plane[uv_index] = U;
        uv_plane[uv_index + 1] = V;
    }
}
