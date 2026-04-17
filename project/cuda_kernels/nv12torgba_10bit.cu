extern "C"
__global__ void nv12_to_rgba_10bit_kernel(
    uint8_t* rgba, size_t rgba_pitch,
    const uint8_t* y_plane, size_t y_pitch,
    const uint8_t* uv_plane, size_t uv_pitch,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    const uint16_t* y_row =
        reinterpret_cast<const uint16_t*>(y_plane + y * y_pitch);

    uint16_t Y10 = y_row[x] >> 6;

    const uint16_t* uv_row =
        reinterpret_cast<const uint16_t*>(uv_plane + (y >> 1) * uv_pitch);

    int uv_x = x & ~1;
    uint16_t U10 = uv_row[uv_x + 0] >> 6;
    uint16_t V10 = uv_row[uv_x + 1] >> 6;

    float Yf = (Y10 - 64.0f) / (940.0f - 64.0f);
    float Uf = (U10 - 512.0f) / (960.0f - 64.0f);
    float Vf = (V10 - 512.0f) / (960.0f - 64.0f);

    Yf = fminf(fmaxf(Yf, 0.0f), 1.0f);

    float R = Yf + 1.5748f * Vf;
    float G = Yf - 0.1873f * Uf - 0.4681f * Vf;
    float B = Yf + 1.8556f * Uf;

    R = fminf(fmaxf(R, 0.0f), 1.0f);
    G = fminf(fmaxf(G, 0.0f), 1.0f);
    B = fminf(fmaxf(B, 0.0f), 1.0f);

    uint8_t* out = rgba + y * rgba_pitch + x * 4;
    out[0] = static_cast<uint8_t>(R * 255.0f);
    out[1] = static_cast<uint8_t>(G * 255.0f);
    out[2] = static_cast<uint8_t>(B * 255.0f);
    out[3] = 255;
}

