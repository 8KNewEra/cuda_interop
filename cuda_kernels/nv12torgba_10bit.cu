extern "C"
__global__ void nv12_to_rgba_10bit_kernel(
    uint16_t* rgba, size_t rgba_pitch,
    const uint8_t* y_plane, size_t y_pitch,
    const uint8_t* uv_plane, size_t uv_pitch,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const uint16_t* y_row =
        (const uint16_t*)(y_plane + y * y_pitch);
    const uint16_t* uv_row =
        (const uint16_t*)(uv_plane + (y >> 1) * uv_pitch);

    uint16_t Yv = y_row[x];
    int uv_x = x & ~1;
    uint16_t Uv = uv_row[uv_x + 0];
    uint16_t Vv = uv_row[uv_x + 1];

    float Yf = (Yv - 64.0f * 64.0f) / ((940.0f - 64.0f) * 64.0f);
    float Uf = (Uv - 512.0f * 64.0f) / ((960.0f - 64.0f) * 64.0f);
    float Vf = (Vv - 512.0f * 64.0f) / ((960.0f - 64.0f) * 64.0f);

    Yf = fminf(fmaxf(Yf, 0.0f), 1.0f);

    float R = Yf + 1.5748f * Vf;
    float G = Yf - 0.1873f * Uf - 0.4681f * Vf;
    float B = Yf + 1.8556f * Uf;

    R = fminf(fmaxf(R, 0.0f), 1.0f);
    G = fminf(fmaxf(G, 0.0f), 1.0f);
    B = fminf(fmaxf(B, 0.0f), 1.0f);

    uint8_t* row = (uint8_t*)rgba + y * rgba_pitch;
    uint16_t* out = (uint16_t*)(row + x * 8);

    out[0] = (uint16_t)(R * 65535.0f);
    out[1] = (uint16_t)(G * 65535.0f);
    out[2] = (uint16_t)(B * 65535.0f);
    out[3] = 65535;
}
