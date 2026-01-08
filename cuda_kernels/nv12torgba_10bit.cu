extern "C"
__global__ void nv12_to_rgba_10bit_kernel(
    float4* rgba, size_t rgba_pitch,
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

    float Y = float(y_row[x] >> 6);
    float U = float(uv_row[(x & ~1) + 0] >> 6);
    float V = float(uv_row[(x & ~1) + 1] >> 6);

    float Yf = (Y - 64.0f)  / (940.0f - 64.0f);
    float Uf = (U - 512.0f) / (960.0f - 64.0f);
    float Vf = (V - 512.0f) / (960.0f - 64.0f);

    Yf = fminf(fmaxf(Yf, 0.0f), 1.0f);

    float R = Yf + 1.5748f * Vf;
    float G = Yf - 0.1873f * Uf - 0.4681f * Vf;
    float B = Yf + 1.8556f * Uf;

    float4* row = (float4*)((uint8_t*)rgba + y * rgba_pitch);
    row[x] = make_float4(
        fminf(fmaxf(R, 0.0f), 1.0f),
        fminf(fmaxf(G, 0.0f), 1.0f),
        fminf(fmaxf(B, 0.0f), 1.0f),
        1.0f
    );
}


