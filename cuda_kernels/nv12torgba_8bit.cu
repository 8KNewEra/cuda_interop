extern "C"
__global__ void nv12_to_rgba_8bit_kernel(
    float4* rgba, size_t rgba_pitch,
    const uint8_t* y_plane, size_t y_pitch,
    const uint8_t* uv_plane, size_t uv_pitch,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float Y = float(y_plane[y * y_pitch + x]);

    int uv_x = x & ~1;
    int uv_y = y >> 1;
    int uv_idx = uv_y * uv_pitch + uv_x;

    float U = float(uv_plane[uv_idx + 0]);
    float V = float(uv_plane[uv_idx + 1]);

    float Yf = (Y - 16.0f)  / 219.0f;
    float Uf = (U - 128.0f) / 224.0f;
    float Vf = (V - 128.0f) / 224.0f;

    Yf = fminf(fmaxf(Yf, 0.0f), 1.0f);

    float R = Yf + 1.5748f * Vf;
    float G = Yf - 0.1873f * Uf - 0.4681f * Vf;
    float B = Yf + 1.8556f * Uf;

    R = fminf(fmaxf(R, 0.0f), 1.0f);
    G = fminf(fmaxf(G, 0.0f), 1.0f);
    B = fminf(fmaxf(B, 0.0f), 1.0f);

    float4* row = (float4*)((uint8_t*)rgba + y * rgba_pitch);
    row[x] = make_float4(R, G, B, 1.0f);
}
