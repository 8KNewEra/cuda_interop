extern "C"
__device__ __forceinline__ uint16_t load_10bit(
    const uint16_t* p, bool is_be)
{
    uint16_t v = *p;
    return is_be ? (v >> 6) : (v & 0x03FF);
}

extern "C"
__device__ __forceinline__ uint8_t clamp_u8(float v)
{
    return (uint8_t)(v < 0.f ? 0.f : (v > 255.f ? 255.f : v));
}

extern "C"
__global__ void yuv420p_to_rgba_10bit_kernel(
    uint8_t* rgba, size_t rgba_pitch,

    const uint8_t* y_plane, size_t y_pitch,
    const uint8_t* u_plane, size_t u_pitch,
    const uint8_t* v_plane, size_t v_pitch,

    int width, int height,
    int is_be
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    const uint16_t* y_row =
        (const uint16_t*)(y_plane + y * y_pitch);
    const uint16_t* u_row =
        (const uint16_t*)(u_plane + (y >> 1) * u_pitch);
    const uint16_t* v_row =
        (const uint16_t*)(v_plane + (y >> 1) * v_pitch);

    uint16_t Y10 = load_10bit(&y_row[x],      is_be);
    uint16_t U10 = load_10bit(&u_row[x >> 1], is_be);
    uint16_t V10 = load_10bit(&v_row[x >> 1], is_be);

    float Y = (float)(Y10 - 64)  * (1.0f / (940 - 64));
    float U = (float)(U10 - 512) * (1.0f / (960 - 64));
    float V = (float)(V10 - 512) * (1.0f / (960 - 64));

    float R = Y + 1.5748f * V;
    float G = Y - 0.1873f * U - 0.4681f * V;
    float B = Y + 1.8556f * U;

    uint8_t* out = rgba + y * rgba_pitch + x * 4;
    out[0] = clamp_u8(R * 255.f);
    out[1] = clamp_u8(G * 255.f);
    out[2] = clamp_u8(B * 255.f);
    out[3] = 255;
}
