// ==============================================================
// CPUデコード系（cpudecode / avidecode）の YUV → RGBA 変換
//   出力先を linear バッファ(d_decode_rgba)ではなく
//   OpenGLテクスチャの surface object に直接書く版。
//
//   従来: YUVプレーン → 変換カーネル → d_decode_rgba(linear)
//                     → cudaMemcpy2DToArray → GLテクスチャ
//   今回: YUVプレーン → 変換カーネル → GLテクスチャ(surface)
//
//   ★ cudaArray は block-linear レイアウトなので、linear からの
//      コピーはスウィズルが入って遅い。surface 経由で直接書けば消える。
//
//   ★ 変換式は既存カーネルからバイト単位で移植している。
//      yuv420p_8bit は limited range、yuv422p_8bit は full range と
//      係数系が揃っていないが、挙動を変えないため意図的にそのまま。
// ==============================================================
#include <cuda_runtime.h>
#include <surface_indirect_functions.h>

// --------------------------------------------------------------
// 共通ヘルパ
//   既存の .cu にある load_10bit / clamp_u8 と名前が衝突しないよう
//   別名にしてある。
// --------------------------------------------------------------
__device__ __forceinline__ unsigned short load10(const unsigned short* p, int is_be)
{
    unsigned short v = *p;
    return is_be ? (unsigned short)(v >> 6) : (unsigned short)(v & 0x03FF);
}

__device__ __forceinline__ unsigned char clampu8(float v)
{
    return (unsigned char)(v < 0.f ? 0.f : (v > 255.f ? 255.f : v));
}

__device__ __forceinline__ void write_rgba(
    cudaSurfaceObject_t out, int x, int y,
    unsigned char R, unsigned char G, unsigned char B)
{
    uchar4 px = make_uchar4(R, G, B, 255);
    surf2Dwrite(px, out, x * (int)sizeof(uchar4), y);   // ★ xはバイト単位
}

// ==============================================================
// YUV420P 8bit（limited range BT.601）
// ==============================================================
extern "C"
__global__ void yuv420p_to_rgba_8bit_surf_kernel(
    cudaSurfaceObject_t out,
    const uint8_t* y_plane, size_t y_pitch,
    const uint8_t* u_plane, size_t u_pitch,
    const uint8_t* v_plane, size_t v_pitch,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    uint8_t Y = y_plane[y * y_pitch + x];

    int uv_x = x >> 1;
    int uv_y = y >> 1;

    uint8_t U = u_plane[uv_y * u_pitch + uv_x];
    uint8_t V = v_plane[uv_y * v_pitch + uv_x];

    int C = (int)Y - 16;
    int D = (int)U - 128;
    int E = (int)V - 128;

    int R = (298 * C + 409 * E + 128) >> 8;
    int G = (298 * C - 100 * D - 208 * E + 128) >> 8;
    int B = (298 * C + 516 * D + 128) >> 8;

    R = min(max(R, 0), 255);
    G = min(max(G, 0), 255);
    B = min(max(B, 0), 255);

    write_rgba(out, x, y, (unsigned char)R, (unsigned char)G, (unsigned char)B);
}

// ==============================================================
// YUV420P 10bit
// ==============================================================
extern "C"
__global__ void yuv420p_to_rgba_10bit_surf_kernel(
    cudaSurfaceObject_t out,
    const uint8_t* y_plane, size_t y_pitch,
    const uint8_t* u_plane, size_t u_pitch,
    const uint8_t* v_plane, size_t v_pitch,
    int width, int height,
    int is_be)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    const unsigned short* y_row = (const unsigned short*)(y_plane + y * y_pitch);
    const unsigned short* u_row = (const unsigned short*)(u_plane + (y >> 1) * u_pitch);
    const unsigned short* v_row = (const unsigned short*)(v_plane + (y >> 1) * v_pitch);

    unsigned short Y10 = load10(&y_row[x],      is_be);
    unsigned short U10 = load10(&u_row[x >> 1], is_be);
    unsigned short V10 = load10(&v_row[x >> 1], is_be);

    float Y = (float)(Y10 - 64)  * (1.0f / (940 - 64));
    float U = (float)(U10 - 512) * (1.0f / (960 - 64));
    float V = (float)(V10 - 512) * (1.0f / (960 - 64));

    float R = Y + 1.5748f * V;
    float G = Y - 0.1873f * U - 0.4681f * V;
    float B = Y + 1.8556f * U;

    write_rgba(out, x, y,
               clampu8(R * 255.f), clampu8(G * 255.f), clampu8(B * 255.f));
}

// ==============================================================
// YUV422P 8bit（既存実装は full range 係数。挙動維持のためそのまま）
// ==============================================================
extern "C"
__global__ void yuv422p_to_rgba_8bit_surf_kernel(
    cudaSurfaceObject_t out,
    const uint8_t* y_plane, size_t y_pitch,
    const uint8_t* u_plane, size_t u_pitch,
    const uint8_t* v_plane, size_t v_pitch,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int Y = (int)y_plane[y * y_pitch + x];
    int U = (int)u_plane[y * u_pitch + (x >> 1)];
    int V = (int)v_plane[y * v_pitch + (x >> 1)];

    int C = Y;
    int D = U - 128;
    int E = V - 128;

    int R = C + ((1436 * E) >> 10);
    int G = C - ((352 * D + 731 * E) >> 10);
    int B = C + ((1814 * D) >> 10);

    R = min(max(R, 0), 255);
    G = min(max(G, 0), 255);
    B = min(max(B, 0), 255);

    write_rgba(out, x, y, (unsigned char)R, (unsigned char)G, (unsigned char)B);
}

// ==============================================================
// YUV422P 10bit
// ==============================================================
extern "C"
__global__ void yuv422p_to_rgba_10bit_surf_kernel(
    cudaSurfaceObject_t out,
    const uint8_t* y_plane, size_t y_pitch,
    const uint8_t* u_plane, size_t u_pitch,
    const uint8_t* v_plane, size_t v_pitch,
    int width, int height,
    int is_be)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    const unsigned short* y_row = (const unsigned short*)(y_plane + y * y_pitch);
    const unsigned short* u_row = (const unsigned short*)(u_plane + y * u_pitch);
    const unsigned short* v_row = (const unsigned short*)(v_plane + y * v_pitch);

    unsigned short Y10 = load10(&y_row[x],      is_be);
    unsigned short U10 = load10(&u_row[x >> 1], is_be);
    unsigned short V10 = load10(&v_row[x >> 1], is_be);

    float Y = (float)(Y10 - 64)  * (1.0f / (940 - 64));
    float U = (float)(U10 - 512) * (1.0f / (960 - 64));
    float V = (float)(V10 - 512) * (1.0f / (960 - 64));

    float R = Y + 1.5748f * V;
    float G = Y - 0.1873f * U - 0.4681f * V;
    float B = Y + 1.8556f * U;

    write_rgba(out, x, y,
               clampu8(R * 255.f), clampu8(G * 255.f), clampu8(B * 255.f));
}

// ==============================================================
// UYVY422 8bit（packed）
// ==============================================================
extern "C"
__global__ void uyvy422_to_rgba_8bit_surf_kernel(
    cudaSurfaceObject_t out,
    const uint8_t* yuv, size_t yuv_pitch,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    const uint8_t* row = yuv + y * yuv_pitch;
    int pair = (x >> 1) * 4;                  // U Y0 V Y1

    uint8_t U = row[pair + 0];
    uint8_t Y = row[pair + 1 + ((x & 1) << 1)];
    uint8_t V = row[pair + 2];

    int C = (int)Y - 16;
    int D = (int)U - 128;
    int E = (int)V - 128;

    int R = (298 * C + 409 * E + 128) >> 8;
    int G = (298 * C - 100 * D - 208 * E + 128) >> 8;
    int B = (298 * C + 516 * D + 128) >> 8;

    R = min(max(R, 0), 255);
    G = min(max(G, 0), 255);
    B = min(max(B, 0), 255);

    write_rgba(out, x, y, (unsigned char)R, (unsigned char)G, (unsigned char)B);
}

// ==============================================================
// GBRP10 → RGBA8
//   呼び出し側が r/g/b の順で正しいプレーンを渡すこと
//   （AVFrame の data[] は G,B,R の順）
// ==============================================================
extern "C"
__global__ void rgb_to_rgba_10bit_surf_kernel(
    cudaSurfaceObject_t out,
    const uint8_t* r_plane, size_t r_pitch,
    const uint8_t* g_plane, size_t g_pitch,
    const uint8_t* b_plane, size_t b_pitch,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    const unsigned short* Rp = (const unsigned short*)(r_plane + y * r_pitch);
    const unsigned short* Gp = (const unsigned short*)(g_plane + y * g_pitch);
    const unsigned short* Bp = (const unsigned short*)(b_plane + y * b_pitch);

    unsigned char R8 = (unsigned char)((Rp[x] + 2) >> 2);
    unsigned char G8 = (unsigned char)((Gp[x] + 2) >> 2);
    unsigned char B8 = (unsigned char)((Bp[x] + 2) >> 2);

    write_rgba(out, x, y, R8, G8, B8);
}
