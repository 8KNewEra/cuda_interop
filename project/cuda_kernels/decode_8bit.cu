// ==============================================================
// 8bit 経路: デコード結果 → RGBA8（OpenGL テクスチャへ surface 経由で直接書く）
//   入力は 8bit / 10bit どちらもあるが、出力は常に RGBA8（10bit 入力も 8bit に落とす）
//
//   前半: NVDEC の NV12（1 ストリーム 8bit / 10bit、タイル分割 x2 / x4 / x8 の結合）
//   後半: CPU デコード（cpudecode / avidecode）の YUV / RGB プレーン
//   起動口は CUDA_ImageProcess（src/imageprocess/cuda_imageprocess.cpp）
//
//   注意: surf2Dwrite の x 座標は「バイト単位」。ピクセル単位で渡すと
//         横 1/4 の範囲しか書けないので必ず sizeof(uchar4) を掛ける。
//   色の行列は全カーネル BT.709（10bit 経路・エンコード側と同じ。HDR は非対応）。
//   レンジは limited（16-235）。yuv422p_8bit だけは旧実装どおり full range（0-255）として読む。
// ==============================================================
#include <cuda_runtime.h>
#include <surface_indirect_functions.h>
#include "tile_rows.cuh"

// ##############################################################
// NVDEC の NV12
// ##############################################################

// --------------------------------------------------------------
// 共通: NV12 の 1 画素を RGBA へ（BT.709 limited range）
// --------------------------------------------------------------
__device__ __forceinline__ uchar4 yuv8_to_rgba_8bit(int Y, int U, int V)
{
    float fY = (float)Y - 16.0f;
    float fU = (float)U - 128.0f;
    float fV = (float)V - 128.0f;

    float R = 1.1644f * fY + 1.7927f * fV;
    float G = 1.1644f * fY - 0.2132f * fU - 0.5329f * fV;
    float B = 1.1644f * fY + 2.1124f * fU;

    uchar4 p;
    p.x = (uint8_t)fminf(fmaxf(R, 0.0f), 255.0f);
    p.y = (uint8_t)fminf(fmaxf(G, 0.0f), 255.0f);
    p.z = (uint8_t)fminf(fmaxf(B, 0.0f), 255.0f);
    p.w = 255;
    return p;
}

__device__ __forceinline__ uchar4 nv12_pixel_to_rgba_8bit(
    const uint8_t* Yp, size_t pitchY,
    const uint8_t* UVp, size_t pitchUV,
    int sx, int sy)
{
    int Y = (int)Yp[sy * pitchY + sx];

    int uv_x = sx & ~1;
    int uv_y = sy >> 1;

    int U = (int)UVp[uv_y * pitchUV + uv_x + 0];
    int V = (int)UVp[uv_y * pitchUV + uv_x + 1];
    return yuv8_to_rgba_8bit(Y, U, V);
}

// ==============================================================
// NVDEC の NV12 タイル → RGBA8（8bit の NVDEC 経路の本線。タイル分割 x1 / x2 / x4 / x8 共通）
//   行ごとの共通部（tile_rows.cuh）で読み、圧縮転送の詰めたタイルは差分を戻しながら変換する。
//   下のタイル数ごとのカーネルは、先頭・pitch が 16 の倍数でないときの保険
// ==============================================================
namespace {
struct WriteRgba8 {
    cudaSurfaceObject_t out;
    __device__ void operator()(int Y, int U, int V, int x, int y) const
    {
        surf2Dwrite(yuv8_to_rgba_8bit(Y, U, V), out, x * (int)sizeof(uchar4), y);
    }
};
}

extern "C"
__global__ void nv12_tiles_rows_to_rgba8_surf_kernel(
    cudaSurfaceObject_t out,
    P010Tiles t,
    int outW, int outH)
{
    tile_rows::convert<uint8_t, 8, 0>(t, outW, outH, WriteRgba8{ out });
}

// ==============================================================
// シングルストリーム 8bit
// ==============================================================
extern "C"
__global__ void nv12_to_rgba_8bit_surf_kernel(
    cudaSurfaceObject_t out,
    const uint8_t* y_plane,  size_t y_pitch,
    const uint8_t* uv_plane, size_t uv_pitch,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    uchar4 pixel = nv12_pixel_to_rgba_8bit(y_plane, y_pitch, uv_plane, uv_pitch, x, y);

    surf2Dwrite(pixel, out, x * (int)sizeof(uchar4), y);
}

// ==============================================================
// シングルストリーム 10bit
// ==============================================================
extern "C"
__global__ void nv12_to_rgba_10bit_surf_kernel(
    cudaSurfaceObject_t out,
    const uint8_t* y_plane,  size_t y_pitch,
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

    uchar4 pixel;
    pixel.x = (uint8_t)(R * 255.0f);
    pixel.y = (uint8_t)(G * 255.0f);
    pixel.z = (uint8_t)(B * 255.0f);
    pixel.w = 255;

    surf2Dwrite(pixel, out, x * (int)sizeof(uchar4), y);
}

// ==============================================================
// 2タイル結合（横2×縦1）
// ==============================================================
extern "C"
__global__ void nv12x2_to_rgba_merge_surf_kernel(
    cudaSurfaceObject_t out,
    const uint8_t* y0,  size_t pitchY0, const uint8_t* uv0, size_t pitchUV0,
    const uint8_t* y1,  size_t pitchY1, const uint8_t* uv1, size_t pitchUV1,
    int outW, int outH,
    int srcW, int srcH)
{
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ox >= outW || oy >= outH) return;

    const uint8_t *Yp, *UVp;
    size_t pitchY, pitchUV;
    int sx, sy;

    int quadX = (ox >= srcW);

    if (quadX == 0) { Yp = y0; UVp = uv0; pitchY = pitchY0; pitchUV = pitchUV0; sx = ox;        sy = oy; }
    else            { Yp = y1; UVp = uv1; pitchY = pitchY1; pitchUV = pitchUV1; sx = ox - srcW; sy = oy; }

    uchar4 pixel = nv12_pixel_to_rgba_8bit(Yp, pitchY, UVp, pitchUV, sx, sy);

    surf2Dwrite(pixel, out, ox * (int)sizeof(uchar4), oy);
}

// ==============================================================
// 4タイル結合（横2×縦2）
// ==============================================================
extern "C"
__global__ void nv12x4_to_rgba_merge_surf_kernel(
    cudaSurfaceObject_t out,
    const uint8_t* y0,  size_t pitchY0, const uint8_t* uv0, size_t pitchUV0,
    const uint8_t* y1,  size_t pitchY1, const uint8_t* uv1, size_t pitchUV1,
    const uint8_t* y2,  size_t pitchY2, const uint8_t* uv2, size_t pitchUV2,
    const uint8_t* y3,  size_t pitchY3, const uint8_t* uv3, size_t pitchUV3,
    int outW, int outH,
    int srcW, int srcH)
{
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ox >= outW || oy >= outH) return;

    const uint8_t *Yp, *UVp;
    size_t pitchY, pitchUV;
    int sx, sy;

    int quadX = (ox >= srcW);
    int quadY = (oy >= srcH);
    int q = quadY * 2 + quadX;

    switch (q) {
    case 0: Yp = y0; UVp = uv0; pitchY = pitchY0; pitchUV = pitchUV0;
            sx = ox;        sy = oy;        break;
    case 1: Yp = y1; UVp = uv1; pitchY = pitchY1; pitchUV = pitchUV1;
            sx = ox - srcW; sy = oy;        break;
    case 2: Yp = y2; UVp = uv2; pitchY = pitchY2; pitchUV = pitchUV2;
            sx = ox;        sy = oy - srcH; break;
    default:Yp = y3; UVp = uv3; pitchY = pitchY3; pitchUV = pitchUV3;
            sx = ox - srcW; sy = oy - srcH; break;
    }

    uchar4 pixel = nv12_pixel_to_rgba_8bit(Yp, pitchY, UVp, pitchUV, sx, sy);

    surf2Dwrite(pixel, out, ox * (int)sizeof(uchar4), oy);
}

// ==============================================================
// 8タイル結合（横4×縦2）
// ==============================================================
extern "C"
__global__ void nv12x8_to_rgba_merge_surf_kernel(
    cudaSurfaceObject_t out,
    const uint8_t* y0, size_t pitchY0, const uint8_t* uv0, size_t pitchUV0,
    const uint8_t* y1, size_t pitchY1, const uint8_t* uv1, size_t pitchUV1,
    const uint8_t* y2, size_t pitchY2, const uint8_t* uv2, size_t pitchUV2,
    const uint8_t* y3, size_t pitchY3, const uint8_t* uv3, size_t pitchUV3,
    const uint8_t* y4, size_t pitchY4, const uint8_t* uv4, size_t pitchUV4,
    const uint8_t* y5, size_t pitchY5, const uint8_t* uv5, size_t pitchUV5,
    const uint8_t* y6, size_t pitchY6, const uint8_t* uv6, size_t pitchUV6,
    const uint8_t* y7, size_t pitchY7, const uint8_t* uv7, size_t pitchUV7,
    int outW, int outH,
    int srcW, int srcH)
{
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ox >= outW || oy >= outH) return;

    int quadX = ox / srcW;
    int quadY = oy / srcH;

    if (quadX >= 4 || quadY >= 2) return;

    int q = quadY * 4 + quadX;

    const uint8_t *Yp, *UVp;
    size_t pitchY, pitchUV;

    switch (q) {
    case 0: Yp = y0; UVp = uv0; pitchY = pitchY0; pitchUV = pitchUV0; break;
    case 1: Yp = y1; UVp = uv1; pitchY = pitchY1; pitchUV = pitchUV1; break;
    case 2: Yp = y2; UVp = uv2; pitchY = pitchY2; pitchUV = pitchUV2; break;
    case 3: Yp = y3; UVp = uv3; pitchY = pitchY3; pitchUV = pitchUV3; break;
    case 4: Yp = y4; UVp = uv4; pitchY = pitchY4; pitchUV = pitchUV4; break;
    case 5: Yp = y5; UVp = uv5; pitchY = pitchY5; pitchUV = pitchUV5; break;
    case 6: Yp = y6; UVp = uv6; pitchY = pitchY6; pitchUV = pitchUV6; break;
    default:Yp = y7; UVp = uv7; pitchY = pitchY7; pitchUV = pitchUV7; break;
    }

    int sx = ox - quadX * srcW;
    int sy = oy - quadY * srcH;

    uchar4 pixel = nv12_pixel_to_rgba_8bit(Yp, pitchY, UVp, pitchUV, sx, sy);

    surf2Dwrite(pixel, out, ox * (int)sizeof(uchar4), oy);
}


// ##############################################################
// CPU デコードの YUV / RGB プレーン
// ##############################################################

// --------------------------------------------------------------
// 共通ヘルパ（CPU デコード用）
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
// YUV420P 8bit（limited range BT.709）
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

    int R = (298 * C + 459 * E + 128) >> 8;             // BT.709 limited（×256 の整数係数）
    int G = (298 * C -  55 * D - 136 * E + 128) >> 8;
    int B = (298 * C + 541 * D + 128) >> 8;

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
// YUV422P 8bit（full range BT.709。レンジは旧実装どおり full）
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

    int R = C + ((1613 * E) >> 10);                     // BT.709 full（×1024 の整数係数）
    int G = C - ((192 * D + 479 * E) >> 10);
    int B = C + ((1900 * D) >> 10);

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

    int R = (298 * C + 459 * E + 128) >> 8;             // BT.709 limited（×256 の整数係数）
    int G = (298 * C -  55 * D - 136 * E + 128) >> 8;
    int B = (298 * C + 541 * D + 128) >> 8;

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
