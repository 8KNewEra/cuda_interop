// ==============================================================
// 10bit 経路: デコード結果 → RGBA16（OpenGL テクスチャ GL_RGBA16 へ surface 経由で直接書く）
//
//   10bit 以上の素材のときだけ使う（8bit 素材は decode_8bit.cu の RGBA8 経路）。
//   切り替えは VideoEngine::upload が素材ごとに自動で行う。
//
//   前半: NVDEC の P010（10bit を 16bit の上位に詰めた NV12 形式）。タイル分割 x1 / x2 / x4 / x8 は
//         1 本のカーネルで扱う（タイルは左上から行ごとの順。列数 cols で位置を決める）
//   後半: CPU デコード（cpudecode / avidecode）の 10bit プレーン（YUV420P / YUV422P / GBRP）
//
//   変換式は decode_8bit.cu の 10bit 入力用カーネルと同じ（BT.709 limited range）。
//   違いは出力の精度だけ（8bit に丸めず 16bit で書く）。
//   起動口は CUDA_ImageProcess（src/imageprocess/cuda_imageprocess.cpp）
//
//   注意: surf2Dwrite の x 座標は「バイト単位」。RGBA16 は 1 画素 8 バイト（sizeof(ushort4)）。
// ==============================================================
#include <cuda_runtime.h>
#include <surface_indirect_functions.h>
#include <stdint.h>
#include "tiles_10bit.h"   // P010Tiles
#include "tile_rows.cuh"

// 0..1 → RGBA16 の 1 チャンネル（四捨五入）
__device__ __forceinline__ unsigned short to_u16(float v)
{
    v = fminf(fmaxf(v, 0.0f), 1.0f);
    return (unsigned short)(v * 65535.0f + 0.5f);
}

__device__ __forceinline__ void write_rgba16(cudaSurfaceObject_t out, int x, int y, float R, float G, float B)
{
    ushort4 px = make_ushort4(to_u16(R), to_u16(G), to_u16(B), 65535);
    surf2Dwrite(px, out, x * (int)sizeof(ushort4), y);
}

// 10bit の Y / U / V（limited range）→ RGB（BT.709）。decode_8bit.cu の 10bit 用と同じ式
__device__ __forceinline__ void yuv10_to_rgb(int Y10, int U10, int V10, float& R, float& G, float& B)
{
    float Yf = (Y10 - 64.0f)  / (940.0f - 64.0f);
    float Uf = (U10 - 512.0f) / (960.0f - 64.0f);
    float Vf = (V10 - 512.0f) / (960.0f - 64.0f);
    R = Yf + 1.5748f * Vf;
    G = Yf - 0.1873f * Uf - 0.4681f * Vf;
    B = Yf + 1.8556f * Uf;
}

// ##############################################################
// NVDEC の P010（タイル分割 x1 / x2 / x4 / x8 共通）
// ##############################################################
extern "C"
__global__ void p010_tiles_to_rgba16_surf_kernel(
    cudaSurfaceObject_t out,
    P010Tiles t,
    int outW, int outH)
{
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= outW || oy >= outH) return;

    const int tx = ox / t.srcW;
    const int ty = oy / t.srcH;
    const int q  = ty * t.cols + tx;
    if (tx >= t.cols || q >= t.count) return;

    const int sx = ox - tx * t.srcW;
    const int sy = oy - ty * t.srcH;

    const uint16_t* y_row  = reinterpret_cast<const uint16_t*>(t.y[q]  + sy * t.yp[q]);
    const uint16_t* uv_row = reinterpret_cast<const uint16_t*>(t.uv[q] + (sy >> 1) * t.uvp[q]);
    const int uv_x = sx & ~1;

    float R, G, B;
    yuv10_to_rgb(y_row[sx] >> 6, uv_row[uv_x] >> 6, uv_row[uv_x + 1] >> 6, R, G, B);
    write_rgba16(out, ox, oy, R, G, B);
}

// ##############################################################
// NVDEC の P010 タイル → RGBA16（10bit の NVDEC 経路の本線。タイル分割 x1 / x2 / x4 / x8 共通）
//   行ごとの共通部（tile_rows.cuh）で読み、圧縮転送の詰めたタイルは差分を戻しながら変換する
// ##############################################################
namespace {
struct WriteRgba16 {
    cudaSurfaceObject_t out;
    __device__ void operator()(int Y, int U, int V, int x, int y) const
    {
        float R, G, B;
        yuv10_to_rgb(Y, U, V, R, G, B);
        write_rgba16(out, x, y, R, G, B);
    }
};
}

extern "C"
__global__ void p010_tiles_rows_to_rgba16_surf_kernel(
    cudaSurfaceObject_t out,
    P010Tiles t,
    int outW, int outH)
{
    tile_rows::convert<unsigned short, 10, 6>(t, outW, outH, WriteRgba16{ out });
}

// ##############################################################
// CPU デコードの 10bit プレーン
// ##############################################################

// decode_8bit.cu の load10 と同じ（BE は上位詰め、LE は下位 10bit）
__device__ __forceinline__ int load10_16(const unsigned short* p, int is_be)
{
    unsigned short v = *p;
    return is_be ? (int)(v >> 6) : (int)(v & 0x03FF);
}

// YUV420P 10bit
extern "C"
__global__ void yuv420p10_to_rgba16_surf_kernel(
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

    float R, G, B;
    yuv10_to_rgb(load10_16(&y_row[x], is_be), load10_16(&u_row[x >> 1], is_be), load10_16(&v_row[x >> 1], is_be), R, G, B);
    write_rgba16(out, x, y, R, G, B);
}

// YUV422P 10bit
extern "C"
__global__ void yuv422p10_to_rgba16_surf_kernel(
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

    float R, G, B;
    yuv10_to_rgb(load10_16(&y_row[x], is_be), load10_16(&u_row[x >> 1], is_be), load10_16(&v_row[x >> 1], is_be), R, G, B);
    write_rgba16(out, x, y, R, G, B);
}

// GBRP 10bit（呼び出し側が r / g / b の順でプレーンを渡す。AVFrame の data[] は G, B, R の順）
extern "C"
__global__ void gbrp10_to_rgba16_surf_kernel(
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

    // 10bit（0..1023）→ 0..1
    write_rgba16(out, x, y, (Rp[x] & 0x03FF) / 1023.0f, (Gp[x] & 0x03FF) / 1023.0f, (Bp[x] & 0x03FF) / 1023.0f);
}
