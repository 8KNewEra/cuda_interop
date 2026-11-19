// ==============================================================
// 10bit 経路: RGBA16（OpenGL FBO、GL_RGBA16）からの出力変換
//
//   encode_rows_rgba16_to_nv12_kernel / encode_rows_rgba16_to_p010_kernel
//                                   : エンコード入力の NV12（8bit 出力）/ P010（10bit 出力。HEVC Main10 / AV1 10bit）へ。
//                                     行ごと（encode_rows.cuh）、タイル分割共通。圧縮転送のタイルは詰めた並びへ直接書く
//   rgba16_to_rgba8_tex_kernel      : 画像保存用の RGBA8 へ（JPEG / BMP、8bit を選んだ PNG / TIFF）
//
//   8bit に落とす 2 本（NV12 / RGBA8）は dither != 0 のとき、四捨五入の前に ±0.5 段階のノイズを足す
//   （なめらかなグラデーションの縞を目立たなくする）。ノイズは座標とチャンネルから決まるので毎回同じ
//   （フレーム間でちらつかない）。値をそのまま見たいとき（解析）は dither = 0
//
//   FBO（cudaArray）を texture object 経由で直接読む。
//   NV12 の変換式は encode_8bit.cu と同じ（BT.709 limited range、四捨五入、色差は 2×2 の平均）。
//   P010 は decode_10bit.cu の逆変換（BT.709 limited range）。色差は 2×2 の平均。
//   flip: FBO は GL の向き（下の行から）なので通常 1。
//   起動口は CUDA_ImageProcess（src/imageprocess/cuda_imageprocess.cpp）
// ==============================================================
#include <cuda_runtime.h>
#include <texture_indirect_functions.h>
#include <stdint.h>
#include "tiles_10bit.h"   // NV12Tiles
#include "encode_rows.cuh"

// テクスチャから 1 画素読み、0..255 の float にする（unnormalized / point sampling）
__device__ __forceinline__ float3 fetch_rgb255(cudaTextureObject_t In, int x, int y)
{
    ushort4 p = tex2D<ushort4>(In, (float)x + 0.5f, (float)y + 0.5f);
    const float k = 255.0f / 65535.0f;
    return make_float3(p.x * k, p.y * k, p.z * k);
}

// 0..255 に収めて四捨五入
__device__ __forceinline__ uint8_t round_u8(float v)
{
    return (uint8_t)(fminf(fmaxf(v, 0.0f), 255.0f) + 0.5f);
}

// ディザ用のノイズ（-0.5 .. +0.5。座標とチャンネル c から決まるハッシュ）
__device__ __forceinline__ float dither_noise(int x, int y, int c)
{
    unsigned int h = (unsigned int)x * 0x8da6b343u ^ (unsigned int)y * 0xd8163841u ^ (unsigned int)c * 0xcb1ab31fu;
    h ^= h >> 13;  h *= 0x5bd1e995u;  h ^= h >> 15;
    return (float)(h & 0xFFFFu) * (1.0f / 65536.0f) - 0.5f;
}

// ---- 色の計算 ----
// NV12（8bit 出力）: c は 0..255（fetch_rgb255）。dn = 1 ならディザ（(x, y) は画像の座標、色差は 2×2 の左上）
__device__ __forceinline__ int nv12_luma_255(float3 c, float dn, int x, int y)
{
    const float yf = 0.1826f * c.x + 0.6142f * c.y + 0.0620f * c.z + 16.0f;
    return round_u8(yf + dn * dither_noise(x, y, 0));
}
__device__ __forceinline__ void nv12_chroma_255(float3 a, float dn, int x, int y, int& u, int& v)
{
    const float uf = -0.1006f * a.x - 0.3386f * a.y + 0.4392f * a.z + 128.0f;
    const float vf =  0.4392f * a.x - 0.3989f * a.y - 0.0403f * a.z + 128.0f;
    u = round_u8(uf + dn * dither_noise(x, y, 1));
    v = round_u8(vf + dn * dither_noise(x, y, 2));
}

// P010（10bit 出力）: c は 0..1（fetch_rgb01）。戻り値は 0..1023（P010 の並びへ書くときに <<6）
__device__ __forceinline__ float3 fetch_rgb01(cudaTextureObject_t In, int x, int fy)
{
    ushort4 p = tex2D<ushort4>(In, (float)x + 0.5f, (float)fy + 0.5f);
    return make_float3(p.x / 65535.0f, p.y / 65535.0f, p.z / 65535.0f);
}
__device__ __forceinline__ float luma709(float3 c) { return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z; }
__device__ __forceinline__ int round10(float v) { return (int)(fminf(fmaxf(v, 0.0f), 1023.0f) + 0.5f); }   // 0..1023 に収めて四捨五入
__device__ __forceinline__ int p010_luma(float3 c) { return round10(64.0f + 876.0f * luma709(c)); }
__device__ __forceinline__ void p010_chroma(float3 a, int& u, int& v)
{
    const float ya = luma709(a);
    u = round10(512.0f + 896.0f * (a.z - ya) / 1.8556f);   // Cb
    v = round10(512.0f + 896.0f * (a.x - ya) / 1.5748f);   // Cr
}

// ==============================================================
// RGBA16 → NV12 / P010（行ごと。タイル分割共通。encode_rows.cuh）
// ==============================================================
namespace {
struct ConvRgba16ToNv12 {
    using T = uint8_t;
    static constexpr int BITS = 8, SHIFT = 0;
    cudaTextureObject_t in;
    float dn;   // ディザ（1 / 0）
    __device__ float3 fetch(int x, int fy) const { return fetch_rgb255(in, x, fy); }
    __device__ int luma(float3 c, int x, int y) const { return nv12_luma_255(c, dn, x, y); }
    __device__ void chroma(float3 a, int x, int y, int& u, int& v) const { nv12_chroma_255(a, dn, x, y, u, v); }
};
struct ConvRgba16ToP010 {
    using T = uint16_t;
    static constexpr int BITS = 10, SHIFT = 6;
    cudaTextureObject_t in;
    __device__ float3 fetch(int x, int fy) const { return fetch_rgb01(in, x, fy); }
    __device__ int luma(float3 c, int, int) const { return p010_luma(c); }
    __device__ void chroma(float3 a, int, int, int& u, int& v) const { p010_chroma(a, u, v); }
};
}

extern "C"
__global__ void encode_rows_rgba16_to_nv12_kernel(
    cudaTextureObject_t In, NV12Tiles t, int srcW, int srcH, int flip, int dither)
{
    encode_rows::convert(t, srcW, srcH, flip, ConvRgba16ToNv12{ In, dither ? 1.0f : 0.0f });
}

extern "C"
__global__ void encode_rows_rgba16_to_p010_kernel(
    cudaTextureObject_t In, NV12Tiles t, int srcW, int srcH, int flip)
{
    encode_rows::convert(t, srcW, srcH, flip, ConvRgba16ToP010{ In });
}

// RGBA16 → RGBA8（画像保存用。上下はそのまま。保存側で反転する）
extern "C"
__global__ void rgba16_to_rgba8_tex_kernel(
    cudaTextureObject_t In,
    uint8_t* out, size_t pitch,
    int width, int height,
    int dither)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const float dn = dither ? 1.0f : 0.0f;
    const float3 c = fetch_rgb255(In, x, y);
    uchar4* row = reinterpret_cast<uchar4*>(out + y * pitch);
    row[x] = make_uchar4(round_u8(c.x + dn * dither_noise(x, y, 0)),
                         round_u8(c.y + dn * dither_noise(x, y, 1)),
                         round_u8(c.z + dn * dither_noise(x, y, 2)), 255);
}
