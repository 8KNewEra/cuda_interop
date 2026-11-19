// ==============================================================
// 8bit 経路: RGBA8（OpenGL FBO）→ NV12（エンコード入力。タイル分割 x1 / x2 / x4 / x8 共通）
//   FBO（cudaArray）を texture object 経由で直接読む（リニアへのコピーなし）
//   起動口は CUDA_ImageProcess（src/imageprocess/cuda_imageprocess.cpp）
//
//   変換は行ごとのカーネル encode_rows_rgba8_to_nv12_kernel（encode_rows.cuh。圧縮転送のタイルは
//   詰めた並びへ直接書く）。色の計算は下の luma_8bit / chroma_8bit
//
//   flip: FBO は GL の向き（下の行から）なので通常 1。
//         GL を通さず直接 NV12 化する経路を作る場合は 0 を渡す。
// ==============================================================
#include <cuda_runtime.h>
#include <texture_indirect_functions.h>
#include "encode_rows.cuh"

// --------------------------------------------------------------
// 色の計算（BT.709 limited range。デコード側と同じ行列）
//   R, G, B は 0..255。戻り値は四捨五入した 0..255
// --------------------------------------------------------------
__device__ __forceinline__ int luma_8bit(float R, float G, float B)
{
    const float yf = 0.1826f * R + 0.6142f * G + 0.0620f * B + 16.0f;
    return (int)(fminf(fmaxf(yf, 0.0f), 255.0f) + 0.5f);   // 四捨五入
}

__device__ __forceinline__ void chroma_8bit(float R, float G, float B, int& U, int& V)
{
    const float uf = -0.1006f * R - 0.3386f * G + 0.4392f * B + 128.0f;
    const float vf =  0.4392f * R - 0.3989f * G - 0.0403f * B + 128.0f;
    U = (int)(fminf(fmaxf(uf, 0.0f), 255.0f) + 0.5f);
    V = (int)(fminf(fmaxf(vf, 0.0f), 255.0f) + 0.5f);
}

// --------------------------------------------------------------
// 共通: テクスチャから1画素読む（unnormalized / point sampling）
// --------------------------------------------------------------
__device__ __forceinline__ uchar4 fetch_rgba(cudaTextureObject_t In, int x, int y)
{
    return tex2D<uchar4>(In, (float)x + 0.5f, (float)y + 0.5f);
}

// ==============================================================
// RGBA8 → NV12（行ごと。タイル分割 x1 / x2 / x4 / x8 共通。encode_rows.cuh）
// ==============================================================
namespace {
struct ConvRgba8ToNv12 {
    using T = uint8_t;
    static constexpr int BITS = 8, SHIFT = 0;
    cudaTextureObject_t in;
    __device__ float3 fetch(int x, int fy) const
    {
        const uchar4 p = fetch_rgba(in, x, fy);
        return make_float3(p.x, p.y, p.z);
    }
    __device__ int luma(float3 c, int, int) const { return luma_8bit(c.x, c.y, c.z); }
    __device__ void chroma(float3 a, int, int, int& u, int& v) const { chroma_8bit(a.x, a.y, a.z, u, v); }
};
}

extern "C"
__global__ void encode_rows_rgba8_to_nv12_kernel(
    cudaTextureObject_t In,
    NV12Tiles t,
    int srcW, int srcH,
    int flip)
{
    encode_rows::convert(t, srcW, srcH, flip, ConvRgba8ToNv12{ In });
}
