// ==============================================================
// 10bit 経路: RGBA16（OpenGL FBO、GL_RGBA16）からの出力変換
//
//   rgba16_to_nv12_split_tex_kernel : エンコード入力の NV12（8bit 出力）へ。タイル分割 x1 / x2 / x4 / x8 共通
//   rgba16_to_p010_split_tex_kernel : エンコード入力の P010（10bit 出力。HEVC Main10 / AV1 10bit）へ
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

// RGBA16 → NV12（全体を走査し、該当するタイルへ書く）
extern "C"
__global__ void rgba16_to_nv12_split_tex_kernel(
    cudaTextureObject_t In,
    NV12Tiles t,
    int srcW, int srcH,
    int flip,
    int dither)
{
    int sx = blockIdx.x * blockDim.x + threadIdx.x;
    int sy = blockIdx.y * blockDim.y + threadIdx.y;
    if (sx >= srcW || sy >= srcH) return;

    const int tx = sx / t.outW;
    const int ty = sy / t.outH;
    const int q  = ty * t.cols + tx;
    if (tx >= t.cols || q >= t.count) return;

    const float dn = dither ? 1.0f : 0.0f;
    const int fy = flip ? (srcH - 1 - sy) : sy;
    const float3 c = fetch_rgb255(In, sx, fy);

    const float yf =  0.1826f * c.x + 0.6142f * c.y + 0.0620f * c.z + 16.0f;
    const int dx = sx - tx * t.outW;
    const int dy = sy - ty * t.outH;
    t.y[q][dy * t.yp[q] + dx] = round_u8(yf + dn * dither_noise(sx, sy, 0));

    if ((dx & 1) == 0 && (dy & 1) == 0) {
        // 色差は 2×2 の平均（画像の端では端の画素を使う。タイルの境界はまたがない）
        const int x1 = min(sx + 1, srcW - 1), y1 = min(sy + 1, srcH - 1);
        const int f1 = flip ? (srcH - 1 - y1) : y1;
        const float3 c01 = fetch_rgb255(In, x1, fy), c10 = fetch_rgb255(In, sx, f1), c11 = fetch_rgb255(In, x1, f1);
        const float3 a = make_float3((c.x + c01.x + c10.x + c11.x) * 0.25f,
                                     (c.y + c01.y + c10.y + c11.y) * 0.25f,
                                     (c.z + c01.z + c10.z + c11.z) * 0.25f);
        const float uf = -0.1006f * a.x - 0.3386f * a.y + 0.4392f * a.z + 128.0f;
        const float vf =  0.4392f * a.x - 0.3989f * a.y - 0.0403f * a.z + 128.0f;
        uint8_t* uv = t.uv[q] + (dy >> 1) * t.uvp[q] + dx;
        uv[0] = round_u8(uf + dn * dither_noise(sx, sy, 1));
        uv[1] = round_u8(vf + dn * dither_noise(sx, sy, 2));
    }
}

// RGBA16 → P010（10bit 出力。値は 16bit の上位 10bit に詰める）
//   NV12Tiles のポインタ・ピッチはバイト単位のまま（1 サンプル 2 バイト）
extern "C"
__global__ void rgba16_to_p010_split_tex_kernel(
    cudaTextureObject_t In,
    NV12Tiles t,
    int srcW, int srcH,
    int flip)
{
    int sx = blockIdx.x * blockDim.x + threadIdx.x;
    int sy = blockIdx.y * blockDim.y + threadIdx.y;
    if (sx >= srcW || sy >= srcH) return;

    const int tx = sx / t.outW;
    const int ty = sy / t.outH;
    const int q  = ty * t.cols + tx;
    if (tx >= t.cols || q >= t.count) return;

    const int dx = sx - tx * t.outW;
    const int dy = sy - ty * t.outH;

    // RGB（0..1）→ Y'（BT.709）
    auto fetch01 = [&](int x, int y) {
        const int fy = flip ? (srcH - 1 - y) : y;
        ushort4 p = tex2D<ushort4>(In, (float)x + 0.5f, (float)fy + 0.5f);
        return make_float3(p.x / 65535.0f, p.y / 65535.0f, p.z / 65535.0f);
    };
    auto luma = [](float3 c) { return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z; };
    auto pack10 = [](float v) {   // 0..1023 に収めて四捨五入し、上位 10bit へ
        int i = (int)(fminf(fmaxf(v, 0.0f), 1023.0f) + 0.5f);
        return (uint16_t)(i << 6);
    };

    const float3 c = fetch01(sx, sy);
    uint16_t* yrow = reinterpret_cast<uint16_t*>(t.y[q] + dy * t.yp[q]);
    yrow[dx] = pack10(64.0f + 876.0f * luma(c));

    if ((dx & 1) == 0 && (dy & 1) == 0) {
        // 色差は 2×2 の平均（タイルの端ではみ出す分は端の画素を使う）
        const int x1 = min(sx + 1, srcW - 1), y1 = min(sy + 1, srcH - 1);
        const float3 c01 = fetch01(x1, sy), c10 = fetch01(sx, y1), c11 = fetch01(x1, y1);
        const float3 a = make_float3((c.x + c01.x + c10.x + c11.x) * 0.25f,
                                     (c.y + c01.y + c10.y + c11.y) * 0.25f,
                                     (c.z + c01.z + c10.z + c11.z) * 0.25f);
        const float ya = luma(a);
        uint16_t* uv = reinterpret_cast<uint16_t*>(t.uv[q] + (dy >> 1) * t.uvp[q]) + dx;
        uv[0] = pack10(512.0f + 896.0f * (a.z - ya) / 1.8556f);   // Cb
        uv[1] = pack10(512.0f + 896.0f * (a.x - ya) / 1.5748f);   // Cr
    }
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
