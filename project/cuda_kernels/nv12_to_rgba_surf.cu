// ==============================================================
// NV12 → RGBA（OpenGLテクスチャへ surface 経由で直接書き込む版）
//
//  従来: merge カーネル → d_decode_rgba(linear) → cudaMemcpy2DToArray → GLテクスチャ
//  今回: merge カーネル → GLテクスチャ(cudaArray) へ直接
//
//  ★ surf2Dwrite の x 座標は「バイト単位」。ピクセル単位で渡すと
//     横 1/4 の範囲しか書けないので必ず sizeof(uchar4) を掛けること。
// ==============================================================
#include <cuda_runtime.h>
#include <surface_indirect_functions.h>

// --------------------------------------------------------------
// 共通: NV12 の 1 画素を RGBA へ（BT.601 limited range、既存カーネルと同一係数）
// --------------------------------------------------------------
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

    float fY = (float)Y - 16.0f;
    float fU = (float)U - 128.0f;
    float fV = (float)V - 128.0f;

    float R = 1.164f * fY + 1.596f * fV;
    float G = 1.164f * fY - 0.392f * fU - 0.813f * fV;
    float B = 1.164f * fY + 2.017f * fU;

    uchar4 p;
    p.x = (uint8_t)fminf(fmaxf(R, 0.0f), 255.0f);
    p.y = (uint8_t)fminf(fmaxf(G, 0.0f), 255.0f);
    p.z = (uint8_t)fminf(fmaxf(B, 0.0f), 255.0f);
    p.w = 255;
    return p;
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
