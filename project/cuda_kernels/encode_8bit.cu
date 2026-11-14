// ==============================================================
// 8bit 経路: RGBA8（OpenGL FBO）→ NV12（エンコード入力。タイル分割 x1 / x2 / x4 / x8）
//   FBO（cudaArray）を texture object 経由で直接読む（リニアへのコピーなし）
//   起動口は CUDA_ImageProcess（src/imageprocess/cuda_imageprocess.cpp）
//
//   flip: FBO は GL の向き（下の行から）なので通常 1。
//         GL を通さず直接 NV12 化する経路を作る場合は 0 を渡す。
// ==============================================================
#include <cuda_runtime.h>
#include <texture_indirect_functions.h>

// --------------------------------------------------------------
// 共通: RGBA → YUV（BT.709 limited range。デコード側と同じ行列）
// --------------------------------------------------------------
__device__ __forceinline__ void rgba_to_yuv_8bit(
    uchar4 rgba, uint8_t& Y, uint8_t& U, uint8_t& V)
{
    float R = rgba.x;
    float G = rgba.y;
    float B = rgba.z;

    float yf =  0.1826f * R + 0.6142f * G + 0.0620f * B + 16.0f;
    float uf = -0.1006f * R - 0.3386f * G + 0.4392f * B + 128.0f;
    float vf =  0.4392f * R - 0.3989f * G - 0.0403f * B + 128.0f;

    Y = (uint8_t)(fminf(fmaxf(yf, 0.0f), 255.0f) + 0.5f);   // 四捨五入
    U = (uint8_t)(fminf(fmaxf(uf, 0.0f), 255.0f) + 0.5f);
    V = (uint8_t)(fminf(fmaxf(vf, 0.0f), 255.0f) + 0.5f);
}

// --------------------------------------------------------------
// 共通: テクスチャから1画素読む（unnormalized / point sampling）
// --------------------------------------------------------------
__device__ __forceinline__ uchar4 fetch_rgba(cudaTextureObject_t In, int x, int y)
{
    return tex2D<uchar4>(In, (float)x + 0.5f, (float)y + 0.5f);
}

// --------------------------------------------------------------
// 共通: 色差（U / V）を 2×2 の平均から作る（P010 出力の encode_10bit.cu と同じ作り）
//   (sx, sy) は 2×2 の左上（出力の向き）。右・下の画素は画像の端で止める。
//   タイル幅・高さは偶数なので 2×2 がタイルをまたぐことはない
// --------------------------------------------------------------
__device__ __forceinline__ void uv_avg_8bit(
    cudaTextureObject_t In, int sx, int sy, int srcW, int srcH, int flip,
    uint8_t& U, uint8_t& V)
{
    const int x1 = min(sx + 1, srcW - 1);
    const int y1 = min(sy + 1, srcH - 1);
    const int f0 = flip ? (srcH - 1 - sy) : sy;
    const int f1 = flip ? (srcH - 1 - y1) : y1;
    const uchar4 a = fetch_rgba(In, sx, f0), b = fetch_rgba(In, x1, f0);
    const uchar4 c = fetch_rgba(In, sx, f1), d = fetch_rgba(In, x1, f1);
    const float R = (a.x + b.x + c.x + d.x) * 0.25f;
    const float G = (a.y + b.y + c.y + d.y) * 0.25f;
    const float B = (a.z + b.z + c.z + d.z) * 0.25f;
    const float uf = -0.1006f * R - 0.3386f * G + 0.4392f * B + 128.0f;
    const float vf =  0.4392f * R - 0.3989f * G - 0.0403f * B + 128.0f;
    U = (uint8_t)(fminf(fmaxf(uf, 0.0f), 255.0f) + 0.5f);
    V = (uint8_t)(fminf(fmaxf(vf, 0.0f), 255.0f) + 0.5f);
}

// ==============================================================
// シングルストリーム（分割なし）
// ==============================================================
extern "C"
__global__ void rgba_to_nv12x1_split_tex_kernel(
    cudaTextureObject_t In,
    uint8_t* y_plane,  size_t y_step,
    uint8_t* uv_plane, size_t uv_step,
    int width, int height,
    int flip)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int fy = flip ? (height - 1 - y) : y;

    uint8_t Y, U, V;
    rgba_to_yuv_8bit(fetch_rgba(In, x, fy), Y, U, V);

    y_plane[y * y_step + x] = Y;

    if ((x % 2 == 0) && (y % 2 == 0)) {
        uv_avg_8bit(In, x, y, width, height, flip, U, V);
        int uv_index = (y / 2) * uv_step + (x / 2) * 2;
        uv_plane[uv_index]     = U;
        uv_plane[uv_index + 1] = V;
    }
}

// ==============================================================
// 2分割（横2×縦1）
// ==============================================================
extern "C"
__global__ void rgba_to_nv12x2_split_tex_kernel(
    cudaTextureObject_t In,
    uint8_t* y0, size_t pitchY0, uint8_t* uv0, size_t pitchUV0,
    uint8_t* y1, size_t pitchY1, uint8_t* uv1, size_t pitchUV1,
    int srcW, int srcH,
    int outW, int outH,
    int flip)
{
    int sx = blockIdx.x * blockDim.x + threadIdx.x;
    int sy = blockIdx.y * blockDim.y + threadIdx.y;

    if (sx >= srcW || sy >= srcH) return;

    int fy = flip ? (srcH - 1 - sy) : sy;

    uint8_t Y, U, V;
    rgba_to_yuv_8bit(fetch_rgba(In, sx, fy), Y, U, V);

    int quadX = (sx >= outW);

    uint8_t *Yp, *UVp;
    size_t pitchY, pitchUV;

    if (quadX == 0) { Yp = y0; UVp = uv0; pitchY = pitchY0; pitchUV = pitchUV0; }
    else            { Yp = y1; UVp = uv1; pitchY = pitchY1; pitchUV = pitchUV1; }

    int dx = sx - quadX * outW;
    int dy = sy;

    Yp[dy * pitchY + dx] = Y;

    if ((dx & 1) == 0 && (dy & 1) == 0) {
        uv_avg_8bit(In, sx, sy, srcW, srcH, flip, U, V);
        uint8_t* uv = UVp + (dy >> 1) * pitchUV + dx;
        uv[0] = U;
        uv[1] = V;
    }
}

// ==============================================================
// 4分割（横2×縦2）
// ==============================================================
extern "C"
__global__ void rgba_to_nv12x4_split_tex_kernel(
    cudaTextureObject_t In,
    uint8_t* y0, size_t pitchY0, uint8_t* uv0, size_t pitchUV0,
    uint8_t* y1, size_t pitchY1, uint8_t* uv1, size_t pitchUV1,
    uint8_t* y2, size_t pitchY2, uint8_t* uv2, size_t pitchUV2,
    uint8_t* y3, size_t pitchY3, uint8_t* uv3, size_t pitchUV3,
    int srcW, int srcH,
    int outW, int outH,
    int flip)
{
    int sx = blockIdx.x * blockDim.x + threadIdx.x;
    int sy = blockIdx.y * blockDim.y + threadIdx.y;

    if (sx >= srcW || sy >= srcH) return;

    int fy = flip ? (srcH - 1 - sy) : sy;

    uint8_t Y, U, V;
    rgba_to_yuv_8bit(fetch_rgba(In, sx, fy), Y, U, V);

    int quadX = (sx >= outW);
    int quadY = (sy >= outH);
    int q = quadY * 2 + quadX;

    uint8_t *Yp, *UVp;
    size_t pitchY, pitchUV;

    switch (q) {
    case 0: Yp = y0; UVp = uv0; pitchY = pitchY0; pitchUV = pitchUV0; break;
    case 1: Yp = y1; UVp = uv1; pitchY = pitchY1; pitchUV = pitchUV1; break;
    case 2: Yp = y2; UVp = uv2; pitchY = pitchY2; pitchUV = pitchUV2; break;
    default:Yp = y3; UVp = uv3; pitchY = pitchY3; pitchUV = pitchUV3; break;
    }

    int dx = sx - quadX * outW;
    int dy = sy - quadY * outH;

    Yp[dy * pitchY + dx] = Y;

    if ((dx & 1) == 0 && (dy & 1) == 0) {
        uv_avg_8bit(In, sx, sy, srcW, srcH, flip, U, V);
        uint8_t* uv = UVp + (dy >> 1) * pitchUV + dx;
        uv[0] = U;
        uv[1] = V;
    }
}

// ==============================================================
// 8分割（横4×縦2）
// ==============================================================
extern "C"
__global__ void rgba_to_nv12x8_split_tex_kernel(
    cudaTextureObject_t In,
    uint8_t* y0, size_t pitchY0, uint8_t* uv0, size_t pitchUV0,
    uint8_t* y1, size_t pitchY1, uint8_t* uv1, size_t pitchUV1,
    uint8_t* y2, size_t pitchY2, uint8_t* uv2, size_t pitchUV2,
    uint8_t* y3, size_t pitchY3, uint8_t* uv3, size_t pitchUV3,
    uint8_t* y4, size_t pitchY4, uint8_t* uv4, size_t pitchUV4,
    uint8_t* y5, size_t pitchY5, uint8_t* uv5, size_t pitchUV5,
    uint8_t* y6, size_t pitchY6, uint8_t* uv6, size_t pitchUV6,
    uint8_t* y7, size_t pitchY7, uint8_t* uv7, size_t pitchUV7,
    int srcW, int srcH,
    int outW, int outH,
    int flip)
{
    int sx = blockIdx.x * blockDim.x + threadIdx.x;
    int sy = blockIdx.y * blockDim.y + threadIdx.y;

    if (sx >= srcW || sy >= srcH) return;

    int fy = flip ? (srcH - 1 - sy) : sy;

    uint8_t Y, U, V;
    rgba_to_yuv_8bit(fetch_rgba(In, sx, fy), Y, U, V);

    int quadX = sx / outW;
    int quadY = sy / outH;

    if (quadX >= 4 || quadY >= 2) return;

    int q = quadY * 4 + quadX;

    uint8_t *Yp, *UVp;
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

    int dx = sx - quadX * outW;
    int dy = sy - quadY * outH;

    Yp[dy * pitchY + dx] = Y;

    if ((dx & 1) == 0 && (dy & 1) == 0) {
        uv_avg_8bit(In, sx, sy, srcW, srcH, flip, U, V);
        uint8_t* uv = UVp + (dy >> 1) * pitchUV + dx;
        uv[0] = U;
        uv[1] = V;
    }
}
