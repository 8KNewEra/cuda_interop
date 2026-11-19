#include "src/imageprocess/cuda_imageprocess.h"

// CUDA カーネルの起動口（cudaLaunchKernel で呼ぶ）。
//   カーネル本体は cuda_kernels/ の
//     8bit 経路（RGBA8 テクスチャ） : decode_8bit / encode_8bit / histogram_8bit
//     10bit 経路（RGBA16 テクスチャ）: decode_10bit / encode_10bit / histogram_10bit
//   どちらを使うかは VideoEngine が素材のビット深度で決める（Widgets 版は常に 8bit 経路）。
//   カーネルを足すときは .cu に extern "C" で書き、下の extern 宣言とメソッドを足す。

// CUDA カーネル（cuda_kernels/*.cu。extern "C" で名前をそのまま公開）
extern "C"{
    // ---- decode_8bit.cu: デコード結果 → GL テクスチャ（RGBA8、surface に直接書く） ----
    __global__ void nv12_to_rgba_8bit_surf_kernel(
        cudaSurfaceObject_t out,
        const uint8_t* y_plane, size_t y_pitch,
        const uint8_t* uv_plane, size_t uv_pitch,
        int width, int height);
    __global__ void nv12_to_rgba_10bit_surf_kernel(
        cudaSurfaceObject_t out,
        const uint8_t* y_plane, size_t y_pitch,
        const uint8_t* uv_plane, size_t uv_pitch,
        int width, int height);
    __global__ void nv12x2_to_rgba_merge_surf_kernel(
        cudaSurfaceObject_t out,
        const uint8_t* y0, size_t pitchY0, const uint8_t* uv0, size_t pitchUV0,
        const uint8_t* y1, size_t pitchY1, const uint8_t* uv1, size_t pitchUV1,
        int outW, int outH, int srcW, int srcH);
    __global__ void nv12x4_to_rgba_merge_surf_kernel(
        cudaSurfaceObject_t out,
        const uint8_t* y0, size_t pitchY0, const uint8_t* uv0, size_t pitchUV0,
        const uint8_t* y1, size_t pitchY1, const uint8_t* uv1, size_t pitchUV1,
        const uint8_t* y2, size_t pitchY2, const uint8_t* uv2, size_t pitchUV2,
        const uint8_t* y3, size_t pitchY3, const uint8_t* uv3, size_t pitchUV3,
        int outW, int outH, int srcW, int srcH);
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
        int outW, int outH, int srcW, int srcH);
    __global__ void yuv420p_to_rgba_8bit_surf_kernel(
        cudaSurfaceObject_t out,
        const uint8_t* y_plane, size_t y_pitch,
        const uint8_t* u_plane, size_t u_pitch,
        const uint8_t* v_plane, size_t v_pitch,
        int width, int height);
    __global__ void yuv420p_to_rgba_10bit_surf_kernel(
        cudaSurfaceObject_t out,
        const uint8_t* y_plane, size_t y_pitch,
        const uint8_t* u_plane, size_t u_pitch,
        const uint8_t* v_plane, size_t v_pitch,
        int width, int height, int is_be);
    __global__ void yuv422p_to_rgba_8bit_surf_kernel(
        cudaSurfaceObject_t out,
        const uint8_t* y_plane, size_t y_pitch,
        const uint8_t* u_plane, size_t u_pitch,
        const uint8_t* v_plane, size_t v_pitch,
        int width, int height);
    __global__ void yuv422p_to_rgba_10bit_surf_kernel(
        cudaSurfaceObject_t out,
        const uint8_t* y_plane, size_t y_pitch,
        const uint8_t* u_plane, size_t u_pitch,
        const uint8_t* v_plane, size_t v_pitch,
        int width, int height, int is_be);
    __global__ void uyvy422_to_rgba_8bit_surf_kernel(
        cudaSurfaceObject_t out,
        const uint8_t* yuv, size_t yuv_pitch,
        int width, int height);
    __global__ void rgb_to_rgba_10bit_surf_kernel(
        cudaSurfaceObject_t out,
        const uint8_t* r_plane, size_t r_pitch,
        const uint8_t* g_plane, size_t g_pitch,
        const uint8_t* b_plane, size_t b_pitch,
        int width, int height);


    // ---- histogram_8bit.cu: ヒストグラム ----
    __global__ void calc_histogram_shared_kernel(HistData* Histdata,cudaTextureObject_t texObj, int width, int height);
    __global__ void histgram_status_kernel(HistData* Histdata,HistStats* out_stats);
    __global__ void histgram_normalize_kernel(float* vbo,int num_bins,HistData* Histdata,HistStats* input_stats);

    // ---- decode_10bit.cu: デコード結果 → GL テクスチャ（RGBA16） ----
    __global__ void p010_tiles_to_rgba16_surf_kernel(cudaSurfaceObject_t out, P010Tiles t, int outW, int outH);
    __global__ void decode_rows_p010_to_rgba16_kernel(cudaSurfaceObject_t out, P010Tiles t, int outW, int outH);
    __global__ void decode_rows_nv12_to_rgba8_kernel(cudaSurfaceObject_t out, P010Tiles t, int outW, int outH);
    __global__ void yuv420p10_to_rgba16_surf_kernel(
        cudaSurfaceObject_t out,
        const uint8_t* y_plane, size_t y_pitch,
        const uint8_t* u_plane, size_t u_pitch,
        const uint8_t* v_plane, size_t v_pitch,
        int width, int height, int is_be);
    __global__ void yuv422p10_to_rgba16_surf_kernel(
        cudaSurfaceObject_t out,
        const uint8_t* y_plane, size_t y_pitch,
        const uint8_t* u_plane, size_t u_pitch,
        const uint8_t* v_plane, size_t v_pitch,
        int width, int height, int is_be);
    __global__ void gbrp10_to_rgba16_surf_kernel(
        cudaSurfaceObject_t out,
        const uint8_t* r_plane, size_t r_pitch,
        const uint8_t* g_plane, size_t g_pitch,
        const uint8_t* b_plane, size_t b_pitch,
        int width, int height);

    // ---- encode_10bit.cu: RGBA16 の FBO → NV12 / P010（エンコード入力）/ RGBA8（画像保存） ----
    __global__ void rgba16_to_rgba8_tex_kernel(cudaTextureObject_t In, uint8_t* out, size_t pitch, int width, int height, int dither);
    __global__ void encode_rows_rgba16_to_nv12_kernel(cudaTextureObject_t In, NV12Tiles t, int srcW, int srcH, int flip, int dither);
    __global__ void encode_rows_rgba16_to_p010_kernel(cudaTextureObject_t In, NV12Tiles t, int srcW, int srcH, int flip);
    // ---- encode_8bit.cu: RGBA8 の FBO → NV12（エンコード入力。行ごとの本線）----
    __global__ void encode_rows_rgba8_to_nv12_kernel(cudaTextureObject_t In, NV12Tiles t, int srcW, int srcH, int flip);

    // ---- histogram_10bit.cu: RGBA16 の FBO からヒストグラム（1024 段） ----
    __global__ void calc_histogram_rgba16_kernel(HistData10* Histdata, cudaTextureObject_t texObj, int width, int height);
    __global__ void histgram10_status_kernel(const HistData10* Histdata, HistStats* out_stats);
    __global__ void histgram10_normalize_kernel(float* vbo, const HistData10* Histdata, const HistStats* input_stats);
}

// --------------------------------------------------------------
// カーネル起動の共通チェック
// --------------------------------------------------------------
static bool launch_check(cudaError_t err, const char* name)
{
    if (err != cudaSuccess) {
        qDebug() << "cudaLaunchKernel failed(" << name << "): " << cudaGetErrorString(err);
        return false;
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        qDebug() << "Kernel launch error(" << name << "): " << cudaGetErrorString(err);
        return false;
    }
    return true;
}

CUDA_ImageProcess::CUDA_ImageProcess(){
    qDebug() << "CUDA_ImageProces: Constructor called";
}

CUDA_ImageProcess::~CUDA_ImageProcess(){
    qDebug() << "CUDA_ImageProces: Destructor called";
}

// ==============================================================
// NVDEC の NV12 → GL テクスチャ（RGBA8。surface に直接書く。decode_8bit.cu）
// ==============================================================

//NV12→RGBA 8bit (surface)
bool CUDA_ImageProcess::NV12_to_RGBA_8bit_surf(cudaSurfaceObject_t out,
                                               uint8_t* d_y, size_t pitch_y,
                                               uint8_t* d_uv, size_t pitch_uv,
                                               int width, int height, cudaStream_t stream)
{
    void* args[] = {&out,
                    &d_y, &pitch_y,
                    &d_uv, &pitch_uv,
                    &width, &height };

    dim3 block(32,32);
    dim3 grid((width+block.x-1)/block.x, (height+block.y-1)/block.y);

    cudaError_t err = cudaLaunchKernel((const void*)nv12_to_rgba_8bit_surf_kernel,
                                       grid, block, args, 0, stream);

    return launch_check(err, "nv12_to_rgba_8bit_surf_kernel");
}

//NV12→RGBA 10bit (surface)
bool CUDA_ImageProcess::NV12_to_RGBA_10bit_surf(cudaSurfaceObject_t out,
                                                uint8_t* d_y, size_t pitch_y,
                                                uint8_t* d_uv, size_t pitch_uv,
                                                int width, int height, cudaStream_t stream)
{
    void* args[] = {&out,
                    &d_y, &pitch_y,
                    &d_uv, &pitch_uv,
                    &width, &height };

    dim3 block(32,32);
    dim3 grid((width+block.x-1)/block.x, (height+block.y-1)/block.y);

    cudaError_t err = cudaLaunchKernel((const void*)nv12_to_rgba_10bit_surf_kernel,
                                       grid, block, args, 0, stream);

    return launch_check(err, "nv12_to_rgba_10bit_surf_kernel");
}

//NV12×2 → RGBA結合 (surface)
bool CUDA_ImageProcess::nv12x2_to_rgba_merge_surf(cudaSurfaceObject_t out,
                                                  uint8_t* y0, size_t pitchY0, uint8_t* uv0, size_t pitchUV0,
                                                  uint8_t* y1, size_t pitchY1, uint8_t* uv1, size_t pitchUV1,
                                                  int outW, int outH, int srcW, int srcH, cudaStream_t stream)
{
    void* args[] = {&out,
                    &y0, &pitchY0, &uv0, &pitchUV0,
                    &y1, &pitchY1, &uv1, &pitchUV1,
                    &outW, &outH, &srcW, &srcH };

    dim3 block(16,16);
    dim3 grid((outW+block.x-1)/block.x, (outH+block.y-1)/block.y);

    cudaError_t err = cudaLaunchKernel((const void*)nv12x2_to_rgba_merge_surf_kernel,
                                       grid, block, args, 0, stream);

    return launch_check(err, "nv12x2_to_rgba_merge_surf_kernel");
}

//NV12×4 → RGBA結合 (surface)
bool CUDA_ImageProcess::nv12x4_to_rgba_merge_surf(cudaSurfaceObject_t out,
                                                  uint8_t* y0, size_t pitchY0, uint8_t* uv0, size_t pitchUV0,
                                                  uint8_t* y1, size_t pitchY1, uint8_t* uv1, size_t pitchUV1,
                                                  uint8_t* y2, size_t pitchY2, uint8_t* uv2, size_t pitchUV2,
                                                  uint8_t* y3, size_t pitchY3, uint8_t* uv3, size_t pitchUV3,
                                                  int outW, int outH, int srcW, int srcH, cudaStream_t stream)
{
    void* args[] = {&out,
                    &y0, &pitchY0, &uv0, &pitchUV0,
                    &y1, &pitchY1, &uv1, &pitchUV1,
                    &y2, &pitchY2, &uv2, &pitchUV2,
                    &y3, &pitchY3, &uv3, &pitchUV3,
                    &outW, &outH, &srcW, &srcH };

    dim3 block(16,16);
    dim3 grid((outW+block.x-1)/block.x, (outH+block.y-1)/block.y);

    cudaError_t err = cudaLaunchKernel((const void*)nv12x4_to_rgba_merge_surf_kernel,
                                       grid, block, args, 0, stream);

    return launch_check(err, "nv12x4_to_rgba_merge_surf_kernel");
}

//NV12×8 → RGBA結合 (surface)
bool CUDA_ImageProcess::nv12x8_to_rgba_merge_surf(cudaSurfaceObject_t out,
                                                  uint8_t* y0, size_t pitchY0, uint8_t* uv0, size_t pitchUV0,
                                                  uint8_t* y1, size_t pitchY1, uint8_t* uv1, size_t pitchUV1,
                                                  uint8_t* y2, size_t pitchY2, uint8_t* uv2, size_t pitchUV2,
                                                  uint8_t* y3, size_t pitchY3, uint8_t* uv3, size_t pitchUV3,
                                                  uint8_t* y4, size_t pitchY4, uint8_t* uv4, size_t pitchUV4,
                                                  uint8_t* y5, size_t pitchY5, uint8_t* uv5, size_t pitchUV5,
                                                  uint8_t* y6, size_t pitchY6, uint8_t* uv6, size_t pitchUV6,
                                                  uint8_t* y7, size_t pitchY7, uint8_t* uv7, size_t pitchUV7,
                                                  int outW, int outH, int srcW, int srcH, cudaStream_t stream)
{
    void* args[] = {&out,
                    &y0, &pitchY0, &uv0, &pitchUV0,
                    &y1, &pitchY1, &uv1, &pitchUV1,
                    &y2, &pitchY2, &uv2, &pitchUV2,
                    &y3, &pitchY3, &uv3, &pitchUV3,
                    &y4, &pitchY4, &uv4, &pitchUV4,
                    &y5, &pitchY5, &uv5, &pitchUV5,
                    &y6, &pitchY6, &uv6, &pitchUV6,
                    &y7, &pitchY7, &uv7, &pitchUV7,
                    &outW, &outH, &srcW, &srcH };

    dim3 block(16,16);
    dim3 grid((outW+block.x-1)/block.x, (outH+block.y-1)/block.y);

    cudaError_t err = cudaLaunchKernel((const void*)nv12x8_to_rgba_merge_surf_kernel,
                                       grid, block, args, 0, stream);

    return launch_check(err, "nv12x8_to_rgba_merge_surf_kernel");
}

// ==============================================================
// CPU デコードの YUV / RGB プレーン → GL テクスチャ（RGBA8。decode_8bit.cu）
// ==============================================================

//YUV420P 8bit → RGBA(surface)
bool CUDA_ImageProcess::yuv420p_to_RGBA_8bit_surf(cudaSurfaceObject_t out,
                                                  uint8_t* d_y, size_t pitch_y,
                                                  uint8_t* d_u, size_t pitch_u,
                                                  uint8_t* d_v, size_t pitch_v,
                                                  int width, int height, cudaStream_t stream)
{
    void* args[] = {&out,
                    &d_y, &pitch_y, &d_u, &pitch_u, &d_v, &pitch_v,
                    &width, &height };

    dim3 block(16,16);
    dim3 grid((width+block.x-1)/block.x, (height+block.y-1)/block.y);

    cudaError_t err = cudaLaunchKernel((const void*)yuv420p_to_rgba_8bit_surf_kernel,
                                       grid, block, args, 0, stream);

    return launch_check(err, "yuv420p_to_rgba_8bit_surf_kernel");
}

//YUV420P 10bit → RGBA(surface)
bool CUDA_ImageProcess::yuv420p_to_RGBA_10bit_surf(cudaSurfaceObject_t out,
                                                   uint8_t* d_y, size_t pitch_y,
                                                   uint8_t* d_u, size_t pitch_u,
                                                   uint8_t* d_v, size_t pitch_v,
                                                   int width, int height, int is_be, cudaStream_t stream)
{
    void* args[] = {&out,
                    &d_y, &pitch_y, &d_u, &pitch_u, &d_v, &pitch_v,
                    &width, &height, &is_be };

    dim3 block(16,16);
    dim3 grid((width+block.x-1)/block.x, (height+block.y-1)/block.y);

    cudaError_t err = cudaLaunchKernel((const void*)yuv420p_to_rgba_10bit_surf_kernel,
                                       grid, block, args, 0, stream);

    return launch_check(err, "yuv420p_to_rgba_10bit_surf_kernel");
}

//YUV422P 8bit → RGBA(surface)
bool CUDA_ImageProcess::yuv422p_to_RGBA_8bit_surf(cudaSurfaceObject_t out,
                                                  uint8_t* d_y, size_t pitch_y,
                                                  uint8_t* d_u, size_t pitch_u,
                                                  uint8_t* d_v, size_t pitch_v,
                                                  int width, int height, cudaStream_t stream)
{
    void* args[] = {&out,
                    &d_y, &pitch_y, &d_u, &pitch_u, &d_v, &pitch_v,
                    &width, &height };

    dim3 block(16,16);
    dim3 grid((width+block.x-1)/block.x, (height+block.y-1)/block.y);

    cudaError_t err = cudaLaunchKernel((const void*)yuv422p_to_rgba_8bit_surf_kernel,
                                       grid, block, args, 0, stream);

    return launch_check(err, "yuv422p_to_rgba_8bit_surf_kernel");
}

//YUV422P 10bit → RGBA(surface)
bool CUDA_ImageProcess::yuv422p_to_RGBA_10bit_surf(cudaSurfaceObject_t out,
                                                   uint8_t* d_y, size_t pitch_y,
                                                   uint8_t* d_u, size_t pitch_u,
                                                   uint8_t* d_v, size_t pitch_v,
                                                   int width, int height, int is_be, cudaStream_t stream)
{
    void* args[] = {&out,
                    &d_y, &pitch_y, &d_u, &pitch_u, &d_v, &pitch_v,
                    &width, &height, &is_be };

    dim3 block(16,16);
    dim3 grid((width+block.x-1)/block.x, (height+block.y-1)/block.y);

    cudaError_t err = cudaLaunchKernel((const void*)yuv422p_to_rgba_10bit_surf_kernel,
                                       grid, block, args, 0, stream);

    return launch_check(err, "yuv422p_to_rgba_10bit_surf_kernel");
}

//UYVY422 8bit(packed) → RGBA(surface)
bool CUDA_ImageProcess::uyvy422_to_RGBA_8bit_surf(cudaSurfaceObject_t out,
                                                  uint8_t* d_yuv, size_t pitch_yuv,
                                                  int width, int height, cudaStream_t stream)
{
    void* args[] = {&out, &d_yuv, &pitch_yuv, &width, &height };

    dim3 block(16,16);
    dim3 grid((width+block.x-1)/block.x, (height+block.y-1)/block.y);

    cudaError_t err = cudaLaunchKernel((const void*)uyvy422_to_rgba_8bit_surf_kernel,
                                       grid, block, args, 0, stream);

    return launch_check(err, "uyvy422_to_rgba_8bit_surf_kernel");
}

//GBRP10 → RGBA(surface)
bool CUDA_ImageProcess::rgb_to_RGBA_10bit_surf(cudaSurfaceObject_t out,
                                               uint8_t* d_r, size_t pitch_r,
                                               uint8_t* d_g, size_t pitch_g,
                                               uint8_t* d_b, size_t pitch_b,
                                               int width, int height, cudaStream_t stream)
{
    void* args[] = {&out,
                    &d_r, &pitch_r, &d_g, &pitch_g, &d_b, &pitch_b,
                    &width, &height };

    dim3 block(16,16);
    dim3 grid((width+block.x-1)/block.x, (height+block.y-1)/block.y);

    cudaError_t err = cudaLaunchKernel((const void*)rgb_to_rgba_10bit_surf_kernel,
                                       grid, block, args, 0, stream);

    return launch_check(err, "rgb_to_rgba_10bit_surf_kernel");
}

// ==============================================================
// ヒストグラム（RGBA8 の FBO から。histogram_8bit.cu）
// ==============================================================

//ヒストグラム計算
bool CUDA_ImageProcess::calc_histgram(HistData* Histdata,cudaTextureObject_t texObj,int width,int height)
{
    void* args[] = {
        &Histdata,
        &texObj,
        &width,
        &height
    };

    dim3 block(16,16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    cudaError_t err = cudaLaunchKernel(
        (void*)calc_histogram_shared_kernel,
        grid, block,
        args,
        0, nullptr);

    return launch_check(err, "calc_histogram_shared_kernel");
}

//ヒストグラム詳細を計算
bool CUDA_ImageProcess::histogram_status(HistData* Histdata,HistStats* out_stats){
    void* args[] = {
        &Histdata,
        &out_stats,
    };

    dim3 block(256);
    dim3 grid(1);

    cudaError_t err = cudaLaunchKernel(
        (void*)histgram_status_kernel,
        grid, block,
        args,
        0, nullptr);

    return launch_check(err, "histgram_status_kernel");
}

//ヒストグラム正規化→VBO
bool CUDA_ImageProcess::histgram_normalize(float* vbo,int num_bins,HistData* Histdata,HistStats* input_stats){
    void* args[] = {
        &vbo,
        &num_bins,
        &Histdata,
        &input_stats
    };

    dim3 block(256);
    dim3 grid(1);

    cudaError_t err = cudaLaunchKernel(
        (void*)histgram_normalize_kernel,
        grid, block,
        args,
        0, nullptr);

    return launch_check(err, "histgram_normalize_kernel");
}


// ##############################################################
// 10bit 経路（RGBA16 テクスチャ）
// ##############################################################

// NVDEC の P010 → GL テクスチャ（RGBA16。タイル分割 x1 / x2 / x4 / x8 共通。decode_10bit.cu）
bool CUDA_ImageProcess::p010_tiles_to_rgba16_surf(cudaSurfaceObject_t out, const P010Tiles& tiles,
                                                  int outW, int outH, cudaStream_t stream)
{
    P010Tiles t = tiles;
    void* args[] = { &out, &t, &outW, &outH };

    dim3 block(32,32);
    dim3 grid((outW+block.x-1)/block.x, (outH+block.y-1)/block.y);

    cudaError_t err = cudaLaunchKernel((const void*)p010_tiles_to_rgba16_surf_kernel,
                                       grid, block, args, 0, stream);
    return launch_check(err, "p010_tiles_to_rgba16_surf_kernel");
}

// NVDEC の NV12 タイル → RGBA8（8bit の NVDEC 経路の本線。詰めたタイルは差分を戻しながら。decode_8bit.cu / decode_rows.cuh）
bool CUDA_ImageProcess::decode_rows_nv12_to_rgba8(cudaSurfaceObject_t out, const P010Tiles& tiles,
                                                      int outW, int outH, cudaStream_t stream)
{
    P010Tiles t = tiles;
    void* args[] = { &out, &t, &outW, &outH };
    dim3 block(256);
    dim3 grid((t.srcH + 1) / 2, t.count);   // 1 ブロック = 1 タイルの Y 2 行 + UV 1 行
    cudaError_t err = cudaLaunchKernel((const void*)decode_rows_nv12_to_rgba8_kernel,
                                       grid, block, args, 0, stream);
    return launch_check(err, "decode_rows_nv12_to_rgba8_kernel");
}

// NVDEC の P010 タイル → RGBA16（10bit の NVDEC 経路の本線。詰めたタイルは差分を戻しながら。decode_10bit.cu / decode_rows.cuh）
bool CUDA_ImageProcess::decode_rows_p010_to_rgba16(cudaSurfaceObject_t out, const P010Tiles& tiles,
                                                       int outW, int outH, cudaStream_t stream)
{
    P010Tiles t = tiles;
    void* args[] = { &out, &t, &outW, &outH };
    dim3 block(256);
    dim3 grid((t.srcH + 1) / 2, t.count);   // 1 ブロック = 1 タイルの Y 2 行 + UV 1 行
    cudaError_t err = cudaLaunchKernel((const void*)decode_rows_p010_to_rgba16_kernel,
                                       grid, block, args, 0, stream);
    return launch_check(err, "decode_rows_p010_to_rgba16_kernel");
}

// CPU デコードの YUV420P 10bit → RGBA16
bool CUDA_ImageProcess::yuv420p10_to_rgba16_surf(cudaSurfaceObject_t out,
                                                 uint8_t* d_y, size_t pitch_y,
                                                 uint8_t* d_u, size_t pitch_u,
                                                 uint8_t* d_v, size_t pitch_v,
                                                 int width, int height, int is_be, cudaStream_t stream)
{
    void* args[] = { &out, &d_y, &pitch_y, &d_u, &pitch_u, &d_v, &pitch_v, &width, &height, &is_be };

    dim3 block(32,32);
    dim3 grid((width+block.x-1)/block.x, (height+block.y-1)/block.y);

    cudaError_t err = cudaLaunchKernel((const void*)yuv420p10_to_rgba16_surf_kernel,
                                       grid, block, args, 0, stream);
    return launch_check(err, "yuv420p10_to_rgba16_surf_kernel");
}

// CPU デコードの YUV422P 10bit → RGBA16
bool CUDA_ImageProcess::yuv422p10_to_rgba16_surf(cudaSurfaceObject_t out,
                                                 uint8_t* d_y, size_t pitch_y,
                                                 uint8_t* d_u, size_t pitch_u,
                                                 uint8_t* d_v, size_t pitch_v,
                                                 int width, int height, int is_be, cudaStream_t stream)
{
    void* args[] = { &out, &d_y, &pitch_y, &d_u, &pitch_u, &d_v, &pitch_v, &width, &height, &is_be };

    dim3 block(32,32);
    dim3 grid((width+block.x-1)/block.x, (height+block.y-1)/block.y);

    cudaError_t err = cudaLaunchKernel((const void*)yuv422p10_to_rgba16_surf_kernel,
                                       grid, block, args, 0, stream);
    return launch_check(err, "yuv422p10_to_rgba16_surf_kernel");
}

// CPU デコードの GBRP 10bit → RGBA16（r / g / b の順でプレーンを渡す）
bool CUDA_ImageProcess::gbrp10_to_rgba16_surf(cudaSurfaceObject_t out,
                                              uint8_t* d_r, size_t pitch_r,
                                              uint8_t* d_g, size_t pitch_g,
                                              uint8_t* d_b, size_t pitch_b,
                                              int width, int height, cudaStream_t stream)
{
    void* args[] = { &out, &d_r, &pitch_r, &d_g, &pitch_g, &d_b, &pitch_b, &width, &height };

    dim3 block(32,32);
    dim3 grid((width+block.x-1)/block.x, (height+block.y-1)/block.y);

    cudaError_t err = cudaLaunchKernel((const void*)gbrp10_to_rgba16_surf_kernel,
                                       grid, block, args, 0, stream);
    return launch_check(err, "gbrp10_to_rgba16_surf_kernel");
}

// ---- エンコード入力: 行ごと（encode_rows.cuh）。1 ブロック = 1 タイルの Y 2 行 + UV 1 行 ----
//   圧縮転送のタイル（tiles.packed[q]）は XferCodec の詰めた並びへ直接書く
static bool launch_encode_rows(const void* kernel, const char* name, void** args, const NV12Tiles& t, cudaStream_t stream)
{
    dim3 block(256);
    dim3 grid((t.outH + 1) / 2, t.count);
    return launch_check(cudaLaunchKernel(kernel, grid, block, args, 0, stream), name);
}

bool CUDA_ImageProcess::encode_rows_rgba8_to_nv12(cudaTextureObject_t in, const NV12Tiles& tiles,
                                                  int srcW, int srcH, int flip, cudaStream_t stream)
{
    NV12Tiles t = tiles;
    void* args[] = { &in, &t, &srcW, &srcH, &flip };
    return launch_encode_rows((const void*)encode_rows_rgba8_to_nv12_kernel, "encode_rows_rgba8_to_nv12_kernel", args, t, stream);
}

bool CUDA_ImageProcess::encode_rows_rgba16_to_nv12(cudaTextureObject_t in, const NV12Tiles& tiles,
                                                   int srcW, int srcH, int flip, bool dither, cudaStream_t stream)
{
    NV12Tiles t = tiles;
    int d = dither ? 1 : 0;
    void* args[] = { &in, &t, &srcW, &srcH, &flip, &d };
    return launch_encode_rows((const void*)encode_rows_rgba16_to_nv12_kernel, "encode_rows_rgba16_to_nv12_kernel", args, t, stream);
}

bool CUDA_ImageProcess::encode_rows_rgba16_to_p010(cudaTextureObject_t in, const NV12Tiles& tiles,
                                                   int srcW, int srcH, int flip, cudaStream_t stream)
{
    NV12Tiles t = tiles;
    void* args[] = { &in, &t, &srcW, &srcH, &flip };
    return launch_encode_rows((const void*)encode_rows_rgba16_to_p010_kernel, "encode_rows_rgba16_to_p010_kernel", args, t, stream);
}

// RGBA16 の FBO → RGBA8（画像保存用）
bool CUDA_ImageProcess::rgba16_to_rgba8_tex(cudaTextureObject_t in, uint8_t* out, size_t pitch,
                                            int width, int height, bool dither, cudaStream_t stream)
{
    int d = dither ? 1 : 0;
    void* args[] = { &in, &out, &pitch, &width, &height, &d };

    dim3 block(32,32);
    dim3 grid((width+block.x-1)/block.x, (height+block.y-1)/block.y);

    cudaError_t err = cudaLaunchKernel((const void*)rgba16_to_rgba8_tex_kernel,
                                       grid, block, args, 0, stream);
    return launch_check(err, "rgba16_to_rgba8_tex_kernel");
}

// RGBA16 の FBO からヒストグラム（histogram_10bit.cu。1024 段）
bool CUDA_ImageProcess::calc_histgram_rgba16(HistData10* Histdata, cudaTextureObject_t texObj, int width, int height)
{
    void* args[] = { &Histdata, &texObj, &width, &height };

    dim3 block(16,16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    cudaError_t err = cudaLaunchKernel((void*)calc_histogram_rgba16_kernel, grid, block, args, 0, nullptr);
    return launch_check(err, "calc_histogram_rgba16_kernel");
}

// 10bit 経路のヒストグラム統計（最小・最大・平均は 0..1023）
bool CUDA_ImageProcess::histogram_status10(HistData10* Histdata, HistStats* out_stats)
{
    void* args[] = { &Histdata, &out_stats };
    cudaError_t err = cudaLaunchKernel((void*)histgram10_status_kernel, dim3(1), dim3(256), args, 0, nullptr);
    return launch_check(err, "histgram10_status_kernel");
}

// 10bit 経路のヒストグラム → 描画用の頂点（1024 点 × 3 色）
bool CUDA_ImageProcess::histgram_normalize10(float* vbo, HistData10* Histdata, HistStats* input_stats)
{
    void* args[] = { &vbo, &Histdata, &input_stats };
    dim3 block(256);
    dim3 grid((HIST10_BINS + block.x - 1) / block.x);
    cudaError_t err = cudaLaunchKernel((void*)histgram10_normalize_kernel, grid, block, args, 0, nullptr);
    return launch_check(err, "histgram10_normalize_kernel");
}
