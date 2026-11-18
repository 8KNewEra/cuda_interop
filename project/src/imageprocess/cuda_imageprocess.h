#ifndef CUDA_IMAGEPROCESS_H
#define CUDA_IMAGEPROCESS_H
#pragma once
#include <QThread>
#include <cuda_runtime.h>
#include <QDebug>
#include <QFile>
#include "qdebug.h"
#include "src/main/__global__.h"
#include "cuda_kernels/tiles_10bit.h"      // P010Tiles / NV12Tiles（10bit 経路の引数）
#include "cuda_kernels/histogram_10bit.h"  // HistData10（10bit 経路のヒストグラム、1024 段）

// CUDA カーネルの起動口（実装は cuda_imageprocess.cpp、カーネルは cuda_kernels/*.cu）
class CUDA_ImageProcess {
public:
    CUDA_ImageProcess();
    ~CUDA_ImageProcess();

    // ---- NVDEC の NV12 → GL テクスチャ（RGBA8。surface に直接書く）----
    //   x2 / x4 / x8 はタイル分割の動画を 1 枚につなぐ
    bool NV12_to_RGBA_8bit_surf(cudaSurfaceObject_t out,
                                uint8_t* d_y, size_t pitch_y,
                                uint8_t* d_uv, size_t pitch_uv,
                                int width, int height, cudaStream_t stream);
    bool NV12_to_RGBA_10bit_surf(cudaSurfaceObject_t out,
                                 uint8_t* d_y, size_t pitch_y,
                                 uint8_t* d_uv, size_t pitch_uv,
                                 int width, int height, cudaStream_t stream);
    bool nv12x2_to_rgba_merge_surf(cudaSurfaceObject_t out,
                                   uint8_t* y0, size_t pitchY0, uint8_t* uv0, size_t pitchUV0,
                                   uint8_t* y1, size_t pitchY1, uint8_t* uv1, size_t pitchUV1,
                                   int outW, int outH, int srcW, int srcH, cudaStream_t stream);
    bool nv12x4_to_rgba_merge_surf(cudaSurfaceObject_t out,
                                   uint8_t* y0, size_t pitchY0, uint8_t* uv0, size_t pitchUV0,
                                   uint8_t* y1, size_t pitchY1, uint8_t* uv1, size_t pitchUV1,
                                   uint8_t* y2, size_t pitchY2, uint8_t* uv2, size_t pitchUV2,
                                   uint8_t* y3, size_t pitchY3, uint8_t* uv3, size_t pitchUV3,
                                   int outW, int outH, int srcW, int srcH, cudaStream_t stream);
    bool nv12x8_to_rgba_merge_surf(cudaSurfaceObject_t out,
                                   uint8_t* y0, size_t pitchY0, uint8_t* uv0, size_t pitchUV0,
                                   uint8_t* y1, size_t pitchY1, uint8_t* uv1, size_t pitchUV1,
                                   uint8_t* y2, size_t pitchY2, uint8_t* uv2, size_t pitchUV2,
                                   uint8_t* y3, size_t pitchY3, uint8_t* uv3, size_t pitchUV3,
                                   uint8_t* y4, size_t pitchY4, uint8_t* uv4, size_t pitchUV4,
                                   uint8_t* y5, size_t pitchY5, uint8_t* uv5, size_t pitchUV5,
                                   uint8_t* y6, size_t pitchY6, uint8_t* uv6, size_t pitchUV6,
                                   uint8_t* y7, size_t pitchY7, uint8_t* uv7, size_t pitchUV7,
                                   int outW, int outH, int srcW, int srcH, cudaStream_t stream);

    // ---- CPU デコードの YUV / RGB プレーン → GL テクスチャ（RGBA8）----
    bool yuv420p_to_RGBA_8bit_surf(cudaSurfaceObject_t out,
                                   uint8_t* d_y, size_t pitch_y,
                                   uint8_t* d_u, size_t pitch_u,
                                   uint8_t* d_v, size_t pitch_v,
                                   int width, int height, cudaStream_t stream);
    bool yuv420p_to_RGBA_10bit_surf(cudaSurfaceObject_t out,
                                    uint8_t* d_y, size_t pitch_y,
                                    uint8_t* d_u, size_t pitch_u,
                                    uint8_t* d_v, size_t pitch_v,
                                    int width, int height, int is_be, cudaStream_t stream);
    bool yuv422p_to_RGBA_8bit_surf(cudaSurfaceObject_t out,
                                   uint8_t* d_y, size_t pitch_y,
                                   uint8_t* d_u, size_t pitch_u,
                                   uint8_t* d_v, size_t pitch_v,
                                   int width, int height, cudaStream_t stream);
    bool yuv422p_to_RGBA_10bit_surf(cudaSurfaceObject_t out,
                                    uint8_t* d_y, size_t pitch_y,
                                    uint8_t* d_u, size_t pitch_u,
                                    uint8_t* d_v, size_t pitch_v,
                                    int width, int height, int is_be, cudaStream_t stream);
    bool uyvy422_to_RGBA_8bit_surf(cudaSurfaceObject_t out,
                                   uint8_t* d_yuv, size_t pitch_yuv,
                                   int width, int height, cudaStream_t stream);
    bool rgb_to_RGBA_10bit_surf(cudaSurfaceObject_t out,
                                uint8_t* d_r, size_t pitch_r,
                                uint8_t* d_g, size_t pitch_g,
                                uint8_t* d_b, size_t pitch_b,
                                int width, int height, cudaStream_t stream);

    // ---- FBO（texture object）→ NV12（エンコード入力。x2 / x4 / x8 はタイル分割）----
    //   flip: FBO は上下反転しているので通常 1
    bool rgba_to_nv12x1_split_tex(cudaTextureObject_t in,
                                  uint8_t* d_y, size_t pitch_y, uint8_t* d_uv, size_t pitch_uv,
                                  int width, int height, int flip, cudaStream_t stream);
    bool rgba_to_nv12x2_split_tex(cudaTextureObject_t in,
                                  uint8_t* y0, size_t pitchY0, uint8_t* uv0, size_t pitchUV0,
                                  uint8_t* y1, size_t pitchY1, uint8_t* uv1, size_t pitchUV1,
                                  int srcW, int srcH, int outW, int outH, int flip, cudaStream_t stream);
    bool rgba_to_nv12x4_split_tex(cudaTextureObject_t in,
                                  uint8_t* y0, size_t pitchY0, uint8_t* uv0, size_t pitchUV0,
                                  uint8_t* y1, size_t pitchY1, uint8_t* uv1, size_t pitchUV1,
                                  uint8_t* y2, size_t pitchY2, uint8_t* uv2, size_t pitchUV2,
                                  uint8_t* y3, size_t pitchY3, uint8_t* uv3, size_t pitchUV3,
                                  int srcW, int srcH, int outW, int outH, int flip, cudaStream_t stream);
    bool rgba_to_nv12x8_split_tex(cudaTextureObject_t in,
                                  uint8_t* y0, size_t pitchY0, uint8_t* uv0, size_t pitchUV0,
                                  uint8_t* y1, size_t pitchY1, uint8_t* uv1, size_t pitchUV1,
                                  uint8_t* y2, size_t pitchY2, uint8_t* uv2, size_t pitchUV2,
                                  uint8_t* y3, size_t pitchY3, uint8_t* uv3, size_t pitchUV3,
                                  uint8_t* y4, size_t pitchY4, uint8_t* uv4, size_t pitchUV4,
                                  uint8_t* y5, size_t pitchY5, uint8_t* uv5, size_t pitchUV5,
                                  uint8_t* y6, size_t pitchY6, uint8_t* uv6, size_t pitchUV6,
                                  uint8_t* y7, size_t pitchY7, uint8_t* uv7, size_t pitchUV7,
                                  int srcW, int srcH, int outW, int outH, int flip, cudaStream_t stream);

    // ---- ヒストグラム ----
    bool calc_histgram(HistData* Histdata,cudaTextureObject_t texObj,int width,int height);
    bool histgram_normalize(float* vbo,int num_bins,HistData* Histdata,HistStats* input_stats);
    bool histogram_status(HistData* Histdata,HistStats* input_stats);


    // ==========================================================
    // 10bit 経路（RGBA16 テクスチャ。10bit 以上の素材のとき VideoEngine が使う）
    // ==========================================================
    // NVDEC の P010 → GL テクスチャ（RGBA16）。タイル分割 x1 / x2 / x4 / x8 共通
    //   rows: 10bit の NVDEC 経路の本線（行ごと。圧縮転送の詰めたタイルも扱う。先頭と pitch が 16 の倍数）
    //   to  : 画素ごとの旧版。先頭・pitch が 16 の倍数でないときの保険（詰めたタイルは扱えない）
    bool p010_tiles_to_rgba16_surf(cudaSurfaceObject_t out, const P010Tiles& tiles,
                                   int outW, int outH, cudaStream_t stream);
    // 8bit の NVDEC 経路の本線（行ごと。P010Tiles に NV12 のタイルを入れて渡す。条件は rows と同じ）
    bool nv12_tiles_rows_to_rgba8_surf(cudaSurfaceObject_t out, const P010Tiles& tiles,
                                       int outW, int outH, cudaStream_t stream);
    bool p010_tiles_rows_to_rgba16_surf(cudaSurfaceObject_t out, const P010Tiles& tiles,
                                        int outW, int outH, cudaStream_t stream);
    // CPU デコードの 10bit プレーン → GL テクスチャ（RGBA16）
    bool yuv420p10_to_rgba16_surf(cudaSurfaceObject_t out,
                                  uint8_t* d_y, size_t pitch_y,
                                  uint8_t* d_u, size_t pitch_u,
                                  uint8_t* d_v, size_t pitch_v,
                                  int width, int height, int is_be, cudaStream_t stream);
    bool yuv422p10_to_rgba16_surf(cudaSurfaceObject_t out,
                                  uint8_t* d_y, size_t pitch_y,
                                  uint8_t* d_u, size_t pitch_u,
                                  uint8_t* d_v, size_t pitch_v,
                                  int width, int height, int is_be, cudaStream_t stream);
    bool gbrp10_to_rgba16_surf(cudaSurfaceObject_t out,
                               uint8_t* d_r, size_t pitch_r,
                               uint8_t* d_g, size_t pitch_g,
                               uint8_t* d_b, size_t pitch_b,
                               int width, int height, cudaStream_t stream);
    // RGBA16 の FBO → NV12（エンコード入力。8bit 出力）/ P010（10bit 出力。ポインタ・ピッチはバイト単位）
    //   / RGBA8（画像保存用）。NV12 / P010 はタイル分割共通
    //   dither: 8bit に落とすとき（NV12 / RGBA8）に ±0.5 段階のノイズを足して縞を目立たなくする
    bool rgba16_to_nv12_split_tex(cudaTextureObject_t in, const NV12Tiles& tiles,
                                  int srcW, int srcH, int flip, bool dither, cudaStream_t stream);
    bool rgba16_to_p010_split_tex(cudaTextureObject_t in, const NV12Tiles& tiles,
                                  int srcW, int srcH, int flip, cudaStream_t stream);
    bool rgba16_to_rgba8_tex(cudaTextureObject_t in, uint8_t* out, size_t pitch,
                             int width, int height, bool dither, cudaStream_t stream);
    // RGBA16 の FBO からヒストグラム（1024 段 = 10bit の値ごと）。統計の値は 0..1023
    bool calc_histgram_rgba16(HistData10* Histdata, cudaTextureObject_t texObj, int width, int height);
    bool histogram_status10(HistData10* Histdata, HistStats* out_stats);
    bool histgram_normalize10(float* vbo, HistData10* Histdata, HistStats* input_stats);
};

#endif // CUDA_IMAGEPROCESS_H
