#ifndef CUDA_IMAGEPROCESS_H
#define CUDA_IMAGEPROCESS_H
#pragma once
#include <QThread>
#include <cuda_runtime.h>
#include <QDebug>
#include <QFile>
#include "qdebug.h"
#include "__global__.h"

class CUDA_ImageProcess {
public:
    CUDA_ImageProcess();
    ~CUDA_ImageProcess();

    //ダミー
    bool Dummy(cudaStream_t stream);

    //directstorage確認用
    bool dump(uint8_t *cudaptr,cudaStream_t stream);

    //NV12↔RGBA
    bool NV12_to_RGBA_8bit(uint8_t* d_rgba, size_t pitch_rgba,uint8_t* d_y, size_t pitch_y,uint8_t* d_uv, size_t pitch_uv,int height, int width,cudaStream_t stream);
    bool NV12_to_RGBA_10bit(uint8_t* d_rgba, size_t pitch_rgba,uint8_t* d_y, size_t pitch_y,uint8_t* d_uv, size_t pitch_uv,int height, int width,cudaStream_t stream);
    bool Flip_RGBA_to_NV12(uint8_t* d_y, size_t pitch_y,uint8_t* d_uv, size_t pitch_uv,uint8_t* d_rgba, size_t pitch_rgba,int height, int width,cudaStream_t stream);

    //YUV420p→RGBA
    bool yuv420p_to_RGBA_8bit(uint8_t* d_rgba, size_t pitch_rgba,uint8_t* d_y, size_t pitch_y,uint8_t* d_u, size_t pitch_u,uint8_t* d_v, size_t pitch_v,int width, int height,cudaStream_t stream);
    bool yuv420p_to_RGBA_10bit(uint8_t* d_rgba, size_t pitch_rgba,uint8_t* d_y, size_t pitch_y,uint8_t* d_u, size_t pitch_u,uint8_t* d_v, size_t pitch_v,int width, int height,int is_be,cudaStream_t stream);

    //YUV422P→RGBA
    bool yuv422p_to_RGBA_8bit(uint8_t* d_rgba, size_t pitch_rgba,uint8_t* d_y, size_t pitch_y,uint8_t* d_u, size_t pitch_u,uint8_t* d_v, size_t pitch_v,int width, int height,cudaStream_t stream);
    bool yuv422p_to_RGBA_10bit(uint8_t* d_rgba, size_t pitch_rgba,uint8_t* d_y, size_t pitch_y,uint8_t* d_u, size_t pitch_u,uint8_t* d_v, size_t pitch_v,int width, int height,int is_be,cudaStream_t stream);

    //特殊(Davinci Resolve YUV422P 8bit→RGBA)
    bool uyvy422_to_RGBA_8bit(uint8_t* d_rgba, size_t pitch_rgba,uint8_t* d_yuv, size_t pitch_yuv,int width, int height,cudaStream_t stream);

    //RGB→RGBA
    bool rgb_to_RGBA_8bit(uint8_t* d_rgba, size_t pitch_rgba,uint8_t* d_r, size_t pitch_r,uint8_t* d_g, size_t pitch_g,uint8_t* d_b, size_t pitch_b,int width, int height,cudaStream_t stream);
    bool rgb_to_RGBA_10bit(uint8_t* d_rgba, size_t pitch_rgba,uint8_t* d_r, size_t pitch_r,uint8_t* d_g, size_t pitch_g,uint8_t* d_b, size_t pitch_b,int width, int height,cudaStream_t stream);
    bool rgb_to_RGBA_8bit_packed(uint8_t* d_rgba, size_t pitch_rgba,uint8_t* d_rgb, size_t pitch_rgb,int width, int height,int mode,cudaStream_t stream);
    bool rgb_to_RGBA_10bit_packed(uint8_t* d_rgba, size_t pitch_rgba,uint8_t* d_rgb, size_t pitch_rgb,int width, int height,int mode,cudaStream_t stream);

    //結合及び分割の処理
    bool nv12x2_to_rgba_merge(uint8_t* y0,  size_t pitchY0,uint8_t* uv0, size_t pitchUV0,
                              uint8_t* y1,  size_t pitchY1,uint8_t* uv1, size_t pitchUV1,
                              uint8_t* out, size_t pitchOut,int outW, int outH,int srcW, int srcH,cudaStream_t stream);
    bool rgba_to_nv12x2_flip_split(uint8_t* In, size_t pitchIn,
                                   uint8_t* y0,  size_t pitchY0, uint8_t* uv0, size_t pitchUV0,
                                   uint8_t* y1,  size_t pitchY1, uint8_t* uv1, size_t pitchUV1,
                                   int srcW, int srcH, int outW, int outH, cudaStream_t stream);
    bool nv12x4_to_rgba_merge(uint8_t* y0,  size_t pitchY0,uint8_t* uv0, size_t pitchUV0,
                              uint8_t* y1,  size_t pitchY1,uint8_t* uv1, size_t pitchUV1,
                              uint8_t* y2,  size_t pitchY2,uint8_t* uv2, size_t pitchUV2,
                              uint8_t* y3,  size_t pitchY3,uint8_t* uv3, size_t pitchUV3,
                              uint8_t* out, size_t pitchOut,int outW, int outH,int srcW, int srcH,cudaStream_t stream);
    bool rgba_to_nv12x4_flip_split(uint8_t* In, size_t pitchIn,
                                   uint8_t* y0,  size_t pitchY0,uint8_t* uv0, size_t pitchUV0,
                                   uint8_t* y1,  size_t pitchY1,uint8_t* uv1, size_t pitchUV1,
                                   uint8_t* y2,  size_t pitchY2,uint8_t* uv2, size_t pitchUV2,
                                   uint8_t* y3,  size_t pitchY3,uint8_t* uv3, size_t pitchUV3,
                                   int outW, int outH,int srcW, int srcH, cudaStream_t stream);
    bool nv12x8_to_rgba_merge(uint8_t* y0,  size_t pitchY0,uint8_t* uv0, size_t pitchUV0,
                              uint8_t* y1,  size_t pitchY1,uint8_t* uv1, size_t pitchUV1,
                              uint8_t* y2,  size_t pitchY2,uint8_t* uv2, size_t pitchUV2,
                              uint8_t* y3,  size_t pitchY3,uint8_t* uv3, size_t pitchUV3,
                              uint8_t* y4,  size_t pitchY4,uint8_t* uv4, size_t pitchUV4,
                              uint8_t* y5,  size_t pitchY5,uint8_t* uv5, size_t pitchUV5,
                              uint8_t* y6,  size_t pitchY6,uint8_t* uv6, size_t pitchUV6,
                              uint8_t* y7,  size_t pitchY7,uint8_t* uv7, size_t pitchUV7,
                              uint8_t* out, size_t pitchOut,int outW, int outH,int srcW, int srcH,cudaStream_t stream);
    bool rgba_to_nv12x8_flip_split(uint8_t* In, size_t pitchIn,
                                   uint8_t* y0,  size_t pitchY0, uint8_t* uv0, size_t pitchUV0,
                                   uint8_t* y1,  size_t pitchY1, uint8_t* uv1, size_t pitchUV1,
                                   uint8_t* y2,  size_t pitchY2, uint8_t* uv2, size_t pitchUV2,
                                   uint8_t* y3,  size_t pitchY3, uint8_t* uv3, size_t pitchUV3,
                                   uint8_t* y4,  size_t pitchY4, uint8_t* uv4, size_t pitchUV4,
                                   uint8_t* y5,  size_t pitchY5, uint8_t* uv5, size_t pitchUV5,
                                   uint8_t* y6,  size_t pitchY6, uint8_t* uv6, size_t pitchUV6,
                                   uint8_t* y7,  size_t pitchY7, uint8_t* uv7, size_t pitchUV7,
                                   int srcW, int srcH, int outW, int outH, cudaStream_t stream);

    //ヒストグラム解析
    bool calc_histgram(HistData* Histdata,cudaTextureObject_t texObj,int width,int height);
    bool histgram_normalize(float* vbo,int num_bins,HistData* Histdata,HistStats* input_stats);
    bool histogram_status(HistData* Histdata,HistStats* input_stats);

    //使わないやつ
    bool image_combine_x2(uint8_t* out, size_t pitchOut,uint8_t* img1, size_t pitch1,uint8_t* img2, size_t pitch2,int width, int height,cudaStream_t stream);
    bool image_combine_x4(uint8_t* out, size_t pitchOut,uint8_t* img1, size_t pitch1,uint8_t* img2, size_t pitch2,uint8_t* img3, size_t pitch3,uint8_t* img4, size_t pitch4,int width, int height, int blend,cudaStream_t stream);
    bool image_split_x4(uint8_t* Out[4], size_t pitch[4],uint8_t* In, size_t pitchIn,int width, int height,cudaStream_t stream);
    bool Gradation(uint8_t *output,size_t pitch_output,uint8_t *input,size_t pitch_input,int height,int width,cudaStream_t stream);
};

#endif // CUDA_IMAGEPROCESS_H
