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

    bool Flip_RGBA_to_NV12(uint8_t* d_y, size_t pitch_y,uint8_t* d_uv, size_t pitch_uv,uint8_t* d_rgba, size_t pitch_rgba,int height, int width);
    bool NV12_to_RGBA(uint8_t* d_rgba, size_t pitch_rgba,uint8_t* d_y, size_t pitch_y,uint8_t* d_uv, size_t pitch_uv,int height, int width);
    bool Gradation(uint8_t *output,size_t pitch_output,uint8_t *input,size_t pitch_input,int height,int width);
    bool calc_histgram(HistData* Histdata,cudaTextureObject_t texObj,int width,int height);
    bool draw_histgram(cudaSurfaceObject_t surfOut,int width,int height,uint32_t* d_hist_r,uint32_t* d_hist_g,uint32_t* d_hist_b,unsigned int* max_r,unsigned int* max_g,unsigned int* max_b);
    bool histgram_normalize(float* vbo,int num_bins,HistData* Histdata,HistStats* input_stats);
    bool histogram_status(HistData* Histdata,HistStats* input_stats);
};


#endif // CUDA_IMAGEPROCESS_H
