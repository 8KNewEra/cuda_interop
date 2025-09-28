#ifndef CUDA_IMAGEPROCESS_H
#define CUDA_IMAGEPROCESS_H

#pragma once

#include <QThread>
#include <cuda_runtime.h>
#include <cuda.h>
#include <QDebug>
#include <QFile>

class CUDA_ImageProcess {
public:
    CUDA_ImageProcess();
    ~CUDA_ImageProcess();
    bool Flip_RGBA_to_NV12(uint8_t* d_rgba, size_t pitch_rgba,uint8_t* d_y, size_t pitch_y,uint8_t* d_uv, size_t pitch_uv,int height, int width);
    bool NV12_to_RGBA(uint8_t* d_y, size_t pitch_y,uint8_t* d_uv, size_t pitch_uv,uint8_t* d_rgba, size_t pitch_rgba,int height, int width);
    bool Gradation(uint8_t *output,size_t pitch_output,uint8_t *input,size_t pitch_input,int height,int width);

protected:
    struct CudaKernelModule {
        QString ptxPath;
        QString kernelName;
        CUmodule module=nullptr;
        CUfunction function=nullptr;
    };

    CudaKernelModule fliprgbToNv12Kernel; // 1つ目のカーネルを保持
    CudaKernelModule nv12TorgbaKernel; // 2つ目のカーネルを保持
    CudaKernelModule gradationKernel; // 2つ目のカーネルを保持
    bool load_CUDA_Kernel(CudaKernelModule& kernelModule);
};


#endif // CUDA_IMAGEPROCESS_H
