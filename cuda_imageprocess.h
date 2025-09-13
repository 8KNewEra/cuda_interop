#ifndef CUDA_IMAGEPROCESS_H
#define CUDA_IMAGEPROCESS_H

#pragma once

#include <QThread>
#include <opencv2/core.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudalegacy.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudafilters.hpp>
#include <cuda_runtime.h>
#include <cuda.h>
#include <QDebug>
#include <QFile>

class CUDA_ImageProcess {
public:
    CUDA_ImageProcess();
    ~CUDA_ImageProcess();
    bool RGBA_to_NV12(cv::cuda::GpuMat &rgba_gpu,cv::cuda::GpuMat &gpu_y,cv::cuda::GpuMat &gpu_uv,int hetght,int width);
    bool NV12_to_BGR(cv::cuda::GpuMat &gpu_y,cv::cuda::GpuMat &gpu_uv,cv::cuda::GpuMat &rgba_gpu,int height,int width) ;
    bool Gradation(cv::cuda::GpuMat &output,cv::cuda::GpuMat &input,int height,int width);

protected:
    struct CudaKernelModule {
        QString ptxPath;
        QString kernelName;
        CUmodule module=nullptr;
        CUfunction function=nullptr;
    };

    CudaKernelModule rgbToNv12Kernel; // 1つ目のカーネルを保持
    CudaKernelModule nv12ToBgrKernel; // 2つ目のカーネルを保持
    CudaKernelModule gradationKernel; // 2つ目のカーネルを保持
    bool load_CUDA_Kernel(CudaKernelModule& kernelModule);
};


#endif // CUDA_IMAGEPROCESS_H
