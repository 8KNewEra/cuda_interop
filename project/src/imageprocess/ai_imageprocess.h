#ifndef AI_IMAGEPROCESS_H
#define AI_IMAGEPROCESS_H

#include "src/main/__global__.h"
#include "src/imageprocess/cuda_imageprocess.h"
#pragma once

#include <QThread>
#include <NvInfer.h>
#include <cuda_runtime_api.h> // CUDAのメモリ操作関数を使うために必須
#include <QFile>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include "qdir.h"

class AI_ImageProcess : public QThread {
    Q_OBJECT
public:

    AI_ImageProcess(QObject* parent = nullptr);
    void testBuildTensorRTEngine();
    void initYoloTensorRT();
    void yolo_analysis(gpuFrame img);
    void initRifeTensorRT(int width,int height);
    void rife_interpolate(const gpuFrame& frame0, const gpuFrame& frame1,
                          std::vector<gpuFrame>& out_frames,
                          cudaStream_t stream, CUDA_ImageProcess *CUDA_Img_Proc) ;

private:
    // Yolo関連
    // // TensorRT関連
    // nvinfer1::IRuntime* m_runtime = nullptr;
    // nvinfer1::ICudaEngine* m_engine = nullptr;
    // nvinfer1::IExecutionContext* m_context = nullptr;
    // cv::cuda::Stream m_stream;

    // // GPU上のメモリポインタ
    // void* m_d_input = nullptr;  // TensorRTへの入力（640x640 CHW）
    // void* m_d_output = nullptr; // TensorRTからの出力

    // // CPU上の結果受け取り用配列
    // std::vector<float> m_h_output;

    // // 前処理用のGpuMat（メモリ再確保を防ぐための使い回しバッファ）
    // cv::cuda::GpuMat m_gpu_resized;
    // cv::cuda::GpuMat m_gpu_rgb;
    // cv::cuda::GpuMat m_gpu_float;
    // std::vector<cv::cuda::GpuMat> m_input_channels; // ゼロコピー転送用の魔法の配列

    // RIFE関連
    // 1つのRIFEエンジンインスタンスを管理する構造体
    struct RifeEngineInstance {
        int interpolateRatio = 2;  // 補間倍率 (2, 3, 4...)

        // TensorRTコアリソース
        nvinfer1::IRuntime* runtime = nullptr;
        nvinfer1::ICudaEngine* engine = nullptr;
        nvinfer1::IExecutionContext* context = nullptr;

        // VRAM生のポインタ
        void* d_img0 = nullptr;
        void* d_img1 = nullptr;
        std::vector<void*> d_outputs;

        // カーネルへ渡す用ラッパー
        gpuFrame gpu_float_img0;
        gpuFrame gpu_float_img1;
        std::vector<gpuFrame> gpu_float_outputs;
    };

    std::map<int, RifeEngineInstance> m_rife_instances;
};

#endif // AI_IMAGEPROCESS_H
