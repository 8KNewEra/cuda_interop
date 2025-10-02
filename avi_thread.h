#ifndef AVI_THREAD_H
#define AVI_THREAD_H

#include <QThread>
#include <QObject>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/cuda.hpp>  // CUDAサポート
#include "dstorage.h"
#include <QWaitCondition>
#include <QMutex>

#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <iostream>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
#include <libavutil/hwcontext.h>
}

#include <npp.h>

extern int g_gpu_usage;
// extern bool g_video_play_flag;

struct FrameIndex {
    size_t offset;  // ファイル内のフレーム開始位置（moviチャンク内の相対位置）
    size_t size;    // フレームサイズ（バイト）
    int width;      // 省略可（後で埋める）
    int height;     // 省略可（後で埋める）
    int format;     // 省略可（RGB24, RGBAなど）
};

class avi_thread : public QObject {
    Q_OBJECT

public:
    explicit avi_thread(QObject *parent = nullptr);
    ~avi_thread() override;

signals:
    void send_decode_image(uint8_t* d_y, size_t pitch_y,uint8_t* d_uv, size_t pitch_uv,int height, int width);
    void send_slider(int frame_no);
    void slider_max_min(int max,int min,int frame);

public slots:
    void get_decode_image();
    void resumePlayback();
    void pausePlayback();
    void startProcessing();
    void stopProcessing();

protected:


private:
    void processFrame();

    bool g_video_play_flag;

    cv::cuda::GpuMat GPU_Image;
    cv::cuda::GpuMat GPU_Processed_Image;

    // FFmpeg関連
    AVPacket* packet;
    AVFrame* hw_frame;
    AVFormatContext* fmt_ctx = nullptr;
    AVCodecContext* codec_ctx = nullptr;
    AVHWDeviceContext* hw_ctx = nullptr;
    AVBufferRef* hw_device_ctx = nullptr;
    const AVCodec* decoder;
    int video_stream_index;
    int frame_number=2;

    cv::cuda::GpuMat gpu_y, gpu_uv, bgr_image;
    NppiSize roi;
    const Npp8u* src[2];

    int Get_Frame_No;
    int Slider_Frame_No;
    double pts_per_frame ;

    QTimer *timer;
    QElapsedTimer elapsedTimer;
    int interval_ms;
    QWaitCondition waitCondition;
    QMutex mutex;

    bool initialize(); // AVI解析＆DirectStorage準備
    bool loadFrameToGpu(int frameNumber, cv::cuda::GpuMat& outGpuMat);
    std::string aviFilePath;
    std::vector<FrameIndex> frameList;
    void* dstorageQueue = nullptr;
    void* dstorageFactory = nullptr;
    void* gpuFileHandle = nullptr;
    IDStorageFile* gpuFile = nullptr; // ← ★ これを追加
    ID3D12Device* g_d3d12Device;

    IDStorageQueue* queue = static_cast<IDStorageQueue*>(dstorageQueue);
    int slider_No;

    std::vector<FrameIndex> ParseAviIndex(const std::string& path);
    bool initDirectStorage();
    bool readFrameToGpu(const FrameIndex& frame, cv::cuda::GpuMat& outGpuMat);
};

#endif // AVI_THREAD_H
