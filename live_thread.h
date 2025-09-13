#ifndef LIVE_THREAD_H
#define LIVE_THREAD_H

#include <QThread>
#include <QObject>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/cuda.hpp>  // CUDAサポート
#include "cuda_imageprocess.h" // CUDA処理用クラスl
#include <QWaitCondition>
#include <QMutex>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
#include <libavutil/hwcontext.h>
}

#include <npp.h>

extern int g_slider_No;
extern int g_gpu_usage;
// extern bool g_video_play_flag;

class live_thread : public QObject {
    Q_OBJECT

public:
    explicit live_thread(QString FilePath,QObject *parent = nullptr);
    ~live_thread() override;

signals:
    void send_live_image(cv::cuda::GpuMat image);
    void send_slider(int frame_no);
    void slider_max_min(int max,int min);

public slots:
    void get_live_image();
    void resumePlayback();
    void pausePlayback();
    void reversePlayback();
    void startProcessing();
    void stopProcessing();

protected:


private:
    void initialized_ffmpeg();
    const char* selectDecoder(const char* codec_name);
    void processFrame();
    void enableVSync();
    double getFrameRate(AVFormatContext* fmt_ctx, int video_stream_index);
    void ffmpeg_to_CUDA();

    bool video_play_flag;
    bool video_reverse_flag;

    cv::cuda::GpuMat GPU_Image;
    cv::cuda::GpuMat GPU_Processed_Image;

    // FFmpeg関連
    const char* input_filename;
    QByteArray File_byteArray;
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

    CUDA_ImageProcess* processorThread;  // CUDA処理用スレッド
    int Get_Frame_No;
    int Slider_Frame_No;
    double pts_per_frame ;

    QTimer *timer;
    QElapsedTimer elapsedTimer;
    int interval_ms;
    QWaitCondition waitCondition;
    QMutex mutex;
};

#endif // LIVE_THREAD_H
