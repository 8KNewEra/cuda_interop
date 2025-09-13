#ifndef SAVE_ENCODE_H
#define SAVE_ENCODE_H

#include <QThread>
#include <QObject>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/cuda.hpp>  // CUDAサポート
#include "cuda_imageprocess.h" // CUDA処理用クラスl
#include <cuda_runtime.h>
#include <cuda.h>
#include <QDebug>
#include <QFile>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
#include <libavutil/hwcontext.h>
}

extern int g_fps;

class save_encode
{
public:
    save_encode(int h,int w);
    ~save_encode();
    void initialized_ffmpeg();
    void initialized_output(const std::string& path);
    bool encode(cv::cuda::GpuMat& frame);
    void encode_finished();
    void initGpuMats();
    cv::cuda::GpuMat gpu_y;
    cv::cuda::GpuMat gpu_uv;
    int height_,width_;

    // --- FFmpeg 関連 ---
    AVCodecContext* codec_ctx = nullptr;         // エンコード用のコーデックコンテキスト
    AVFormatContext* fmt_ctx = nullptr;          // 出力ファイルのフォーマットコンテキスト
    AVStream* stream = nullptr;                  // 出力ファイル内の映像ストリーム
    AVFrame* hw_frame = nullptr;                 // CUDAデバイス上の映像フレーム
    AVPacket* pkt = nullptr;                     // エンコード後のパケット
    AVBufferRef* hw_device_ctx = nullptr;        // CUDA デバイスのコンテキスト
    AVBufferRef* hw_frames_ctx = nullptr;        // CUDA フレーム用のコンテキスト

    int64_t frame_index = 0;                         // PTS 管理用
    int64_t step=0;
    int ret = 0;

    CUDA_ImageProcess* CUDA_IMG_processor=nullptr;

    int No=0;
    int No2=0;
};

#endif // SAVE_ENCODE_H
