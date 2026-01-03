#ifndef SAVE_ENCODE_H
#define SAVE_ENCODE_H

#include <QThread>
#include <QObject>
#include <cuda_runtime.h>
#include <QDebug>
#include <QFile>
#include "cuda_imageprocess.h"
#include "__global__.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
#include <libavutil/hwcontext.h>
}

extern int g_cudaDeviceID;

struct VideoEncoder {
    AVCodecContext* codec_ctx = nullptr;
    AVStream*       stream    = nullptr;
    AVFrame*        hw_frame  = nullptr;
    AVBufferRef*    hw_frames_ctx = nullptr;
};


class save_encode
{
public:
    save_encode(int h,int w);
    ~save_encode();
    void encode(uint8_t* d_rgba, size_t pitch_rgba);

    struct QueuedPacket {
        AVPacket* pkt;
        int stream_index;
    };

private:
    void initialized_ffmpeg_hardware_context(int i);
    void initialized_ffmpeg_codec_context(int i,int max_split);
    int height_,width_;

    // --- FFmpeg 関連 ---
    AVPacket* packet = nullptr;
    std::vector<VideoEncoder> ve;   // デフォルトコンストラクタで N 個作成
    AVFormatContext* fmt_ctx = nullptr;          // 出力ファイルのフォーマットコンテキスト
    AVBufferRef* hw_device_ctx = nullptr;        // CUDA デバイスのコンテキスト
    AVRational tb;
    AVRational fr;
    int fps;  // 可変FPSでもOK
    int pts_step;  // 1フレームあたりの刻み
    int64_t frame_index = 0;                         // PTS 管理用

    //エンコード設定
    const EncodeSettings& encode_settings = EncodeSettingsManager::getInstance().getSettings();

    //CUDA周り
    CUDA_ImageProcess* CUDA_IMG_Proc=nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t event = nullptr;
};

#endif // SAVE_ENCODE_H
