#ifndef SAVE_ENCODE_H
#define SAVE_ENCODE_H

#include <QThread>
#include <QObject>
#include <cuda_runtime.h>
#include <QDebug>
#include <QFile>
#include <queue>
#include "qmutex.h"
#include "src/imageprocess/cuda_imageprocess.h"
#include "src/main/__global__.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/audio_fifo.h>
#include <libswscale/swscale.h>
#include <libavutil/hwcontext.h>
#include "libswresample/swresample.h"
}

struct FrameSlot {
    AVFrame* frame = nullptr;
    cudaEvent_t ready = nullptr;   // このslotのコピー完了通知

    uint8_t* d_y = nullptr;
    size_t y_pitch = 0;
    uint8_t* d_uv = nullptr;
    size_t uv_pitch = 0;
};

struct VideoEncoder {
    AVBufferRef* hw_device_ctx = nullptr;        // CUDA デバイスのコンテキスト
    AVCodecContext* codec_ctx = nullptr;
    AVStream*       stream    = nullptr;

    // リングバッファ
    AVBufferRef*    hw_frames_ctx = nullptr;
    std::vector<FrameSlot> hw_frames;

    std::queue<int> inflight;
    std::mutex inflight_mtx;
    std::condition_variable inflight_cv;
    int max_inflight = 32; // ★これが重要（調整ポイント）

    cudaStream_t st = nullptr;
};

struct AudioJob
{
    QVector<QByteArray> audio_pcm;
    QVector<int> audio_pts;
};


class save_encode
{
public:
    save_encode(int h,int w);
    ~save_encode();
    void encode(VideoFrame Frame);

private:
    void initialized_ffmpeg_hardware_context(int i);
    void initialized_ffmpeg_codec_context(int i,int max_split);
    int height_,width_;

    void encode_video(VideoFrame Frame);;

    // --- FFmpeg 関連 ---
    AVPacket* packet = nullptr;
    std::vector<std::unique_ptr<VideoEncoder>> ve;   // デフォルトコンストラクタで N 個作成
    AVFormatContext* fmt_ctx = nullptr;          // 出力ファイルのフォーマットコンテキスト
    int64_t frame_index = 0;                         // PTS 管理用

    //エンコード設定
    const EncodeSettings& encodeSettings = EncodeSettingsManager::getInstance().getSettings();
    const DecodeInfo& VideoInfo = DecodeInfoManager::getInstance().getSettings();

    //CUDA周り
    CUDA_ImageProcess* CUDA_IMG_Proc=nullptr;
    cudaStream_t st = nullptr;
    cudaEvent_t ev = nullptr;
    QMutex muxMutex;

    //音声
    void init_audio_encoder();
    void encode_audio(AudioJob Frame);
    SwrContext* swr_enc = nullptr;
    AVCodecContext* audio_enc_ctx = nullptr;
    AVStream*       audio_stream  = nullptr;
    int64_t         audio_pts     = 0;
    AVAudioFifo* audio_fifo = nullptr;

    //音声エンコードスレッド関連
    std::queue<AudioJob> audioQueue;
    std::mutex audioMutex;
    std::condition_variable audioCV;
    std::thread audioThread;
    std::atomic<bool> audioRunning = false;
    std::mutex audioEncMutex;
    void audio_loop();
    void stop_audio_thread();
    void audio_flush();

    //リング設定(映像エンコード)
    void wait_inflight(VideoEncoder& enc);
    void drain_encoder(VideoEncoder& enc, AVFormatContext* fmt_ctx, AVPacket* packet);
};

#endif // SAVE_ENCODE_H
