#ifndef DECODE_THREAD_H
#define DECODE_THREAD_H

#include <QThread>
#include <QObject>
#include <QWaitCondition>
#include <QMutex>
#include <cuda_runtime.h>
#include <QDebug>
#include <QFile>
#include "__global__.h"
#include "cuda_imageprocess.h"
#include "qaudiosink.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
#include <libavutil/hwcontext.h>
#include "libswresample/swresample.h"
}

extern int g_cudaDeviceID;

struct VideoDecorder {
    AVCodecContext* codec_ctx = nullptr;
    AVStream* stream = nullptr;
    int stream_index=0;
    const AVCodec* decoder;
    AVFrame* hw_frame;
};

class decode_thread : public QObject {
    Q_OBJECT

public:
    explicit decode_thread(QString FilePath,bool audio_m,QObject *parent = nullptr);
    ~decode_thread() override;
    bool encode_flag=false;
    bool audio_mode=false;

signals:
    void send_decode_image(uint8_t* d_rgba, size_t pitch_rgba,int slider);
    void send_audio(QByteArray pcm);
    void send_video_info();
    void finished();
    void decode_end();
    void decode_error(QString error);

public slots:
    void sliderPlayback(int value);
    void resumePlayback();
    void pausePlayback();
    void reversePlayback();
    void startProcessing();
    void stopProcessing();
    void processFrame();
    void receve_decode_flag();
    void set_decode_speed(int speed);

private:
    enum DecodeState {
        STATE_DECODE_READY,
        STATE_DECODING,
        STATE_WAIT_DECODE_FLAG
    };

    void initialized_ffmpeg();
    QString ffmpegErrStr(int errnum);
    const char* selectDecoder(const char* codec_name);
    double getFrameRate(AVFormatContext* fmt_ctx, int video_stream_index);
    void get_last_frame_pts();
    void get_gpudecode_image();
    void get_decode_audio(AVPacket* pkt);
    void CUDA_RGBA_to_merge();

    bool video_play_flag;
    bool video_reverse_flag;
    bool thread_stop_flag =false;

    // FFmpeg関連
    const char* input_filename;
    QByteArray File_byteArray;
    std::vector<VideoDecorder> vd;   // デフォルトコンストラクタで N 個作成
    AVFormatContext* fmt_ctx = nullptr;
    AVBufferRef* hw_device_ctx = nullptr;
    DecodeState decode_state = STATE_DECODE_READY;
    DecodeInfo& VideoInfo = DecodeInfoManager::getInstance().getSettingsNonConst();
    int slider_No;
    QMutex mutex;

    //タイマー関連
    QTimer *timer;
    QElapsedTimer elapsedTimer;
    int interval_ms;


    //CUDA
    CUDA_ImageProcess* CUDA_IMG_Proc=nullptr;
    uint8_t* d_rgba;
    size_t pitch_rgba;
    cudaStream_t stream;
    cudaEvent_t events;

    //音声
    // ----- Audio -----
    int audio_stream_index = -1;
    AVCodecContext* audio_ctx = nullptr;
    const AVCodec* audio_decoder = nullptr;

    SwrContext* swr = nullptr;
    uint8_t* audio_buffer = nullptr;
    int audio_buffer_size = 0;

    AVChannelLayout in_ch_layout = {};
    AVChannelLayout out_ch_layout = {};

    int in_sample_rate  = 0;
    int out_sample_rate = 0;

    AVSampleFormat in_format  = AV_SAMPLE_FMT_NONE;
    AVSampleFormat out_format = AV_SAMPLE_FMT_S16;  // S16 にリサンプルする

    QByteArray pcm;
    QAudioSink* audioSink = nullptr;
    QIODevice* audioOutput = nullptr;

    int a=0;
};

#endif // DECODE_THREAD_H
