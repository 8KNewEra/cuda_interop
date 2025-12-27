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
    AVHWDeviceContext* hw_ctx = nullptr;
    AVFrame* hw_frame;
    AVPacket* packet;
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
    void get_multistream_decode_image();
    void get_singlestream_gpudecode_image(int i);
    void get_decode_audio(AVPacket* pkt);
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
    void ffmpeg_to_CUDA(int i);
    void CUDA_merge();
    void get_last_frame_pts();
    void CUDA_RGBA_to_merge();
    bool all_same_pts();


    bool video_play_flag;
    bool video_reverse_flag;
    bool thread_stop_flag =false;

    // FFmpeg関連
    const char* input_filename;
    QByteArray File_byteArray;
    std::vector<VideoDecorder> vd;   // デフォルトコンストラクタで N 個作成
    std::vector<AVPacket*> pkt{} ;
    AVFormatContext* fmt_ctx = nullptr;
    AVBufferRef* hw_device_ctx = nullptr;
    DecodeState decode_state = STATE_DECODE_READY;

    int slider_No;
    QMutex mutex;
    QMutex merge_mutex;


    QTimer *timer;
    QElapsedTimer elapsedTimer;
    int interval_ms;

    DecodeInfo& VideoInfo = DecodeInfoManager::getInstance().getSettingsNonConst();

    //CUDA
    CUDA_ImageProcess* CUDA_IMG_Proc=nullptr;
    static constexpr int BUF = 16;
    std::vector<std::vector<uint8_t*>> d_rgba;      // [stream][buf]
    std::vector<std::vector<size_t>> pitch_rgba;
    std::vector<cudaStream_t> stream;
    std::vector<std::vector<cudaEvent_t>> events;   // [stream][buf]
    std::vector<int> write_idx;   // 書き込み側 index (0/1)
    std::vector<int> ready_idx;   // merge に使う buf index
    std::vector<int64_t> frame_no;
    std::vector<bool>    rgba_finish;

    uint8_t* d_merged=nullptr;
    size_t pitch_merged=0;


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
