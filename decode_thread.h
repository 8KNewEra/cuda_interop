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

class decode_thread : public QObject {
    Q_OBJECT

public:
    explicit decode_thread(QString FilePath,QObject *parent = nullptr);
    ~decode_thread() override;

signals:
    void send_decode_image(uint8_t* d_rgba, size_t pitch_rgba,int slider);
    void send_audio(QByteArray pcm);
    void send_video_info();
    void finished();
    void decode_end();
    void decode_error(QString error);

public slots:
    void get_decode_image();
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
    void ffmpeg_to_CUDA();
    void get_last_frame_pts();
    void ffmpeg_software_process();


    bool video_play_flag;
    bool video_reverse_flag;
    bool thread_stop_flag =false;

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

    DecodeState decode_state = STATE_DECODE_READY;

    int slider_No;
    QMutex mutex;

    QTimer *timer;
    QElapsedTimer elapsedTimer;
    int interval_ms;

    uint8_t *d_rgba;
    size_t pitch_rgba = 0;
    CUDA_ImageProcess* CUDA_IMG_Proc=nullptr;

    DecodeInfo& VideoInfo = DecodeInfoManager::getInstance().getSettingsNonConst();


    //音声
    // ----- Audio -----
    int audio_stream_index = -1;
    AVCodecContext* audio_ctx = nullptr;
    const AVCodec* audio_decoder = nullptr;

    SwrContext* swr = nullptr;
    uint8_t* audio_buffer = nullptr;
    int audio_buffer_size = 0;

    AVSampleFormat out_format = AV_SAMPLE_FMT_S16;
    uint64_t out_ch_layout = AV_CH_LAYOUT_STEREO;
    QByteArray pcm;

    QAudioSink* audioSink = nullptr;
    QIODevice* audioOutput = nullptr;
    int out_sample_rate = 48000;
};

#endif // DECODE_THREAD_H
