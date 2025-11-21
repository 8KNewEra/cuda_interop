#ifndef DECODE_THREAD_H
#define DECODE_THREAD_H

#include <QThread>
#include <QObject>
#include <QWaitCondition>
#include <QMutex>
#include <QDebug>
#include <QFile>
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

class decode_thread : public QObject {
    Q_OBJECT

public:
    explicit decode_thread(QString FilePath,QObject *parent = nullptr);
    ~decode_thread() override;

signals:
    void send_decode_image(AVFrame* rgbaFrame,int slider);
    void send_video_info();
    void send_software_image(AVFrame *rgba_frame);
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
    AVFrame* sw_frame;
    AVFrame* rgbaFrame;
    AVFormatContext* fmt_ctx = nullptr;
    AVCodecContext* codec_ctx = nullptr;
    AVHWDeviceContext* hw_ctx = nullptr;
    AVBufferRef* hw_device_ctx = nullptr;
    const AVCodec* decoder;
    int video_stream_index;
    SwsContext* sws_ctx;

    DecodeState decode_state = STATE_DECODE_READY;

    int slider_No;
    QMutex mutex;

    QTimer *timer;
    QElapsedTimer elapsedTimer;
    int interval_ms;

    DecodeInfo& VideoInfo = DecodeInfoManager::getInstance().getSettingsNonConst();
};

#endif // DECODE_THREAD_H
