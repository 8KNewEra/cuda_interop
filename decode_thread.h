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

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
#include <libavutil/hwcontext.h>
}

extern int g_gpu_usage;

class decode_thread : public QObject {
    Q_OBJECT

public:
    explicit decode_thread(QString FilePath,QObject *parent = nullptr);
    ~decode_thread() override;

signals:
    void send_decode_image(uint8_t* d_y, size_t pitch_y,uint8_t* d_uv, size_t pitch_uv,int slider);
    void send_video_info();
    void send_software_image(AVFrame *rgba_frame);
    void finished();
    void decode_end();

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

    uint64_t Get_Frame_No;
    int slider_No;
    double pts_per_frame ;
    int maxFrames;
    QMutex mutex;

    QTimer *timer;
    QElapsedTimer elapsedTimer;
    int interval_ms;

    DecodeState decode_state = STATE_DECODE_READY;
    int No=0;

    uint8_t *d_y = nullptr, *d_uv = nullptr;
    size_t pitch_y = 0, pitch_uv = 0;

    DecodeInfo& VideoInfo = DecodeInfoManager::getInstance().getSettingsNonConst();
};

#endif // DECODE_THREAD_H
