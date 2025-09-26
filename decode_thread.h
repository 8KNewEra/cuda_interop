#ifndef DECODE_THREAD_H
#define DECODE_THREAD_H

#include <QThread>
#include <QObject>
#include "cuda_imageprocess.h" // CUDA処理用クラスl
#include <QWaitCondition>
#include <QMutex>
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

extern int g_gpu_usage;
extern int g_fps;

class decode_thread : public QObject {
    Q_OBJECT

public:
    explicit decode_thread(QString FilePath,QObject *parent = nullptr);
    ~decode_thread() override;

signals:
    void send_decode_image(uint8_t *d_rgba,int width,int height,size_t pitch_rgba);
    void send_slider(int frame_no);
    void send_video_info(int pts,int maxframe,int framerate);

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

    bool video_play_flag;
    bool video_reverse_flag;

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

    CUDA_ImageProcess* CUDA_IMG_Proc=nullptr;  // CUDA処理用スレッド
    int Get_Frame_No;
    int Slider_Frame_No;
    int slider_No;
    double pts_per_frame ;
    QMutex mutex;

    QTimer *timer;
    QElapsedTimer elapsedTimer;
    int interval_ms;

    DecodeState decode_state = STATE_DECODE_READY;
    int No=0;

    uint8_t *d_y = nullptr, *d_uv = nullptr, *d_rgba = nullptr;
    size_t pitch_y = 0, pitch_uv = 0, pitch_rgba = 0;
};

#endif // DECODE_THREAD_H
