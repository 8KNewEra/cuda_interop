#ifndef DECODE_THREAD_H
#define DECODE_THREAD_H

#include <QThread>
#include <QObject>
#include <QWaitCondition>
#include <QMutex>
#include <cuda_runtime.h>
#include <QDebug>
#include <QFile>
#include "src/main/__global__.h"
#include "src/imageprocess/cuda_imageprocess.h"
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

extern int g_openglDeviceID;

struct VideoDecorder {
    AVCodecContext* codec_ctx = nullptr;
    int stream_index=0;
    const AVCodec* decoder;
    std::vector<AVFrame*> hw_frames;
    cudaStream_t st = nullptr;
    cudaEvent_t ev = nullptr;
};

class decode_thread : public QObject {
    Q_OBJECT

public:
    explicit decode_thread(QString FilePath,QObject *parent = nullptr);
    ~decode_thread() override;
    int encode_state=STATE_NOT_ENCODE;
    bool back1frame_flag = false;

signals:
    void send_decode_image(VideoFrame Frame,bool pause,bool reverse);
    void send_audio(QByteArray pcm);
    void send_slider_info();
    void finished();
    void decode_end();
    void decode_error(QString error);
    void drop_decode();
    void heavy_process_signal(bool flag);

public slots:
    void sliderPlayback(int value);
    void slider_range_end(int value);
    void slider_range_start(int value);
    void resumePlayback();
    void pausePlayback();
    void reversePlayback();
    void back1frame();
    void go1frame();
    void high_res_sliderPlayback(int value);
    void startProcessing();
    void stopProcessing();
    void processFrame();
    void receve_decode_flag();

protected:
    enum DecodeState {
        STATE_DECODE_READY,
        STATE_DECODING,
        STATE_WAIT_DECODE_FLAG
    };

    //デコード周り
    virtual bool initialized_ffmpeg()=0;
    virtual void get_decode_image()=0;
    virtual void high_res_seek_frame(int targetFrameNo,bool heavy_UI_flag)=0;
    const char* selectDecoder(const char* codec_name);
    double getFrameRate(AVFormatContext* fmt_ctx, int video_stream_index);
    bool get_last_frame_pts();
    void get_decode_audio();

    //エラー処理
    QString Error_String="";
    QString ffmpegErrStr(int errnum);

    //制御
    bool video_play_flag;
    bool video_reverse_flag;
    bool thread_stop_flag =false;
    bool seek_flag = false;
    bool high_res_slider_flag = false;
    int slider_No;
    QMutex mutex;
    bool drop_flag=false;
    bool go1frame_flag = false;
    int back1FrameNo = 0;
    int go1FrameNo = 0;
    int high_res_sliderNo = 0;

    //FFmpeg関連
    AVPacket* packet = nullptr;
    AVFrame* audio_frame = nullptr;
    QByteArray File_byteArray;
    const char* input_filename;
    std::vector<VideoDecorder> vd;   // デフォルトコンストラクタで N 個作成
    AVFormatContext* fmt_ctx = nullptr;
    AVBufferRef* hw_device_ctx = nullptr;
    DecodeState decode_state = STATE_DECODE_READY;
    DecodeInfo& VideoInfo = DecodeInfoManager::getInstance().getSettingsNonConst();

    //CUDA
    CUDA_ImageProcess* CUDA_IMG_Proc=nullptr;
    cudaStream_t stream=nullptr;
    cudaEvent_t events=nullptr;
    VideoFrame Frame{};

    //CPUデコード用
    uint8_t *d_y=nullptr,*d_u=nullptr,*d_v=nullptr,*d_r=nullptr,*d_g=nullptr,*d_b=nullptr,*d_yuv=nullptr,*d_rgb=nullptr;
    size_t pitch_y,pitch_u,pitch_v,pitch_r,pitch_g,pitch_b,pitch_yuv,pitch_rgb;

    //音声
    int audio_stream_index = -1;
    AVCodecContext* audio_ctx = nullptr;
    const AVCodec* audio_decoder = nullptr;
    SwrContext* swr = nullptr;
    uint8_t* audio_buffer = nullptr;
    int audio_buffer_size = 0;
    AVChannelLayout in_ch_layout = {};
    AVChannelLayout out_ch_layout = {};
    QAudioSink* audioSink = nullptr;
    QIODevice* audioOutput = nullptr;
    AVFrame* audio_Frame=nullptr;

    //タイマー関連
    QTimer *timer;
    QElapsedTimer elapsedTimer;
    int interval_ms;

    //リング設定
    int ringNo = 0;
    int ringSize = 1;
};

#endif // DECODE_THREAD_H
