#include "decode_thread.h"
#include <QDebug>
#include <QOpenGLContext>
#include <QOpenGLFunctions>
#include <QDebug>
#include <QTimer>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
#include <libavutil/hwcontext.h>
#include <libavutil/frame.h>
}

decode_thread::decode_thread(QString FilePath, QObject *parent)
    : QObject(parent), video_play_flag(true), timer(new QTimer(this))  {
    File_byteArray = FilePath.toUtf8();
    input_filename = File_byteArray.constData();
}

decode_thread::~decode_thread() {
    stopProcessing();

    if (d_y) {
        cudaFree(d_y);
        cudaFree(d_uv);
    }

    qDebug() << "Live Thread: Destructor called";
}

void decode_thread::receve_decode_flag(){
    QMutexLocker locker(&mutex);
    if(decode_state==STATE_WAIT_DECODE_FLAG){
        decode_state = STATE_DECODE_READY;
    }
}

void decode_thread::set_decode_speed(int speed){
    interval_ms = static_cast<int>(1000.0 / speed);
    elapsedTimer.start();
    timer->start(interval_ms);
}

void decode_thread::startProcessing() {
    initialized_ffmpeg();
    qDebug() << "Live Thread: Start Processing";
}

void decode_thread::stopProcessing() {
    qDebug() << "Live Thread: Stop Processing";
}

void decode_thread::sliderPlayback(int value){
    pausePlayback();
    slider_No=value*pts_per_frame;
    video_play_flag = false;
    video_reverse_flag = false;
}

void decode_thread::resumePlayback() {
    receve_decode_flag();
    QMutexLocker locker(&mutex);
    video_play_flag = true;
    video_reverse_flag = false;
}

void decode_thread::pausePlayback() {
    QMutexLocker locker(&mutex);
    video_play_flag = false;
}

void decode_thread::reversePlayback(){
    receve_decode_flag();
    QMutexLocker locker(&mutex);
    video_play_flag = true;
    video_reverse_flag = true;
}

void decode_thread::processFrame() {
    QMutexLocker locker(&mutex);

    if (!video_play_flag && slider_No == Get_Frame_No){
        return;
    }

    if(decode_state==STATE_DECODE_READY){
        decode_state=STATE_DECODING;
        get_decode_image();
    }
}

// FFmpeg 初期化
void decode_thread::initialized_ffmpeg() {
    packet = av_packet_alloc();
    hw_frame = av_frame_alloc();
    int ret;

    //CUDA デバイスコンテキスト作成
    ret = av_hwdevice_ctx_create(&hw_device_ctx, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0);
    if (ret < 0) {
        qDebug() << "Failed to create CUDA device context";
        return;
    }

    //ファイルを開く
    ret = avformat_open_input(&fmt_ctx, input_filename, nullptr, nullptr);
    if (ret < 0) {
        qDebug() << "Could not open input file";
        return;
    }

    ret = avformat_find_stream_info(fmt_ctx, nullptr);
    if (ret < 0) {
        qDebug() << "Failed to retrieve input stream information";
        return;
    }

    video_stream_index = -1;
    for (unsigned i = 0; i < fmt_ctx->nb_streams; i++) {
        if (fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_index = i;
            break;
        }
    }

    if (video_stream_index == -1) {
        qDebug() << "Could not find video stream";
        return;
    }

    double framerate = getFrameRate(fmt_ctx, video_stream_index);

    //フレームレートを元にタイマー設定
    const char* codec_name = avcodec_get_name(fmt_ctx->streams[video_stream_index]->codecpar->codec_id);
    const char* decoder_name = selectDecoder(codec_name);

    if (!decoder_name) {
        qDebug() << "Unsupported codec:" << codec_name;
        return;
    }

    decoder = avcodec_find_decoder_by_name(decoder_name);
    if (!decoder) {
        qDebug() << "Decoder not found:" << decoder_name;
        return;
    }

    AVDictionary* opts = nullptr;
    av_dict_set(&opts, "split_decode_mode", "1", 0);

    codec_ctx = avcodec_alloc_context3(decoder);
    codec_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);
    avcodec_parameters_to_context(codec_ctx, fmt_ctx->streams[video_stream_index]->codecpar);
    avcodec_open2(codec_ctx, decoder, &opts);

    //1フレームのPTSを計算
    double time_base_d = av_q2d(fmt_ctx->streams[video_stream_index]->time_base);
    pts_per_frame = 1.0 / (framerate * time_base_d);
    qDebug() << "1フレームのPTS数:" << pts_per_frame;

    //スライダー設定
    emit send_video_info(pts_per_frame, (fmt_ctx->streams[video_stream_index]->duration)/pts_per_frame-1,framerate);

    //再生
    video_play_flag = true;
    video_reverse_flag = false;

    interval_ms = static_cast<double>(1000.0 / 1000);
    elapsedTimer.start();
    timer->start(interval_ms);
    connect(timer, &QTimer::timeout, this, &decode_thread::processFrame);
}

// デコーダ設定
const char*decode_thread::selectDecoder(const char* codec_name) {
    if (strcmp(codec_name, "h264") == 0) {
        qDebug()<<"h264_cuvid";
        return "h264_cuvid";
    } else if (strcmp(codec_name, "hevc") == 0) {
        qDebug()<<"hevc_cuvid";
        return "hevc_cuvid";
    }else if (strcmp(codec_name, "av1") == 0){
        qDebug()<<"av1_cuvid";
        return "av1_cuvid";
    }
    return nullptr;
}

//フレームレートを取得する関数
double decode_thread::getFrameRate(AVFormatContext* fmt_ctx, int video_stream_index) {
    if (video_stream_index < 0 || video_stream_index >= fmt_ctx->nb_streams) {
        qDebug() << "Invalid video stream index";
        return 0.0;
    }

    //映像ストリームの取得
    AVStream* video_stream = fmt_ctx->streams[video_stream_index];

    //avg_frame_rateを取得
    AVRational frame_rate = video_stream->avg_frame_rate;
    if (frame_rate.num == 0 || frame_rate.den == 0) {
        // avg_frame_rate が0の時は r_frame_rate を使用
        frame_rate = video_stream->r_frame_rate;
    }

    //フレームレートを計算
    double frame_rate_value = static_cast<double>(frame_rate.num) / static_cast<double>(frame_rate.den);

    return frame_rate_value;
}

// フレーム取得
void decode_thread::get_decode_image() {
    //シーク処理
    if (slider_No != Get_Frame_No || video_reverse_flag == true) {
        if (video_reverse_flag == true) {
            slider_No = slider_No - pts_per_frame;
        }
        avcodec_flush_buffers(codec_ctx);
        av_seek_frame(fmt_ctx, video_stream_index, slider_No, AVSEEK_FLAG_BACKWARD);
        Get_Frame_No = slider_No;
    }

    while (true) {
        if(av_read_frame(fmt_ctx, packet) >= 0){
            if (packet->stream_index == video_stream_index) {
                if (avcodec_send_packet(codec_ctx, packet) < 0) continue;

                if (avcodec_receive_frame(codec_ctx, hw_frame) == 0) {
                    ffmpeg_to_CUDA();
                    break;
                }
            }
        }else{
            avcodec_send_packet(codec_ctx, nullptr);
            if (avcodec_receive_frame(codec_ctx, hw_frame) == 0) {
                ffmpeg_to_CUDA();
                break;
            } else {
                break;
            }
        }
    }
}

//CUDAに渡して画像処理
void decode_thread::ffmpeg_to_CUDA(){
    // QElapsedTimer timer;
    // timer.start();
    int width = hw_frame->width;
    int height = hw_frame->height;

    //初回時にのみmalloc
    if (!d_y || !d_uv) {
        cudaMallocPitch(&d_y, &pitch_y, width, height);
        cudaMallocPitch(&d_uv, &pitch_uv, width, height / 2);
    }

    cudaMemcpy2D(d_y, pitch_y,
                 hw_frame->data[0], hw_frame->linesize[0],
                 width, height,
                 cudaMemcpyDeviceToDevice);

    cudaMemcpy2D(d_uv, pitch_uv,
                 hw_frame->data[1], hw_frame->linesize[1],
                 width, height / 2,
                 cudaMemcpyDeviceToDevice);

    //フレーム番号更新
    AVRational time_base = fmt_ctx->streams[video_stream_index]->time_base;
    Get_Frame_No = hw_frame->best_effort_timestamp;
    slider_No = Get_Frame_No;

    emit send_decode_image(d_y,pitch_y,d_uv,pitch_uv,width,height);
    emit send_slider(Get_Frame_No/pts_per_frame);

    decode_state=STATE_WAIT_DECODE_FLAG;
    // double seconds = timer.nsecsElapsed() / 1e6; // ナノ秒 →  ミリ秒
    // qDebug()<<seconds;
}





