#include "decode_thread.h"
#include "opencv2/imgcodecs.hpp"
#include <QDebug>
#include <QOpenGLContext>
#include <QOpenGLFunctions>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
#include <libavutil/hwcontext.h>
#include <libavutil/frame.h>
}

// コンストラクタ
#include "decode_thread.h"
#include <QDebug>

#include <QTimer>

decode_thread::decode_thread(QString FilePath, QObject *parent)
    : QObject(parent), video_play_flag(true), timer(new QTimer(this))  {
    File_byteArray = FilePath.toUtf8();
    input_filename = File_byteArray.constData();
}

decode_thread::~decode_thread() {
    stopProcessing();
    delete CUDA_IMG_processor;
    CUDA_IMG_processor=nullptr;
    qDebug() << "Live Thread: Destructor called";
}

void decode_thread::receve_decode_flag(){
    decode_flag = true;
}

void decode_thread::set_decode_speed(int speed){
    interval_ms = static_cast<int>(1000.0 / speed);
    elapsedTimer.start();
    timer->start(interval_ms); // 接続し直さなくてもOK
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
    decode_flag=true;
    slider_No=value*pts_per_frame;
    video_play_flag = false;
    video_reverse_flag = false;
}

void decode_thread::resumePlayback() {
    decode_flag=true;
    processFrame();
    QMutexLocker locker(&mutex);
    video_play_flag = true;
    video_reverse_flag = false;
}

void decode_thread::pausePlayback() {
    QMutexLocker locker(&mutex);
    video_play_flag = false;
}

void decode_thread::reversePlayback(){
    decode_flag=true;
    processFrame();
    QMutexLocker locker(&mutex);
    video_play_flag = true;
    video_reverse_flag = true;
}


void decode_thread::processFrame() {
    QMutexLocker locker(&mutex);

    if (!video_play_flag && slider_No == Get_Frame_No){
        return;
    }

    if(decode_flag){
        get_decode_image();
        decode_flag=false;
    }
}

// FFmpeg 初期化
void decode_thread::initialized_ffmpeg() {
    if(CUDA_IMG_processor==nullptr){
        CUDA_IMG_processor=new CUDA_ImageProcess();
    }

    packet = av_packet_alloc();
    hw_frame = av_frame_alloc();
    int ret;

    // CUDA デバイスコンテキスト作成
    ret = av_hwdevice_ctx_create(&hw_device_ctx, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0);
    if (ret < 0) {
        qDebug() << "Failed to create CUDA device context";
        return;
    }


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
    interval_ms = static_cast<double>(1000.0 / 1000);
    elapsedTimer.start();
    timer->start(interval_ms);

    connect(timer, &QTimer::timeout, this, &decode_thread::processFrame);

    // フレームレートを元にタイマー設定
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

    codec_ctx = avcodec_alloc_context3(decoder);
    codec_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);
    avcodec_parameters_to_context(codec_ctx, fmt_ctx->streams[video_stream_index]->codecpar);
    avcodec_open2(codec_ctx, decoder, nullptr);

    //1フレームのPTSを計算
    double time_base_d = av_q2d(fmt_ctx->streams[video_stream_index]->time_base);
    pts_per_frame = 1.0 / (framerate * time_base_d);
    qDebug() << "1フレームのPTS数:" << pts_per_frame;

    //スライダー設定
    emit send_video_info(pts_per_frame, (fmt_ctx->streams[video_stream_index]->duration)/pts_per_frame-1,framerate);

    //再生
    video_play_flag = true;
    video_reverse_flag = false;

    processFrame();
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

// フレームレートを取得する関数
double decode_thread::getFrameRate(AVFormatContext* fmt_ctx, int video_stream_index) {
    if (video_stream_index < 0 || video_stream_index >= fmt_ctx->nb_streams) {
        qDebug() << "Invalid video stream index";
        return 0.0;
    }

    // 映像ストリームの取得
    AVStream* video_stream = fmt_ctx->streams[video_stream_index];

    // avg_frame_rate を取得
    AVRational frame_rate = video_stream->avg_frame_rate;
    if (frame_rate.num == 0 || frame_rate.den == 0) {
        // avg_frame_rate が0の時は r_frame_rate を使用
        frame_rate = video_stream->r_frame_rate;
    }

    // フレームレートを計算（分子 / 分母）
    double frame_rate_value = static_cast<double>(frame_rate.num) / static_cast<double>(frame_rate.den);

    return frame_rate_value;
}

// フレーム取得
// 1フレーム取得を試みる（タイマーから呼ばれる）
void decode_thread::get_decode_image() {
    // --- シーク処理（必要なら）
    if (slider_No != Get_Frame_No || video_reverse_flag == true) {
        if (video_reverse_flag == true) {
            slider_No = slider_No - pts_per_frame;
        }
        avcodec_flush_buffers(codec_ctx);  // デコーダのバッファをフラッシュ
        av_seek_frame(fmt_ctx, video_stream_index, slider_No, AVSEEK_FLAG_BACKWARD);  // シーク
        Get_Frame_No = slider_No;
    }

    while (true) {
        if(av_read_frame(fmt_ctx, packet) >= 0){
            if (packet->stream_index == video_stream_index) {
                if (avcodec_send_packet(codec_ctx, packet) < 0) continue;

                if (avcodec_receive_frame(codec_ctx, hw_frame) == 0) {
                    ffmpeg_to_CUDA(); // ここでCUDA転送など
                    break;
                }
            }
        }else{
            // デコーダをフラッシュして残りフレームを取り出す
            avcodec_send_packet(codec_ctx, nullptr);
            if (avcodec_receive_frame(codec_ctx, hw_frame) == 0) {
                ffmpeg_to_CUDA();
                break;
            }
        }
    }
}

void decode_thread::ffmpeg_to_CUDA(){
    // QElapsedTimer timer;
    // timer.start();
    int width = hw_frame->width;
    int height = hw_frame->height;

    if (gpu_y.empty() || gpu_y.size() != cv::Size(width, height)) {
        gpu_y.create(height, width, CV_8UC1);
        gpu_uv.create(height / 2, width / 2, CV_8UC2);
        bgr_image.create(height, width, CV_8UC3);
    }

    cudaMemcpy(gpu_y.ptr(), hw_frame->data[0], height * hw_frame->linesize[0], cudaMemcpyDeviceToDevice);
    cudaMemcpy(gpu_uv.ptr(), hw_frame->data[1], (height / 2) * hw_frame->linesize[1], cudaMemcpyDeviceToDevice);

    // フレーム番号更新
    AVRational time_base = fmt_ctx->streams[video_stream_index]->time_base;
    Get_Frame_No = hw_frame->best_effort_timestamp;
    slider_No = Get_Frame_No;

    if(CUDA_IMG_processor->NV12_to_BGR(gpu_y,gpu_uv,bgr_image,height,width)){
        // cv::Mat g;
        // bgr_image.download(g);
        // bgr_image.upload(g);
        // cv::imwrite("debug.bmp",g);

        CUDA_IMG_processor->Gradation(bgr_image,bgr_image,height,width);

        emit send_decode_image(bgr_image);
        emit send_slider(Get_Frame_No/pts_per_frame);
        //qDebug()<<Get_Frame_No/pts_per_frame;
    }
    // double seconds = timer.nsecsElapsed() / 1e6; // ナノ秒 →  ミリ秒
    // qDebug()<<seconds;
}



