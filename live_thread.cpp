#include "live_thread.h"
#include "opencv2/imgcodecs.hpp"
#include <QDebug>
#include <QOpenGLContext>
#include <QOpenGLFunctions>
#include <QTimer>
#include <npp.h>

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
#include "live_thread.h"
#include <QDebug>

live_thread::live_thread(QString FilePath, QObject *parent)
    : QObject(parent), video_play_flag(true), timer(new QTimer(this)) {
    File_byteArray = FilePath.toUtf8();
    input_filename = File_byteArray.constData();
}

live_thread::~live_thread() {
    stopProcessing();
    qDebug() << "Live Thread: Destructor called";
}

void live_thread::startProcessing() {
    initialized_ffmpeg();
    qDebug() << "Live Thread: Start Processing";
}

void live_thread::stopProcessing() {
    qDebug() << "Live Thread: Stop Processing";
    timer->stop();
}

void live_thread::resumePlayback() {
    QMutexLocker locker(&mutex);
    video_play_flag = true;
    video_reverse_flag = false;
}

void live_thread::pausePlayback() {
    QMutexLocker locker(&mutex);
    video_play_flag = false;
}

void live_thread::reversePlayback(){
    QMutexLocker locker(&mutex);
    video_play_flag = true;
    video_reverse_flag = true;
}

void live_thread::processFrame() {
    QMutexLocker locker(&mutex);

    if (!video_play_flag && g_slider_No == Get_Frame_No)
        return;

    // // フレーム間隔チェック: 経過時間が interval_ms を超えなければ処理しない
    // qint64 elapsed = elapsedTimer.elapsed();
    // if (elapsed < interval_ms) return;
    // elapsedTimer.restart();

    get_live_image();
}


// V-Sync 有効化
void live_thread::enableVSync() {
    QOpenGLContext* context = QOpenGLContext::currentContext();
    if (context) {
        context->functions()->glFinish();
        if (context->format().swapInterval() != 1) {
            QSurfaceFormat format = context->format();
            format.setSwapInterval(1);
            context->setFormat(format);
        }
    }
}

// FFmpeg 初期化
void live_thread::initialized_ffmpeg() {
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

    //フレームレートを元にタイマー設定
    double framerate = 90;
    qDebug() << "Framerate:" << framerate;
    interval_ms = static_cast<double>(1000.0 / framerate);
    elapsedTimer.start();
    timer->start(interval_ms);

    connect(timer, &QTimer::timeout, this, &live_thread::processFrame);

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

    //スライダー設定
    emit slider_max_min(1, fmt_ctx->streams[video_stream_index]->duration);

    //1フレームのPTSを計算
    double time_base_d = av_q2d(fmt_ctx->streams[video_stream_index]->time_base);
    pts_per_frame = 1.0 / (framerate * time_base_d);
    qDebug() << "1フレームのPTS数:" << pts_per_frame;

    //再生
    video_play_flag = true;
    video_reverse_flag = false;
}

// デコーダ設定
const char* live_thread::selectDecoder(const char* codec_name) {
    if (strcmp(codec_name, "h264") == 0) {
        return "h264_cuvid";
    } else if (strcmp(codec_name, "hevc") == 0) {
        return "hevc_cuvid";
    }
    return nullptr;
}

// フレームレートを取得する関数
double live_thread::getFrameRate(AVFormatContext* fmt_ctx, int video_stream_index) {
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
void live_thread::get_live_image() {
    if (g_slider_No != Get_Frame_No||video_reverse_flag==true) {
        if(video_reverse_flag==true){
            g_slider_No=g_slider_No-pts_per_frame;
        }
        avcodec_flush_buffers(codec_ctx);
        av_seek_frame(fmt_ctx, video_stream_index, g_slider_No, AVSEEK_FLAG_BACKWARD);
        Get_Frame_No = g_slider_No;
    }

    while (av_read_frame(fmt_ctx, packet) >= 0) {
        if (packet->stream_index == video_stream_index) {
            if (avcodec_send_packet(codec_ctx, packet) < 0) {
                av_packet_unref(packet);
                continue;
            }

            if (avcodec_receive_frame(codec_ctx, hw_frame) == 0) {
                ffmpeg_to_CUDA();
                av_frame_unref(hw_frame);
                av_packet_unref(packet);
                break;
            }
        }
        av_packet_unref(packet);
    }
}

void live_thread::ffmpeg_to_CUDA(){
    int width = hw_frame->width;
    int height = hw_frame->height;

    if (gpu_y.empty() || gpu_y.size() != cv::Size(width, height)) {
        gpu_y.create(height, width, CV_8UC1);
        gpu_uv.create(height / 2, width / 2, CV_8UC2);
        bgr_image.create(height, width, CV_8UC3);
        roi.width = width;
        roi.height = height;
    }

    cudaMemcpy(gpu_y.ptr(), hw_frame->data[0], height * hw_frame->linesize[0], cudaMemcpyDeviceToDevice);
    cudaMemcpy(gpu_uv.ptr(), hw_frame->data[1], (height / 2) * hw_frame->linesize[1], cudaMemcpyDeviceToDevice);

    const Npp8u* src[2] = { gpu_y.ptr<Npp8u>(), gpu_uv.ptr<Npp8u>() };
    nppiNV12ToBGR_8u_P2C3R(src, gpu_y.step, bgr_image.ptr<Npp8u>(), bgr_image.step, roi);

    // フレーム番号更新
    AVRational time_base = fmt_ctx->streams[video_stream_index]->time_base;
    Get_Frame_No = hw_frame->best_effort_timestamp;
    g_slider_No = Get_Frame_No;

    emit send_live_image(bgr_image);
    emit send_slider(Get_Frame_No);
}



