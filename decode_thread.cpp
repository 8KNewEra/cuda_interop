#include "decode_thread.h"
#include "qfileinfo.h"
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
    QFileInfo fileInfo(FilePath);
    VideoInfo.Name = fileInfo.completeBaseName().toStdString()+".mp4";

    VideoInfo.Path = FilePath.toStdString();
    File_byteArray = FilePath.toUtf8();
    input_filename = File_byteArray.constData();
}

decode_thread::~decode_thread() {
    if (timer) {
        QMetaObject::invokeMethod(timer, "stop", Qt::QueuedConnection);
        QMetaObject::invokeMethod(timer, "deleteLater", Qt::QueuedConnection);
        timer = nullptr;
        delete timer;
    }

    // 1) デコードループ停止要求
    QThread::msleep(10);

    // 2) CUDA同期（非同期処理完了待ち）
    cudaError_t cerr = cudaDeviceSynchronize();
    if (cerr != cudaSuccess) {
        qWarning() << "cudaDeviceSynchronize failed:"
                   << cudaGetErrorString(cerr);
    }

    // 3) FFmpeg関係の安全な解放
    auto safe_free_codec = [&]() {
        if (codec_ctx) {
            avcodec_flush_buffers(codec_ctx);
            avcodec_free_context(&codec_ctx);
            codec_ctx = nullptr;
        }
    };

    auto safe_free_format = [&]() {
        if (fmt_ctx) {
            avformat_close_input(&fmt_ctx);
            fmt_ctx = nullptr;
        }
    };

    auto safe_free_frames = [&]() {
        if (hw_frame) {
            av_frame_free(&hw_frame);
            hw_frame = nullptr;
        }
        if (packet) {
            av_packet_free(&packet);
            packet = nullptr;
        }
    };

    auto safe_free_hwctx = [&]() {
        if (hw_device_ctx) {
            av_buffer_unref(&hw_device_ctx);
            hw_device_ctx = nullptr;
        }
    };

    try {
        safe_free_codec();
        safe_free_format();
        safe_free_frames();
        safe_free_hwctx();
    } catch (...) {
        qWarning() << "Exception during FFmpeg cleanup (ignored)";
    }

    // 4) CUDA メモリの安全な解放
    auto safe_cuda_free = [&](void*& ptr, const char* name) {
        if (ptr) {
            cudaError_t err = cudaFree(ptr);
            if (err != cudaSuccess)
                qWarning() << name << "cudaFree failed:"
                           << cudaGetErrorString(err);
            ptr = nullptr;
        }
    };

    safe_cuda_free((void*&)d_y, "d_y");
    safe_cuda_free((void*&)d_uv, "d_uv");

    qDebug() << "decode_thread: resources released cleanly";
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
    thread_stop_flag = true;

    qDebug() << "decode_thread: stopProcessing called";
}

void decode_thread::sliderPlayback(int value){
    pausePlayback();
    slider_No=value*VideoInfo.pts_per_frame;
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

    //停止ボタン押下でシークしていない場合は停止
    if (!video_play_flag && slider_No == Get_Frame_No){
        return;
    }

    if(decode_state==STATE_DECODE_READY&&!decode_state){
        decode_state=STATE_DECODING;
        get_decode_image();
        decode_state=STATE_WAIT_DECODE_FLAG;
    }

    //デコード修了指示が出た場合は全ての処理を完了してから修了を通知
    if(thread_stop_flag){
        emit finished();
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

    VideoInfo.fps = getFrameRate(fmt_ctx, video_stream_index);

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
    VideoInfo.pts_per_frame = 1.0 / (VideoInfo.fps * time_base_d);
    qDebug() << "1フレームのPTS数:" << VideoInfo.pts_per_frame;

    //最終フレームptsを取得
    get_last_frame_pts();

    //スライダー設定
    emit send_video_info();

    //再生
    video_play_flag = true;
    video_reverse_flag = false;

    interval_ms = static_cast<double>(1000.0 / 33);
    elapsedTimer.start();
    timer->start(interval_ms);
    connect(timer, &QTimer::timeout, this, &decode_thread::processFrame);
}

// デコーダ設定
const char*decode_thread::selectDecoder(const char* codec_name) {
    const char*codec="";
    if (strcmp(codec_name, "h264") == 0) {
        codec="h264_cuvid";
        qDebug()<<codec;
    } else if (strcmp(codec_name, "hevc") == 0) {
        codec="hevc_cuvid";
        qDebug()<<codec;
    }else if (strcmp(codec_name, "av1") == 0){
        codec="av1_cuvid";
        qDebug()<<codec;
    }
    VideoInfo.Codec=codec;
    return codec;
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

void decode_thread::get_last_frame_pts() {
    avcodec_flush_buffers(codec_ctx);

    int64_t duration_pts = fmt_ctx->streams[video_stream_index]->duration;
    if (duration_pts <= 0) {
        // durationが不明な場合はファイル末尾にシーク
        duration_pts = INT64_MAX;
    }

    // 後方シーク（できるだけ終端に近づく）
    if (av_seek_frame(fmt_ctx, video_stream_index, duration_pts, AVSEEK_FLAG_BACKWARD) < 0) {
        qDebug() << "seek failed";
        return;
    }

    avcodec_flush_buffers(codec_ctx);

    int64_t last_pts = AV_NOPTS_VALUE;
    bool frame_received = false;

    while (true) {
        int ret = av_read_frame(fmt_ctx, packet);
        if (ret < 0) {
            // EOFに到達：デコーダに残りを流す
            avcodec_send_packet(codec_ctx, nullptr);
            while (avcodec_receive_frame(codec_ctx, hw_frame) == 0) {
                last_pts = hw_frame->best_effort_timestamp;
                frame_received = true;
            }
            break;
        }

        if (packet->stream_index == video_stream_index) {
            if (avcodec_send_packet(codec_ctx, packet) == 0) {
                while (avcodec_receive_frame(codec_ctx, hw_frame) == 0) {
                    last_pts = hw_frame->best_effort_timestamp;
                    frame_received = true;
                }
            }
        }
        av_packet_unref(packet);
    }

    if (frame_received) {
        AVRational tb = fmt_ctx->streams[video_stream_index]->time_base;
        double seconds = last_pts * av_q2d(tb);
        qDebug() << "Last PTS:" << last_pts << " (" << seconds << "sec)";
        VideoInfo.max_frames_pts = last_pts;
        VideoInfo.max_framesNo = fmt_ctx->streams[video_stream_index]->duration/VideoInfo.pts_per_frame-1;
    } else {
        qDebug() << "No frame found at end.";
    }
}


// フレーム取得
void decode_thread::get_decode_image() {
    //シーク処理
    if (slider_No != Get_Frame_No || video_reverse_flag == true) {
        if (video_reverse_flag == true) {
            slider_No = slider_No - VideoInfo.pts_per_frame;
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
                // EOFならループで再シークして続行
                uint64_t seek_frame{};
                if (Get_Frame_No < VideoInfo.max_frames_pts - VideoInfo.pts_per_frame) {
                    seek_frame=Get_Frame_No+VideoInfo.pts_per_frame;
                }else{
                    emit decode_end();
                    seek_frame=0;
                }
                avcodec_flush_buffers(codec_ctx);
                av_seek_frame(fmt_ctx, video_stream_index, seek_frame, AVSEEK_FLAG_ANY);
                continue;
            }
        }
    }
}

//CUDAに渡して画像処理
void decode_thread::ffmpeg_to_CUDA(){
    // QElapsedTimer timer;
    // timer.start();
    VideoInfo.width = hw_frame->width;
    VideoInfo.height = hw_frame->height;

    //初回時にのみmalloc
    if (!d_y || !d_uv) {
        cudaMallocPitch(&d_y, &pitch_y,VideoInfo.width, VideoInfo.height);
        cudaMallocPitch(&d_uv, &pitch_uv, VideoInfo.width, VideoInfo.height / 2);
    }

    cudaMemcpy2D(d_y, pitch_y,
                 hw_frame->data[0], hw_frame->linesize[0],
                 VideoInfo.width, VideoInfo.height,
                 cudaMemcpyDeviceToDevice);

    cudaMemcpy2D(d_uv, pitch_uv,
                 hw_frame->data[1], hw_frame->linesize[1],
                 VideoInfo.width, VideoInfo.height / 2,
                 cudaMemcpyDeviceToDevice);

    //フレーム番号更新
    AVRational time_base = fmt_ctx->streams[video_stream_index]->time_base;
    Get_Frame_No = hw_frame->best_effort_timestamp;
    slider_No = Get_Frame_No;
    VideoInfo.current_frameNo= Get_Frame_No/VideoInfo.pts_per_frame;

    emit send_decode_image(d_y,pitch_y,d_uv,pitch_uv,VideoInfo.current_frameNo);
    // double seconds = timer.nsecsElapsed() / 1e6; // ナノ秒 →  ミリ秒
    // qDebug()<<seconds;
}





