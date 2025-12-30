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
#include <libavutil/error.h>
}

decode_thread::decode_thread(QString FilePath,bool audio_m,QObject *parent)
    : QObject(parent), video_play_flag(true), timer(new QTimer(this))  {
    QFileInfo fileInfo(FilePath);
    VideoInfo.Name = fileInfo.completeBaseName().toStdString()+".mp4";

    VideoInfo.Path = FilePath.toStdString();
    File_byteArray = FilePath.toUtf8();
    input_filename = File_byteArray.constData();

    if(CUDA_IMG_Proc==nullptr){
        CUDA_IMG_Proc=new CUDA_ImageProcess();
    }

    audio_mode=audio_m;
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
        for (int i = 0; i < vd.size(); i++) {
            if (vd[i].codec_ctx) {
                if (vd[i].codec_ctx->codec && vd[i].codec_ctx->internal)
                    avcodec_flush_buffers(vd[i].codec_ctx);
                avcodec_free_context(&vd[i].codec_ctx);
                vd[i].codec_ctx = nullptr;
            }
        }
    };

    auto safe_free_format = [&]() {
        if (fmt_ctx) {
            avformat_close_input(&fmt_ctx);
            fmt_ctx = nullptr;
        }
    };

    auto safe_free_frames = [&]() {
        for (int i = 0; i < vd.size(); i++) {
            if (vd[i].Frame) {
                av_frame_free(&vd[i].Frame);
                vd[i].Frame = nullptr;
            }
        }
    };

    auto safe_free_hwctx = [&]() {
        if (hw_device_ctx) {
            av_buffer_unref(&hw_device_ctx);
            hw_device_ctx = nullptr;
        }
    };

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

    try {
        safe_free_codec();
        safe_free_format();
        safe_free_frames();
        safe_free_hwctx();
    } catch (...) {
        qWarning() << "Exception during FFmpeg cleanup (ignored)";
    }

    if(stream){
        cudaStreamDestroy(stream);
        stream=nullptr;
    }

    if (events) {
        cudaEventDestroy(events);
        events = nullptr;
    }

    if(d_rgba){
        safe_cuda_free((void*&)d_rgba, "d_rgba");
    }

    delete CUDA_IMG_Proc;
    CUDA_IMG_Proc=nullptr;

    qDebug() << "decode_thread: resources released cleanly";
}

QString decode_thread::ffmpegErrStr(int errnum) {
    char buf[AV_ERROR_MAX_STRING_SIZE] = {0};
    av_strerror(errnum, buf, sizeof(buf));
    return QString::fromUtf8(buf);
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

    interval_ms = static_cast<double>(1000.0 / 33);
    connect(timer, &QTimer::timeout, this, &decode_thread::processFrame);
    elapsedTimer.start();
    timer->start(interval_ms);

    qDebug() << "Live Thread: Start Processing";
}

void decode_thread::stopProcessing() {
    thread_stop_flag = true;

    qDebug() << "decode_thread: stopProcessing called";
}

void decode_thread::sliderPlayback(int value){
    pausePlayback();
    slider_No=value;
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

//デコードループ
void decode_thread::processFrame() {
    QMutexLocker locker(&mutex);

    //停止ボタン押下でシークしていない場合は停止
    if (!video_play_flag && slider_No == VideoInfo.current_frameNo){
        if(decode_state==STATE_DECODE_READY){
            decode_state=STATE_DECODING;
            emit send_decode_image(nullptr,0,VideoInfo.current_frameNo);
            decode_state=STATE_WAIT_DECODE_FLAG;
        }
        return;
    }

    if(decode_state==STATE_DECODE_READY){
        decode_state=STATE_DECODING;
        get_decode_image();
        decode_state=STATE_WAIT_DECODE_FLAG;
    }

    //デコード修了指示が出た場合は全ての処理を完了してから修了を通知
    if(thread_stop_flag){
        emit finished();
    }
}
