#include "src/videoprocess/decode_thread.h"
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

decode_thread::decode_thread(QString FilePath,QObject *parent)
    : QObject(parent), video_play_flag(true){
    QFileInfo fileInfo(FilePath);
    QString ext = QFileInfo(fileInfo).suffix().toLower();
    VideoInfo.Name = fileInfo.completeBaseName().toStdString() + "." + ext.toStdString();

    VideoInfo.Path = FilePath.toStdString();
    File_byteArray = FilePath.toUtf8();
    input_filename = File_byteArray.constData();

    if(CUDA_IMG_Proc==nullptr){
        CUDA_IMG_Proc=new CUDA_ImageProcess();
    }
}

decode_thread::~decode_thread() {
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
            for (int j = 0; j < ringSize; j++) {
                if (vd[i].hw_frames[j]) {
                    av_frame_free(&vd[i].hw_frames[j]);
                    vd[i].hw_frames[j] = nullptr;
                }
            }
        }
        if (audio_frame) {
            av_frame_free(&audio_frame);
            audio_frame = nullptr;
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

    auto safe_free_packet = [&]() {
        if (packet) {
            av_packet_free(&packet);
            packet = nullptr;
        }
    };

    try {
        safe_free_codec();
        safe_free_format();
        safe_free_frames();
        safe_free_hwctx();
        safe_free_packet();
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

    // ★ d_decode_rgba はリングで持っているのでまとめて解放する
    //   （Frame.d_decode_rgba はリング内の1面を指しているだけなので個別解放しない）
    free_decode_rgba_ring();

    if(Frame.d_encode_rgba){
        safe_cuda_free((void*&)Frame.d_encode_rgba, "d_encode_rgba");
    }

    if(d_y){
        safe_cuda_free((void*&)d_y, "d_y");
    }

    if(d_u){
        safe_cuda_free((void*&)d_u, "d_u");
    }

    if(d_v){
        safe_cuda_free((void*&)d_v, "d_v");
    }

    if(d_r){
        safe_cuda_free((void*&)d_r, "d_r");
    }

    if(d_g){
        safe_cuda_free((void*&)d_g, "d_g");
    }

    if(d_b){
        safe_cuda_free((void*&)d_b, "d_b");
    }

    if(d_yuv){
        safe_cuda_free((void*&)d_yuv, "d_yuv");
    }

    if(d_rgb){
        safe_cuda_free((void*&)d_rgb, "d_rgb");
    }

    delete CUDA_IMG_Proc;
    CUDA_IMG_Proc=nullptr;

    qDebug() << "decode_thread: resources released cleanly";
}

// ==============================================================
// ★ デコードRGBAリング
// ==============================================================

//リング確保
bool decode_thread::alloc_decode_rgba_ring(int width, int height)
{
    free_decode_rgba_ring();

    for (int i = 0; i < rgbaRingSize; i++) {
        cudaError_t err = cudaMallocPitch(
            &d_decode_rgba_ring[i],
            &decode_pitch_ring[i],
            (size_t)width * 4,
            (size_t)height
            );

        if (err != cudaSuccess) {
            Error_String = QString("cudaMallocPitch(d_decode_rgba_ring[%1]) failed: %2")
                               .arg(i)
                               .arg(QString::fromUtf8(cudaGetErrorString(err)));
            free_decode_rgba_ring();
            return false;
        }
    }

    rgbaRingNo = 0;
    select_decode_rgba_slot();

    qDebug() << "[decode_thread] d_decode_rgba ring:" << rgbaRingSize << "faces,"
             << (double)decode_pitch_ring[0] * height / (1024.0*1024.0) << "MB/face";

    return true;
}

//リング解放
void decode_thread::free_decode_rgba_ring()
{
    for (int i = 0; i < rgbaRingSize; i++) {
        if (d_decode_rgba_ring[i]) {
            cudaError_t err = cudaFree(d_decode_rgba_ring[i]);
            if (err != cudaSuccess)
                qWarning() << "d_decode_rgba_ring cudaFree failed:"
                           << cudaGetErrorString(err);
            d_decode_rgba_ring[i] = nullptr;
        }
        decode_pitch_ring[i] = 0;
    }

    Frame.d_decode_rgba = nullptr;
    Frame.decode_pitch  = 0;
    rgbaRingNo = 0;
}

//現在のslotを Frame に割り当て（★カーネル起動より前に呼ぶ）
void decode_thread::select_decode_rgba_slot()
{
    Frame.d_decode_rgba = d_decode_rgba_ring[rgbaRingNo];
    Frame.decode_pitch  = decode_pitch_ring[rgbaRingNo];
}

//次のslotへ（★emit の後に呼ぶ）
void decode_thread::advance_decode_rgba_slot()
{
    rgbaRingNo++;
    if (rgbaRingNo >= rgbaRingSize) rgbaRingNo = 0;
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

    if(drop_flag){
        QMetaObject::invokeMethod(this, "processFrame", Qt::QueuedConnection);
        drop_flag=false;
    }
}

void decode_thread::startProcessing() {
    if(initialized_ffmpeg()){
        Error_String="";
        qDebug() << "Live Thread: Start Processing";
    }else{
        emit decode_error(Error_String);
        emit finished();
    }
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

void decode_thread::slider_range_end(int value){
    if(value>VideoInfo.max_framesNo){
        value = VideoInfo.max_framesNo;
    }
    VideoInfo.end_range_framesNo = value;
}

void decode_thread::slider_range_start(int value){
    if(value<0){
        value = 0;
    }
    VideoInfo.start_range_framesNo = value;
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

//1フレーム戻し
void decode_thread::back1frame(){
    QMutexLocker locker(&mutex);
    back1frame_flag = true;
    video_play_flag = false;
    video_reverse_flag = false;
}

//1フレーム送り
void decode_thread::go1frame(){
    QMutexLocker locker(&mutex);
    go1FrameNo = Frame.FrameNo;
    go1frame_flag = true;
    video_play_flag = false;
    video_reverse_flag = false;
}

//高精度スライダー
void decode_thread::high_res_sliderPlayback(int value){
    QMutexLocker locker(&mutex);
    high_res_sliderNo = value;
    high_res_slider_flag = true;
}

//デコードループ
void decode_thread::processFrame() {
    QMutexLocker locker(&mutex);
    Frame.audio_pcm.clear();
    Frame.audio_pts.clear();

    //1フレーム戻し
    if(back1frame_flag){
        if(Frame.FrameNo-1 < VideoInfo.start_range_framesNo){
            high_res_seek_frame(Frame.FrameNo-1,true);
        }else{
            high_res_seek_frame(back1FrameNo,true);
        }
        back1frame_flag = false;
        return;
    }

    //高精度スライダー
    if(high_res_slider_flag){
        high_res_seek_frame(high_res_sliderNo,true);
        high_res_slider_flag = false;
        return;
    }

    //1フレーム送り
    if(go1frame_flag){
        if(Frame.FrameNo+1 > VideoInfo.end_range_framesNo){
            high_res_seek_frame(Frame.FrameNo+1,true);
        }else{
            get_decode_image();
        }
        go1frame_flag = false;
        return;
    }

    //停止ボタン押下でシークしていない場合は停止
    if (!video_play_flag && slider_No == Frame.FrameNo){
        if(decode_state==STATE_DECODE_READY){
            decode_state=STATE_DECODING;

            // ★ この経路は画像を書き換えないので slot は進めない
            //    （エンコード中にここを通っていないかの確認用ログ）
            if (encode_state == STATE_ENCODING)
                qWarning() << "[PAUSE EMIT during encode] FrameNo:" << Frame.FrameNo;

            emit send_decode_image(Frame,true,video_reverse_flag);
            decode_state=STATE_WAIT_DECODE_FLAG;
        }
        return;
    }

    //デコード
    if(decode_state==STATE_DECODE_READY){
        decode_state=STATE_DECODING;
        get_decode_image();
        decode_state=STATE_WAIT_DECODE_FLAG;
    }else{
        drop_flag=true;
    }

    //デコード修了指示が出た場合は全ての処理を完了してから修了を通知
    if(thread_stop_flag){
        emit finished();
    }
}
