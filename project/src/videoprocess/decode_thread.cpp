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
            // ★ 受信用スクラッチも解放する
            if (vd[i].tmp_frame) {
                av_frame_free(&vd[i].tmp_frame);
                vd[i].tmp_frame = nullptr;
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

    // ★ NVDEC 転送完了イベントの破棄
    for (auto& e : decode_ready_ev) {
        if (e) { cudaEventDestroy(e); e = nullptr; }
    }
    decode_ready_ev.clear();
    dec_stream = nullptr;

    VideoInfo.video_open_flag = false;
    // ★ デコード面のメモリはここでは持たない。
    //   NVDEC経路 : vd[].hw_frames（FFmpegが解放）
    //   CPUデコード系 : 各派生クラスの YuvPlaneRing（RAIIで解放）

    qDebug() << "decode_thread: resources released cleanly";
}

QString decode_thread::ffmpegErrStr(int errnum) {
    char buf[AV_ERROR_MAX_STRING_SIZE] = {0};
    av_strerror(errnum, buf, sizeof(buf));
    return QString::fromUtf8(buf);
}

// ==============================================================
// GUIが1フレーム消費したことの通知（★デコード駆動の唯一の入口）
//
//   以前は「エンコード中は decode_view が processFrame を直結でも
//   叩く」構成だったため、1表示フレームにつき最大2回デコードが走り、
//   hw_frames リングを追い越して GUI が変換中の面を上書きしていた。
//   （= エンコード結果がたまに帯状にズレる原因）
//
//   ここで
//     ・通常再生 : fps_thread が駆動。取りこぼし時のみ再投入
//     ・エンコード中 : fps_thread を切り離し、「1枚消費 → 1枚生成」
//   に統一する。
// ==============================================================
void decode_thread::receve_decode_flag(){
    QMutexLocker locker(&mutex);

    // GUIが1枚消費した
    if (inflight_frames.load() > 0)
        --inflight_frames;

    if(decode_state==STATE_WAIT_DECODE_FLAG){
        decode_state = STATE_DECODE_READY;
    }

    // ★ エンコード中は自己駆動。ただし一時停止中は駆動しない
    //   （停止中フレームの emit で無限ループになるのを防ぐ）。
    const bool self_drive =
        (encode_state.load() == STATE_ENCODING) && video_play_flag;

    if(self_drive || drop_flag){
        drop_flag = false;
        QMetaObject::invokeMethod(this, "processFrame", Qt::QueuedConnection);
    }
}

void decode_thread::startProcessing() {
    if(initialized_ffmpeg()){
        Error_String="";
        VideoInfo.video_open_flag = true;
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
    QMutexLocker locker(&mutex);
    video_play_flag = true;
    video_reverse_flag = false;
}

void decode_thread::pausePlayback() {
    QMutexLocker locker(&mutex);
    video_play_flag = false;
}

void decode_thread::reversePlayback(){
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
