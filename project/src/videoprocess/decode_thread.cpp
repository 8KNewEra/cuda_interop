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
#include <libavutil/mathematics.h>
}

#include <cmath>

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
// ★ 以下の3つは全派生クラス(nvgpudecode / cpudecode / avidecode)で
//   同一実装だったものを基底へ集約したもの。
//   派生固有のメンバには触れないので、ここに置いて差し支えない。
// ==============================================================

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

//テストでコード、最終フレームpts取得
bool decode_thread::get_last_frame_pts() {
    avcodec_flush_buffers(vd[0].codec_ctx);

    int64_t duration_pts = fmt_ctx->streams[vd[0].stream_index]->duration;
    if (duration_pts <= 0) {
        // durationが不明な場合はファイル末尾にシーク
        duration_pts = INT64_MAX;
    }

    // 後方シーク（できるだけ終端に近づく）
    if (av_seek_frame(fmt_ctx, vd[0].stream_index, duration_pts, AVSEEK_FLAG_BACKWARD) < 0) {
        Error_String = "seek failed";
        return false;
    }

    avcodec_flush_buffers(vd[0].codec_ctx);

    int64_t last_pts = AV_NOPTS_VALUE;
    bool frame_received = false;

    while (true) {
        int ret = av_read_frame(fmt_ctx, packet);
        if (ret < 0) {
            // EOFに到達：デコーダに残りを流す
            avcodec_send_packet(vd[0].codec_ctx, nullptr);
            while (avcodec_receive_frame(vd[0].codec_ctx, vd[0].hw_frames[0]) == 0) {
                last_pts = vd[0].hw_frames[0]->best_effort_timestamp;
                VideoInfo.width = vd[0].hw_frames[0]->width;
                VideoInfo.height = vd[0].hw_frames[0]->height;
                frame_received = true;
            }
            break;
        }

        if (packet->stream_index == vd[0].stream_index) {
            if (avcodec_send_packet(vd[0].codec_ctx, packet) == 0) {
                while (avcodec_receive_frame(vd[0].codec_ctx, vd[0].hw_frames[0]) == 0) {
                    last_pts = vd[0].hw_frames[0]->best_effort_timestamp;
                    VideoInfo.width = vd[0].hw_frames[0]->width;
                    VideoInfo.height = vd[0].hw_frames[0]->height;
                    frame_received = true;
                }
            }
        }
    }

    //packet解放
    av_packet_unref(packet);

    if (frame_received) {
        AVRational tb = fmt_ctx->streams[vd[0].stream_index]->time_base;
        double time = last_pts * av_q2d(tb);
        qDebug() << "Last PTS:" << last_pts << " (" << time << "sec)";
        VideoInfo.max_framesNo = last_pts/VideoInfo.pts_per_frame;
        VideoInfo.start_range_framesNo = 0;
        VideoInfo.end_range_framesNo = VideoInfo.max_framesNo;
        Frame.FrameNo = VideoInfo.max_framesNo;

        //最大時間を計算 時:分:秒
        VideoInfo.max_hour = time / 3600;
        VideoInfo.max_minute = (int(time) % 3600) / 60;
        VideoInfo.max_second = fmod(time, 60.0);
    } else {
        Error_String = "No frame found at end.";
        return false;
    }

    return true;
}

//オーディオ
void decode_thread::get_decode_audio()
{
    if (avcodec_send_packet(audio_ctx, packet) < 0)
        return;

    while (avcodec_receive_frame(audio_ctx, audio_frame) >= 0) {

        const int out_channels = out_ch_layout.nb_channels;
        const int bps = av_get_bytes_per_sample(VideoInfo.out_format);

        int in_rate = audio_frame->sample_rate > 0
                          ? audio_frame->sample_rate
                          : audio_ctx->sample_rate;

        int max_out_samples = av_rescale_rnd(
            swr_get_delay(swr, in_rate) + audio_frame->nb_samples,
            VideoInfo.out_sample_rate,
            in_rate,
            AV_ROUND_UP
            );


        QByteArray pcm_tmp;
        pcm_tmp.resize(max_out_samples * out_channels * bps);

        uint8_t* out_planes[] = {
            reinterpret_cast<uint8_t*>(pcm_tmp.data())
        };

        int out_samples = swr_convert(
            swr,
            out_planes,
            max_out_samples,
            (const uint8_t**)audio_frame->extended_data,
            audio_frame->nb_samples
            );

        if (out_samples <= 0)
            continue;

        pcm_tmp.resize(out_samples * out_channels * bps);
        VideoInfo.audio_channels = out_channels;

        // ★ここで確定コピーを作る
        QByteArray pcm=pcm_tmp;

        // 低遅延再生（同一スレッド）
        Frame.audio_pcm.push_back(QByteArray(pcm));
        Frame.audio_pts.push_back(audio_frame->pts);

        //低遅延音声再生
        if (encode_state == STATE_NOT_ENCODE) {
            if (g_AppSettings.audio_low_laytency_flag && audioOutput) {
                if (audioSink->bytesFree() >= pcm.size()) {

                    float volume = g_AppSettings.audio_volume / 100.0f;

                    int16_t* samples = reinterpret_cast<int16_t*>(pcm.data());
                    int sampleCount = pcm.size() / sizeof(int16_t);

                    for (int i = 0; i < sampleCount; i++) {
                        int32_t v = samples[i] * volume;

                        // クリップ防止
                        if (v > 32767) v = 32767;
                        if (v < -32768) v = -32768;

                        samples[i] = static_cast<int16_t>(v);
                    }

                    audioOutput->write(pcm);
                }
            }
        }

        // エンコードスレッドへ（別スレッド）
        emit send_audio(pcm);
    }
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
