#include "src/videoprocess/decode_thread.h"
#include "src/videoprocess/audio_decode.h"
#include "qfileinfo.h"
#include <QDebug>
#include <QThread>
#include <cmath>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
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
    // ★ 音声スレッドを先に畳む（自前の AVFormatContext しか触らないので
    //   こちらの解放順とは独立）。
    stop_audio_decode();

    // ★ デコードスレッドが最後の 1 周を抜けるのを待つ。
    //   stopProcessing() で thread_stop_flag は立っている前提。
    QThread::msleep(10);

    // CUDA同期（非同期処理完了待ち）
    cudaError_t cerr = cudaDeviceSynchronize();
    if (cerr != cudaSuccess) {
        qWarning() << "cudaDeviceSynchronize failed:"
                   << cudaGetErrorString(cerr);
    }

    // FFmpeg関係の安全な解放
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

    // 音声側（エンコード入力用）。再生の QAudioSink は audio_decode が持つ。
    auto safe_free_audio = [&]() {
        if (swr)       swr_free(&swr);
        if (audio_ctx) avcodec_free_context(&audio_ctx);
        av_channel_layout_uninit(&in_ch_layout);
        av_channel_layout_uninit(&out_ch_layout);
        audio_stream_index = -1;
    };

    try {
        safe_free_codec();
        safe_free_format();
        safe_free_frames();
        safe_free_hwctx();
        safe_free_packet();
        safe_free_audio();
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

// =============================================================================
// ★ フレーム間隔を実測で補正する
//
//   MP4/MOV に fps の宣言欄は無く、avg_frame_rate は「実フレーム数 ÷ 尺」で
//   出る。コマ落ちの多いファイルでは下振れし、pts_per_frame が何倍にもなって
//   番号もシーク先の pts も全部ずれる（実測: 60 -> 15 / 256 -> 1024）。
//   インデックスのタイムスタンプの最小間隔を本来の 1 コマとみなす。
// =============================================================================
int decode_thread::detect_pts_per_frame(int stream_index)
{
    AVStream* st = fmt_ctx->streams[stream_index];

    // ★ パケットは読まずインデックスだけ見る（I/O ゼロ）。
    //   16K は 1 パケットが MB 単位なので、先頭を数百パケット読む作りだと
    //   ファイルを開くだけで数百 MB 読むことになる。
    //   MKV のインデックスはキーフレームだけなので GOP 間隔が出るが、
    //   それは fix_frame_grid() 側の条件で弾かれる。
    const int n = avformat_index_get_entries_count(st);
    if (n < 8) return 0;

    const int lim = std::min(n, 256);
    int64_t best = 0;
    int64_t prev = AV_NOPTS_VALUE;
    for (int i = 0; i < lim; i++) {
        const AVIndexEntry* e = avformat_index_get_entry(st, i);
        if (!e) break;
        if (prev != AV_NOPTS_VALUE) {
            const int64_t d = e->timestamp - prev;   // インデックスは昇順
            if (d > 0 && (best == 0 || d < best)) best = d;
        }
        prev = e->timestamp;
    }
    return (int)best;
}

void decode_thread::fix_frame_grid(int stream_index, double time_base_d)
{
    // audio_decode へ渡す位置を pts から作るのに使う（play_pos_us_ 参照）
    video_time_base_ = time_base_d;

    const int observed = detect_pts_per_frame(stream_index);

    // ★ 寄せるのは「宣言値のほうが明らかに大きい」ときだけ。
    //   MKV は time_base 1/1000 で 60fps が割り切れず（16.67 -> 16）、
    //   宣言値と実測値が 1 だけ食い違う。そこで差し替えると逆に悪化する。
    if (observed > 0 && time_base_d > 0.0 &&
        VideoInfo.pts_per_frame >= observed * 3 / 2) {

        const double fixed_fps = 1.0 / (observed * time_base_d);
        qWarning().noquote()
            << QString("[decode] フレーム間隔が宣言値と食い違うので実測値を使います"
                       " pts_per_frame %1 -> %2 / fps %3 -> %4")
                   .arg(VideoInfo.pts_per_frame).arg(observed)
                   .arg(VideoInfo.fps).arg(fixed_fps);

        VideoInfo.pts_per_frame = observed;
        VideoInfo.fps           = fixed_fps;
    }
}

// =============================================================================
// ★ デコードできたコマの pts から、表示用の番号と時刻を決める（3 派生共通）
//
//   pts_per_frame が 0、pts が AV_NOPTS_VALUE のどちらもありうる。
//   決められないときは番号を動かさない（前のコマの位置を保つ）。
//
//   back1FrameNo（1 コマ戻しの行き先）だけ経路で違う。
//     シーク直後 : 着地点の 1 つ前
//     戻し/スライダー : 呼び出し側が決めているので触らない
//     通常再生   : 直前に出したコマ（欠番があっても実在する番号になる）
// =============================================================================
void decode_thread::advance_frame_no(int64_t ts)
{
    const bool ok = (VideoInfo.pts_per_frame > 0.0 && ts != AV_NOPTS_VALUE);
    const int  no = ok ? pts_to_frame(ts) : Frame.FrameNo;

    if (seek_flag) {
        back1FrameNo = no - 1;
    } else if (!back1frame_flag.load() && !high_res_slider_flag) {
        back1FrameNo = Frame.FrameNo;
    }

    Frame.FrameNo = no;
    slider_No     = no;
    seek_flag     = false;

    // ★ audio_decode が読む。渡すのは番号ではなく pts を秒へ直したもの
    //   （no / fps で作り直すと丸めが毎コマ積もる。play_pos_us_ 参照）。
    if (ok && video_time_base_ > 0.0) {
        play_pos_us_.store((qint64)llround((double)ts * video_time_base_ * 1e6),
                           std::memory_order_release);
    }

    const double time = (VideoInfo.fps > 0.0) ? Frame.FrameNo / VideoInfo.fps : 0.0;
    Frame.hour   = time / 3600;
    Frame.minute = (int(time) % 3600) / 60;
    Frame.second = fmod(time, 60.0);
}

//ファイル末尾のフレーム番号を調べる（スライダーの終端になる）
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
        // ★ 毎周解放する。ここは最後のキーフレームから EOF まで読むので、
        //   取りこぼすとファイルを開くたびにその区間ぶん漏れる。
        av_packet_unref(packet);
    }

    av_packet_unref(packet);

    if (frame_received) {
        AVRational tb = fmt_ctx->streams[vd[0].stream_index]->time_base;
        double time = last_pts * av_q2d(tb);
        qDebug() << "Last PTS:" << last_pts << " (" << time << "sec)";
        VideoInfo.max_framesNo =
            (VideoInfo.pts_per_frame > 0.0) ? pts_to_frame(last_pts) : 0;
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

// 再生用の音声スレッド（設計理由は audio_decode.h の先頭）
void decode_thread::start_audio_decode()
{
    if (audio_decode_ || !VideoInfo.audio) return;

    audio_thread_ = new QThread;
    audio_decode_ = new audio_decode(QString::fromStdString(VideoInfo.Path),
                                     &play_pos_us_);
    audio_decode_->moveToThread(audio_thread_);

    connect(audio_thread_, &QThread::started, audio_decode_, &audio_decode::start);
    connect(audio_decode_, &audio_decode::finished, audio_thread_, &QThread::quit);

    // ★ 16K のデコード中に 10ms ごとの pump が遅れると、そのまま音が切れる
    audio_thread_->start(QThread::HighPriority);
    sync_audio_state();
}

void decode_thread::stop_audio_decode()
{
    if (!audio_thread_) return;

    if (audio_decode_) {
        audio_decode_->setPlaying(false);
        // ★ 所有スレッドで閉じさせる（QAudioSink はそのスレッドのもの）
        QMetaObject::invokeMethod(audio_decode_, "shutdown", Qt::BlockingQueuedConnection);
    }

    audio_thread_->quit();
    audio_thread_->wait();

    delete audio_decode_;  audio_decode_ = nullptr;
    delete audio_thread_;  audio_thread_ = nullptr;
}

// ★ 音を出してよいのは「順方向の通常再生中」だけ。
//   逆再生・コマ送り・スライダーのドラッグ中・エンコード中は止める。
//   位置の飛びは audio_decode 側が映像との差で気づいて合わせ直す。
void decode_thread::sync_audio_state()
{
    if (!audio_decode_) return;
    audio_decode_->setPlaying(video_play_flag && !video_reverse_flag &&
                              !high_res_slider_flag &&
                              encode_state.load() == STATE_NOT_ENCODE);
}

// ★ エンコード入力の音声を作る（再生は audio_decode が別に行う）。
//   Frame.audio_pcm を使うのは save_encode だけなので、エンコード中以外は
//   呼ばれても何もしない。
void decode_thread::get_decode_audio()
{
    if (encode_state.load() == STATE_NOT_ENCODE) return;

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


        QByteArray pcm;
        pcm.resize(max_out_samples * out_channels * bps);

        uint8_t* out_planes[] = {
            reinterpret_cast<uint8_t*>(pcm.data())
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

        pcm.resize(out_samples * out_channels * bps);
        VideoInfo.audio_channels = out_channels;

        // ★ エンコード入力なので音量は掛けない（掛けるとスライダーの位置が
        //   録画結果に焼き付く）。
        Frame.audio_pcm.push_back(pcm);
    }
}

// ==============================================================
// GUIが1フレーム消費したことの通知（★デコード駆動の唯一の入口）
//
//   通常再生     : fps_thread が駆動。取りこぼし時のみ再投入
//   エンコード中 : fps_thread を切り離し「1枚消費 -> 1枚生成」
//
//   ★ ここ以外から processFrame を叩かないこと。1 表示フレームに 2 回
//     デコードが走ると hw_frames リングを追い越し、GUI が変換中の面を
//     上書きする（エンコード結果が帯状にズレる）。
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
        start_audio_decode();
        qDebug() << "Live Thread: Start Processing";
    }else{
        emit decode_error(Error_String);
        emit finished();
    }
}

void decode_thread::stopProcessing() {
    thread_stop_flag = true;
    if (audio_decode_) audio_decode_->setPlaying(false);   // 先に黙らせる
    qDebug() << "decode_thread: stopProcessing called";
}

void decode_thread::sliderPlayback(int value){
    pausePlayback();
    // ★ 他のスロットと同じくロックの中で書く。slider_No はデコードスレッドが
    //   毎フレーム読む（get_*decode_image の先頭のシーク判定）。
    QMutexLocker locker(&mutex);
    slider_No=value;
    video_play_flag = false;
    video_reverse_flag = false;
    sync_audio_state();
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
    sync_audio_state();
}

void decode_thread::pausePlayback() {
    QMutexLocker locker(&mutex);
    video_play_flag = false;
    sync_audio_state();
}

void decode_thread::reversePlayback(){
    QMutexLocker locker(&mutex);
    video_play_flag = true;
    video_reverse_flag = true;
    sync_audio_state();   // 逆再生では鳴らさない
}

//1フレーム戻し
void decode_thread::back1frame(){
    QMutexLocker locker(&mutex);
    back1frame_flag = true;
    video_play_flag = false;
    video_reverse_flag = false;
    sync_audio_state();
}

//1フレーム送り
void decode_thread::go1frame(){
    QMutexLocker locker(&mutex);
    go1frame_flag = true;
    video_play_flag = false;
    video_reverse_flag = false;
    sync_audio_state();
}

//高精度スライダー
void decode_thread::high_res_sliderPlayback(int value){
    QMutexLocker locker(&mutex);
    high_res_sliderNo = value;
    high_res_slider_flag = true;
    sync_audio_state();
}

//デコードループ
void decode_thread::processFrame() {
    QMutexLocker locker(&mutex);

    // ★ エンコードの開始/終了はスロットを通らないので、ここでも見る。
    sync_audio_state();

    Frame.audio_pcm.clear();

    //1フレーム戻し
    if(back1frame_flag){
        if(Frame.FrameNo-1 < VideoInfo.start_range_framesNo){
            // ★ 先頭での 1 コマ戻しは終端へ回す（1 コマ送りと対称）。
            //   high_res_seek_frame() に範囲外の番号を渡すと、停止中は
            //   「始端で待機」に丸められてその場から動かない。あの丸めは
            //   3/10/30 秒戻しのためのもので、1 コマ戻しには合わない。
            high_res_seek_frame(VideoInfo.end_range_framesNo,true);
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
            // ★ 終端での 1 コマ送りは先頭へ回す。
            //   high_res_seek_frame() に範囲外の番号を渡すと、停止中は
            //   「終端で待機」に丸められてその場から動かない。あの丸めは
            //   3/10/30 秒送りのためのもので、1 コマ送りには合わない。
            high_res_seek_frame(VideoInfo.start_range_framesNo,true);
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
