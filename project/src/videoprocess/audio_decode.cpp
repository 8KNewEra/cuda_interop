#include "src/videoprocess/audio_decode.h"

#include <QTimer>
#include <QStringList>
#include <QDebug>
#include <QAudioDevice>
#include <QMediaDevices>
#include <algorithm>
#include <cmath>

audio_decode::audio_decode(const QString& path,
                           const std::atomic<qint64>* video_pos_us,
                           QObject* parent)
    : QObject(parent), path_(path), video_pos_us_(video_pos_us)
{
}

audio_decode::~audio_decode()
{
    shutdown();
}

// =============================================================================
// ★ 音量（音割れの分かれ目）
//
//   S16 へ落としてから掛けると 0dBFS 超えが変換の時点で ±32767 に飽和し、
//   後から下げても潰れた波形が小さくなるだけ。float のうちに掛ければ、
//   下げた結果が ±1.0 に収まった時点で歪みが消える。
// =============================================================================
QByteArray audio_decode::apply_volume(const QByteArray& f32, float volume,
                                      bool sink_is_float)
{
    const int    n   = f32.size() / (int)sizeof(float);
    const float* src = reinterpret_cast<const float*>(f32.constData());

    if (sink_is_float) {
        QByteArray out(f32.size(), Qt::Uninitialized);
        float* dst = reinterpret_cast<float*>(out.data());
        for (int i = 0; i < n; i++) dst[i] = src[i] * volume;
        return out;
    }

    // Int16 デバイス。音量を掛けた「後」に飽和させるのが肝心。
    QByteArray out(n * (int)sizeof(int16_t), Qt::Uninitialized);
    int16_t* dst = reinterpret_cast<int16_t*>(out.data());
    for (int i = 0; i < n; i++) {
        float v = src[i] * volume * 32767.0f;
        if (v >  32767.0f) v =  32767.0f;
        if (v < -32768.0f) v = -32768.0f;
        dst[i] = (int16_t)lrintf(v);
    }
    return out;
}

// =============================================================================
// 起動 / 終了
// =============================================================================
void audio_decode::start()
{
    if (!openFile()) {
        emit finished();
        return;
    }

    timer_ = new QTimer(this);
    timer_->setTimerType(Qt::PreciseTimer);
    connect(timer_, &QTimer::timeout, this, &audio_decode::pump);
    timer_->start(kPumpMs);
}

void audio_decode::shutdown()
{
    if (timer_) { timer_->stop(); delete timer_; timer_ = nullptr; }

    if (pump_n_) {
        const double avg = (double)pump_us_ / pump_n_ / 1000.0;
        const double dec = (double)decode_us_ / pump_n_ / 1000.0;
        qDebug().noquote()
            << QString("[audio] 追従中のズレ最大 %1 ms (合わせ直し %2 ms で発動)"
                       "  枯渇 %3 回 / 位置合わせ %4 回"
                       "  pump 間隔 平均 %5 ms (最大 %6 ms, 目標 %7 ms)"
                       "  うちデコード %8 ms  (sink %9 ms)")
                   .arg(err_worst_ * 1000.0, 0, 'f', 1)
                   .arg(kResyncSec * 1000.0, 0, 'f', 0)
                   .arg((qulonglong)underrun_).arg((qulonglong)resync_)
                   .arg(avg, 0, 'f', 1)
                   .arg(pump_worst_us_ / 1000.0, 0, 'f', 1)
                   .arg(kPumpMs).arg(dec, 0, 'f', 1).arg(kBufferMs);
    }

    if (pump_n_) {
        qDebug().noquote()
            << QString("[audio] 映像の実効速度 %1 (設定 %2)  最後に鳴らしていた速度 %3")
                   .arg(rate_ema_, 0, 'f', 3).arg(setting_speed_, 0, 'f', 3)
                   .arg(speed_, 0, 'f', 3);
    }

    if (sink_) { sink_->stop(); delete sink_; sink_ = nullptr; dev_ = nullptr; }
    closeFile();
    emit finished();
}

// =============================================================================
// ★ 自前で開く（decode_thread とは別の AVFormatContext）
//   音声ストリームしか読まないので実質ただのファイルハンドル。映像デコードの
//   都合に縛られないのがこの構成の利点。
// =============================================================================
bool audio_decode::openFile()
{
    int ret = avformat_open_input(&fmt_ctx_, path_.toUtf8().constData(), nullptr, nullptr);
    if (ret < 0) { qWarning() << "[audio] avformat_open_input failed"; return false; }

    if (avformat_find_stream_info(fmt_ctx_, nullptr) < 0) {
        qWarning() << "[audio] find_stream_info failed"; return false;
    }

    for (unsigned i = 0; i < fmt_ctx_->nb_streams; i++) {
        if (fmt_ctx_->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            stream_index_ = (int)i; break;
        }
    }
    if (stream_index_ < 0) return false;          // 音声なし。鳴らすものが無い

    // ★ 音声以外は demuxer に捨てさせる。
    //   これが無いと av_read_frame() が巨大な映像パケットを読んでから捨てる
    //   （16K・212Mbps なら 1 秒の音声に 26MB）。AVDISCARD_ALL なら index から
    //   次の音声サンプルへ直接 seek するので映像の中身は読まれない。
    for (unsigned i = 0; i < fmt_ctx_->nb_streams; i++)
        fmt_ctx_->streams[i]->discard =
            ((int)i == stream_index_) ? AVDISCARD_DEFAULT : AVDISCARD_ALL;

    AVStream* st = fmt_ctx_->streams[stream_index_];
    time_base_ = av_q2d(st->time_base);
    stream_tb_ = st->time_base;

    // ★ 映像と音声で「0 秒」の位置を合わせる。
    //   どちらも各ストリームの pts をそのまま秒にした値なので、開始 pts が
    //   違うファイルでは同じ瞬間を指さない。差を控えて引く（無ければ 0）。
    {
        const AVStream* vst = nullptr;
        for (unsigned i = 0; i < fmt_ctx_->nb_streams; i++) {
            if (fmt_ctx_->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
                vst = fmt_ctx_->streams[i];
                break;
            }
        }

        const double a0 = (st->start_time != AV_NOPTS_VALUE)
                              ? st->start_time * time_base_ : 0.0;
        const double v0 = (vst && vst->start_time != AV_NOPTS_VALUE)
                              ? vst->start_time * av_q2d(vst->time_base) : 0.0;
        origin_shift_ = v0 - a0;

        if (std::fabs(origin_shift_) > 0.001) {
            qWarning().noquote()
                << QString("[audio] 映像と音声で開始位置が %1 ms ずれています（補正します）")
                       .arg(origin_shift_ * 1000.0, 0, 'f', 1);
        }
    }

    const AVCodec* dec = avcodec_find_decoder(st->codecpar->codec_id);
    if (!dec) { qWarning() << "[audio] decoder not found"; return false; }

    ctx_ = avcodec_alloc_context3(dec);
    if (!ctx_ || avcodec_parameters_to_context(ctx_, st->codecpar) < 0) return false;
    ctx_->pkt_timebase = st->time_base;
    if (avcodec_open2(ctx_, dec, nullptr) < 0) {
        qWarning() << "[audio] avcodec_open2 failed"; return false;
    }

    pkt_        = av_packet_alloc();
    frame_      = av_frame_alloc();
    filt_frame_ = av_frame_alloc();
    if (!pkt_ || !frame_ || !filt_frame_) return false;

    // ---- 出力デバイス ----
    //   float を優先する。Windows(WASAPI 共有) の内部形式が float なので
    //   変換が 1 段減り、音量も float のまま掛けられる。
    const int channels = 2;
    const int rate     = ctx_->sample_rate;

    QAudioFormat afmt;
    afmt.setSampleRate(rate);
    afmt.setChannelCount(channels);
    afmt.setSampleFormat(QAudioFormat::Float);

    const QAudioDevice adev = QMediaDevices::defaultAudioOutput();
    sink_float_ = adev.isFormatSupported(afmt);
    if (!sink_float_) {
        afmt.setSampleFormat(QAudioFormat::Int16);
        qWarning() << "[audio] float 出力が使えないので Int16 で開きます";
    }

    const int bps = sink_float_ ? 4 : 2;
    frame_bytes_  = channels * bps;
    byte_per_sec_ = rate * frame_bytes_;

    sink_ = new QAudioSink(adev, afmt);
    sink_->setBufferSize(byte_per_sec_ * kBufferMs / 1000);
    dev_  = sink_->start();

    // ---- フィルタ（出力は常にインターリーブ float。音量を掛けるため）----
    av_channel_layout_copy(&in_ch_, &ctx_->ch_layout);
    av_channel_layout_default(&out_ch_, channels);
    out_rate_ = ctx_->sample_rate;
    src_rate_ = ctx_->sample_rate;
    src_fmt_  = ctx_->sample_fmt;

    setting_speed_ = g_AppSettings.video_speed_ratio;
    if (!buildFilter(setting_speed_, false)) {
        qWarning() << "[audio] フィルタグラフを作れません"; return false;
    }

    return true;
}

// =============================================================================
// ★ 再生速度（音程を保つ）
//
//     abuffer -> atempo(xN) -> aformat=flt/stereo/実レート -> abuffersink
//
//   以前は swr に「入力レートが speed 倍だ」と申告して伸縮していたので、
//   音程も速度に比例して動いた（実効速度 0.67 倍で約 7 半音低い）。
//   atempo は時間軸だけを伸縮するので音程が変わらない。
//
//   ★ atempo 1 段の範囲は 0.5〜2.0（古い FFmpeg）。外れる速度は段を重ねる。
//   ★ 速度 1.0 では atempo を入れない（素通しが一番きれい）。
//   ★ 形式・チャンネル・レートの変換は aformat が自動で aresample を挿入して
//     行う（中身は swr）。swr を自前で持つ必要はもう無い。
//   ★ 速度・EOF 以外にも、デコーダの出力形式が途中で変わると abuffer が
//     弾く。そのときは decodeOne が作り直す。
// =============================================================================
void audio_decode::freeFilter()
{
    avfilter_graph_free(&graph_);   // 中のフィルタごと解放。src/sink は所有していない
    src_ctx_  = nullptr;
    sink_ctx_ = nullptr;
    filter_eof_ = false;
}

bool audio_decode::buildFilter(double speed, bool drain)
{
    if (!ctx_) return false;
    if (!(speed > 0.01)) speed = 1.0;

    // ---- 旧グラフの残りを吐かせる（速度だけ変えるとき）----
    //   atempo は窓の分（数十 ms）を内部に抱えている。捨てるとそこが欠ける。
    if (drain && graph_ && src_ctx_ && sink_ctx_ && !filter_eof_) {
        if (av_buffersrc_add_frame(src_ctx_, nullptr) >= 0) {
            while (av_buffersink_get_frame(sink_ctx_, filt_frame_) >= 0) {
                const int n = filt_frame_->nb_samples;
                const int bytes = n * out_ch_.nb_channels * (int)sizeof(float);
                drained_.append(reinterpret_cast<const char*>(filt_frame_->data[0]), bytes);
                out_samples_ += n;
                av_frame_unref(filt_frame_);
            }
            if (src_anchor_ >= 0.0)
                drained_end_sec_ = src_anchor_ + (double)out_samples_ * speed_ / out_rate_;
            else
                drained_end_sec_ = cur_sec_;
        }
    }

    freeFilter();
    src_anchor_  = -1.0;     // 次に入れるフレームの位置から数え直す
    out_samples_ = 0;

    graph_ = avfilter_graph_alloc();
    if (!graph_) return false;

    // ---- 入口: デコーダの出力そのまま ----
    char layout[128] = {};
    av_channel_layout_describe(&in_ch_, layout, sizeof(layout));
    const QString src_args =
        QString("time_base=%1/%2:sample_rate=%3:sample_fmt=%4:channel_layout=%5")
            .arg(stream_tb_.num).arg(stream_tb_.den)
            .arg(src_rate_)
            .arg(av_get_sample_fmt_name(src_fmt_))
            .arg(layout);

    int ret = avfilter_graph_create_filter(&src_ctx_, avfilter_get_by_name("abuffer"),
                                           "in", src_args.toUtf8().constData(),
                                           nullptr, graph_);
    if (ret < 0) {
        qWarning() << "[audio] abuffer の作成に失敗:" << src_args;
        freeFilter(); return false;
    }

    ret = avfilter_graph_create_filter(&sink_ctx_, avfilter_get_by_name("abuffersink"),
                                       "out", nullptr, nullptr, graph_);
    if (ret < 0) { qWarning() << "[audio] abuffersink の作成に失敗"; freeFilter(); return false; }

    // ---- 中身: atempo を必要な段数だけ + 出口の形式 ----
    QStringList chain;
    if (std::fabs(speed - 1.0) > 1e-6) {
        double rest = speed;
        while (rest < 0.5) { chain << "atempo=0.5"; rest /= 0.5; }
        while (rest > 2.0) { chain << "atempo=2.0"; rest /= 2.0; }
        chain << QString("atempo=%1").arg(rest, 0, 'f', 6);
    }
    char out_layout[64] = {};
    av_channel_layout_describe(&out_ch_, out_layout, sizeof(out_layout));
    chain << QString("aformat=sample_fmts=flt:sample_rates=%1:channel_layouts=%2")
                 .arg(out_rate_).arg(out_layout);
    const QByteArray desc = chain.join(',').toUtf8();

    AVFilterInOut* outputs = avfilter_inout_alloc();   // グラフ側から見た「出口へ繋ぐ口」= src
    AVFilterInOut* inputs  = avfilter_inout_alloc();   // 同「入口へ繋ぐ口」= sink
    if (!outputs || !inputs) {
        avfilter_inout_free(&outputs); avfilter_inout_free(&inputs);
        freeFilter(); return false;
    }
    outputs->name = av_strdup("in");  outputs->filter_ctx = src_ctx_;  outputs->pad_idx = 0; outputs->next = nullptr;
    inputs->name  = av_strdup("out"); inputs->filter_ctx  = sink_ctx_; inputs->pad_idx  = 0; inputs->next  = nullptr;

    ret = avfilter_graph_parse_ptr(graph_, desc.constData(), &inputs, &outputs, nullptr);
    avfilter_inout_free(&inputs);
    avfilter_inout_free(&outputs);
    if (ret < 0 || avfilter_graph_config(graph_, nullptr) < 0) {
        qWarning() << "[audio] フィルタグラフの構成に失敗:" << desc;
        freeFilter(); return false;
    }

    speed_ = speed;
    return true;
}

void audio_decode::closeFile()
{
    freeFilter();
    if (filt_frame_) av_frame_free(&filt_frame_);
    drained_.clear();
    if (ctx_)     avcodec_free_context(&ctx_);
    if (fmt_ctx_) avformat_close_input(&fmt_ctx_);
    if (pkt_)     av_packet_free(&pkt_);
    if (frame_)   av_frame_free(&frame_);
    av_channel_layout_uninit(&in_ch_);
    av_channel_layout_uninit(&out_ch_);
    stream_index_ = -1;
}

// =============================================================================
// 位置
// =============================================================================
// ★ 映像が「いま出している位置」を秒で返す。decode_thread が pts から
//   作って置いたものを読むだけ（フレーム番号から作り直さないこと）。
double audio_decode::videoSeconds() const
{
    if (!video_pos_us_) return -1.0;
    const qint64 us = video_pos_us_->load(std::memory_order_acquire);
    if (us < 0) return -1.0;
    const double v = (double)us / 1000000.0 - origin_shift_;
    return (v > 0.0) ? v : 0.0;
}

double audio_decode::queuedSeconds() const
{
    if (!sink_ || byte_per_sec_ <= 0) return 0.0;
    const qint64 queued = (qint64)sink_->bufferSize() - sink_->bytesFree();
    return (queued > 0) ? (double)queued / byte_per_sec_ : 0.0;
}

void audio_decode::flushDevice()
{
    if (!sink_) return;
    carry_.clear();          // 持ち越しも捨てる（これも鳴る）
    sink_->reset();          // 積んである PCM を破棄
    dev_ = sink_->start();   // reset で停止するので張り直す
}

void audio_decode::seekToSeconds(double sec)
{
    if (!fmt_ctx_ || stream_index_ < 0) return;
    if (sec < 0.0) sec = 0.0;

    avcodec_flush_buffers(ctx_);
    const int64_t ts = (time_base_ > 0.0) ? (int64_t)(sec / time_base_) : 0;
    av_seek_frame(fmt_ctx_, stream_index_, ts, AVSEEK_FLAG_BACKWARD);

    flushDevice();

    // ★ グラフも作り直す。atempo が抱えているシーク前の音を出させないため。
    //   速度切り替え用の残り（drained_）もシーク前の音なので捨てる。
    drained_.clear();
    buildFilter(speed_, false);

    cur_sec_   = sec;
    skip_until_ = sec;        // ★ 目標に届くまで捨てる（上の説明）
    eof_       = false;
    since_resync_.restart();
}

// =============================================================================
// ★ 1 かたまりぶんの PCM(インターリーブ float) を返す
//
//   引く順番: 速度切り替えの残り -> グラフの出力 -> デコーダ -> パケット
//   グラフは入力 1 フレームにつき 0〜数フレームを出す（atempo は窓が溜まる
//   まで出さない）ので、出口から先に引き、空なら入口へ足す。
// =============================================================================
bool audio_decode::decodeOne(QByteArray& pcm)
{
    if (!drained_.isEmpty()) {
        pcm.swap(drained_);
        drained_.clear();
        cur_sec_ = drained_end_sec_;
        return true;
    }

    if (!graph_ && !buildFilter(speed_, false)) return false;

    for (;;) {
        // ---- ① グラフの出口 ----
        int r = av_buffersink_get_frame(sink_ctx_, filt_frame_);
        if (r >= 0) {
            const int n = filt_frame_->nb_samples;
            const int bytes = n * out_ch_.nb_channels * (int)sizeof(float);
            pcm = QByteArray(reinterpret_cast<const char*>(filt_frame_->data[0]), bytes);
            av_frame_unref(filt_frame_);
            if (n <= 0) continue;

            // ★ 「このかたまりの終わり」を素材の時間で指す。
            //   出力 n 本は素材の n*speed/out_rate 秒ぶんに当たる。
            out_samples_ += n;
            if (src_anchor_ >= 0.0)
                cur_sec_ = src_anchor_ + (double)out_samples_ * speed_ / out_rate_;
            else
                cur_sec_ += (double)n * speed_ / out_rate_;
            return true;
        }
        if (r == AVERROR_EOF) return false;            // 終端まで出し切った
        if (r != AVERROR(EAGAIN)) return false;

        // ---- ② デコーダ ----
        r = avcodec_receive_frame(ctx_, frame_);
        if (r == 0) {
            if (src_anchor_ < 0.0) {
                src_anchor_ = (frame_->best_effort_timestamp != AV_NOPTS_VALUE)
                                  ? frame_->best_effort_timestamp * time_base_
                                  : cur_sec_;
                out_samples_ = 0;
            }

            // ★ 形式が途中で変わった（チャンネル数・レートの切り替わり）。
            //   abuffer は宣言と違うフレームを受け付けないので作り直す。
            if (frame_->sample_rate != src_rate_ ||
                frame_->format != src_fmt_ ||
                av_channel_layout_compare(&frame_->ch_layout, &in_ch_) != 0) {
                av_channel_layout_uninit(&in_ch_);
                av_channel_layout_copy(&in_ch_, &frame_->ch_layout);
                src_rate_ = frame_->sample_rate;
                src_fmt_  = (AVSampleFormat)frame_->format;
                const double anchor = src_anchor_;
                const qint64 done   = out_samples_;
                if (!buildFilter(speed_, true)) { av_frame_unref(frame_); return false; }
                // drained_ に残りがあれば次の呼び出しで返る。位置は数え直し。
                src_anchor_ = (frame_->best_effort_timestamp != AV_NOPTS_VALUE)
                                  ? frame_->best_effort_timestamp * time_base_
                                  : anchor + (double)done * speed_ / out_rate_;
            }

            // KEEP_REF: グラフは参照を足して受け取る。frame_ はこちらで解放する。
            if (av_buffersrc_add_frame_flags(src_ctx_, frame_, AV_BUFFERSRC_FLAG_KEEP_REF) < 0)
                qWarning() << "[audio] フィルタへの投入に失敗";
            av_frame_unref(frame_);
            continue;
        }

        if (r == AVERROR_EOF) {
            // デコーダが空になった。グラフにも終わりを伝え、残りを出させる。
            if (!filter_eof_) {
                (void)av_buffersrc_add_frame(src_ctx_, nullptr);   // 失敗しても下の get_frame が EOF を返す
                filter_eof_ = true;
                continue;
            }
            return false;
        }
        if (r != AVERROR(EAGAIN)) return false;

        // ---- ③ 次のパケットを入れる ----
        if (eof_) return false;

        const int rr = av_read_frame(fmt_ctx_, pkt_);
        if (rr < 0) {
            eof_ = true;
            avcodec_send_packet(ctx_, nullptr);     // 残りを吐かせる
            continue;
        }

        if (pkt_->stream_index == stream_index_)
            avcodec_send_packet(ctx_, pkt_);

        av_packet_unref(pkt_);
    }
}

// =============================================================================
// ★ 映像の実効速度（素材秒 / 実秒）を測って、鳴らす速度をそれに合わせる
//
//   映像の位置は fps_thread の周期ではなく「デコードできたコマ」でしか進まない。
//   デコーダが飽和すると設定速度より遅くなるので、その速さで鳴らせば
//   ズレがそもそも溜まらない。
//
//   ★ 速度を変えるときにシークしないこと。積んである音はそのまま鳴らし、
//     ここから先を新しい速度で作る（シークすると切り替えのたびに音が途切れる）。
//   ★ 伸縮は atempo なので音程は変わらない（buildFilter 参照）。
// =============================================================================
void audio_decode::resetVideoRate()
{
    rate_v0_    = -1.0;
    last_v_     = -1.0;
    rate_votes_ = 0;
    rate_clock_.invalidate();
    // rate_ema_ は残す。シークしてもデコーダの性能は変わらないので、
    // 測り直しが済むまでは前の値のほうが当たっている。
}

// 切り替え直後に鳴らし始める速度。上限が分かっていれば最初からそこに合わせる。
//   丸め方は updateVideoRate の目標速度と揃えること（揃わないと最初の計測で
//   同じ速度へ作り直しが走る）。
double audio_decode::initialSpeed() const
{
    if (capacity_ <= 0.0 || capacity_ >= setting_speed_ * (1.0 - kSpeedTolerance))
        return setting_speed_;
    const double t = std::round(capacity_ / kSpeedStep) * kSpeedStep;
    return std::max((double)kSpeedStep, std::min(t, setting_speed_));
}

// 出せるか分からない速度か。
//   飽和を見ていれば initialSpeed() で当てられる。確認済みの範囲なら設定どおり。
bool audio_decode::needProbe() const
{
    if (capacity_ > 0.0) return false;
    return setting_speed_ > proven_ * (1.0 + kSpeedTolerance);
}

void audio_decode::startProbe()
{
    probing_       = true;
    waiting_       = false;
    fresh_setting_ = false;     // プローブの結果で直接決めるので不要
    rate_ema_      = 0.0;       // 古い計測を使わない
    resetVideoRate();
    flushDevice();
    probe_clock_.restart();
    qDebug().noquote()
        << QString("[audio] 速度 %1 が出せるか分からないので、測ってから鳴らします")
               .arg(setting_speed_, 0, 'f', 2);
}

void audio_decode::updateVideoRate(double v)
{
    if (v < 0.0) return;

    // ---- 飛び（シーク・コマ送り・終端の折り返し）を跨いで測らない ----
    //   kPumpMs(10ms) の間に 0.5 秒以上進む、または戻るのは再生ではない。
    if (last_v_ >= 0.0) {
        const double dv = v - last_v_;
        if (dv < -1e-6 || dv > 0.5) {
            resetVideoRate();
            last_v_ = v;
            return;
        }
    }
    last_v_ = v;

    if (rate_v0_ < 0.0 || !rate_clock_.isValid()) {
        rate_v0_ = v;
        rate_clock_.start();
        return;
    }

    const qint64 ms = rate_clock_.elapsed();
    if (ms < kRateWindowMs) return;

    const double rate = (v - rate_v0_) * 1000.0 / (double)ms;
    rate_v0_ = v;
    rate_clock_.restart();

    // 映像が止まっていた区間（停止へ入る直前など）は使わない
    if (!(rate > 0.05)) return;

    rate_ema_ = (rate_ema_ > 0.0) ? (rate_ema_ * 0.5 + rate * 0.5) : rate;

    // ---- デコーダの上限を覚える ----
    //   設定に届いていない = 飽和している。そのときの速度が上限。
    //   届いている = 上限は設定以上。覚えていた値より速く回れたなら引き上げる
    //   （軽い場面に入った）。
    if (rate_ema_ < setting_speed_ * (1.0 - kSpeedTolerance)) {
        capacity_ = rate_ema_;
    } else {
        proven_ = std::max(proven_, rate_ema_);
        if (capacity_ > 0.0 && rate_ema_ > capacity_)
            capacity_ = rate_ema_;
    }

    // プローブ中は pump 側が結果を見て鳴らし始める
    if (probing_) return;

    // ---- 目標速度 ----
    //   設定より速くはしない（映像が設定より速く進むことは無い。あれば計測の揺れ）。
    //   ほぼ設定どおりなら設定値そのもの。外れていれば kSpeedStep に丸める。
    double target;
    if (rate_ema_ >= setting_speed_ * (1.0 - kSpeedTolerance)) {
        target = setting_speed_;
    } else {
        target = std::round(rate_ema_ / kSpeedStep) * kSpeedStep;
        // ★ std::clamp は lo > hi で未定義なので使わない（設定が極端に遅い場合）
        target = std::max((double)kSpeedStep, std::min(target, setting_speed_));
    }

    const bool fresh = fresh_setting_;
    fresh_setting_ = false;

    if (std::fabs(target - speed_) < 1e-6) { rate_votes_ = 0; return; }

    // 1 回の計測だけでは変えない（重い GOP を 1 回踏んだだけで下げない）。
    // ★ ただし切り替え直後は 1 回で反映する。initialSpeed() は前の設定での
    //   観測から推した値なので、外れていたら早く直したほうがいい。
    if (++rate_votes_ < (fresh ? 1 : 2)) return;
    rate_votes_ = 0;

    const double from = speed_;
    if (buildFilter(target, true)) {
        qDebug().noquote()
            << QString("[audio] 映像の実効速度 %1 に合わせて %2 -> %3 で鳴らします (設定 %4)")
                   .arg(rate_ema_, 0, 'f', 3).arg(from, 0, 'f', 2)
                   .arg(target, 0, 'f', 2).arg(setting_speed_, 0, 'f', 2);

        // ★ queuedSeconds() * speed_ は、積んである古い速度の音を新しい速度で
        //   素材時間に直してしまう。積み直しが済むまで合わせ直しを止める。
        since_resync_.restart();
    }
}

// =============================================================================
// ★ 本体。デバイスの空きを見て、空いているぶんだけ詰める。
// =============================================================================
void audio_decode::pump()
{
    if (!sink_ || !dev_) return;

    // ---- 診断: 実際に何 ms おきに回れているか ----
    if (pump_clock_.isValid()) {
        const qint64 us = pump_clock_.nsecsElapsed() / 1000;
        pump_us_ += us;
        if ((quint64)us > pump_worst_us_) pump_worst_us_ = (quint64)us;
        pump_n_++;
    }
    pump_clock_.restart();

    const bool play = playing_.load(std::memory_order_acquire);

    // ---- 停止へ入った瞬間に、積んである音を捨てる ----
    //   ここを通さないと sink に残った音が鳴り切ってしまい、
    //   「止めたのに音だけ遅れて止まる」になる。
    if (!play) {
        if (was_playing_) { flushDevice(); was_playing_ = false; }
        return;
    }

    // ---- 再生へ入った瞬間は、映像の位置から始める ----
    if (!was_playing_) {
        was_playing_ = true;
        waiting_     = false;
        probing_     = false;
        resetVideoRate();
        if (needProbe()) {
            startProbe();
        } else {
            const double v = videoSeconds();
            if (v >= 0.0) seekToSeconds(v);
        }
    }

    // ---- ユーザーが再生速度を変えたらグラフを作り直す ----
    //   変えた直後はキューの中身が古い速度のものなので捨てる。
    //   ★ speed_ とではなく設定の控えと比べること。speed_ は実効速度への
    //     追従で設定値から外れるので、比べると毎周ここで作り直してしまう。
    if (std::fabs(g_AppSettings.video_speed_ratio - setting_speed_) > 1e-6) {
        setting_speed_ = g_AppSettings.video_speed_ratio;
        rate_ema_      = 0.0;      // 速度が変われば実効速度も変わるので測り直す
        resetVideoRate();
        fresh_setting_ = true;

        // 出せるか分からない速度なら、測ってから鳴らす
        if (needProbe()) {
            probing_ = false;
            startProbe();
            return;
        }
        probing_ = false;

        // ★ 設定値ではなく「出せるはずの速度」から始める（capacity_ 参照）。
        //   シークはこれまでどおり行う。積んである音は古い速度のもので、
        //   残すと切り替え直後にその分だけ映像とずれたまま固定される。
        const double start = initialSpeed();
        if (std::fabs(start - setting_speed_) > 1e-6) {
            qDebug().noquote()
                << QString("[audio] 速度 %1 はデコーダの上限 %2 を超えるので %3 から始めます")
                       .arg(setting_speed_, 0, 'f', 2).arg(capacity_, 0, 'f', 3)
                       .arg(start, 0, 'f', 2);
        }
        if (buildFilter(start, false)) {
            const double at = videoSeconds();
            seekToSeconds(at >= 0.0 ? at : cur_sec_);
            return;
        }
    }

    // ---- 映像とずれていたら合わせ直す ----
    //   シーク・コマ送り・終端の折り返しは全部ここで吸収される。
    //   cur_sec_ はデコード済みの位置なので、まだ鳴っていないキューを引く。
    const double v = videoSeconds();

    // ---- 映像の実効速度に合わせる（デコーダが飽和しているとき）----
    updateVideoRate(v);

    // ---- プローブ中: 最初の計測が出るまで黙る ----
    if (probing_) {
        const bool measured = (rate_ema_ > 0.0);
        if (!measured && probe_clock_.elapsed() < kProbeMaxMs) return;

        probing_ = false;
        const double start = measured ? initialSpeed() : setting_speed_;
        qDebug().noquote()
            << (measured
                    ? QString("[audio] 映像の実効速度 %1 (設定 %2) -> %3 で鳴らし始めます")
                          .arg(rate_ema_, 0, 'f', 3).arg(setting_speed_, 0, 'f', 2)
                          .arg(start, 0, 'f', 2)
                    : QString("[audio] 映像の速度を測れなかったので設定どおり %1 で鳴らします")
                          .arg(start, 0, 'f', 2));
        if (std::fabs(start - speed_) > 1e-6) buildFilter(start, false);
        resetVideoRate();
        if (v >= 0.0) seekToSeconds(v);   // 映像の今の位置から鳴らす
        return;
    }

    const bool   settled = !since_resync_.isValid() ||
                           since_resync_.elapsed() > kResyncHoldMs;
    if (v >= 0.0 && settled) {
        // ★ queuedSeconds() は実時間。素材の時間に直すには speed を掛ける。
        const double heard = cur_sec_ - queuedSeconds() * speed_;
        const double err   = heard - v;

        // ★ 先行は巻き戻さずに待つ。
        //   映像が遅れているだけなら、後ろへシークすると今聞いた所をもう一度
        //   鳴らすことになる（デコーダ飽和時に 1 秒周期で繰り返していた原因）。
        //   書き込みを止めれば積んだぶんが鳴り切り、映像が追いついた所から
        //   続きを鳴らせる。途切れはするが、同じ音は二度鳴らない。
        //   kWaitMaxSec を超える先行はユーザーの後方シークかループの折り返しな
        //   ので、従来どおり下でシークする。
        if (!waiting_ && err > kResyncSec && err < kWaitMaxSec)
            waiting_ = true;

        if (waiting_) {
            if (err > 0.0 && err < kWaitMaxSec)
                return;                      // まだ先行している。書かない
            waiting_ = false;
            // 再開直後はデバイスが空なので、枯渇として数えないよう少し待たせる
            if (err <= 0.0) since_resync_.restart();
        }

        if (std::fabs(err) > kResyncSec) {
            resync_++;
            seekToSeconds(v);
            return;           // 捨てる作業は次の周から
        }

        // ★ 同期の指標。合わせ直しの「後」で取ること。前で取ると終端の
        //   折り返しや大ジャンプを拾って尺に近い値が出る（それらは resync_）。
        if (std::fabs(err) > err_worst_) err_worst_ = std::fabs(err);
    }

    // ★ 合わせ直した直後は、目標まで読み飛ばす間どうしてもデバイスが空になる。
    //   これは枯渇ではないので数えない（数えるとシークのたびに増える）。
    if (settled && skip_until_ < 0.0 &&
        sink_->bytesFree() >= sink_->bufferSize()) {
        underrun_++;
    }

    // ★ 書けるのはサンプル枠の倍数まで。途中で切ると左右が入れ替わる。
    const auto writable = [&](qint64 want) {
        const int f = (int)std::min<qint64>(sink_->bytesFree(), want);
        return f - (f % frame_bytes_);
    };

    // ---- 前の周の端数を先に吐く ----
    if (!carry_.isEmpty()) {
        const int n = writable(carry_.size());
        if (n > 0) { dev_->write(carry_.constData(), n); carry_.remove(0, n); }
        if (!carry_.isEmpty()) return;        // まだ空いていない
    }

    // ---- 空いているぶんだけ詰める ----
    const float volume = g_AppSettings.audio_volume / 100.0f;

    int guard = 0;
    while (sink_->bytesFree() > frame_bytes_ && guard++ < 128) {
        QByteArray pcm;
        QElapsedTimer dt; dt.start();
        const bool got = decodeOne(pcm);
        decode_us_ += dt.nsecsElapsed() / 1000;
        if (!got) break;                     // 終端。折り返しは resync が拾う
        if (pcm.isEmpty()) continue;

        // ★ シーク直後。目標より前はまだ鳴らさない（巻き戻って聞こえる）
        if (skip_until_ >= 0.0) {
            if (cur_sec_ <= skip_until_) continue;
            skip_until_ = -1.0;
        }

        const QByteArray out = apply_volume(pcm, volume, sink_float_);

        const int n = writable(out.size());
        if (n > 0) dev_->write(out.constData(), n);

        if (n < out.size()) {                 // 端数は次の周へ
            carry_ = out.mid(n);
            break;
        }
    }
}
