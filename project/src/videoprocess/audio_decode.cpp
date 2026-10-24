#include "src/videoprocess/audio_decode.h"

#include <QTimer>
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

    pkt_   = av_packet_alloc();
    frame_ = av_frame_alloc();
    if (!pkt_ || !frame_) return false;

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

    // ---- swr（出力は常にインターリーブ float。音量を掛けるため）----
    av_channel_layout_copy(&in_ch_, &ctx_->ch_layout);
    av_channel_layout_default(&out_ch_, channels);

    if (!buildSwr(g_AppSettings.video_speed_ratio)) {
        qWarning() << "[audio] swr_init failed"; return false;
    }

    return true;
}

// =============================================================================
// ★ 再生速度
//
//   「入力レートが speed 倍だ」と swr に申告すると出力が 1/speed 本になり、
//   デバイスのレートはそのままで速く鳴る。音程も上がるが、これは従来と同じ。
//   保ちたいなら libavfilter の atempo が要る。
//
//   ★ 出すのは「設定どおりの速度」だけ。デコーダが設定どおりに回らない場合
//     （16K の 2 倍速など）は音声が先行し、合わせ直しが入る。映像主導である
//     以上これは仕様で、飽和時に崩れるのは音声の方になる。
// =============================================================================
bool audio_decode::buildSwr(double speed)
{
    if (!ctx_) return false;
    if (!(speed > 0.01)) speed = 1.0;

    if (swr_) swr_free(&swr_);

    out_rate_ = ctx_->sample_rate;
    in_rate_  = (int)llround((double)ctx_->sample_rate * speed);
    if (in_rate_ < 1) in_rate_ = ctx_->sample_rate;

    const int ret = swr_alloc_set_opts2(&swr_, &out_ch_, AV_SAMPLE_FMT_FLT, out_rate_,
                                        &in_ch_, ctx_->sample_fmt, in_rate_, 0, nullptr);
    if (ret < 0 || !swr_ || swr_init(swr_) < 0) return false;

    speed_ = speed;
    return true;
}

void audio_decode::closeFile()
{
    if (swr_)     swr_free(&swr_);
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
    cur_sec_   = sec;
    skip_until_ = sec;        // ★ 目標に届くまで捨てる（上の説明）
    eof_       = false;
    since_resync_.restart();
}

// =============================================================================
// ★ 1 パケットぶんデコードして PCM(float) を返す
// =============================================================================
bool audio_decode::decodeOne(QByteArray& pcm)
{
    for (;;) {
        // まず溜まっている出力を引く
        int r = avcodec_receive_frame(ctx_, frame_);

        if (r == 0) {
            const int ch = out_ch_.nb_channels;

            // ★ 遅い再生（speed < 1）では出力が入力より多くなる。
            //   実レートで割らず、必ず in_rate_/out_rate_ で見積もること。
            const int max_out = (int)av_rescale_rnd(
                swr_get_delay(swr_, in_rate_) + frame_->nb_samples,
                out_rate_, in_rate_, AV_ROUND_UP) + 64;

            pcm.resize(max_out * ch * (int)sizeof(float));
            uint8_t* out[1] = { reinterpret_cast<uint8_t*>(pcm.data()) };

            const int n = swr_convert(swr_, out, max_out,
                                      (const uint8_t**)frame_->extended_data,
                                      frame_->nb_samples);
            if (n <= 0) { pcm.clear(); continue; }

            pcm.resize(n * ch * (int)sizeof(float));

            // ★ 「このフレームの終わり」を素材の時間で指すこと。
            //   出力 n 本は素材の n*speed/out_rate 秒ぶんに当たる。
            const double adv = (double)n * speed_ / out_rate_;
            if (frame_->best_effort_timestamp != AV_NOPTS_VALUE)
                cur_sec_ = frame_->best_effort_timestamp * time_base_ + adv;
            else
                cur_sec_ += adv;

            return true;
        }

        if (r != AVERROR(EAGAIN)) return false;     // EOF など

        // 次のパケットを入れる
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
        const double v = videoSeconds();
        if (v >= 0.0) seekToSeconds(v);
    }

    // ---- 再生速度が変わっていたら swr を作り直す ----
    //   変えた直後はキューの中身が古い速度のものなので捨てる。
    if (std::fabs(g_AppSettings.video_speed_ratio - speed_) > 1e-6) {
        if (buildSwr(g_AppSettings.video_speed_ratio)) {
            const double at = videoSeconds();
            seekToSeconds(at >= 0.0 ? at : cur_sec_);
            return;
        }
    }

    // ---- 映像とずれていたら合わせ直す ----
    //   シーク・コマ送り・終端の折り返しは全部ここで吸収される。
    //   cur_sec_ はデコード済みの位置なので、まだ鳴っていないキューを引く。
    const double v = videoSeconds();
    const bool   settled = !since_resync_.isValid() ||
                           since_resync_.elapsed() > kResyncHoldMs;
    if (v >= 0.0 && settled) {
        // ★ queuedSeconds() は実時間。素材の時間に直すには speed を掛ける。
        const double heard = cur_sec_ - queuedSeconds() * speed_;
        const double err   = heard - v;

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
