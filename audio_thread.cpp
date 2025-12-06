#include "audio_thread.h"
#include <QDebug>

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

audio_thread::audio_thread(QObject *parent) : QThread(parent)
{
}

audio_thread::~audio_thread()
{
    stop();
}

bool audio_thread::open(const QString &filename)
{
    inputFile = filename;

    // open and get format context
    if (avformat_open_input(&fmtCtx, filename.toUtf8().data(), nullptr, nullptr) < 0) {
        qDebug() << "Failed to open audio file";
        return false;
    }

    if (avformat_find_stream_info(fmtCtx, nullptr) < 0) {
        qDebug() << "Failed to get stream info";
        return false;
    }

    audioStream = av_find_best_stream(fmtCtx, AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
    if (audioStream < 0) {
        qDebug() << "Audio stream not found";
        return false;
    }

    AVCodecParameters *par = fmtCtx->streams[audioStream]->codecpar;
    const AVCodec *codec = avcodec_find_decoder(par->codec_id);
    if (!codec) {
        qDebug() << "Decoder not found";
        return false;
    }

    codecCtx = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(codecCtx, par);

    if (avcodec_open2(codecCtx, codec, nullptr) < 0) {
        qDebug() << "Failed to open decoder";
        return false;
    }

    // -----------------------------
    // ★ SwrContext（リサンプル）
    // -----------------------------
    // swrCtx = swr_alloc_set_opts2(
    //     nullptr,
    //     &codecCtx->ch_layout,          // 出力：同じ Layout 使用
    //     AV_SAMPLE_FMT_S16,             // 出力：Qt が扱いやすい PCM16
    //     codecCtx->sample_rate,         // 出力サンプルレート
    //     &codecCtx->ch_layout,          // 入力 Layout
    //     codecCtx->sample_fmt,          // 入力フォーマット
    //     codecCtx->sample_rate,         // 入力サンプルレート
    //     0, nullptr);

    if (!swrCtx || swr_init(swrCtx) < 0) {
        qDebug() << "swr_init failed";
        return false;
    }

    // -----------------------------
    // ★ Qt Audio Format 設定
    // -----------------------------
    QAudioFormat format;
    format.setSampleRate(codecCtx->sample_rate);
    format.setChannelCount(codecCtx->ch_layout.nb_channels);
    format.setSampleFormat(QAudioFormat::Int16);

    audioSink = new QAudioSink(format);
    audioDevice = audioSink->start();

    isRunning = true;
    start(); // thread 起動

    return true;
}

void audio_thread::stop()
{
    QMutexLocker locker(&mutex);
    isRunning = false;

    if (audioSink) {
        audioSink->stop();
        delete audioSink;
        audioSink = nullptr;
    }

    if (codecCtx) {
        avcodec_free_context(&codecCtx);
        codecCtx = nullptr;
    }

    if (fmtCtx) {
        avformat_close_input(&fmtCtx);
        fmtCtx = nullptr;
    }

    if (swrCtx) {
        swr_free(&swrCtx);
    }
}

void audio_thread::run()
{

}
