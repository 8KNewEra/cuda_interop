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

audio_thread::audio_thread(QObject *parent)
    : QThread(parent)
{
}

audio_thread::~audio_thread()
{
    stop();
}

bool audio_thread::open(const QString &filename)
{
    inputFile = filename;

    if (avformat_open_input(&fmtCtx, filename.toUtf8().data(), nullptr, nullptr) < 0)
        return false;

    if (avformat_find_stream_info(fmtCtx, nullptr) < 0)
        return false;

    audioStream = av_find_best_stream(fmtCtx, AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
    if (audioStream < 0)
        return false;

    AVCodecParameters *par = fmtCtx->streams[audioStream]->codecpar;
    const AVCodec *codec = avcodec_find_decoder(par->codec_id);

    codecCtx = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(codecCtx, par);

    if (avcodec_open2(codecCtx, codec, nullptr) < 0)
        return false;

    // ★ 出力レイアウトは必ず 2ch（stereo）
    AVChannelLayout out_layout;
    av_channel_layout_default(&out_layout, 2);

    // ★ swr セットアップ（S16 / stereo）
    if (swr_alloc_set_opts2(
            &swrCtx,
            &out_layout,
            AV_SAMPLE_FMT_S16,
            codecCtx->sample_rate,
            &codecCtx->ch_layout,
            codecCtx->sample_fmt,
            codecCtx->sample_rate,
            0, nullptr) < 0)
        return false;

    if (swr_init(swrCtx) < 0)
        return false;

    // ★ Qt AudioSink フォーマット
    QAudioFormat fmt;
    fmt.setSampleRate(codecCtx->sample_rate);
    fmt.setChannelCount(2);
    fmt.setSampleFormat(QAudioFormat::Int16);

    audioSink = new QAudioSink(fmt);
    audioDevice = audioSink->start();

    isRunning = true;
    start();

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

    if (codecCtx)
        avcodec_free_context(&codecCtx);

    if (fmtCtx)
        avformat_close_input(&fmtCtx);

    if (swrCtx)
        swr_free(&swrCtx);

    if (buffer)
        free(buffer), buffer = nullptr;
}

void audio_thread::run()
{
    AVPacket *pkt = av_packet_alloc();
    AVFrame  *frm = av_frame_alloc();

    while (true)
    {
        {
            QMutexLocker locker(&mutex);
            if (!isRunning) break;
        }

        if (av_read_frame(fmtCtx, pkt) < 0)
            continue;

        if (pkt->stream_index != audioStream) {
            av_packet_unref(pkt);
            continue;
        }

        if (avcodec_send_packet(codecCtx, pkt) < 0) {
            av_packet_unref(pkt);
            continue;
        }

        while (avcodec_receive_frame(codecCtx, frm) == 0)
        {
            int out_samples = swr_get_out_samples(swrCtx, frm->nb_samples);

            int required = out_samples * 2 * av_get_bytes_per_sample(AV_SAMPLE_FMT_S16);
            if (bufferSize < required) {
                buffer = (uint8_t*)realloc(buffer, required);
                bufferSize = required;
            }

            uint8_t *out[] = { buffer };

            int samples = swr_convert(
                swrCtx,
                out,
                out_samples,
                (const uint8_t**)frm->extended_data,
                frm->nb_samples
                );

            int bytes = samples * 2 * av_get_bytes_per_sample(AV_SAMPLE_FMT_S16);

            if (audioDevice && bytes > 0)
                audioDevice->write((char*)buffer, bytes);
        }

        av_packet_unref(pkt);
    }

    av_packet_free(&pkt);
    av_frame_free(&frm);
}
