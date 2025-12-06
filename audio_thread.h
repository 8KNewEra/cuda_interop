#ifndef AUDIO_THREAD_H
#define AUDIO_THREAD_H

#include <QThread>
#include <QAudioSink>
#include <QIODevice>
#include <QMutex>
#include <QWaitCondition>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
#include <libavutil/hwcontext.h>
#include "libswresample/swresample.h"
}

class audio_thread : public QThread
{
    Q_OBJECT
public:
    explicit audio_thread(QObject *parent = nullptr);
    ~audio_thread();

    bool open(const QString &filename);
    void stop();

protected:
    void run() override;

private:
    QString inputFile;

    AVFormatContext *fmtCtx = nullptr;
    AVCodecContext  *codecCtx = nullptr;
    SwrContext      *swrCtx = nullptr;

    int audioStream = -1;

    QAudioSink  *audioSink = nullptr;
    QIODevice   *audioDevice = nullptr;

    bool isRunning = false;

    QMutex mutex;
};

#endif // AUDIO_THREAD_H
