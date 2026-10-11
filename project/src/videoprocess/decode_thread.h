#ifndef DECODE_THREAD_H
#define DECODE_THREAD_H

#include <QThread>
#include <QObject>
#include <QWaitCondition>
#include <QMutex>
#include <atomic>
#include <cuda_runtime.h>
#include <QDebug>
#include <QFile>
#include "src/main/__global__.h"
#include "qaudiosink.h"
#include "qelapsedtimer.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
#include <libavutil/hwcontext.h>
#include "libswresample/swresample.h"
}

struct VideoDecorder {
    AVCodecContext* codec_ctx = nullptr;
    int stream_index=0;
    const AVCodec* decoder;
    std::vector<AVFrame*> hw_frames;

    // ==========================================================
    // ★ 受信用スクラッチ（リングのコマを壊さないための緩衝材）
    //
    //   avcodec_receive_frame() は公式ドキュメントに
    //     "the function will always call av_frame_unref(frame)
    //      before doing anything else"
    //   と明記されている。つまり EAGAIN で失敗しても、
    //   渡した AVFrame の中身は先に消される。
    //
    //   hw_frames[ringNo] を直接渡していると、
    //   「すでにこのラウンドで取得済みのコマ」が、後続の
    //   『進めてみる』receive の失敗で無言で消滅する。
    //   NVDEC が飽和しているときほど EAGAIN を踏むので、
    //   負荷が高いときだけタイルが別フレームになる＝今回の症状。
    //
    //   そこで必ずここへ受けて、成功したときだけ
    //   av_frame_move_ref() でリングへ移す。
    //   失敗しても ring は無傷。
    // ==========================================================
    AVFrame* tmp_frame = nullptr;

    cudaStream_t st = nullptr;
    cudaEvent_t ev = nullptr;
};

class decode_thread : public QObject {
    Q_OBJECT

public:
    explicit decode_thread(QString FilePath,QObject *parent = nullptr);
    ~decode_thread() override;

    // ★ GUIスレッドが書き、デコードスレッドが読むので atomic 必須。
    //   非atomicだとループ内でキャッシュされ、状態遷移が見えない。
    std::atomic<int>  encode_state{STATE_NOT_ENCODE};
    std::atomic<bool> back1frame_flag{false};

signals:
    void send_decode_image(VideoFrame Frame,bool pause,bool reverse);
    void send_audio(QByteArray pcm);
    void send_slider_info();
    void finished();
    void decode_end();
    void decode_error(QString error);
    void drop_decode();
    void heavy_process_signal(bool flag);

public slots:
    void sliderPlayback(int value);
    void slider_range_end(int value);
    void slider_range_start(int value);
    void resumePlayback();
    void pausePlayback();
    void reversePlayback();
    void back1frame();
    void go1frame();
    void high_res_sliderPlayback(int value);
    void startProcessing();
    void stopProcessing();
    void processFrame();
    void receve_decode_flag();

protected:
    enum DecodeState {
        STATE_DECODE_READY,
        STATE_DECODING,
        STATE_WAIT_DECODE_FLAG
    };

    //デコード周り
    virtual bool initialized_ffmpeg()=0;
    virtual void get_decode_image()=0;
    virtual void high_res_seek_frame(int targetFrameNo,bool heavy_UI_flag)=0;
    const char* selectDecoder(const char* codec_name);
    double getFrameRate(AVFormatContext* fmt_ctx, int video_stream_index);
    bool get_last_frame_pts();
    void get_decode_audio();

    //エラー処理
    QString Error_String="";
    QString ffmpegErrStr(int errnum);

    //制御
    bool video_play_flag;
    bool video_reverse_flag;
    // ★ GUIスレッドから stopProcessing() で直接書かれるので atomic。
    //   非atomicだと while(!thread_stop_flag) がレジスタにキャッシュされ、
    //   終了要求が永久に見えずクローズ時にハングする。
    std::atomic<bool> thread_stop_flag{false};
    bool seek_flag = false;
    bool high_res_slider_flag = false;
    int slider_No;
    QMutex mutex;
    bool drop_flag=false;
    bool go1frame_flag = false;
    int back1FrameNo = 0;
    int go1FrameNo = 0;
    int high_res_sliderNo = 0;

    //FFmpeg関連
    AVPacket* packet = nullptr;
    AVFrame* audio_frame = nullptr;
    QByteArray File_byteArray;
    const char* input_filename;
    std::vector<VideoDecorder> vd;   // デフォルトコンストラクタで N 個作成
    AVFormatContext* fmt_ctx = nullptr;
    AVBufferRef* hw_device_ctx = nullptr;
    DecodeState decode_state = STATE_DECODE_READY;
    DecodeInfo& VideoInfo = DecodeInfoManager::getInstance().getSettingsNonConst();

    //CUDA
    //  ★ 画像変換はすべてGUIスレッド(glwidget)へ移ったので、
    //    デコードスレッド側は CUDA_ImageProcess を持たない。
    cudaStream_t stream=nullptr;
    cudaEvent_t events=nullptr;
    VideoFrame Frame{};

    //音声
    int audio_stream_index = -1;
    AVCodecContext* audio_ctx = nullptr;
    const AVCodec* audio_decoder = nullptr;
    SwrContext* swr = nullptr;
    uint8_t* audio_buffer = nullptr;
    int audio_buffer_size = 0;
    AVChannelLayout in_ch_layout = {};
    AVChannelLayout out_ch_layout = {};
    QAudioSink* audioSink = nullptr;
    QIODevice* audioOutput = nullptr;
    AVFrame* audio_Frame=nullptr;

    // AVFrame リング（vd[].hw_frames の面数）
    //   ★ デコード結果のRGBA面はもう持たない。
    //     NVDEC経路は hw_frames を、CPUデコード系は各派生クラスが持つ
    //     YuvPlaneRing を GUIスレッドへ直接渡す（ゼロコピー）。
    //
    //   ★ 段数は「GUIの先行許容(kMaxFramesInFlight=2)」×
    //     「1表示フレームあたりのデコード数」より大きくなければならない。
    //     ハンドシェイクを 1:1 に直したので必要なのは 3 段だが、
    //     安全マージンを見て 6 段とする。
    //
    //   ★ これ以上むやみに増やさないこと。
    //     *_cuvid はアプリが掴んだぶんだけ自前プール（surfaces、既定25枚）
    //     から出力サーフェスが減る。ここを大きくしすぎると DPB が枯渇して
    //     デコードが止まる。増やすなら cuvid 側の surfaces も一緒に上げる。
    int ringNo = 0;
    int ringSize = 6;

    // ==========================================================
    // ★ NVDEC の出力コピー完了イベント（リング各段に1つ）
    //
    //   cuvid は出力を device_hwctx->stream へ cuMemcpy2DAsync で
    //   積むだけで同期しない（cuviddec.c 参照）。そこで全タイルを
    //   受け取った後にこのイベントを「そのストリームへ」記録し、
    //   GUI側に cudaStreamWaitEvent させる。
    //
    //   ★ 記録先が ffmpeg のストリームなので、hw_device_ctx を
    //     プライマリコンテキストで作れていることが前提。
    //     作れなかった場合は dec_stream = nullptr のままにして、
    //     publish 側がホスト同期にフォールバックする。
    // ==========================================================
    std::vector<cudaEvent_t> decode_ready_ev;

    // = AVCUDADeviceContext::stream。
    //   ★ hwcontext_cuda.c は cuda_context_init で
    //       hwctx->stream = NULL;   // 既定ストリームを使う、の意味
    //     を設定する。プライマリかどうかに関係なく常に NULL。
    //     「NULL だから駄目」と判定してはいけない（前版の誤り）。
    cudaStream_t dec_stream = nullptr;

    // ★ イベント方式で NVDEC の完了を待てるか。
    //   false のときはホスト同期へ落ちるが、ffmpeg が別コンテキストに
    //   いる場合 cudaDeviceSynchronize() では *待てない* ので、
    //   その状況は初期化時に警告する。
    bool dec_sync_by_event = false;

    // ★ 追い越し検出用。
    //   emit した枚数 - GUIが消費した枚数。ringSize に達したら
    //   「GUIがまだ読んでいる面を上書きしかけている」＝バグ。
    std::atomic<int> inflight_frames{0};
};

#endif // DECODE_THREAD_H
