#ifndef DECODE_THREAD_H
#define DECODE_THREAD_H

#include <QThread>
#include <QObject>
#include <QMutex>
#include <atomic>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>
#include <QDebug>
#include "src/main/__global__.h"
#include "qaudiosink.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include "libswresample/swresample.h"
}

struct VideoDecorder {
    AVCodecContext* codec_ctx = nullptr;
    int stream_index=0;
    const AVCodec* decoder;
    std::vector<AVFrame*> hw_frames;

    // ★ 受信用スクラッチ（NVDEC 経路のみ使用）。
    //   avcodec_receive_frame() は「何をするより先に必ず av_frame_unref する」
    //   仕様なので、hw_frames を直接渡すと EAGAIN で失敗しただけで
    //   取得済みのコマが消える。タイルを並べる NVDEC 経路では致命的なので、
    //   必ずここへ受けて成功時だけ move_ref する（receive_into_slot）。
    //   CPU デコード経路は 1 枚ずつ同期的に使い切るので直接受けている。
    AVFrame* tmp_frame = nullptr;
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
    double getFrameRate(AVFormatContext* fmt_ctx, int video_stream_index);
    // インデックスから 1 フレームぶんの pts を実測する。0 = 判定不能。
    int    detect_pts_per_frame(int stream_index);
    // 宣言値と実測値を突き合わせ、食い違っていれば実測値へ寄せる。
    void   fix_frame_grid(int stream_index, double time_base_d);
    bool get_last_frame_pts();
    void get_decode_audio();

    // デコードできたコマの pts から、表示用のフレーム番号と時刻を決める。
    //   3 派生クラス共通。0 除算と AV_NOPTS_VALUE もここでふさいでいる。
    void advance_frame_no(int64_t ts);

    // ★ スピーカーへ流してよい状況か。順方向の通常再生中だけ true。
    bool audio_playable() const;
    // 出力デバイスに積んである再生待ちの PCM を捨てる。
    void reset_audio_sink();

    //エラー処理
    QString Error_String="";
    QString ffmpegErrStr(int errnum);

    //制御
    // ★ GUIスレッドが書き、デコードスレッドが読む値は atomic にする。
    //   非atomicだとループ内でレジスタにキャッシュされ、状態の変化が
    //   いつまでも見えない（thread_stop_flag ではクローズ時のハングになった）。
    std::atomic<bool> video_play_flag{false};
    std::atomic<bool> video_reverse_flag{false};
    std::atomic<bool> thread_stop_flag{false};
    bool seek_flag = false;
    bool high_res_slider_flag = false;
    int slider_No;
    QMutex mutex;
    bool drop_flag=false;
    bool go1frame_flag = false;
    int back1FrameNo = 0;
    int high_res_sliderNo = 0;

    //FFmpeg関連
    AVPacket* packet = nullptr;
    AVFrame* audio_frame = nullptr;
    QByteArray File_byteArray;
    const char* input_filename;
    // タイルの本数ぶん。実体は各派生の initialized_ffmpeg() が push_back する。
    std::vector<VideoDecorder> vd;
    AVFormatContext* fmt_ctx = nullptr;
    AVBufferRef* hw_device_ctx = nullptr;
    DecodeState decode_state = STATE_DECODE_READY;
    DecodeInfo& VideoInfo = DecodeInfoManager::getInstance().getSettingsNonConst();

    //  ★ 画像変換はすべて GUI スレッド(glwidget)が行う。ここは
    //    デコード結果のポインタを載せて渡すだけ。
    VideoFrame Frame{};

    //音声
    int audio_stream_index = -1;
    AVCodecContext* audio_ctx = nullptr;
    const AVCodec* audio_decoder = nullptr;
    SwrContext* swr = nullptr;
    AVChannelLayout in_ch_layout = {};
    AVChannelLayout out_ch_layout = {};
    QAudioSink* audioSink = nullptr;
    QIODevice* audioOutput = nullptr;

    // ★ 消音中かどうか。消音へ入る最初の 1 回だけ出力デバイスを捨てる
    //   （毎回 reset するとドラッグ中ずっとデバイスを開き直すことになる）。
    bool audio_muted_ = false;

    // ★ high_res_seek_frame() の実行中だけ true。この経路は目標まで
    //   パケットを読み進めるので、通り道の音声が鳴ってしまう。
    //   デコードスレッド専用なので atomic は不要。
    bool in_seek_scan_ = false;

    // in_seek_scan_ の RAII（high_res_seek_frame() は return が多い）
    struct SeekScanScope {
        decode_thread* t;
        explicit SeekScanScope(decode_thread* p) : t(p) { t->in_seek_scan_ = true; }
        ~SeekScanScope() { t->in_seek_scan_ = false; }
        SeekScanScope(const SeekScanScope&)            = delete;
        SeekScanScope& operator=(const SeekScanScope&) = delete;
    };

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

    // ★ NVDEC の出力コピー完了イベント（リング各段に1つ）。
    //   全タイルを受け取った後に dec_stream へ記録し、GUI 側に待たせる。
    std::vector<cudaEvent_t> decode_ready_ev;

    // = AVCUDADeviceContext::stream。
    //   ★ hwcontext_cuda.c は常に NULL を入れる（= 既定ストリーム）。
    //     NULL は正常値なので、非 NULL 判定で弾かないこと。
    cudaStream_t dec_stream = nullptr;

    // イベント方式で待てるか。false ならホスト同期だが、ffmpeg が
    // 別コンテキストにいる場合はそれでも待てないので初期化時に警告する。
    bool dec_sync_by_event = false;

    // ★ 追い越し検出用。
    //   emit した枚数 - GUIが消費した枚数。ringSize に達したら
    //   「GUIがまだ読んでいる面を上書きしかけている」＝バグ。
    std::atomic<int> inflight_frames{0};
};

#endif // DECODE_THREAD_H
