#ifndef AUDIO_DECODE_H
#define AUDIO_DECODE_H

#include <QObject>
#include <QByteArray>
#include <QString>
#include <QElapsedTimer>
#include <atomic>
#include "qaudiosink.h"
#include "src/main/__global__.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include "libswresample/swresample.h"
}

class QTimer;

// =============================================================================
// ★ 再生専用の音声スレッド
//
//   自前の AVFormatContext を持ち、出力デバイスの空きを見て自分のペースで
//   読む。以前のように映像 1 枚ごとに取り出す形だと、生成ペースが映像の
//   フレームレートに縛られ、ファイルの交互配置が粗いと音がまとめて出てきて
//   途切れた（スマホの動画）。停止も通知で受け取れずポーリングが要った。
//
//   時計は映像（fps_thread）。自分の位置が映像からずれたら合わせ直すだけなので、
//   シーク・コマ送り・終端の折り返しに経路ごとのフックは要らない。
//
//   エンコード入力の音声は映像フレームとの対応が要るので decode_thread が作る。
//   こちらは再生専用で、エンコード中は止める。
// =============================================================================
class audio_decode : public QObject
{
    Q_OBJECT
public:
    // video_pos_us : 映像の位置（コンテナ時間・µs）。フレーム番号ではない。
    explicit audio_decode(const QString& path,
                          const std::atomic<qint64>* video_pos_us,
                          QObject* parent = nullptr);
    ~audio_decode() override;

    // 再生/停止。どのスレッドからでも呼べる（積んだ音を捨てるのは pump 側）。
    void setPlaying(bool on) { playing_.store(on, std::memory_order_release); }

    // ★ 音量。入力はインターリーブ float(±1.0)。必ず float のうちに掛ける。
    //   S16 へ落としてからでは 0dBFS 超えが変換の時点で飽和しており、
    //   音量を下げても歪みが残る。
    static QByteArray apply_volume(const QByteArray& f32, float volume,
                                   bool sink_is_float);

public slots:
    void start();        // スレッド側で開く（moveToThread 後に呼ぶこと）
    void shutdown();     // スレッド側で閉じる

signals:
    void finished();

private slots:
    void pump();

private:
    bool openFile();

    // ★ 再生速度。「入力レートが speed 倍だ」と swr に申告すると出力が
    //   1/speed 本になる = 速く鳴る。音程も上がるが、これは従来（映像 1 枚に
    //   つき音声を取り出していた頃）と同じ挙動。保ちたいなら atempo が要る。
    bool buildSwr(double speed);
    void closeFile();
    void flushDevice();          // デバイスに積んである音を捨てる
    void seekToSeconds(double sec);
    bool decodeOne(QByteArray& pcm);   // 1 パケットぶん。false = これ以上出ない

    double videoSeconds() const;   // 鳴らすべき位置（秒）
    double queuedSeconds() const;  // いまデバイスに積んである秒数

    // ★ 出力バッファの深さ（ms）。深くしても映像とのズレは増えない
    //   （位置の比較で queuedSeconds を引くため）。代償は停止・シーク時に
    //   捨てる量だけ。浅いと pump が 1 回遅れただけで枯渇する（実測 30 分、
    //   120ms で 1280 回）。
    static const int kBufferMs = 300;

    // pump の周期（ms）。バッファの深さより十分短くすること。
    static const int kPumpMs = 10;

    // ★ 映像とこれ以上ずれたら合わせ直す（秒）。バッファの深さより大きく。
    static constexpr double kResyncSec = 0.35;

    // ★ 合わせ直した直後の待ち（ms）。シークは目標の手前へ着地するので、
    //   追いつくまでは必ず「遅れている」ように見える。そこで再度合わせると
    //   同じ場所へ戻り続けて音が繰り返す。
    static const int kResyncHoldMs = 400;

    QString path_;

    // ★ 映像の位置（µs）。フレーム番号で受け取ってはいけない。番号を fps で
    //   割り戻すと pts_per_frame の丸めが毎コマ積もり、音声だけ先行する。
    //   しかもこちらから見た誤差は 0 のままなので合わせ直しも効かない。
    const std::atomic<qint64>* video_pos_us_ = nullptr;

    // 映像と音声の開始 pts の差（秒）。同じ pts 値が同じ瞬間を指さない
    // ファイルがあるので、openFile() で控えて引く。
    double origin_shift_ = 0.0;

    std::atomic<bool> playing_{false};
    bool was_playing_ = false;

    QTimer* timer_ = nullptr;

    AVFormatContext* fmt_ctx_ = nullptr;
    AVCodecContext*  ctx_     = nullptr;
    AVPacket*        pkt_     = nullptr;
    AVFrame*         frame_   = nullptr;
    SwrContext*      swr_     = nullptr;
    AVChannelLayout  in_ch_{};
    AVChannelLayout  out_ch_{};
    int              stream_index_ = -1;
    int              in_rate_  = 0;   // swr へ渡す「嘘の」入力レート = 実レート x speed
    int              out_rate_ = 0;   // デバイスのレート（実レート）
    double           speed_    = 1.0; // いま swr に入れてある再生速度
    double           time_base_ = 0.0;
    bool             eof_ = false;

    QAudioSink* sink_ = nullptr;
    QIODevice*  dev_  = nullptr;
    bool        sink_float_ = false;
    int         frame_bytes_ = 4;      // ch * bps（ここで切らないと左右が入れ替わる）
    int         byte_per_sec_ = 0;

    // デコードし終えた位置（秒）。フレームの「終わり」を指す。
    // queuedSeconds() を引くと「いま聞こえている位置」になる。
    double cur_sec_ = 0.0;

    // ★ シーク直後に読み飛ばす目標（秒）。-1 なら飛ばさない。
    //   AVSEEK_FLAG_BACKWARD は目標の手前（粗いファイルでは 1 秒近く前）へ
    //   着地する。そこから鳴らすと巻き戻って聞こえ、さらに「まだ遅れている」
    //   と判定されて同じ場所へシークし続ける。届くまで捨ててから鳴らすこと。
    double skip_until_ = -1.0;

    QElapsedTimer since_resync_;

    // =========================================================================
    // ★ 映像の実効速度への追従
    //
    //   映像主導なので、デコーダが設定どおりに回らない（60fps の素材が 40fps
    //   しか出ない）と映像の位置は実時間の 2/3 でしか進まない。音声を設定速度
    //   のまま鳴らすと毎秒 0.33 秒ずつ先行し、合わせ直しで「巻き戻って同じ所を
    //   繰り返す」になる。映像の進み方を実測し、その速度で鳴らす。
    // =========================================================================
    void   updateVideoRate(double v);   // pump ごとに呼ぶ
    void   resetVideoRate();            // シーク・再生開始で測り直す

    double setting_speed_ = 1.0;   // ユーザー設定（video_speed_ratio）の控え
    QElapsedTimer rate_clock_;     // 計測区間の開始
    double rate_v0_    = -1.0;     // 計測区間の開始時の映像位置
    double last_v_     = -1.0;     // 直前の pump で見た映像位置（飛びの検出）
    double rate_ema_   = 0.0;      // 実効速度（素材秒 / 実秒）の平滑値。0 = 未計測
    int    rate_votes_ = 0;        // 速度を変えるべき計測が何回続いたか

    // 計測区間（ms）。映像位置はコマ単位でしか進まないので短すぎると揺れる。
    static const int kRateWindowMs = 500;

    // 速度を変える閾値（設定速度に対する比）と刻み。細かく変えると swr の
    // 作り直しが続くので、刻んで揃え、2 回続いたときだけ変える。
    static constexpr double kSpeedTolerance = 0.05;
    static constexpr double kSpeedStep      = 0.05;

    // ★ 先行がこれ未満なら巻き戻さずに待つ（秒）。これ以上は意図的な後方
    //   シークかループの折り返しとみなしてシークする。
    static constexpr double kWaitMaxSec = 1.5;

    // 映像待ちで書き込みを止めているか（枯渇の計数から外す）
    bool waiting_ = false;

    // デバイスに入りきらなかった端数。捨てるとそこが穴になるので持ち越す。
    QByteArray carry_;

    // --- 診断 ---
    quint64 underrun_ = 0;   // 詰めに来た時点でデバイスが空だった回数
    quint64 resync_   = 0;   // 映像に合わせ直した回数

    // ★ 追従できている間のズレの最大値（秒）。しきい値を超えたものは
    //   合わせ直して resync_ に数えるので、ここには入らない。
    double err_worst_ = 0.0;

    // 枯渇の切り分け: pump 間隔が開く -> スレッドが回っていない /
    //                 デコードが遅い -> 読み出しが重い
    QElapsedTimer pump_clock_;
    quint64 pump_n_ = 0, pump_us_ = 0, pump_worst_us_ = 0, decode_us_ = 0;
};

#endif // AUDIO_DECODE_H
