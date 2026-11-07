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
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersrc.h>
#include <libavfilter/buffersink.h>
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

    // ★ 再生速度。libavfilter の atempo で音程を保ったまま伸縮する。
    //     abuffer -> atempo(xN) -> aformat(flt/stereo/実レート) -> abuffersink
    //   drain = true なら、作り直す前に旧グラフの残りを吐かせて drained_ に
    //   取っておく（実効速度への追従はシークせずに切り替えるので、捨てると
    //   その数十 ms が欠ける）。シーク・初期化では false。
    bool buildFilter(double speed, bool drain);
    void freeFilter();
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
    // ---- フィルタグラフ（swr の代わり。形式変換もここで行う）----
    AVFilterGraph*   graph_    = nullptr;
    AVFilterContext* src_ctx_  = nullptr;   // abuffer
    AVFilterContext* sink_ctx_ = nullptr;   // abuffersink
    AVFrame*         filt_frame_ = nullptr;
    bool             filter_eof_ = false;   // グラフへ EOF を送った

    // ★ 出力の位置（素材の秒）はグラフの出力本数から逆算する。
    //   atempo の出力 pts は「伸縮後の時間」なので素材の位置には使えない。
    //   素材位置 = src_anchor_ + out_samples_ * speed_ / out_rate_
    double           src_anchor_  = -1.0;   // このグラフに最初に入れたフレームの位置。-1 = 未定
    qint64           out_samples_ = 0;      // このグラフが出したサンプル数

    // 速度切り替え時に旧グラフから吐かせた残り。decodeOne が先に返す。
    QByteArray       drained_;
    double           drained_end_sec_ = 0.0;
    AVChannelLayout  in_ch_{};
    AVChannelLayout  out_ch_{};
    int              stream_index_ = -1;
    int              out_rate_ = 0;   // デバイスのレート（実レート）
    double           speed_    = 1.0; // いまグラフに入れてある再生速度
    AVRational       stream_tb_{0, 1};

    // abuffer に宣言した入力形式。デコーダの出力が途中で変わったら（チャンネル
    // 数・レートの切り替わり）ここを書き換えてグラフを作り直す。
    // ★ ctx_ のフィールドを書き換えて代用しないこと（デコーダの持ち物）。
    int              src_rate_ = 0;
    AVSampleFormat   src_fmt_  = AV_SAMPLE_FMT_NONE;
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

    // ★ デコーダが出せる上限の速度（素材秒 / 実秒）。飽和を観測したときの
    //   実効速度を覚えておく。0 = 飽和を観測していない（上限は設定より上）。
    //   ユーザーが速度を切り替えた直後は、計測を待たずに min(設定, これ) で
    //   鳴らし始める。待つと切り替え直後の約 1 秒（計測 2 回ぶん）、出せない
    //   速度で鳴らして待機と再開を繰り返し、音が途切れ途切れになる。
    double capacity_ = 0.0;

    // 切り替え直後の最初の計測か（1 回で反映する）
    bool   fresh_setting_ = false;

    double initialSpeed() const;   // 切り替え直後に鳴らし始める速度

    // ★ 飽和せずに出せると確認できた速度の最大。0 = 何も確認していない。
    //   capacity_（飽和したときの上限）と合わせて「この速度は出せるか」を判断する。
    double proven_ = 0.0;

    // =========================================================================
    // ★ 測ってから鳴らす（プローブ）
    //
    //   出せるか分からない速度で再生を始めるとき（ファイルを開いた直後に
    //   既に 4 倍速になっている、確認済みより速い速度へ切り替えた）は、
    //   鳴らしてから直すと最初の計測が揃うまで待機と再開を繰り返す。
    //   最初の計測（kRateWindowMs）が終わるまで黙り、測れた速度で鳴らし始める。
    // =========================================================================
    bool   needProbe() const;
    void   startProbe();
    bool   probing_ = false;
    QElapsedTimer probe_clock_;

    // 映像が進まず測れないときの打ち切り（ms）。超えたら設定値で鳴らす。
    static const int kProbeMaxMs = 1500;

    // 計測区間（ms）。映像位置はコマ単位でしか進まないので短すぎると揺れる。
    static const int kRateWindowMs = 500;

    // 速度を変える閾値（設定速度に対する比）と刻み。細かく変えるとグラフの
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
