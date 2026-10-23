#ifndef AUDIO_PLAYER_H
#define AUDIO_PLAYER_H

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
//   【なぜ分けたか】
//   以前は音声を decode_thread が「映像を 1 枚作るついで」に取り出していた。
//   そのため
//     ・生成ペースが映像のフレームレートに縛られる
//       -> ファイルの交互配置（インターリーブ）が粗いと音がまとめて出てきて、
//          その塊を抱えきれず途切れる。スマホの動画でよく起きた。
//     ・停止したことを通知で受け取れない
//       -> 毎周期ポーリングして気づくしかなく、止めても音が鳴り残る。
//   という症状が出ていた。どちらも「音声に独立した存在が無い」ことの派生。
//
//   ここでは自前の AVFormatContext を持ち、出力デバイスの空きを見て
//   自分のペースで読む。ファイルの偏りも映像側の詰まりも無関係になる。
//
//   【映像との関係】
//   このプレーヤーは fps_thread が映像を進める構成なので、映像が時計。
//   音声は自分の位置が映像からずれたら合わせ直す（seekToFrame）。
//   シーク・コマ送り・終端の折り返しは全部このズレとして現れるので、
//   経路ごとにフックを置く必要が無い。
//
//   【エンコードとの関係】
//   エンコード入力の音声（Frame.audio_pcm）は映像フレームと対応していないと
//   いけないので、従来どおり decode_thread が作る。こちらは再生専用で、
//   エンコード中は停止させる。用途が違うので重複ではない。
// =============================================================================
class audio_player : public QObject
{
    Q_OBJECT
public:
    explicit audio_player(const QString& path,
                          const std::atomic<int>* video_frame_no,
                          QObject* parent = nullptr);
    ~audio_player() override;

    // ★ 再生/停止。どのスレッドからでも呼べる（atomic を置くだけ）。
    //   止めた瞬間に積んである音を捨てるのは pump() 側で行う。
    void setPlaying(bool on) { playing_.store(on, std::memory_order_release); }

    // ★ 音量を掛ける。入力は常にインターリーブ float(±1.0 基準)。
    //
    //   S16 へ落としてから音量を掛けると、0dBFS を超えた素材は変換の時点で
    //   潰れていて、音量を下げても歪みが残る。必ず float のうちに掛ける。
    //   sink が float なら頭打ちもしない（下げれば歪みが消える）。
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

    // ★ 再生速度に合わせて swr を作り直す。
    //   「入力のサンプリングレートが speed 倍である」と swr に嘘をつくと、
    //   出力が 1/speed 本になる = 速く鳴る。音程も上がるが、これは
    //   decode_thread 内で鳴らしていた従来の挙動と同じ
    //   （あちらは映像 1 枚につき音声を取り出していたので、
    //     速度を上げると自然に音声も速くなっていた）。
    //   音程を保ちたい場合は libavfilter の atempo が要る。
    bool buildSwr(double speed);
    void closeFile();
    void flushDevice();          // デバイスに積んである音を捨てる
    void seekToSeconds(double sec);
    bool decodeOne(QByteArray& pcm);   // 1 パケットぶん。false = これ以上出ない

    // 鳴らすべき位置（秒）。映像のフレーム番号から引く。
    double videoSeconds() const;
    // いまデバイスに積んである秒数
    double queuedSeconds() const;

    // -------------------------------------------------------------------------
    // ★ 出力バッファの深さ（ms）
    //
    //   ★ ここを深くしても映像とのズレは増えない。
    //     位置の比較は「デコード位置 - まだ鳴っていないキュー」で行う
    //     （queuedSeconds を引く）ので、積んだぶんは差し引かれる。
    //     深さの代償は停止・シーク時に捨てる量だけで、そこは flush する。
    //
    //   浅いと pump() が 1 回遅れただけで枯渇する。実測（30 分）で
    //   120ms だと 1280 回枯渇していたので、素直に余裕を持たせる。
    // -------------------------------------------------------------------------
    static const int kBufferMs = 300;

    // pump の周期（ms）。バッファの深さより十分短くすること。
    static const int kPumpMs = 10;

    // ★ 映像の位置とこれ以上ずれたら合わせ直す（秒）。
    //   小さすぎると普通の再生中に鳴り続けるので、バッファの深さより大きく。
    static constexpr double kResyncSec = 0.35;

    // ★ 合わせ直した直後は少し待つ（ms）。
    //   シークは目標の手前へ着地するので、捨てて追いつくまでの間は
    //   必ず「遅れている」ように見える。そこで再度合わせ直すと、
    //   同じ場所へ戻り続けて音が繰り返す。
    static const int kResyncHoldMs = 400;

    QString path_;
    const std::atomic<int>* video_frame_no_ = nullptr;
    const DecodeInfo& VideoInfo = DecodeInfoManager::getInstance().getSettings();

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

    // ★ デコードし終えた位置（秒）。フレームの「終わり」を指す。
    //   queuedSeconds() を引くと「いま聞こえている位置」になる。
    double cur_sec_ = 0.0;

    // -------------------------------------------------------------------------
    // ★ シーク直後に読み飛ばす目標（秒）。-1 なら飛ばさない。
    //
    //   av_seek_frame(AVSEEK_FLAG_BACKWARD) は目標の「手前」へ着地する。
    //   音声がまとめて置かれているファイルでは 1 秒近く手前に落ちることが
    //   あり、そこから鳴らすと巻き戻って聞こえる。さらに「まだ遅れている」
    //   と判定されて再び同じ場所へシークし、延々と繰り返す。
    //   （実測でこれが「シーク時に音が繰り返される」の正体）
    //
    //   目標に届くまでデコード結果を捨ててから鳴らし始めること。
    // -------------------------------------------------------------------------
    double skip_until_ = -1.0;

    QElapsedTimer since_resync_;

    // ★ デバイスに入りきらなかったぶん。捨てるとそこが穴になるので持ち越す。
    //   pump() は空きを見てから詰めるので普通は空のままだが、
    //   パケットの切れ目と空きの境目でどうしても端数が出る。
    QByteArray carry_;

    // 診断
    quint64 underrun_ = 0;             // 詰めに来た時点でデバイスが空だった回数
    quint64 resync_   = 0;             // 映像に合わせ直した回数

    // ★ 枯渇の原因を切り分ける 2 つ。
    //   pump 間隔が開いている  -> スレッドが回してもらえていない（優先度・負荷）
    //   デコードが遅い         -> 読み出しが重い（巨大な映像パケットの読み飛ばし）
    QElapsedTimer pump_clock_;
    quint64 pump_n_ = 0, pump_us_ = 0, pump_worst_us_ = 0, decode_us_ = 0;
};

#endif // AUDIO_PLAYER_H
