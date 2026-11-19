#ifndef SAVE_ENCODE_H
#define SAVE_ENCODE_H

#include <QThread>
#include <QObject>
#include <cuda_runtime.h>
#include <QDebug>
#include <QFile>
#include <queue>
#include <vector>
#include <memory>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include "qmutex.h"
#include "src/imageprocess/cuda_imageprocess.h"
#include "src/main/__global__.h"
#include "src/videoprocess/xfer_codec.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/audio_fifo.h>
#include <libswscale/swscale.h>
#include <libavutil/hwcontext.h>
#include <libavutil/buffer.h>          // ★ av_buffer_get_ref_count
#include "libswresample/swresample.h"
}

struct FrameSlot {
    AVFrame* frame = nullptr;      // NVENC の入力フレーム（エンコードする GPU 上）
    cudaEvent_t ready = nullptr;   // frame が書き上がった（ワーカーがこれを待って NVENC へ渡す）

    // 変換カーネルの書き込み先（表示用 GPU 上）
    //   表示用 GPU のタイル: frame の面そのもの（encode_video が毎回入れる）
    //   別 GPU のタイル    : 転送の中継（cudaMallocPitch）。圧縮転送するタイルは使わない（nullptr）
    uint8_t* d_y = nullptr;
    size_t y_pitch = 0;
    uint8_t* d_uv = nullptr;
    size_t uv_pitch = 0;

    // ★ submit 済みで、ワーカーがまだ avcodec_send_frame を終えていない。
    //   VideoEncoder::mtx で保護する。
    bool submitted_to_worker = false;
};

// エンコーダスレッドへ渡す1フレーム分の仕事
struct EncodeJob {
    int     slot = -1;
    int64_t pts  = 0;
    bool    eos  = false;   // true なら flush して終了
};

struct VideoEncoder {
    AVBufferRef* hw_device_ctx = nullptr;        // CUDA デバイスのコンテキスト
    AVCodecContext* codec_ctx = nullptr;
    AVStream*       stream    = nullptr;

    // ★ このタイルを書き込む先。
    //   fmt     … 書き込み先（1 ファイル構成では全タイルが同じものを指す）
    //   own_fmt … このタイルが所有しているもの。別ファイル構成のときだけ
    //             非 null で、解放の責任はこちらにある。
    AVFormatContext* fmt     = nullptr;
    AVFormatContext* own_fmt = nullptr;

    // リングバッファ
    AVBufferRef*    hw_frames_ctx = nullptr;
    std::vector<FrameSlot> hw_frames;

    int          dev = -1;              // エンコードする GPU
    cudaStream_t st  = nullptr;         // 表示用 GPU 側（変換の完了待ち、転送の D2H）

    // ---- 別 GPU のタイル: メインメモリ経由の転送（save_encode::transfer_remote。docs/ENCODE_CORE.md 6）----
    //   圧縮なし: 2 段コピー（中継 d_y/d_uv → h_y/h_uv[s] → AVFrame）
    //   圧縮あり: XferCodec（変換カーネルが詰めた並びを書き、圧縮 → メインメモリの枠 s → 展開 → AVFrame）
    //   remoteXfer = false（表示用 GPU のタイル、または準備に失敗）なら transfer_remote を使わない
    bool         remoteXfer = false;
    cudaStream_t rst = nullptr;                 // 別 GPU 側（H2D / 展開）
    std::unique_ptr<XferCodec> codec;           // 圧縮ありのとき（設定 xfer_codec / xfer_delta）
    std::vector<uint8_t*>    h_y, h_uv;         // 圧縮なしの中継（ページ固定・Portable、kXferStage 枠）
    std::vector<cudaEvent_t> d2h_ev;            // 表示用 GPU: 枠 s へ書き終えた（圧縮なし）
    std::vector<cudaEvent_t> h2d_ev;            // 別 GPU: 枠 s を読み終えた（枠を再利用してよい）

    // ================= パイプライン用 =================
    std::thread             th;                 // このエンコーダ専用ワーカー
    AVPacket*               pkt = nullptr;      // ワーカー専用パケット

    std::mutex              mtx;
    std::condition_variable job_cv;             // job投入通知   (ワーカーが待つ)
    std::condition_variable slot_cv;            // slot解放通知  (メインが待つ)
    std::queue<EncodeJob>   jobs;

    // ★ 統計・ログ用カウンタ（占有判定には使わない）
    //   旧実装は occupied = submitted - completed で slot の空きを判定して
    //   いたが、これは「出力パケット順 == 入力フレーム順」「1フレーム=1
    //   パケット」を暗黙に仮定しており、B-frame などで崩れると
    //   NVENC が読んでいる最中の面に次フレームを書いてしまう。
    //   現在は FrameSlot::submitted_to_worker と AVFrame の参照カウントで
    //   判定する（wait_slot_free 参照）。
    uint64_t submitted     = 0;                 // メインが投入したフレーム数
    uint64_t completed     = 0;                 // ワーカーが回収したパケット数
    int      ring_capacity = 1;                 // = encodeRingSize
};

struct AudioJob
{
    QVector<QByteArray> audio_pcm;
};


class save_encode
{
public:
    // srcRgba16: 入力（FBO）が RGBA16 か（10bit 経路。QML 版の VideoEngine が渡す。Widgets 版は省略 = false）
    save_encode(int h,int w,bool srcRgba16 = false);
    ~save_encode();

    // ★ ゼロコピー化 + 完全非同期化
    //
    //   RGBA は linear バッファではなく、OpenGL FBO をラップした
    //   texture object から直接読む。
    //
    //   この関数は CPU をブロックせずに戻る（slot 待ちを除く）。
    //   順序は全て GPU 側のイベントで担保する。
    //
    //   glWaitEvent : 呼び出し側が map 後に interop_stream へ記録したイベント。
    //                 これを st に待たせることで「GLの描画完了 → 変換開始」を保証する。
    //   convertDone : 変換完了イベントを返す。呼び出し側はこれを
    //                 interop_stream に待たせてから unmap すること。
    //                 そうしないと変換中の FBO を GL が上書きする。
    void encode(VideoFrame Frame,
                cudaTextureObject_t srcTex,
                cudaEvent_t  glWaitEvent = nullptr,
                cudaEvent_t* convertDone = nullptr);

    // 入力（srcTex の FBO）が RGBA16 か（10bit 経路）。false なら RGBA8（8bit 経路）。コンストラクタで決まる
    bool srcRgba16 = false;
    // 10bit（P010）で書き出すか（encode_out_10bit。HEVC Main10 / AV1 10bit）。コンストラクタで決まる
    bool out10 = false;

    // 途中の失敗（ディスクの空き不足・書き込みエラー・エンコーダのエラー・転送の失敗）。どのスレッドからでも立つ。
    //   呼び出し側は encode の後に failed() を見て、立っていたら close() → delete して failMessage() を画面に出す
    bool    failed() const { return failed_; }
    QString failMessage() const;
    // 終了処理（ワーカー停止 → 吐き切り → trailer → 解放）。成功なら true。デストラクタも呼ぶ（2 回目以降は何もしない）
    bool    close();
    // 呼び出し側（VideoEngine）で起きた続けられない失敗（FBO の CUDA への受け渡しの失敗等）を記録する。以降は failed()
    void    abort(const QString& msg) { fail(msg); }

private:
    void init(int h, int w, bool srcRgba16);    // コンストラクタの本体（失敗したらコンストラクタが release_all して投げ直す）
    void release_all(bool finalize);            // 全部の解放（デストラクタ / 初期化の失敗）。途中まで確保した状態でも動く
    std::vector<QString> openedPaths_;          // avio_open したファイル（初期化に失敗したら消す）
    int  headersWritten_ = 0;                   // ヘッダを書けたファイル数（files_ の先頭から。trailer はここまで）
    bool closed_ = false;
    std::atomic<bool>  failed_{false};
    mutable std::mutex failMutex_;
    QString            failMsg_;
    void fail(const QString& msg);
    bool ck(cudaError_t e, const char* what);   // CUDA API の戻り値（失敗なら fail して false）

    void initialized_ffmpeg_hardware_context(int i);
    void initialized_ffmpeg_codec_context(int i,int max_split);
    int height_,width_;          // 書き出す大きさ（偶数に切り下げ済み）
    int texW_ = 0, texH_ = 0;    // 入力テクスチャ（FBO）の大きさ

    void encode_video(VideoFrame Frame,
                      cudaTextureObject_t srcTex,
                      cudaEvent_t  glWaitEvent,
                      cudaEvent_t* convertDone);

    // --- FFmpeg 関連 ---
    std::vector<std::unique_ptr<VideoEncoder>> ve;   // デフォルトコンストラクタで N 個作成
    // ★ 音声を書き込む先。1 ファイル構成では唯一の出力、別ファイル構成では
    //   タイル 0 のファイル。映像は各 VideoEncoder::fmt へ書く。
    AVFormatContext* fmt_ctx = nullptr;

    // ★ 解放すべきフォーマットコンテキストの一覧（1 本か、タイル数ぶん）。
    //   fmt_ctx と own_fmt は同じものを指すことがあるので、二重解放を避ける
    //   ために所有はここ 1 箇所に集める。
    std::vector<AVFormatContext*> files_;
    bool separate_ = false;

    int64_t frame_index = 0;                         // PTS 管理用

    //エンコード設定
    const EncodeSettings& encodeSettings = EncodeSettingsManager::getInstance().getSettings();
    const DecodeInfo& VideoInfo = DecodeInfoManager::getInstance().getSettings();

    //CUDA周り
    CUDA_ImageProcess* CUDA_IMG_Proc=nullptr;
    cudaStream_t st = nullptr;
    cudaEvent_t ev = nullptr;
    QMutex muxMutex;

    //音声
    void init_audio_encoder();
    void encode_audio(AudioJob Frame);
    SwrContext* swr_enc = nullptr;
    AVCodecContext* audio_enc_ctx = nullptr;
    AVStream*       audio_stream  = nullptr;
    int64_t         audio_pts     = 0;
    AVAudioFifo* audio_fifo = nullptr;

    // ★ 1 回に送るサンプル数。
    //   AAC / FLAC は codec_ctx->frame_size が決まっているのでそれに従う。
    //   PCM は frame_size を持たない（任意長を受け付ける）ので自分で決める。
    //   0 のまま FIFO の条件に使うと無限ループになるため、必ず 1 以上。
    int audio_frame_size_ = 1024;

    // ★ 実際に音声を書き出すか。素材に音声があり、かつ設定で有効なとき。
    //   encode() がキューへ積むかどうかの判定にも使う。false のまま積むと
    //   取り出すスレッドが居ないのでキューが伸び続ける。
    bool audio_on_ = false;

    //音声エンコードスレッド関連
    std::queue<AudioJob> audioQueue;
    std::mutex audioMutex;
    std::condition_variable audioCV;
    std::thread audioThread;
    std::atomic<bool> audioRunning = false;
    std::mutex audioEncMutex;
    void audio_loop();
    void stop_audio_thread();
    void audio_flush();
    AVFrame* alloc_audio_frame(int n);          // 音声フレーム（失敗したら fail して nullptr）
    void send_audio(AVFrame* f);                // エンコーダへ渡して出たパケットを書く（nullptr = 終わり）

    // ================= 映像パイプライン =================
    //   ★ b_frames_ / async_depth_ / nvenc_surfaces_ / encodeRingSize は
    //     コンストラクタ冒頭で encodeSettings.b_frames から動的に決定する。
    //     ここの初期値は「hw_frames_ctx を作る前に必ず上書きされる」前提の
    //     仮値なので、直接いじらないこと。
    //
    //   ★ 守るべき不変条件（これが崩れるとデッドロックする）
    //
    //         encodeRingSize > nvenc_surfaces_
    //
    //     NVENC は入力サーフェスを nvenc_surfaces_ 枚しか持てない。
    //     ring がそれより大きければ、こちらが詰まる前に必ず NVENC の
    //     入力プールが満杯になり、NVENC は出力を出さざるを得なくなる。
    //     逆に ring <= nvenc_surfaces_ だと、NVENC が出力を始める前に
    //     こちらが空き slot 待ちで止まる（wait_slot_free still waiting）。
    //
    //       b_frames_       = min(UI値, 7, エンコーダ上限)
    //                         ※エンコーダ上限は nvenc_max_bframes()
    //                           av1=7 / hevc=5 / h264=4
    //       async_depth_    = b_frames_ ? 1 : 4
    //       nvenc_surfaces_ = b_frames_ ? 3*(b_frames_+1) + 12
    //                                   : async_depth_ + 1 + 3
    //       encodeRingSize  = nvenc_surfaces_ + 4
    //
    //     b_frames  async  surfaces  ring  pool(=ring+4)
    //         0       4        8      12       16
    //         1       1       18      22       26
    //         2       1       21      25       29
    //         3       1       24      28       32
    //         4       1       27      31       35
    //         5       1       30      34       38
    //         7       1       36      40       44
    //
    //   ※ 16K(15360x8640) 1タイルなら 1面 ≒ 199MB。
    //     b_frames=7 では pool 44 面 ≒ 8.8GB を使う。
    //     VRAM が足りない環境では b_frames を下げること。
    int  async_depth_    = 4;                   // NVENC内部のパイプライン段数
    int  b_frames_       = 0;                   // 実効 B-frame 数（クランプ後）
    int  nvenc_surfaces_ = 8;                   // NVENC の入力サーフェス数(明示指定)
    int  encodeRingNo    = 0;
    int  encodeRingSize  = 12;                  // = nvenc_surfaces_ + 4

    // FBO は GL のテクスチャ座標の都合で上下反転しているため 1。
    // GLをバイパスして linear な RGBA を直接渡す経路を作る場合は 0 にする。
    int srcFlip = 1;

    // 1 タイルの大きさ（UV は U / V の組なので偶数要素に切り上げ。タイル幅が奇数でも最後の組を落とさない）
    struct TileSize { int w, h; size_t yBytes, uvBytes; };
    TileSize tileSize() const;

    // 行ごとの変換カーネル（encode_rows.cuh）に渡すタイル情報
    NV12Tiles encodeRowsTiles(int slot);

    // 別 GPU へのメインメモリ経由の転送（VideoEncoder::remoteXfer）。メインメモリの枠はフレームごとに順に使う
    static constexpr int kXferStage = 3;
    int  xferStage_ = 0;
    bool init_remote_xfer(VideoEncoder& e);
    void release_remote_xfer(VideoEncoder& e);
    void transfer_remote(VideoEncoder& e, FrameSlot& fs, int s);
    void transfer_peer(VideoEncoder& e, FrameSlot& fs);   // 準備に失敗したとき（デバイスをまたぐコピー。遅い）
    // 圧縮転送の計測（ログ。xferFrame_ = 今のフレームのタイルの最大、xferSum_ = その合計）
    static constexpr int kXferLogFrames = 120;
    XferCodec::Stats xferFrame_, xferSum_;
    int  xferFrames_ = 0;
    void log_xfer_stats();

    void encoder_loop(int idx);                 // エンコーダ毎ワーカー本体
    void start_encoder_threads();
    void stop_encoder_threads();

    // ★上書き防止（メインがここで待つ）。slot 単位・順序仮定なし
    void wait_slot_free(VideoEncoder& enc, int slot);
    void submit_job(VideoEncoder& enc, int slot, int64_t pts);
    void release_slot(VideoEncoder& enc);        // パケット回収を通知
    void drain_video_encoder(VideoEncoder& enc); // ワーカー内でのみ呼ぶ
};

#endif // SAVE_ENCODE_H
