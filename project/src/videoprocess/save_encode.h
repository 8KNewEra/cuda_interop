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
    AVFrame* frame = nullptr;
    cudaEvent_t ready = nullptr;   // このslotのコピー完了通知

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

    // リングバッファ
    AVBufferRef*    hw_frames_ctx = nullptr;
    std::vector<FrameSlot> hw_frames;

    cudaStream_t st = nullptr;

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
    save_encode(int h,int w);
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

private:
    void initialized_ffmpeg_hardware_context(int i);
    void initialized_ffmpeg_codec_context(int i,int max_split);
    int height_,width_;

    void encode_video(VideoFrame Frame,
                      cudaTextureObject_t srcTex,
                      cudaEvent_t  glWaitEvent,
                      cudaEvent_t* convertDone);

    // --- FFmpeg 関連 ---
    std::vector<std::unique_ptr<VideoEncoder>> ve;   // デフォルトコンストラクタで N 個作成
    AVFormatContext* fmt_ctx = nullptr;          // 出力ファイルのフォーマットコンテキスト
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
