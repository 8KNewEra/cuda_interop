#ifndef NVGPUDECODE_H
#define NVGPUDECODE_H

#include "src/videoprocess/decode_thread.h"
#include "src/videoprocess/xfer_codec.h"

#include <deque>
#include <map>
#include <memory>
#include <vector>

class nvgpudecode:public decode_thread
{
    Q_OBJECT
public:
    using decode_thread::decode_thread;
    ~nvgpudecode() override;

protected:
    bool initialized_ffmpeg() override;
    void get_decode_image()override;
    const char*selectDecoder(const char* codec_name);
    void get_singledecode_image();
    void get_multidecode_image();
    void high_res_seek_frame(int targetFrameNo,bool heavy_UI_flag)override;

    // ★ 旧 CUDA_RGBA_to_merge()
    //   NVDEC が出力した NV12 タイルのポインタを VideoFrame に載せて
    //   GUI スレッドへ渡すだけ。NV12→RGBA 結合は GUI スレッドが
    //   OpenGL テクスチャへ直接書き込む形で行う。
    //
    //   got : 今回のラウンドで各タイルがフレームを取れたか。
    //         1 タイルでも取れていなければ中身が null なので、
    //         新しい Frame は作らず前回の Frame を再送する。
    void publish_decode_frame(const std::vector<bool>* got = nullptr);

    void high_res_seek_frame_single(int FrameNo);
    void high_res_seek_frame_multi(int FrameNo);

    // ★ 目標フレームへ届かないまま終端に当たったときの後始末。
    //   「末尾へシークしたのに画が変わらない」を避けるため、取れている
    //   最後のコマを必ず出す。原因の切り分け用に数値もログへ出す。
    void publish_seek_fallback(int targetFrameNo);

    // ==========================================================
    // ★ タイル間の pts 整列（順再生のまま。シークはしない）
    //
    //   *_cuvid は出力遅延がストリームごとに違うので、「各ストリームから
    //   1 枚ずつ取れた」では別フレームのタイルが混ざる。一番進んだタイル
    //   を基準に、遅れているタイルだけを前へ進める。
    //
    //   ★ 読んだパケットはデコーダではなく pending_ に積むこと。
    //     デコーダに溜めると *_cuvid の出力サーフェス（既定 25 枚）を
    //     食い潰して止まる。投入は pump_stream() だけで、入出力 1:1。
    // ==========================================================
    bool align_tiles_pts(std::vector<int64_t>& pts);

    // pending_[i] からパケットを 1 本取り出してデコーダへ入れ、
    // 出たフレームを受け取る。フレームが出たら true。
    //   ★ 送ったら必ず受け取ること（溜めるとプールが枯渇する）
    bool pump_stream(int i, std::vector<bool>& got,
                     std::vector<int64_t>& pts, int& got_count);

    // ファイルからパケットを 1 本読んで pending_ へ積む（音声はその場で処理）。
    //   ★ デコーダへは入れない。
    //   戻り値 false = EOF
    bool read_into_pending();

    // 保留パケットの破棄（flush / シーク / 終了時）
    void clear_pending();

    // ★ リングのコマを壊さない受信。tmp_frame へ受け、成功時だけ
    //   move_ref でリングへ移す（失敗しても ring は無傷）。
    //   戻り値 true = フレームが 1 枚リングへ入った。
    bool receive_into_slot(int i, int64_t* out_pts = nullptr);

    // タイル間 pts 不一致の検出。false = 揃っていない（絵が割れる）。
    //   publish 側はその場合コマを出さず前フレームを再送する。
    bool check_tile_pts_mismatch();

    // 警告の連打を防ぐ
    // 警告を出してよいタイミングか。文字列を組む前にこれで弾く。
    bool warn_due();
    void warn_once_in_a_while(const QString& msg);

    // ★ 新しい絵を作れなかったとき、前の Frame をそのまま送り直す。
    //   emit しないと receve_decode_flag が返らず再生が止まるので、
    //   ハンドシェイクだけは必ず回すこと。
    void resend_last_frame();

    // ストリームごとの未投入パケット（デコーダには溜めない）
    std::vector<std::deque<AVPacket*>> pending_;

    // pending_ 1 本あたりの上限。これを超えたら揃えるのを諦める。
    //   正常時は 0〜数本しか溜まらない。
    static const int kMaxPending = 240;

    // align_tiles_pts() の空回り上限
    static const int kAlignMaxRounds = 256;

    // 各ループの空回りガード（EAGAIN で送れず受け取れもしない等）
    static const int kLoopGuard = 512;

    // 直近に警告を出したフレーム番号
    int last_warn_frame_ = -100000;

    // pts 不一致で publish を見送った連続回数。
    //   見送り続けると再生が止まるので、上限を超えたら警告して通す。
    int mismatch_skips_ = 0;
    static const int kMaxMismatchSkips = 8;

    // 累計の不一致回数
    int mismatch_total_ = 0;

    // ---- マルチ GPU デコード（docs/MULTI_GPU_DECODE.md）----
    //   別 GPU のタイルは publish_decode_frame → transfer_remote で表示用 GPU へ送ってから GUI に渡す。
    //   圧縮しない: その GPU → メインメモリ（h_y / h_uv）→ 表示用 GPU の r_y / r_uv（2 段コピー）
    //   圧縮する  : XferCodec（別 GPU ごと）。詰めたまま渡せるときは展開先を Frame に載せ、差分は GUI の変換で戻す
    //   メインメモリの受け渡しは kStage 枠を順に使い、同じ枠の再利用は前回の取り込み完了（h2d_ev_）を待つ。
    //   decode_ready は xfer_stream_ の最後（表示用 GPU のタイル local_ev_ も待たせてから）で記録する
    static constexpr int kStage = 3;
    struct RemoteGpu {                       // 圧縮しない転送の D2H（時間は性能 HUD の「転送」）
        cudaEvent_t beg[kStage] = {};
        cudaEvent_t end[kStage] = {};
        bool        used[kStage] = {};
        double      bytes[kStage] = {};
    };
    std::map<int, AVBufferRef*> remote_ctx_;   // GPU ごとの FFmpeg デバイスコンテキスト（primary_ctx）
    std::map<int, RemoteGpu>    remote_;
    cudaStream_t xfer_stream_ = nullptr;       // 表示用 GPU の取り込みと decode_ready（非ブロッキング）
    cudaEvent_t  local_ev_[kStage] = {};       // 表示用 GPU のタイル（cuvid の出力）の完了
    cudaEvent_t  h2d_ev_[kStage]   = {};       // 枠の取り込み完了
    bool         h2d_used_[kStage] = {};
    int          stage_ = 0;
    bool         remote_init_ = false;
    XferCodec::Algo xferAlgo_ = XferCodec::None;   // 開くときの設定
    bool            xferDelta_ = true;
    std::map<int, std::unique_ptr<XferCodec>> codecs_;

    AVBufferRef* remote_hw_ctx(int gpu);                    // 無ければ作る。失敗は nullptr
    bool init_remote();                                     // 初回の転送で作る
    bool packed_eligible(const AVFrame* f) const;           // 圧縮転送のタイルを詰めたまま GUI に渡せるか
    bool alloc_remote_tile(int i, AVFrame* f);              // 転送先リングと中継（要るものだけ）
    bool transfer_remote(const std::vector<int>& tiles);    // 送って decode_ready を記録
    bool transfer_compressed(const std::vector<int>& tiles, int s);
    void record_decode_ready();                             // xfer_stream_ の最後で decode_ready を記録
    void release_remote();

    // ---- 性能 HUD: デコード（待ち）の内訳（get_multidecode_image 1 回分の CPU 実時間 ms）----
    //   cuvid は avcodec_send_packet の中でパース〜出力コピーまで進むので、待ちはほぼ send に出る
    struct DecodeProf {
        double read = 0, xfer = 0;
        std::map<int, double> send, recv;   // GPU 番号 → ms
        void clear() { read = xfer = 0; send.clear(); recv.clear(); }
    };
    DecodeProf prof_;
    int tileGpu(int i) const { return (vd[i].gpu >= 0) ? vd[i].gpu : g_openglDeviceID; }
    void emit_decode_breakdown(double totalMs);
};

#endif // NVGPUDECODE_H
