#ifndef NVGPUDECODE_H
#define NVGPUDECODE_H

#include "src/videoprocess/decode_thread.h"

#include <deque>
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
    double getFrameRate(AVFormatContext* fmt_ctx, int video_stream_index);
    bool get_last_frame_pts();
    void get_singledecode_image();
    void get_multidecode_image();
    void high_res_seek_frame(int targetFrameNo,bool heavy_UI_flag)override;
    void get_decode_audio();

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

    // ==========================================================
    // ★ タイル間の pts 整列（順再生のまま揃える）
    //
    //   h264_cuvid / hevc_cuvid は「ストリームごとに」出力遅延が違い、
    //   GPU 負荷でも変動する。「各ストリームから 1 枚ずつ取れた」で
    //   抜けると、タイルごとに別のフレームを掴んだまま描画してしまう。
    //
    //   一番進んでいるタイル（max）を基準に、遅れているタイルだけを
    //   前へ進めて揃える。シークはしない。
    //
    //   ★★ 遅れているタイルだけを進める、が要点。★★
    //     全ストリームへ一斉にパケットを送ると全員が同じだけ進むので、
    //     差はいつまでも縮まらない。かといって「送るだけで受け取らない」
    //     をやると *_cuvid の出力サーフェス（自前プール、既定 25 枚）を
    //     食い潰してデコードが止まる（decode_thread.h の ringSize 参照）。
    //
    //     そこで、読んだパケットは *デコーダではなく pending_ に積む*。
    //     圧縮パケットは数十〜数百 KB で、出力サーフェス（16K で数十 MB）
    //     とは桁が違うので、こちらに溜めるぶんには安全。
    //     デコーダへ入れるのは pump_stream() だけで、そこでは必ず
    //     送った直後に受け取る（入出力 1:1）。
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

    // ==========================================================
    // ★ リングのコマを壊さない受信
    //
    //   avcodec_receive_frame() は「何をするより先に必ず
    //   av_frame_unref(frame) する」と公式に明記されている。
    //   よって hw_frames[ringNo] を直接渡すと、EAGAIN で
    //   失敗しただけで取得済みのコマが消える。
    //   ここでは vd[i].tmp_frame へ受け、成功したときだけ
    //   move_ref でリングへ移す。失敗時 ring は無傷。
    //
    //   戻り値 true = フレームが 1 枚出てリングへ入った
    // ==========================================================
    bool receive_into_slot(int i, int64_t* out_pts = nullptr);

    // タイル間 pts 不一致の検出。
    //   ★ 戻り値 false = 揃っていない（＝絵が割れる）。
    //     エンコード中にこれを出すとファイルに永久に焼き付くので、
    //     publish 側はこの場合コマを出さず前フレームを再送する。
    bool check_tile_pts_mismatch();

    // 警告の連打を防ぐ
    void warn_once_in_a_while(const QString& msg);

    // ★ ストリームごとの未投入パケット
    //   デコーダに溜めずにここで持つ（上の説明を参照）
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

    // ★ pts 不一致で publish を見送った連続回数。
    //   いつまでも見送ると再生が止まるので、この回数を超えたら
    //   警告を出したうえで通す（止まるよりは出す）。
    int mismatch_skips_ = 0;
    static const int kMaxMismatchSkips = 8;

    // 累計の不一致回数（警告が間引かれても総数は分かるように）
    int mismatch_total_ = 0;

    int a=0;
};

#endif // NVGPUDECODE_H
