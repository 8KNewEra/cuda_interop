#ifndef XFER_CODEC_H
#define XFER_CODEC_H

// GPU 間の圧縮転送（nvCOMP）。送り元 GPU（src）→ メインメモリ → 送り先 GPU（dst）
//   圧縮（src）: 前処理（詰めた並び・差分）→ nvCOMP → xfer_pack（メインメモリへカーネルが直接書く）
//   展開（dst）: xfer_fetch（見込みまで DMA、超過分はカーネル）→ nvCOMP
//                → 詰めたまま GUI へ（変換カーネルが差分を戻す）か、xfer_inv で差分を戻して転送先へ
//
//   デコードとエンコードで共用している（docs/DECODE_CORE.md 8.12）。向きで違うのは入口だけ
//     デコード（別 GPU → 表示用 GPU。1 つ = 1 枚の別 GPU のタイル全部。nvgpudecode::transfer_compressed）
//       compress（前処理は xfer_fwd）→ decompress（詰めたまま渡す ringSize > 0 か、xfer_inv で r_y / r_uv へ）
//     エンコード（表示用 GPU → 別 GPU。1 つ = 1 タイル。save_encode::transfer_remote）
//       nextTight（変換カーネルが前処理済みの値を書く）→ compressPacked → decompress（xfer_inv で NVENC の入力へ）
//   ★ 中身（init / 圧縮 / pack / fetch / 展開）を変えたら両方で動作を確かめること。
//     片方だけ調整を変えたくなったら、init の引数か入口を足して分ける
//   カーネルは cuda_kernels/xfer_compress.cu

#include <cuda_runtime.h>
#include <QString>
#include <cstdint>
#include <vector>

extern "C" {
#include <libavutil/frame.h>
}

class XferCodec
{
public:
    enum Algo { None = 0, Cascaded, Bitcomp, ANS };
    static Algo parse(const QString& s);   // "cascaded" / "bitcomp" / "ans"、それ以外は None

    // 転送先（dst 上の 1 タイル。詰めたまま渡すときは使わない）
    struct Dst { uint8_t* y; size_t yPitch; uint8_t* uv; size_t uvPitch; };

    // 1 フレーム分の計測（性能 HUD）。ms は GPU 上の経過時間
    struct Stats {
        double compressMs = 0;   // src: 前処理 + 圧縮（エンコードは前処理を変換カーネルが済ませるので圧縮だけ）
        double sendMs = 0;       // src: メインメモリへの書き出し（枠の空き待ちは除く）
        double expandMs = 0;     // dst: 取り込み〜差分戻し
        double fetchMs = -1, decompMs = -1, invMs = -1;   // その内訳（-1 = 取れない）
        double rawBytes = 0, compBytes = 0;
    };

    ~XferCodec() { release(); }

    // tileW / tileH: 受け持つタイルの大きさ（タイル順）、bps: 1 = NV12、2 = P010、nStage: メインメモリの枠数
    //   ringSize > 0 なら詰めたまま渡す（差分は GUI の変換カーネルが戻す）。展開先をリングの段数ぶん持つ
    bool init(int srcGpu, int dstGpu, Algo algo, bool delta, int bps,
              const std::vector<int>& tileW, const std::vector<int>& tileH, int nStage, int ringSize = 0);
    void release();

    // src で圧縮してメインメモリの枠 s へ（戻るときは dstGpu が current）。中身は compressImpl 1 本
    //   stageFree: 枠 s を前回読み終えたイベント（無ければ nullptr）

    // デコード用（nvgpudecode::transfer_compressed。別 GPU → 表示用 GPU）
    //   frames: cuvid の出力（data[0] = Y、data[1] = UV、タイル順）。xfer_fwd で前処理してから圧縮。
    //   既定ストリーム（cuvid の出力コピー）の後から始め、既定ストリームにも組の前回の完了を待たせる
    //   （以降の出力コピーが読み中の面を上書きしないため）
    bool compress(int s, const std::vector<const AVFrame*>& frames, cudaEvent_t stageFree);
    // エンコード用（save_encode。表示用 GPU → 別 GPU）。変換カーネル（encode_rows.cuh）が詰めた並びへ直接書く
    //   nextTight: 次の compressPacked が使う組の詰めた並び（面 plane の位置は planeOffset、1 行は planeRowBytes）。
    //     reuseDone: その組を前回使い終えたイベント（src 上。変換カーネルの前に待たせる。初回は nullptr）
    //   compressPacked: xfer_fwd を飛ばして圧縮から（差分は変換カーネルが delta() に合わせて取っておく）。
    //     inputReady: 変換カーネルの完了イベント（src 上）。この後から始める
    uint8_t* nextTight(cudaEvent_t* reuseDone) const;
    bool compressPacked(int s, cudaEvent_t inputReady, cudaEvent_t stageFree);
    size_t planeOffset(int plane) const { return planes_[plane].off; }
    bool   delta() const { return delta_; }
    cudaEvent_t compressedEvent(int s) const { return encEnd_[s]; }

    // dst のストリーム st で枠 s を展開する（st には先に compressedEvent(s) を待たせる。デコード・エンコード共通）
    //   普段は差分を戻して dst へ（デコード: 表示用 GPU のリング、エンコード: 別 GPU の NVENC 入力フレーム）。
    //   詰めたまま渡すとき（デコードだけ）は展開先リングの ring 段へ展開するだけ（packedPlane で読む）
    bool decompress(int s, int ring, cudaStream_t st, const std::vector<Dst>& dst);
    bool     packed() const { return ringSize_ > 0; }
    uint8_t* packedPlane(int ring, int plane) const { return dTight_[ring] + planes_[plane].off; }   // plane: タイル t の Y = 2t、UV = 2t+1
    size_t   planeRowBytes(int plane) const { return size_t(planes_[plane].elems) * size_t(bps_); }

    // 枠 s の前回の計測（終わっていれば true。待たない）
    bool takeStats(int s, Stats& out);

    // GPU 上で見つかった失敗（nvCOMP のチャンクの失敗・展開後の大きさの不一致・ヘッダの破損。Bitcomp はヘッダだけ）。
    //   非同期なので数フレーム遅れて分かる。エンコードは見つけたら中断、デコードはログだけ（clear = 読んだら戻す）
    //   compress / compressPacked / decompress の戻り値は、起動と CUDA API の失敗（その場で分かるもの）
    bool deviceError(bool clear = false);

private:
    static QString name(Algo a);
    struct Src { const uint8_t* y; size_t yPitch; const uint8_t* uv; size_t uvPitch; };   // 前処理の入力（デコード: cuvid の面）
    bool compressImpl(int s, const std::vector<Src>* src, cudaEvent_t inputReady, cudaEvent_t stageFree);
    struct Plane { size_t off; int elems; int rows; int lag; };   // 詰めた並びの 1 面（lag: 差分の相手。Y = 1、UV = 2）
    bool ready_ = false;
    int  src_ = -1, dst_ = -1, nStage_ = 0, bps_ = 1, ringSize_ = 0;
    Algo algo_ = None;
    bool delta_ = true;
    std::vector<Plane> planes_;   // タイルごとに Y, UV の順
    size_t total_ = 0;            // 詰めた並びの合計バイト
    size_t chunk_ = 64 * 1024;
    size_t maxComp_ = 0;          // 1 チャンクの圧縮後の最大（16 バイト境界に切り上げ）
    int    n_ = 0;                // チャンク数

    // 送る側の GPU（src_）の作業領域。2 組を交互に使い、前のフレームの送出と次の圧縮を重ねる
    static constexpr int kSets = 2;
    struct SrcSet {
        cudaStream_t stream = nullptr;   // 非ブロッキング
        cudaEvent_t  ready = nullptr;    // 既定ストリーム（cuvid の出力コピー）の後（デコードのみ）
        cudaEvent_t  done = nullptr;     // この組の圧縮・送出が終わった（組の再利用の待ち。エンコードは nextTight の reuseDone）
        bool         used = false;
        uint8_t*  tight = nullptr;       // 詰めた並び
        void**    inPtrs = nullptr;
        uint8_t*  slotBuf = nullptr;       void**  slotPtrs = nullptr;
        size_t*   compBytes = nullptr;   int*    statuses = nullptr;
        void*     temp = nullptr;
        void*     bcPlan = nullptr;      // Bitcomp（ネイティブ API）のプラン。このストリームに結び付ける
        uint64_t* hdr = nullptr;
    };
    SrcSet    sets_[kSets];
    int       nextSet_ = 0;
    size_t*   sInBytes_ = nullptr;       // チャンクの大きさ（組で共通）
    size_t    sTempBytes_ = 0;
    std::vector<cudaEvent_t> encBeg_, encEnd_;
    std::vector<cudaEvent_t> cmpEnd_, sendBeg_;   // 間は枠の空き待ち（計測から外す）
    std::vector<bool> used_;
    // メインメモリ（ページ固定・マップ済み・Portable。nStage 枠）
    int*                   hFlag_ = nullptr;   // GPU が立てる失敗の印（[0] 圧縮、[1] 展開。Mapped）
    std::vector<uint64_t*> hHdr_;
    std::vector<uint8_t*>  hData_;
    // 受ける側の GPU（dst_）
    uint8_t*  dData_ = nullptr;      // 取り込んだ圧縮データ
    void**    dCompPtrs_ = nullptr;  size_t* dCompBytes_ = nullptr;
    std::vector<uint8_t*> dTight_;   // 展開した詰めた並び（普段は 1 面、詰めたまま渡すときはリングの段数）
    std::vector<void**>   dOutPtrs_; // その展開先のチャンク表
    size_t* dOutBufBytes_ = nullptr; size_t* dOutBytes_ = nullptr;
    int*      dStatus_ = nullptr;
    void*     dTemp_ = nullptr;      size_t  dTempBytes_ = 0;
    std::vector<cudaEvent_t> decBeg_, decEnd_;
    std::vector<cudaEvent_t> fetchEnd_, decompEnd_;
    uint64_t* dHdr_ = nullptr;
    void*     dPlan_ = nullptr;          // Bitcomp の展開用プラン
    size_t    lastComp_ = 0;         // 直近の圧縮後の合計（取り込みの DMA の見込み）
};

#endif // XFER_CODEC_H
