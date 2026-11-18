#ifndef XFER_CODEC_H
#define XFER_CODEC_H

// マルチ GPU デコードの圧縮転送（nvCOMP）。1 つの XferCodec = 1 枚の別 GPU が受け持つタイル
//   encode（別 GPU）    : xfer_fwd（詰めた並び・差分）→ 圧縮 → xfer_pack（メインメモリへカーネルが直接書く）
//   decode（表示用 GPU）: xfer_fetch（見込みまで DMA、超過分はカーネル）→ 展開
//                         → 詰めたまま GUI へ（変換カーネルが差分を戻す）か、xfer_inv で戻して転送先へ
//   流れ・同期・バッファの関係は docs/MULTI_GPU_DECODE.md、カーネルは cuda_kernels/xfer_compress.cu

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

    // 転送先（表示用 GPU 上の 1 タイル。詰めたまま渡すときは使わない）
    struct Dst { uint8_t* y; size_t yPitch; uint8_t* uv; size_t uvPitch; };

    // 1 フレーム分の計測（性能 HUD）。ms は GPU 上の経過時間
    struct Stats {
        double compressMs = 0;   // 別 GPU: 前処理 + 圧縮
        double sendMs = 0;       // 別 GPU: メインメモリへの書き出し（枠の空き待ちは除く）
        double expandMs = 0;     // 表示用 GPU: 取り込み〜差分戻し
        double fetchMs = -1, decompMs = -1, invMs = -1;   // その内訳（-1 = 取れない）
        double rawBytes = 0, compBytes = 0;
    };

    ~XferCodec() { release(); }

    // tileW / tileH: 受け持つタイルの大きさ（タイル順）、bps: 1 = NV12、2 = P010、nStage: メインメモリの枠数
    //   ringSize > 0 なら詰めたまま渡す（差分は GUI の変換カーネルが戻す）。展開先をリングの段数ぶん持つ
    bool init(int srcGpu, int dstGpu, Algo algo, bool delta, int bps,
              const std::vector<int>& tileW, const std::vector<int>& tileH, int nStage, int ringSize = 0);
    void release();

    // 別 GPU で圧縮してメインメモリの枠 s へ（戻るときは dstGpu が current）
    //   frames: タイル順（data[0] = Y、data[1] = UV）。stageFree: 枠 s を前回読み終えたイベント（無ければ nullptr）
    bool encode(int s, const std::vector<const AVFrame*>& frames, cudaEvent_t stageFree);
    cudaEvent_t encodedEvent(int s) const { return encEnd_[s]; }

    // 表示用 GPU のストリーム st で枠 s を展開する（st には先に encodedEvent(s) を待たせる）
    //   普段は差分を戻して dst へ。詰めたまま渡すときは展開先リングの ring 段へ展開するだけ（packedPlane で読む）
    bool decode(int s, int ring, cudaStream_t st, const std::vector<Dst>& dst);
    bool     packed() const { return ringSize_ > 0; }
    uint8_t* packedPlane(int ring, int plane) const { return dTight_[ring] + planes_[plane].off; }   // plane: タイル t の Y = 2t、UV = 2t+1
    size_t   planeRowBytes(int plane) const { return size_t(planes_[plane].elems) * size_t(bps_); }

    // 枠 s の前回の計測（終わっていれば true。待たない）
    bool takeStats(int s, Stats& out);

private:
    static QString name(Algo a);
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

    // 別 GPU 側の作業領域。2 組を交互に使い、前のフレームの送出と次の圧縮を重ねる
    static constexpr int kSets = 2;
    struct SrcSet {
        cudaStream_t stream = nullptr;   // 非ブロッキング
        cudaEvent_t  ready = nullptr;    // 既定ストリーム（cuvid の出力コピー）の後
        cudaEvent_t  done = nullptr;
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
    std::vector<uint64_t*> hHdr_;
    std::vector<uint8_t*>  hData_;
    // 表示用 GPU
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
