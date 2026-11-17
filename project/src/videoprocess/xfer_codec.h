#ifndef XFER_CODEC_H
#define XFER_CODEC_H

// マルチ GPU デコードの圧縮転送（nvCOMP）。1 つの XferCodec = 1 枚の別 GPU が受け持つタイル
//   encode（別 GPU）  : xfer_fwd（詰めた並び・差分）→ nvCOMP 圧縮 → xfer_pack（メインメモリへ直接書く）
//   decode（表示用 GPU）: xfer_fetch（DMA ＋ 見込み超過分はカーネル）→ nvCOMP 展開 → xfer_inv（差分を戻して転送先へ）
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
    static Algo    parse(const QString& s);     // "cascaded" / "bitcomp" / "ans"、それ以外は None
    static QString name(Algo a);

    // 転送先（表示用 GPU 上の面。タイルごと）
    struct Dst { uint8_t* y; size_t yPitch; uint8_t* uv; size_t uvPitch; };

    ~XferCodec() { release(); }

    // tileW / tileH: このコーデックが受け持つタイルの大きさ（タイル順）、bps: 1 = NV12、2 = P010
    bool init(int srcGpu, int dstGpu, Algo algo, bool delta, int bps,
              const std::vector<int>& tileW, const std::vector<int>& tileH, int nStage);
    bool ready() const { return ready_; }

    // 別 GPU で圧縮してメインメモリの枠 s へ（戻るときは dstGpu が current）
    //   frames: タイル順（data[0] = Y、data[1] = UV）。stageFree: 枠 s を前回読み終えたイベント（無ければ nullptr）
    bool encode(int s, const std::vector<const AVFrame*>& frames, cudaEvent_t stageFree);
    cudaEvent_t encodedEvent(int s) const { return encEnd_[s]; }

    // 表示用 GPU のストリーム st で枠 s を展開して転送先へ（st は encodedEvent(s) を待たせてから呼ぶ）
    bool decode(int s, cudaStream_t st, const std::vector<Dst>& dst);

    // 枠 s の前回の計測（終わっていれば true。待たない）。圧縮 / 送出 / 展開と内訳の ms、元と圧縮後のバイト数
    bool takeStats(int s, double& cmpMs, double& sendMs, double& decMs, double& rawBytes, double& compBytes,
                   double* fetchMs = nullptr, double* decompMs = nullptr, double* invMs = nullptr);

    void release();

private:
    struct Plane { size_t off; int elems; int rows; int lag; };   // 詰めた並びの 1 面（lag: 差分の相手。Y = 1、UV = 2）
    bool ready_ = false;
    int  src_ = -1, dst_ = -1, nStage_ = 0, bps_ = 1;
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
    uint8_t*  dTight_ = nullptr;     // 展開した詰めた並び
    void**    dOutPtrs_ = nullptr;   size_t* dOutBufBytes_ = nullptr; size_t* dOutBytes_ = nullptr;
    int*      dStatus_ = nullptr;
    void*     dTemp_ = nullptr;      size_t  dTempBytes_ = 0;
    std::vector<cudaEvent_t> decBeg_, decEnd_;
    std::vector<cudaEvent_t> fetchEnd_, decompEnd_;
    uint64_t* dHdr_ = nullptr;
    size_t    lastComp_ = 0;         // 直近の圧縮後の合計（取り込みの DMA の見込み）
};

#endif // XFER_CODEC_H
