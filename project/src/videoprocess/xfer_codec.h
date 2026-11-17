#ifndef XFER_CODEC_H
#define XFER_CODEC_H

// ==============================================================
// マルチ GPU デコードの圧縮転送（nvCOMP）
//
//   別 GPU でデコードしたタイルを、圧縮してから表示用 GPU へ送る。GPU 間の経路（PCIe）が細い構成向け
//   （例: サブ GPU が PCIe Gen3 x4 で約 3GB/s）。圧縮しない転送は nvgpudecode::transfer_remote の 2 段コピー
//
//   1 フレームの流れ（1 つの XferCodec = 1 枚の別 GPU。その GPU が受け持つタイルをまとめて扱う）
//     encode（デコード側 GPU の非ブロッキングストリーム。既定ストリーム = cuvid の出力コピーの後に記録したイベントを待つ）
//       xfer_fwd   : NV12 / P010（pitched）→ 詰めた並び。Y と UV は別の面として並べる（差分は面ごとの行内）
//       nvCOMP     : 64KB のチャンクに分けて圧縮（Cascaded / Bitcomp / ANS）
//       xfer_pack  : 圧縮チャンクを詰めて、メインメモリ（ページ固定・マップ済み）へカーネルが直接書く
//                    → 圧縮後の実際のバイト数だけが PCIe を流れる。CPU は大きさを知らなくてよい（待たない）
//     decode（表示用 GPU の非ブロッキングストリーム）
//       xfer_fetch : メインメモリ → 表示用 GPU（チャンクの位置と大きさはメインメモリ上のヘッダから）
//       nvCOMP     : 展開
//       xfer_inv   : 差分を戻して、転送先（pitched。GUI が読む面）へ
//   メインメモリの受け渡し枠は nStage 枠。同じ枠を次に使う前に、前回の decode の完了を待たせる（encode の stageFree）
//
//   デコード側 GPU は作業領域とストリームを kSets 組持ち、フレームごとに交互に使う
//     → 前のフレームの書き出し（PCIe）と次のフレームの圧縮が重なり、1 フレームの時間は「圧縮＋書き出し」から
//       ほぼ「遅い方」になる。既定ストリームに積まないので、cuvid の出力コピーも圧縮の終わりを待たない
//     同じ組は同じストリームで順に進むので、組の再利用は自動で順序が付く
//     デコード済みの面（hw_frames）はリングが一周するまで残るが、念のため既定ストリームに
//     「kSets 個前の圧縮の完了」を待たせてから次を積む（面が再利用される前に読み終えていることを保証）
//
//   カーネルは cuda_kernels/xfer_compress.cu、nvCOMP は 3rdparty/nvcomp（NVIDIA nvCOMP SDK）
//   試験的な機能。効果がなければ設定を「なし」に戻せば従来の 2 段コピーになる
// ==============================================================

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

    // デコード側 GPU で圧縮してメインメモリの枠 s へ（cudaSetDevice は中でする。戻るときは dstGpu を current に）
    //   frames: タイル順の AVFrame（data[0] = Y、data[1] = UV）。stageFree: 枠 s を前回読み終えたイベント（無ければ nullptr）
    bool encode(int s, const std::vector<const AVFrame*>& frames, cudaEvent_t stageFree);
    cudaEvent_t encodedEvent(int s) const { return encEnd_[s]; }

    // 表示用 GPU のストリーム st で、枠 s から取り込んで展開し、転送先へ（st は encodedEvent(s) を待たせてから呼ぶ）
    bool decode(int s, cudaStream_t st, const std::vector<Dst>& dst);

    // 枠 s の前回の結果（終わっていれば true。待たない）
    //   cmpMs: 圧縮（差分の前処理を含む。デコード側 GPU）、sendMs: メインメモリへ書き出す時間（PCIe。枠の空き待ちは含まない）
    //   decMs: 取り込みと展開（表示用 GPU）、raw / comp: バイト数
    bool takeStats(int s, double& cmpMs, double& sendMs, double& decMs, double& rawBytes, double& compBytes);

    void release();

private:
    struct Plane { size_t off; int elems; int rows; int lag; };   // 詰めた並びの中の 1 面
    bool ready_ = false;
    int  src_ = -1, dst_ = -1, nStage_ = 0, bps_ = 1;
    Algo algo_ = None;
    bool delta_ = true;
    std::vector<Plane> planes_;   // タイルごとに Y, UV の順
    size_t total_ = 0;            // 詰めた並びの合計バイト
    size_t chunk_ = 64 * 1024;
    size_t maxComp_ = 0;          // 1 チャンクの圧縮後の最大（16 バイト境界に切り上げ）
    int    n_ = 0;                // チャンク数

    // デコード側 GPU（kSets 組を交互に使う）
    static constexpr int kSets = 2;
    struct SrcSet {
        cudaStream_t stream = nullptr;   // 非ブロッキング
        cudaEvent_t  ready = nullptr;    // 既定ストリーム（cuvid の出力コピー）の後
        cudaEvent_t  done = nullptr;     // この組の最後の圧縮・書き出しの完了
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
    std::vector<cudaEvent_t> cmpEnd_, sendBeg_;   // 圧縮の終わり / 書き出しの始まり（間は枠の空き待ち）
    std::vector<bool> used_;
    // メインメモリ（ページ固定・マップ済み・Portable）
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
};

#endif // XFER_CODEC_H
