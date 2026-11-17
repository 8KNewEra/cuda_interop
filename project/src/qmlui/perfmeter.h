#ifndef PERFMETER_H
#define PERFMETER_H

// ==============================================================
// 性能 HUD の計測（PerfMeter）
//
//   1 フレームの処理を段に分けて時間を測る。HUD（qml/PerfHud.qml）が ON のときだけ動き、
//   OFF の間は各フックが即 return する（GL のクエリも CUDA のイベントも発行しない）
//
//   段と測り方
//     デコード     : デコードスレッドで get_decode_image() にかかった時間（CPU の実時間。NVDEC の
//                    完了待ちを含む）。decode_thread::decode_time_ms → PlayerController → addDecode
//     変換         : NV12 / P010 / YUV → RGBA の CUDA カーネル（interop_stream 上の CUDA イベント間）
//     画像処理     : 画像処理のシェーダ（全パス。GL のタイムスタンプ間）
//     表示ツール   : 表示専用の最終パス（比較・ゼブラなど。ON のときだけ。GL のタイムスタンプ間）
//     ヒストグラム : 集計と描画（CPU の実時間。集計は結果を読むため同期している）
//     描画スレッド : VideoEngine::upload 全体の CPU 時間（GPU 待ち・フェンス待ちを含む）
//     転送         : マルチ GPU デコードで別 GPU → メインメモリに送った時間（その GPU の CUDA イベント間）と速度
//     フレーム間隔 : 新しいフレームを受け取った間隔（= 実際の描画のフレーム時間）
//
//   GPU の計測結果は後から届く（待たない）。数フレームぶんのリングで発行して、届いたものから集計する
//   集計は take() を呼ぶまでの区間の平均と最大。フレーム間隔は推移グラフ用に 1 フレームずつ返す
// ==============================================================

#include <QElapsedTimer>
#include <QVariantMap>
#include <cuda_runtime.h>
#include <vector>

class QOpenGLFunctions_4_5_Core;

class PerfMeter
{
public:
    // GL 側の計測点
    enum GlMark { FilterBegin = 0, FilterEnd, DisplayBegin, DisplayEnd, kGlMarks };

    void setEnabled(bool on);
    bool enabled() const { return on_; }

    // ---- VideoEngine から（GL コンテキストが current の状態で呼ぶ）----
    void frameBegin(QOpenGLFunctions_4_5_Core* gl);                 // upload の先頭
    void convertBegin(cudaStream_t s);                              // 変換カーネルの前後
    void convertEnd(cudaStream_t s);
    void glMark(QOpenGLFunctions_4_5_Core* gl, GlMark m);           // シェーダの前後
    void addHistogram(double ms);                                   // ヒストグラム（CPU）
    void frameEnd();                                                // upload の最後
    void destroy(QOpenGLFunctions_4_5_Core* gl);                    // 破棄（コンテキスト current で）

    // ---- PlayerController から ----
    void addDecode(double ms);
    // マルチ GPU デコードの転送（別 GPU → メインメモリの時間 ms と速度 GB/s。nvgpudecode::transfer_remote）
    void addTransfer(double ms, double gbps);
    // 前回からの集計（平均・最大）とフレーム間隔の列。呼ぶと区間をリセットする
    QVariantMap take();

private:
    struct Acc {
        double sum = 0, max = 0;
        int    n = 0;
        void add(double v) { sum += v; if (v > max) max = v; ++n; }
        void clear() { sum = 0; max = 0; n = 0; }
        QVariantMap toMap() const;
    };

    static constexpr int kRing = 4;   // GPU の結果待ちを何フレームぶん持つか
    struct Slot {
        bool        used     = false;   // 発行済みで結果待ち
        bool        mark[kGlMarks] = {};
        unsigned    query[kGlMarks] = {};
        bool        cuda     = false;
        cudaEvent_t ev[2]    = {};
    };

    bool on_ = false;
    bool created_ = false;
    Slot slots_[kRing];
    int  cur_ = -1;                   // このフレームの slot（-1 = 計測しない）
    int  next_ = 0;

    QElapsedTimer frameCpu_;          // upload の CPU 時間
    QElapsedTimer interval_;          // フレーム間隔
    bool haveLast_ = false;

    Acc decode_, convert_, filter_, display_, hist_, cpu_, interval_ms_, transfer_, transferGbps_;
    std::vector<float> intervals_;    // take() までのフレーム間隔（ms）

    void create(QOpenGLFunctions_4_5_Core* gl);
    void collect(QOpenGLFunctions_4_5_Core* gl);   // 届いた結果を集計する
};

#endif // PERFMETER_H
