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
//     圧縮         : 圧縮転送のとき、別 GPU で差分の前処理と圧縮にかかった時間（その GPU の CUDA イベント間）
//     転送         : マルチ GPU デコードで別 GPU → メインメモリに送った時間（その GPU の CUDA イベント間）と速度
//     展開         : 圧縮転送のとき、表示用 GPU で取り込んで展開した時間（詰めたまま渡すときの差分戻しは「変換」に入る）
//     受け取り間隔 : 新しいフレームを upload した間隔（画面に出た間隔ではない）
//     表示間隔     : 新しいフレームが画面に出た間隔（QQuickWindow::frameSwapped の時刻。VideoEngine::onFrameSwapped）
//     upload→表示  : upload の先頭から画面に出るまで
//     番号の飛び   : 画面に出たフレームの番号が 2 以上進んだ分（出なかったコマ数）
//     未表示の上書き: fbo に描いたのに、画面に出る前に次のフレームで上書きした枚数（番号の飛びの内訳）
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
    // 圧縮転送（nvCOMP）の圧縮率と、別 GPU での圧縮・表示用 GPU での取り込みと展開の時間（ms）
    void addTransferCodec(double ratio, double compressMs, double expandMs);
    // 展開の内訳（取り込み / nvCOMP の展開 / 差分を戻して書く。ms、-1 = 無し）
    void addExpandParts(double fetchMs, double decompMs, double invMs);
    // デコード（待ち）の内訳（nvgpudecode::decode_breakdown。[{key, label, ms}]）
    void addDecodeBreakdown(const QVariantList& parts);
    // ---- VideoEngine::onFrameSwapped から（画面に出たフレーム）----
    //   intervalMs: 前に出てからの間隔（< 0 = 起点なし）、latencyMs: upload → 表示、
    //   frameStep: 前に出たフレームからの番号の差、overwritten: その間に出ないまま上書きした枚数
    void addPresent(double intervalMs, double latencyMs, int frameStep, int overwritten);
    // 間隔の起点を捨てる（一時停止・シークの明け。受け取り間隔も）
    void resetTiming() { haveLast_ = false; }
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

    Acc decode_, convert_, filter_, display_, hist_, cpu_, interval_ms_, transfer_, transferGbps_, ratio_, compress_, expand_, fetch_, decomp_, inv_;
    std::vector<float> intervals_;    // take() までのフレーム間隔（ms）
    Acc present_, latency_;           // 表示間隔、upload → 表示
    std::vector<float> presentIntervals_;   // take() までの表示間隔（ms。推移とばらつき用）
    int frameSkips_ = 0, overwritten_ = 0;  // take() までの番号の飛び・未表示の上書き
    struct Part { QString key, label; Acc acc; };
    std::vector<Part> parts_;         // デコード（待ち）の内訳（来た順）

    void create(QOpenGLFunctions_4_5_Core* gl);
    void collect(QOpenGLFunctions_4_5_Core* gl);   // 届いた結果を集計する
};

#endif // PERFMETER_H
