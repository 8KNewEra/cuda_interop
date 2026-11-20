#ifndef VIDEOENGINE_H
#define VIDEOENGINE_H

// ==============================================================
// VideoEngine（Qt Quick 版 UI（--qml）専用の描画・処理部）
//
//   Widgets 版 GLWidget の処理部を QOpenGLWindow から切り離したもの。
//   ・自前の QOpenGLContext + QOffscreenSurface を GUI スレッドで持つ。
//     Qt Quick とは AA_ShareOpenGLContexts で共有しているので、ここで描いた
//     テクスチャを VideoItem がそのまま表示できる。
//   ・CUDA interop / GLSL フィルタ / NVENC への受け渡し / 画像保存 /
//     ヒストグラム集計の手順は GLWidget と同じ。
//   ・画面への描画は持たない（表示は VideoItem / HistogramItem、文字は QML）。
//
//   1 フレームの流れ（FBO_Rendering）:
//     入力テクスチャ（CUDA が書く、上下逆）
//       → 処理パス: colorcorrect（画像補正・色補正・トーンカーブ）→ gaussian → sobel → averaging
//         （ON のものだけ。fbo / tempfbo を交互に使い、最終結果が fbo に来るよう開始側を決める）
//       → fbo = 処理結果。ヒストグラム・画像保存・エンコードはここを読む
//       → 表示専用パス（zebra.frag: 比較・チャンネル・ゼブラ・タイル境界）: fbo → tempfbo
//       → displayTexture() を VideoItem が表示
//
//   コンテキストが 2 つあるので GPU 上の順序はフェンスで渡す:
//     engine → QML : 描き終えたら frameFence。VideoItem / HistogramItem が描画前に glWaitSync
//     QML → engine : シーングラフが描き終えたら displayDoneFence。次の FBO 上書き前に glWaitSync
//   render loop は basic（GUI スレッド）固定なので CPU 側の競合はない。
// ==============================================================

#include "src/videoprocess/save_encode.h"
#include "src/imageprocess/cuda_imageprocess.h"
#include "src/main/__global__.h"
#include "src/main/glwidget.h"          // CudaObjRetireRing を共用する
#include "src/qmlui/perfmeter.h"       // 性能 HUD の計測

#include <QObject>
#include <QOpenGLContext>
#include <QOffscreenSurface>
#include <QOpenGLFunctions_4_5_Core>
#include <QOpenGLShaderProgram>
#include <QElapsedTimer>
#include <QMutex>
#include <QVariantMap>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <atomic>
#include <memory>
#include <vector>

namespace gpuimg { class Encoder; }
class QFile;
class QTextStream;
class QQuickWindow;

class VideoEngine : public QObject, protected QOpenGLFunctions_4_5_Core
{
    Q_OBJECT

public:
    explicit VideoEngine(QObject* parent = nullptr);
    ~VideoEngine() override;

    // QML 側（VideoItem / HistogramItem）が参照する唯一のインスタンス
    static VideoEngine* instance() { return s_instance; }

    // GL コンテキスト作成・シェーダ・CUDA ストリーム。GPU 情報の確定もここ。
    bool initialize();
    bool isInitialized() const { return initialize_completed_flag; }

    // ---- デコード側からの入口（Widgets 版 MainWindow::decode_view 相当）----
    //   upload   : 新しいフレーム（GLWidget::uploadToGLTexture 相当）
    //   rerender : 停止中の再描画（GLWidget::FBO_Rendering 相当）
    void upload(const VideoFrame& Frame);
    void rerender(const VideoFrame& Frame);

    // ファイルを閉じたとき（画面を黒に戻す）
    void reset();

    // エンコード状態（GLWidget::encode_mode と同じ）
    void encode_mode(int flag);

    // フィルタの ON/OFF・パラメータを変えたら立てる（次の描画で setShaderUniformEnable）
    bool filter_change_flag = true;

    // ガウシアン / 平均化の係数（行優先 size×size、正規化済み）
    //   GUI スレッドから呼ぶ（FBO_Rendering も同じスレッド）。次の描画で GPU へ送る
    void setFilterKernel(bool gaussian, int size, const std::vector<float>& coeffs);
    bool screenshot_flag = false;

    int MinFrame = 0;
    int MaxFrame = 0;
    int encode_FrameCount = 0;

    double fps = 0.0;            // OpenGL Rendering FPS（表示用）

    // 性能 HUD の計測（PlayerController が ON/OFF・集計の取り出し・デコード時間の追加をする）
    PerfMeter& perf() { return perf_; }

    // ---- 表示側（Qt Quick のレンダリング中に呼ぶ）----
    bool    hasFrame()        const { return frame_valid_; }
    // 表示するテクスチャ。表示ツールが ON なら tempTexture（fbo に重ねた表示専用）、OFF なら fbo
    GLuint  displayTexture()  const { return displayTex_ ? displayTex_ : fboTextureID; }
    QSize   displaySize()     const { return QSize(width_, height_); }
    // 10bit 経路か（テクスチャが RGBA16）。10bit 以上の素材のとき upload が自動で切り替える
    bool    highBitDepth()    const { return rgba16_; }
    // スポイト（フレーム単位）
    //   setPickRequest で位置（左上原点のピクセル、-1 で停止）を指定すると、描画した
    //   フレームごとに CUDA の非同期コピーでその画素（補正後・補正前）を拾う（GPU を待たない）
    //   pickResult は最後に拾えた値 {frame, x, y, r, g, b, r0, g0, b0}（なければ空）
    void setPickRequest(int x, int y) { pickX_ = x; pickY_ = y; if (x < 0) pickResult_.clear(); }
    QVariantMap pickResult();
    QVariantMap lastPick() const { return pickX_ < 0 ? QVariantMap() : pickResult_; }
    GLuint  histTexture()     const { return histTextureID; }
    QSize   histSize()        const { return histSize_; }
    // HistogramItem の実ピクセルサイズ。次の描画でプロット先をこの大きさに作り直す
    //   （固定サイズの拡大縮小だと、大きな画面でぼやけ、小さな画面で線が消える）
    void    setHistViewport(const QSize& px, qreal dpr);
    bool    histValid()       const { return hist_valid_; }
    quint64 frameSerial()     const { return frame_serial_; }

    // シーングラフのコンテキストがカレントの状態で呼ぶ
    void waitFrameFence();            // engine の描画完了を GPU 側で待つ
    void signalDisplayDone();         // シーングラフの描画完了を記録

    // ---- 表示の計測（画面に出たフレーム。性能 HUD と CSV ログ）----
    //   upload しただけでは画面に出たとは限らない（swap の前に次のフレームで fbo を上書きすると出ない）。
    //   VideoItem が描いたフレームを notePainted で控え、frameSwapped で onFrameSwapped が「出た」と数える
    //   CSV: 環境変数 CUDA_VIEWER_PRESENT_LOG=<パス> で 1 表示ごとに 1 行（HUD の ON/OFF と無関係）
    void notePainted();                       // VideoItem::updatePaintNode から（描くフレームを控える）
    void onFrameSwapped();                    // QQuickWindow::frameSwapped から（swap の直後）
    void setSteadyPlayback(bool on);          // 安定して順方向に再生しているか（PlayerController::decode_view）。
                                              //   false の間は間隔の起点を捨てる（一時停止・シークの明けを数えない）

    // ---- 表示のペース（PlayerController の待ち行列が使う）----
    // 実際の vblank の格子（DWM）。vbNs = 直近の vblank の時刻（mono_now_ns と同じ物差し）、perNs = 周期。取れなければ false
    bool   vblankGrid(qint64& vbNs, double& perNs) const;
    // upload の先頭から swap（frameSwapped）までの時間の移動平均（ms）。upload を始める時刻の逆算に使う
    double uploadToSwapMs() const { return uploadToSwapEmaMs_; }
    // 描いたのにまだ swap していないフレームがあるか（1 回の swap に 2 枚描かないため）
    bool   hasPendingPresent() const { return frame_valid_ && rendered_.serial != 0 && rendered_.serial != presented_.serial; }
    // 映像を出すウィンドウ（main.cpp が渡す）と、その場で描かせる（表示のペースが upload の直後に呼ぶ）。
    //   requestUpdate に任せると、UI の更新（性能 HUD の 250ms など）で先に予約されていた描画に乗って swap が早まり、
    //   狙った vblank の 1 つ前に入ってしまう（upload→swap が 7ms → 2〜3ms）。UpdateRequest を直接送れば
    //   render loop（basic）がその場で描くので、upload→swap が毎回同じになる。CUDA_VIEWER_RENDER_NOW=0 で切る
    void   setPresentWindow(QQuickWindow* w) { presentWindow_ = w; }
    void   renderNow();
    // QQuickWindow::afterRendering（swap の直前。main.cpp）から: swap にかかった時間を測る起点
    void   noteAfterRendering() { afterRenderNs_ = mono_now_ns(); }

    // ヒストグラム統計（QML 表示用）
    QVariantMap histStats() const;

signals:
    void initialized();
    void frameReady();                // 新しい表示フレームができた
    void histogramChanged();
    void pickUpdated();               // スポイトの値が新しいフレームの分に更新された
    void encode_finished();
    void encode_error(QString error);
    void screenshot_finished();

private:
    static VideoEngine* s_instance;

    bool makeCurrent();
    void doneCurrent();

    void FBO_Rendering(const VideoFrame& Frame);
    void downloadToGLTexture_and_Encode(const VideoFrame& Frame);
    void finishEncode();                  // エンコーダを閉じて結果を通知（完了 / 中断）
    void abortEncode(const char* what, cudaError_t err);   // FBO の受け渡しの失敗で中断する（エンコード中だけ）
    void setShaderUniformEnable();
    void uploadKernels();                 // 係数をシェーダの uniform へ

    struct KernelData { int size = 0; float k[81] = {}; };
    KernelData gaussKernel_, avgKernel_;
    bool kernel_change_flag = false;
    void setShaderUniform(int width, int height);

    void initCudaTexture(int width, int height);
    void initTextureCuda(int width, int height);
    void initCudaScreenshot(int width, int height);
    void initCudaHist();
    void initHistTarget(const QSize& size);
    void histgram_Analysys();
    void drawHistogram();
    void doScreenshot(const VideoFrame& Frame);

    bool mergeToSurface(const VideoFrame& Frame, cudaSurfaceObject_t surf);
    bool yuvToSurface(const VideoFrame& Frame, cudaSurfaceObject_t surf);
    bool createInputSurface(cudaArray_t array, cudaSurfaceObject_t& out);
    bool createArrayTexture(cudaArray_t array, cudaTextureObject_t& out, const char* tag);

    void queryCudaGPUs();
    void getCudaDeviceIDFromOpenGL();
    static std::vector<int> make_nice_y_labels(int max_value);

    void waitDisplayDone();           // 自分のコンテキストで displayDoneFence を待つ
    void publishFrameFence();

    void waitScreenshotSaved();

    // ---- GL ----
    QOpenGLContext*    ctx_ = nullptr;
    QOffscreenSurface* surface_ = nullptr;
    bool initialize_completed_flag = false;

    GLuint inputTextureID = 0;
    GLuint fboTextureID   = 0;
    GLuint curveTexID     = 0;      // トーンカーブの対応表（1D、256×RGB）
    // スポイト（フレームごとの非同期コピー）
    int          pickX_ = -1, pickY_ = -1;      // 要求中の位置（-1 = 停止）
    bool         pickPending_ = false;          // コピーを積んで結果待ち
    int          pickReqX_ = 0, pickReqY_ = 0, pickReqFrame_ = 0;
    unsigned char* h_pick_ = nullptr;           // ピン留めしたホストメモリ（16 バイト：補正後 RGBA、補正前 RGBA。8bit 経路は前 8 バイトだけ使う）
    bool         pickReqRgba16_ = false;          // 要求したときの経路（結果の読み方）
    cudaEvent_t  pickEvent_ = nullptr;
    QVariantMap  pickResult_;
    void pickCollect();                         // 終わったコピーの結果を受け取る（待たない）
    void pickIssue(int frameNo);                // この描画フレームの画素のコピーを積む
    GLuint tempTextureID  = 0;
    GLuint fbo     = 0;
    GLuint tempfbo = 0;
    GLuint vao = 0;
    GLuint vbo = 0;
    GLuint ebo = 0;
    // 2 パス目以降用（上下反転しない quad）。vao は 1 パス目で CUDA の上下逆のデータを
    //   反転させる。全パスで反転するとパス数が偶数のときに逆さまになる
    GLuint vaoNoFlip = 0;
    GLuint vboNoFlip = 0;

    // ヒストグラムのプロット先（QML の HistogramItem が貼る）
    QSize  histSize_{480, 320};        // 現在のプロット先
    QSize  histWanted_{480, 320};      // 表示側が求めている大きさ
    qreal  histDpr_ = 1.0;
    GLuint histFbo = 0;
    GLuint histTextureID = 0;
    GLuint vbo_hist = 0;
    bool   hist_valid_ = false;

    QOpenGLShaderProgram Sobel_program;
    QOpenGLShaderProgram Gaussian_program;
    QOpenGLShaderProgram Averaging_program;
    QOpenGLShaderProgram Zebra_program;
    GLuint displayTex_ = 0;            // 最後に描いたフレームの表示用テクスチャ（0 = fbo）
    // 追加の uniform（ソーベルのモード、表示専用パス、ガウシアンのシャープ化）
    //   注意: zebraProg_ / Zebra_program は名前に反して表示専用パス全体（zebra.frag）
    struct ExtraLocs {
        GLint sobelMode = -1, sobelThreshold = -1, sobelColor = -1;
        GLint zTex = -1, zHighOn = -1, zHigh = -1, zLowOn = -1, zLow = -1, zPeriod = -1;
        GLint zOn = -1, zChannel = -1;          // ゼブラ ON/OFF、チャンネル表示
        GLint cOrig = -1, cOn = -1, cDir = -1, cPos = -1, cLine = -1;   // 比較表示
        GLint sOn = -1, sGrid = -1, sSize = -1, sThr = -1, sBand = -1, sColor = -1; // タイル境界の段差
        GLint gMode = -1, gAmount = -1;         // ガウシアン：ぼかし / シャープ化
    } xl_;
    GLuint zebraProg_ = 0;
    struct shader {
        GLuint progId = 0;
        GLint loc_tex = 0;
        GLint loc_texelSize = 0;
        GLint loc_filterEnabled = 0;
        GLint loc_radius = -1;           // 0 = シェーダ内の固定 7x7 カーネル（Widgets 版）
        GLint loc_kernel = -1;
    };
    shader Sobel_shader;
    shader Gaussian_shader;
    shader Averaging_shader;
    shader Color_shader;                 // 画像補正 ＋ 色補正 ＋ トーンカーブ（1 パス）
    QOpenGLShaderProgram Color_program;
    struct ColorLocs { GLint toneOn = -1, brightness = -1, contrast = -1, shadows = -1, highlights = -1, gamma = -1,
                       rgbOn = -1, wb = -1, gain = -1, curveOn = -1, curveTex = -1, offset = -1, rgbGamma = -1, saturation = -1; } cl_;

    // ---- フェンス（GLsync は void* 相当の不透明ポインタ）----
    mutable QMutex fenceMutex_;
    GLsync frameFence_       = nullptr;
    bool   frameFenceWaited_ = false;
    GLsync displayDoneFence_ = nullptr;

    // ---- CUDA interop ----
    cudaGraphicsResource* cudaResource1 = nullptr;   // 入力テクスチャ
    cudaGraphicsResource* cudaResource2 = nullptr;   // FBO（NVENC / ヒストグラム）
    cudaGraphicsResource* cudaResource_hist_draw = nullptr;
    cudaGraphicsResource_t cudaResource_screenshot = nullptr;
    CUDA_ImageProcess* CUDA_IMG_Proc = nullptr;
    cudaStream_t interop_stream = nullptr;
    cudaEvent_t encode_gl_event = nullptr;

    CudaObjRetireRing inputSurfRetire;
    CudaObjRetireRing fboTexRetire;
    CudaObjRetireRing histTexRetire;

    // in-flight クランプ（GLWidget と同じ理由で 2）
    static const int kMaxFramesInFlight = 2;
    cudaEvent_t frameDone[kMaxFramesInFlight] = {};
    bool        frameDoneValid[kMaxFramesInFlight] = {};
    int         frameDoneNo = 0;

    // ---- エンコード ----
    int encode_state = STATE_NOT_ENCODE;
    int prev_encode_state = STATE_NOT_ENCODE;
    save_encode* save_encoder = nullptr;

    // ---- 動画情報 ----
    int width_ = 0, height_ = 0;
    // 処理経路: false = 8bit 経路（テクスチャ RGBA8）、true = 10bit 経路（RGBA16）
    //   入力・fbo・tempfbo の 3 枚がこの形式。CUDA のカーネルも経路ごとに別（cuda_kernels/*_8bit / *_10bit）
    bool rgba16_ = false;
    bool wantsRgba16(const VideoFrame& Frame) const;   // この素材は 10bit 経路で処理するか
    bool p010ToSurface(const VideoFrame& Frame, cudaSurfaceObject_t surf);   // 10bit 経路の NVDEC 入力
    int FrameNo = 0;
    bool frame_valid_ = false;
    quint64 frame_serial_ = 0;
    const DecodeInfo& VideoInfo = DecodeInfoManager::getInstance().getSettings();

    QElapsedTimer fpsTimer;
    int fpsCount = 0;
    PerfMeter perf_;             // 性能 HUD（OFF の間は何もしない）

    // ---- 表示の計測（notePainted / onFrameSwapped）----
    //   時刻はすべて mono_now_ns（__global__.h。QPC の ns。スレッド・DWM の vblank と同じ物差し）
    struct ShownFrame {
        quint64 serial   = 0;    // frame_serial_（0 = なし）
        int     frameNo  = -1;
        qint64  uploadNs = -1;   // upload の先頭（一時停止中の描き直しは描いた時刻）
        qint64  tickNs = 0, decodeBeginNs = 0, emitNs = 0, recvNs = 0;   // VideoFrame の t_*（0 = なし）
        int     paced = 0;                                              // VideoFrame::paced
        qint64  wakePlanNs = 0, wakeNs = 0; int pendWaits = 0;          // VideoFrame::t_wake_plan_ns / t_wake_ns / pend_waits
    };
    qint64     logT0Ns_ = 0;            // CSV の t_ms の起点
    qint64     prevTickNs_ = 0;         // 前に出たフレームの刻み（刻みの間隔）
    qint64     prevVblankIdx_ = -1;     // 前に出たフレームが乗った vblank の番号（DWM。時刻から数える）
    qint64     prevVblankNs_  = 0;      // 前に出たフレームの swap の後の最初の vblank の時刻
    double     uploadToSwapEmaMs_ = 7.0;
    QQuickWindow* presentWindow_ = nullptr;
    int        renderNowEnv_ = -1;      // CUDA_VIEWER_RENDER_NOW（0 で切る）。-1 = 未読
    // swap の回数と時間（CSV の swaps / swap_ms / swap_max_ms。前に出たコマから今のコマが出るまで）
    qint64     afterRenderNs_ = 0;
    int        swapsSincePresent_ = 0;
    double     swapMaxMsSince_ = -1;
    qint64     pendingUploadNs_ = -1;   // upload の先頭の時刻（FBO_Rendering が rendered_ に移す）
    ShownFrame rendered_;               // 最後に fbo に描いたフレーム
    ShownFrame painted_;                // VideoItem が最後に描いた（swap 待ちの）フレーム
    ShownFrame presented_;              // 最後に画面に出たフレーム
    qint64     lastPresentNs_ = -1;     // 前に出た時刻（-1 = 起点なし）
    int        lastPresentFrame_ = -1;
    int        overwrittenSince_ = 0;   // 前に出てから、出る前に上書きした枚数
    bool       steady_ = false;
    std::unique_ptr<QFile>       presentLogFile_;
    std::unique_ptr<QTextStream> presentLog_;     // file より後に宣言（先に壊して flush する）
    bool       presentLogTried_ = false;
    int        presentLogLines_ = 0;

    // ---- ヒストグラム ----
    cudaStream_t hist_stream = nullptr;
    HistData*  d_hist_data  = nullptr;
    HistStats* d_hist_stats = nullptr;
    HistStats  h_hist_stats{};
    std::vector<int> hist_labels_;
    int num_bins = 256;            // 最後に集計した段数（8bit 経路 256、10bit 経路 1024 = HIST10_BINS）
    HistData10* d_hist_data10 = nullptr;   // 10bit 経路の集計先（1024 段）

    // ---- スクリーンショット ----
    uint8_t* d_cudaScreenshot = nullptr;
    size_t d_cudaScreenshotPitch = 0;
    cudaStream_t screenshotStream = nullptr;
    std::unique_ptr<gpuimg::Encoder> screenshotEncoder;
    std::atomic<bool> screenshotSaving{false};
};

#endif // VIDEOENGINE_H
