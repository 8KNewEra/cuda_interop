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

    // ---- 表示側（Qt Quick のレンダリング中に呼ぶ）----
    bool    hasFrame()        const { return frame_valid_; }
    // 表示するテクスチャ。表示ツールが ON なら tempTexture（fbo に重ねた表示専用）、OFF なら fbo
    GLuint  displayTexture()  const { return displayTex_ ? displayTex_ : fboTextureID; }
    QSize   displaySize()     const { return QSize(width_, height_); }
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
    unsigned char* h_pick_ = nullptr;           // ピン留めしたホストメモリ（8 バイト：補正後 RGBA、補正前 RGBA）
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
    int FrameNo = 0;
    bool frame_valid_ = false;
    quint64 frame_serial_ = 0;
    const DecodeInfo& VideoInfo = DecodeInfoManager::getInstance().getSettings();

    QElapsedTimer fpsTimer;
    int fpsCount = 0;

    // ---- ヒストグラム ----
    cudaStream_t hist_stream = nullptr;
    HistData*  d_hist_data  = nullptr;
    HistStats* d_hist_stats = nullptr;
    HistStats  h_hist_stats{};
    std::vector<int> hist_labels_;
    int num_bins = 256;

    // ---- スクリーンショット ----
    uint8_t* d_cudaScreenshot = nullptr;
    size_t d_cudaScreenshotPitch = 0;
    cudaStream_t screenshotStream = nullptr;
    std::unique_ptr<gpuimg::Encoder> screenshotEncoder;
    std::atomic<bool> screenshotSaving{false};
};

#endif // VIDEOENGINE_H
