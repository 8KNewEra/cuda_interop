#ifndef GLWIDGET_H
#define GLWIDGET_H

#include "qelapsedtimer.h"
#include "src/videoprocess/save_encode.h"
#include "src/imageprocess/cuda_imageprocess.h"
#include "src/main/__global__.h"
#include <QOpenGLFunctions_4_5_Core>
#include <QOpenGLWindow>
#include <QOpenGLShader>
#include <QOpenGLShaderProgram>
#include <QPainter>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <QWheelEvent>
#include <QMouseEvent>

// ==============================================================
// ★ CUDA object（texture / surface）の遅延破棄リング
//
//   cudaGraphicsSubResourceGetMappedArray() が返す cudaArray は
//   「その map の間だけ有効」で、次の map でも同じ実体を指す保証は無い
//   （CUDA の仕様上、map のたびに値が変わりうる）。
//   したがって object を map/unmap を跨いでキャッシュするのは仕様違反で、
//   ドライバが裏で実体を差し替えると、古いフレームを読む／書くことになる。
//   ＝「たまにフレームが混ざる」症状を単独で起こせる。
//
//   かといって使い終わった直後に破棄すると、その object を使っている
//   カーネルがまだ走っている可能性があるので CPU同期が必要になり、
//   せっかく外した GUIスレッドの停止が復活してしまう。
//
//   そこで
//     ・object は map のたびに作り直す（仕様どおり）
//     ・使い終わりをイベントで記録し、数フレーム後に
//       そのイベント完了を確認してから破棄する
//   という形にする。仕様違反もCPU同期も無くなる。
//   SIZE は in-flight(kMaxFramesInFlight=2) より十分大きくしておくこと。
// ==============================================================
struct CudaObjRetireRing
{
    static const int SIZE = 4;
    enum Kind { TEXTURE, SURFACE };

    // cudaTextureObject_t / cudaSurfaceObject_t はどちらも unsigned long long
    Kind               kind            = TEXTURE;
    unsigned long long obj    [SIZE]   = {};
    cudaEvent_t        ev     [SIZE]   = {};
    bool               evValid[SIZE]   = {};
    int                no              = 0;

    void init(Kind k);                      // イベント生成（GL/CUDA初期化時に1回）
    int  acquire();                         // 空きslot確保（前の住人を安全に破棄）
    void store(int i, unsigned long long o) { if (i >= 0 && i < SIZE) obj[i] = o; }
    void retire(int i, cudaStream_t s);     // 「このslotの使用終わり」を記録
    void drain();                           // 全slotを同期して object を破棄
    void destroy();                         // drain + イベント破棄

private:
    void destroySlot(int i);
};


class GLWidget : public QOpenGLWindow, protected QOpenGLFunctions_4_5_Core
{
    Q_OBJECT
signals:
    void decode_please();
    void initialized();
    void encode_finished();
    // ★ エンコーダの生成に失敗した（VRAM不足・コーデック初期化失敗など）。
    //   save_encode のコンストラクタは例外を投げるので、ここで受けて通知する。
    void encode_error(QString error);

public:
    explicit GLWidget(QWindow *parent = nullptr);
    ~GLWidget();
    void uploadToGLTexture(VideoFrame Frame);
    void encode_mode(int flag);
    void GLresize();
    void GLreset();
    void FBO_Rendering(VideoFrame Frame);
    void setShaderUniformEnable();

    int MinFrame=0;
    int MaxFrame=0;
    int encode_FrameCount=0;

    //画像処理
    bool filter_change_flag = true;

    double fps = 0.0;

    void resetZoom();

    bool screenshot_flag = false;

protected:
    void initializeGL() override;
    void paintGL() override;

    // ズーム/パン: ホイール=拡大縮小, 左/中ドラッグ=移動, ダブルクリック=等倍
    void wheelEvent(QWheelEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;
    void mouseDoubleClickEvent(QMouseEvent *event) override;

private:
    void initTextureCuda(int width,int height);
    void initCudaTexture(int width,int height);
    void setShaderUniform(int width,int height);
    void initCudaHist();
    void histgram_Analysys();
    void downloadToGLTexture_and_Encode(VideoFrame Frame);

    // ★ NV12タイル → OpenGLテクスチャ へ surface object 経由で直接 merge
    //   （中間バッファを挟まないゼロコピー経路）
    bool mergeToSurface(const VideoFrame& Frame, cudaSurfaceObject_t surf);
    bool yuvToSurface(const VideoFrame& Frame, cudaSurfaceObject_t surf);

    // ==========================================================
    // ★ surface / texture object は map のたびに作り直す
    //
    //   以前は cudaArray のハンドルを覚えておいて使い回していたが、
    //   mapped array の寿命は map 中だけというのが CUDA の仕様で、
    //   ハンドルが同じでも実体が差し替わりうる（＝古い面を読む）。
    //   破棄は CudaObjRetireRing がイベントで遅延させるので、
    //   毎フレーム作り直しても GUIスレッドは止まらない。
    // ==========================================================
    bool createInputSurface(cudaArray_t array, cudaSurfaceObject_t& out);
    bool createArrayTexture(cudaArray_t array, cudaTextureObject_t& out,
                            const char* tag);

    CudaObjRetireRing inputSurfRetire;   // cudaResource1（入力テクスチャ）
    CudaObjRetireRing fboTexRetire;      // cudaResource2（FBO → NVENC）
    CudaObjRetireRing histTexRetire;     // cudaResource2（ヒストグラム読み出し）

    std::vector<int> make_nice_y_labels(int max_value);
    void queryCudaGPUs();
    void getCudaDeviceIDFromOpenGL();

    //CUDA Interop
    cudaGraphicsResource* cudaResource1;
    cudaGraphicsResource* cudaResource2;
    cudaGraphicsResource* cudaResource_hist_draw=nullptr;
    CUDA_ImageProcess* CUDA_IMG_Proc=nullptr;
    cudaStream_t interop_stream = nullptr;

    // ★ FBOのGL描画完了を save_encode の st へ橋渡しするイベント
    cudaEvent_t encode_gl_event = nullptr;

    // ==========================================================
    // ★ GPU投入の上限（in-flight クランプ）
    //
    //   CPU同期を外した結果、GUIスレッドはGPUの完了を待たずに
    //   投げ続けられるようになった。GPUが1フレームでも遅れると
    //   その差分が毎フレーム積み上がり、遅延とキューが青天井になる。
    //   （タスクマネージャで3D使用率がのこぎり波になる）
    //
    //   毎フレーム待つ（＝①以前）のでもなく、無制限でもなく、
    //   K フレームぶんだけ先行を許して、それ以上は待つ。
    //   GPUが間に合っている間はこの待ちはノーコストで通過する。
    //
    //   ★ K はリング段数より小さくすること。
    //     nvgpudecode:  ringSize(4)          → K <= 3
    //     CPUデコード系: YuvPlaneRing(3)      → K <= 2
    //     よって 2 が両方を満たす安全側の値。
    // ==========================================================
    static const int kMaxFramesInFlight = 2;
    cudaEvent_t frameDone     [kMaxFramesInFlight] = {};
    bool        frameDoneValid[kMaxFramesInFlight] = {};
    int         frameDoneNo = 0;

    //OpenGL周り
    bool initialize_completed_flag=false;
    GLuint inputTextureID;  // ← 入力用
    GLuint fboTextureID = 0;    // ← 出力先（FBOバインド用）
    GLuint tempTextureID = 0;
    GLuint vbo_hist = 0;
    GLuint fbo = 0;
    GLuint vao = 0;
    GLuint vbo = 0;
    GLuint tempfbo = 0;
    QPainter painter;

    //シェーダ―
    QOpenGLShaderProgram Sobel_program;
    QOpenGLShaderProgram Gaussian_program;
    QOpenGLShaderProgram Averaging_program;
    struct shader{
        GLuint progId=0;
        GLint loc_tex           = 0;
        GLint loc_texelSize     = 0;
        GLint loc_filterEnabled =0;
    };
    shader Sobel_shader;
    shader Gaussian_shader;
    shader Averaging_shader;

    // 描画領域を計算
    // ★ paintGL は uploadToGLTexture より先に走りうるので必ず0初期化する
    float monitor_scaling=1;
    GLint viewportWidth  = 0;
    GLint viewportHeight = 0;
    GLint x0 = 0, y0 = 0, x1 = 0, y1 = 0;

    // ズーム（viewCx/viewCy は元画像座標の表示中心、左上原点）
    double zoomFactor = 1.0;
    double viewCx = 0.0;
    double viewCy = 0.0;
    bool   panning = false;
    Qt::MouseButton panButton = Qt::NoButton;
    QPointF panLastPos;

    double zoomMax() const;
    void   clampView();
    void   visibleSrcRect(double& sx, double& sy, double& sw, double& sh) const;
    QPointF windowToDisplayUV(const QPointF& pos) const;
    void   drawZoomNavigator(QPainter& p);

    //エンコード関連
    int encode_state=STATE_NOT_ENCODE;
    int prev_encode_state=STATE_NOT_ENCODE;
    save_encode* save_encoder=nullptr;

    //動画情報
    // ★ 未初期化のまま paintGL / GLresize から参照されるのを防ぐ
    int width_ = 0, height_ = 0;
    int FrameNo = 0;
    const DecodeInfo& VideoInfo = DecodeInfoManager::getInstance().getSettings();

    //fpsタイマー
    QElapsedTimer fpsTimer;
    int fpsCount = 0;

    //ヒストグラム
    cudaStream_t hist_stream = nullptr;
    cudaEvent_t hist_event = nullptr;
    HistData* d_hist_data = nullptr;
    HistStats* d_hist_stats = nullptr;
    // ★ map に失敗した回は更新されないまま paintGL が読むので 0 初期化必須。
    //   （d_hist_* と同じ理由で、ここだけ初期化子が抜けていた）
    HistStats h_hist_stats{};
    int num_bins = 256;
};


#endif // GLWIDGET_H
