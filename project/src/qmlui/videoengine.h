#ifndef VIDEOENGINE_H
#define VIDEOENGINE_H

// ==============================================================
// VideoEngine
//
//   GLWidget の「処理部」を QOpenGLWindow から切り離したもの。
//   Qt Quick 版 UI（--qml）専用。
//
//   ・自前の QOpenGLContext + QOffscreenSurface を GUI スレッドで持つ。
//     Qt Quick のコンテキストとは AA_ShareOpenGLContexts で共有している
//     ので、ここで描いた FBO テクスチャを QML 側がそのまま表示できる。
//   ・CUDA interop / GLSL フィルタ / NVENC への受け渡し / スクショ /
//     ヒストグラム集計の手順は GLWidget と同じ（コメントもなるべく残した）。
//   ・画面への描画（paintGL / QPainter の文字）は持たない。
//     表示は VideoItem / HistogramItem、文字は QML が担当する。
//
//   ★ コンテキストが 2 つあるので、GPU 上の順序はフェンスで渡す。
//     engine → QML : フレームを描き終えたら frameFence を張る。
//                    VideoItem / HistogramItem が描画前に glWaitSync する。
//     QML → engine : シーングラフが描き終えたら displayDoneFence を張る。
//                    次のフレームで FBO を上書きする前に glWaitSync する。
//     render loop は basic（GUI スレッド）に固定しているので CPU 側の
//     競合は無いが、GPU の実行順はこれが無いと保証されない。
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

    // ---- MainWindow::decode_view から呼ばれていた入口 ----
    //   upload   : GLWidget::uploadToGLTexture 相当（新しいフレーム）
    //   rerender : GLWidget::FBO_Rendering 相当（停止中の再描画）
    void upload(const VideoFrame& Frame);
    void rerender(const VideoFrame& Frame);

    // ファイルを閉じたとき（画面を黒に戻す）
    void reset();

    // エンコード状態（GLWidget::encode_mode と同じ）
    void encode_mode(int flag);

    // フィルタの ON/OFF を変えたら立てる（次の描画で uniform を更新）
    bool filter_change_flag = true;
    bool screenshot_flag = false;

    int MinFrame = 0;
    int MaxFrame = 0;
    int encode_FrameCount = 0;

    double fps = 0.0;            // OpenGL Rendering FPS（表示用）

    // ---- 表示側（Qt Quick のレンダリング中に呼ぶ）----
    bool    hasFrame()        const { return frame_valid_; }
    GLuint  displayTexture()  const { return fboTextureID; }
    QSize   displaySize()     const { return QSize(width_, height_); }
    GLuint  histTexture()     const { return histTextureID; }
    QSize   histSize()        const { return histSize_; }
    // ★ HistogramItem の実ピクセルサイズ。次の描画でプロット先をこの大きさに作り直す
    //   （固定サイズを拡大縮小すると、大きな画面でぼやけ、小さな画面で線が消える）
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
    GLuint tempTextureID  = 0;
    GLuint fbo     = 0;
    GLuint tempfbo = 0;
    GLuint vao = 0;
    GLuint vbo = 0;
    GLuint ebo = 0;

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
    struct shader {
        GLuint progId = 0;
        GLint loc_tex = 0;
        GLint loc_texelSize = 0;
        GLint loc_filterEnabled = 0;
    };
    shader Sobel_shader;
    shader Gaussian_shader;
    shader Averaging_shader;

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
