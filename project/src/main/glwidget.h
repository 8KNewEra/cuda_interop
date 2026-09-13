#ifndef GLWIDGET_H
#define GLWIDGET_H

#include "qelapsedtimer.h"
#include "src/videoprocess/save_encode.h"
#include "src/imageprocess/cuda_imageprocess.h"
#include "src/main/__global__.h"
#include <QOpenGLFunctions_4_5_Core>
#pragma once
#include <QOpenGLWindow>
#include <QOpenGLShader>
#include <QOpenGLShaderProgram>
#include <QOpenGLFunctions>
#include <QPainter>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <QTimer>

class GLWidget : public QOpenGLWindow, protected QOpenGLFunctions_4_5_Core
{
    Q_OBJECT
signals:
    void decode_please();
    void initialized();
    void encode_finished();

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

protected:
    void initializeGL() override;
    void paintGL() override;

private:
    void initTextureCuda(int width,int height);
    void initCudaTexture(int width,int height);
    void initCudaMalloc(int width,int height);
    void setShaderUniform(int width,int height);
    void initCudaHist();
    void histgram_Analysys();
    void downloadToGLTexture_and_Encode(VideoFrame Frame);

    // ★ NV12タイル → OpenGLテクスチャ へ surface object 経由で直接 merge
    //   （d_decode_rgba と cudaMemcpy2DToArray を経由しないゼロコピー経路）
    bool mergeToSurface(const VideoFrame& Frame, cudaSurfaceObject_t surf);
    bool yuvToSurface(const VideoFrame& Frame, cudaSurfaceObject_t surf);

    // ==========================================================
    // ★ surface / texture object のキャッシュ
    //
    //   毎フレーム Create/Destroy していたが、これには2つ問題がある。
    //     ・API呼び出しのコストを毎フレーム払う
    //     ・「破棄する前にカーネル完了を待つ」必要があるため、
    //       GUIスレッドでの CPU同期を外せない
    //
    //   cudaArray のハンドルは map/unmap を跨いで実際には安定しているが
    //   仕様上の保証はない。そこで前回の値と比較し、
    //   変わったときだけ作り直す。これなら変わっても安全。
    // ==========================================================
    bool ensureInputSurface(cudaArray_t array);   // cudaResource1（入力テクスチャ）
    bool ensureFboTexture  (cudaArray_t array);   // cudaResource2（FBO）
    bool ensureHistTexture (cudaArray_t array);   // cudaResource_hist（FBO・ヒストグラム読み出し）
    void invalidateInputSurface();
    void invalidateFboTexture();
    void invalidateHistTexture();

    cudaSurfaceObject_t inputSurf        = 0;
    cudaArray_t         inputSurfArray   = nullptr;
    cudaTextureObject_t fboTex           = 0;
    cudaArray_t         fboTexArray      = nullptr;
    cudaTextureObject_t histTex          = 0;
    cudaArray_t         histTexArray     = nullptr;

    std::vector<int> make_nice_y_labels(int max_value);
    void queryCudaGPUs();
    void getCudaDeviceIDFromOpenGL();

    //CUDA Interop
    cudaGraphicsResource* cudaResource1;
    cudaGraphicsResource* cudaResource2;
    cudaGraphicsResource* cudaResource_hist=nullptr;
    cudaGraphicsResource* cudaResource_hist_draw=nullptr;
    CUDA_ImageProcess* CUDA_IMG_Proc=nullptr;
    cudaStream_t interop_stream = nullptr;
    cudaEvent_t interop_event = nullptr;

    // ★ FBOのGL描画完了を save_encode の st へ橋渡しするイベント。
    //   interop_event と共用すると同一フレーム内で用途が衝突するので分ける。
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

    //エンコード関連
    int encode_state=STATE_NOT_ENCODE;
    int prev_encode_state=STATE_NOT_ENCODE;
    save_encode* save_encoder=nullptr;

    //動画情報
    // ★ 未初期化のまま paintGL / GLresize から参照されるのを防ぐ
    int width_ = 0, height_ = 0;
    int FrameNo=0;
    const DecodeInfo& VideoInfo = DecodeInfoManager::getInstance().getSettings();

    //fpsタイマー
    QElapsedTimer fpsTimer;
    int fpsCount = 0;

    //ヒストグラム
    cudaStream_t hist_stream = nullptr;
    cudaEvent_t hist_event = nullptr;
    HistData* d_hist_data = nullptr;
    HistStats* d_hist_stats = nullptr;
    HistData h_hist_data;
    HistStats h_hist_stats;
    int num_bins = 256;
    int line_y1,line_y2,line_y3,line_y4;

    // 音声
    bool audio_mode=false;
    QVector<QByteArray> audio_pcm{};
};


#endif // GLWIDGET_H
