#include "src/qmlui/videoengine.h"

#include <QDebug>
#include <QDir>
#include <QFileInfo>
#include <QThread>
#include <QThreadPool>
#include <QRunnable>
#include <QOpenGLExtraFunctions>
#include <QVariantList>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <gpuimg/gpuimg.hpp>

VideoEngine* VideoEngine::s_instance = nullptr;

// スクリーンショットの保存形式（GLWidget の同名関数と同じ）
static gpuimg::EncodeParams makeScreenshotParams(QString& ext)
{
    gpuimg_compression level = GPUIMG_COMPRESSION_FAST;
    if (g_AppSettings.screenshot_mode == "normal")    level = GPUIMG_COMPRESSION_NORMAL;
    else if (g_AppSettings.screenshot_mode == "high") level = GPUIMG_COMPRESSION_HIGH;

    if (g_AppSettings.screenshot_format == "tiff") {
        ext = "tif";
        gpuimg::EncodeParams p(GPUIMG_FORMAT_TIFF);
        if (g_AppSettings.screenshot_tiff_compression == "none") p.tiffUncompressed();
        else                                                       p.tiffDeflate().compressionLevel(level);
        return p;
    }
    if (g_AppSettings.screenshot_format == "bmp") {
        ext = "bmp";
        return gpuimg::EncodeParams(GPUIMG_FORMAT_BMP);
    }
    if (g_AppSettings.screenshot_format == "jpg") {
        ext = "jpg";
        return gpuimg::EncodeParams(GPUIMG_FORMAT_JPEG)
            .jpegQuality(g_AppSettings.screenshot_jpeg_quality)
            .jpegSubsampling(g_AppSettings.screenshot_jpeg_subsampling.toInt());
    }
    ext = "png";
    gpuimg::EncodeParams p(GPUIMG_FORMAT_PNG);
    p.compressionLevel(level);
    return p;
}

VideoEngine::VideoEngine(QObject* parent)
    : QObject(parent)
{
    s_instance = this;
    CUDA_IMG_Proc = new CUDA_ImageProcess();
}

VideoEngine::~VideoEngine()
{
    waitScreenshotSaved();
    screenshotEncoder.reset();

    if (save_encoder) {
        delete save_encoder;
        save_encoder = nullptr;
    }

    const bool current = makeCurrent();

    // ★ surface / texture object は unregister より先に破棄する
    inputSurfRetire.destroy();
    fboTexRetire.destroy();
    histTexRetire.destroy();

    if (cudaResource1)           { cudaGraphicsUnregisterResource(cudaResource1);           cudaResource1 = nullptr; }
    if (cudaResource2)           { cudaGraphicsUnregisterResource(cudaResource2);           cudaResource2 = nullptr; }
    if (cudaResource_screenshot) { cudaGraphicsUnregisterResource(cudaResource_screenshot); cudaResource_screenshot = nullptr; }
    if (cudaResource_hist_draw)  { cudaGraphicsUnregisterResource(cudaResource_hist_draw);  cudaResource_hist_draw = nullptr; }

    if (interop_stream) { cudaStreamSynchronize(interop_stream); cudaStreamDestroy(interop_stream); interop_stream = nullptr; }
    if (encode_gl_event) { cudaEventDestroy(encode_gl_event); encode_gl_event = nullptr; }
    for (int i = 0; i < kMaxFramesInFlight; i++) {
        if (frameDone[i]) { cudaEventDestroy(frameDone[i]); frameDone[i] = nullptr; }
        frameDoneValid[i] = false;
    }
    if (hist_stream)      { cudaStreamSynchronize(hist_stream);      cudaStreamDestroy(hist_stream);      hist_stream = nullptr; }
    if (screenshotStream) { cudaStreamSynchronize(screenshotStream); cudaStreamDestroy(screenshotStream); screenshotStream = nullptr; }

    if (d_hist_stats)     { cudaFree(d_hist_stats);     d_hist_stats = nullptr; }
    if (d_hist_data)      { cudaFree(d_hist_data);      d_hist_data = nullptr; }
    if (d_cudaScreenshot) { cudaFree(d_cudaScreenshot); d_cudaScreenshot = nullptr; }

    if (current) {
        {
            QMutexLocker lock(&fenceMutex_);
            if (frameFence_)       { glDeleteSync(frameFence_);       frameFence_ = nullptr; }
            if (displayDoneFence_) { glDeleteSync(displayDoneFence_); displayDoneFence_ = nullptr; }
        }
        if (inputTextureID) glDeleteTextures(1, &inputTextureID);
        if (fboTextureID)   glDeleteTextures(1, &fboTextureID);
        if (tempTextureID)  glDeleteTextures(1, &tempTextureID);
        if (histTextureID)  glDeleteTextures(1, &histTextureID);
        if (fbo)            glDeleteFramebuffers(1, &fbo);
        if (tempfbo)        glDeleteFramebuffers(1, &tempfbo);
        if (histFbo)        glDeleteFramebuffers(1, &histFbo);
        if (vbo_hist)       glDeleteBuffers(1, &vbo_hist);
        if (vbo)            glDeleteBuffers(1, &vbo);
        if (ebo)            glDeleteBuffers(1, &ebo);
        if (vao)            glDeleteVertexArrays(1, &vao);
        Sobel_program.removeAllShaders();
        Gaussian_program.removeAllShaders();
        Averaging_program.removeAllShaders();
        doneCurrent();
    }

    delete CUDA_IMG_Proc;
    CUDA_IMG_Proc = nullptr;

    if (s_instance == this) s_instance = nullptr;
    qDebug() << "VideoEngine: Destructor called";
}

bool VideoEngine::makeCurrent()
{
    if (!ctx_ || !surface_) return false;
    if (QOpenGLContext::currentContext() == ctx_) return true;
    return ctx_->makeCurrent(surface_);
}

void VideoEngine::doneCurrent()
{
    if (ctx_ && QOpenGLContext::currentContext() == ctx_)
        ctx_->doneCurrent();
}

// ==============================================================
// 初期化（GLWidget::initializeGL 相当）
// ==============================================================
bool VideoEngine::initialize()
{
    if (initialize_completed_flag) return true;

    // ★ Qt Quick 側と同じフォーマット（main で既定フォーマットを設定済み）で作り、
    //   グローバル共有コンテキストと共有する。これで FBO テクスチャを
    //   シーングラフからそのまま参照できる。
    ctx_ = new QOpenGLContext(this);
    ctx_->setFormat(QSurfaceFormat::defaultFormat());
    ctx_->setShareContext(QOpenGLContext::globalShareContext());
    if (!ctx_->create()) {
        qWarning() << "[VideoEngine] QOpenGLContext::create failed";
        return false;
    }
    if (!QOpenGLContext::globalShareContext() ||
        !QOpenGLContext::areSharing(ctx_, QOpenGLContext::globalShareContext())) {
        qWarning() << "[VideoEngine] グローバル共有コンテキストと共有できていません"
                      "（AA_ShareOpenGLContexts を QApplication より前に設定すること）";
    }

    surface_ = new QOffscreenSurface(nullptr, this);
    surface_->setFormat(ctx_->format());
    surface_->create();

    if (!makeCurrent()) {
        qWarning() << "[VideoEngine] makeCurrent failed";
        return false;
    }
    if (!initializeOpenGLFunctions()) {
        qWarning() << "[VideoEngine] OpenGL 4.5 の関数を取得できません。format ="
                   << ctx_->format();
        return false;
    }
    qDebug() << "[VideoEngine] GL context:" << ctx_->format();

    // === 頂点データ（GLWidget と同じ）===
    static const float vertices[] = {
        -1.0f, -1.0f, 0.0f, 1.0f,
         1.0f, -1.0f, 1.0f, 1.0f,
         1.0f,  1.0f, 1.0f, 0.0f,
        -1.0f,  1.0f, 0.0f, 0.0f
    };
    static const GLuint indices[] = { 0, 1, 2, 2, 3, 0 };

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // === シェーダ（パスは GLWidget と同じ。作業ディレクトリ基準）===
    Sobel_program.addShaderFromSourceFile(QOpenGLShader::Vertex,   "../../project/shaders/sobel.vert");
    Sobel_program.addShaderFromSourceFile(QOpenGLShader::Fragment, "../../project/shaders/sobel.frag");
    Gaussian_program.addShaderFromSourceFile(QOpenGLShader::Vertex,   "../../project/shaders/gaussian.vert");
    Gaussian_program.addShaderFromSourceFile(QOpenGLShader::Fragment, "../../project/shaders/gaussian.frag");
    Averaging_program.addShaderFromSourceFile(QOpenGLShader::Vertex,   "../../project/shaders/averaging.vert");
    Averaging_program.addShaderFromSourceFile(QOpenGLShader::Fragment, "../../project/shaders/averaging.frag");

    if (!Sobel_program.link())     { qWarning() << "Shader link failed:" << Sobel_program.log();     return false; }
    if (!Gaussian_program.link())  { qWarning() << "Shader link failed:" << Gaussian_program.log();  return false; }
    if (!Averaging_program.link()) { qWarning() << "Shader link failed:" << Averaging_program.log(); return false; }

    auto setupShader = [this](shader& s, QOpenGLShaderProgram& p) {
        s.progId            = p.programId();
        s.loc_tex           = glGetUniformLocation(s.progId, "tex");
        s.loc_texelSize     = glGetUniformLocation(s.progId, "texelSize");
        s.loc_filterEnabled = glGetUniformLocation(s.progId, "u_filterEnabled");
    };
    setupShader(Sobel_shader, Sobel_program);
    setupShader(Gaussian_shader, Gaussian_program);
    setupShader(Averaging_shader, Averaging_program);

    // GPU 情報（ini と突き合わせ）と OpenGL を描いている CUDA デバイスの特定
    queryCudaGPUs();
    getCudaDeviceIDFromOpenGL();

    cudaStreamCreate(&interop_stream);
    cudaEventCreateWithFlags(&encode_gl_event, cudaEventDisableTiming);

    inputSurfRetire.init(CudaObjRetireRing::SURFACE);
    fboTexRetire.init(CudaObjRetireRing::TEXTURE);
    histTexRetire.init(CudaObjRetireRing::TEXTURE);

    for (int i = 0; i < kMaxFramesInFlight; i++) {
        cudaEventCreateWithFlags(&frameDone[i], cudaEventDisableTiming);
        frameDoneValid[i] = false;
    }
    cudaStreamCreate(&hist_stream);
    cudaStreamCreate(&screenshotStream);

    initHistTarget(histSize_);

    doneCurrent();

    fpsTimer.start();
    initialize_completed_flag = true;
    emit initialized();
    return true;
}

// ==============================================================
// フェンス
// ==============================================================
void VideoEngine::publishFrameFence()
{
    GLsync s = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    // ★ 他のコンテキストから待つ場合、張った側で flush しておかないと
    //   フェンスがドライバ内に留まり、待つ側が永久に待つことがある
    glFlush();

    QMutexLocker lock(&fenceMutex_);
    if (frameFence_) glDeleteSync(frameFence_);
    frameFence_ = s;
    frameFenceWaited_ = false;
}

void VideoEngine::waitDisplayDone()
{
    QMutexLocker lock(&fenceMutex_);
    if (displayDoneFence_) {
        glWaitSync(displayDoneFence_, 0, GL_TIMEOUT_IGNORED);
        glDeleteSync(displayDoneFence_);
        displayDoneFence_ = nullptr;
    }
}

// シーングラフ側（VideoItem::updatePaintNode）から呼ぶ
void VideoEngine::waitFrameFence()
{
    QOpenGLContext* cur = QOpenGLContext::currentContext();
    if (!cur || cur == ctx_) return;

    QMutexLocker lock(&fenceMutex_);
    if (frameFence_ && !frameFenceWaited_) {
        cur->extraFunctions()->glWaitSync(frameFence_, 0, GL_TIMEOUT_IGNORED);
        frameFenceWaited_ = true;
    }
}

// QQuickWindow::frameSwapped（シーングラフのコンテキストがカレント）から呼ぶ
void VideoEngine::signalDisplayDone()
{
    QOpenGLContext* cur = QOpenGLContext::currentContext();
    if (!cur || cur == ctx_) return;

    QOpenGLExtraFunctions* f = cur->extraFunctions();
    GLsync s = f->glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    f->glFlush();

    QMutexLocker lock(&fenceMutex_);
    if (displayDoneFence_) f->glDeleteSync(displayDoneFence_);
    displayDoneFence_ = s;
}

// ==============================================================
// フレーム入力（GLWidget::uploadToGLTexture 相当）
// ==============================================================
void VideoEngine::upload(const VideoFrame& Frame)
{
    if (!initialize_completed_flag) return;
    if (!makeCurrent()) return;

    // 解像度の変更に対応（寸法は 1 度だけ計算して使い回す）
    const int want_w = VideoInfo.width  * VideoInfo.width_scale;
    const int want_h = VideoInfo.height * VideoInfo.height_scale;
    if (want_w != width_ || want_h != height_) {
        initCudaTexture(want_w, want_h);
        initTextureCuda(want_w, want_h);
        initCudaScreenshot(want_w, want_h);
        setShaderUniform(want_w, want_h);
        initCudaHist();
        width_  = want_w;
        height_ = want_h;
        frame_valid_ = false;
        return;
    }

    FrameNo = Frame.FrameNo;

    if (Frame.src_mode == SRC_NV12_TILE) {
        if (Frame.tile_count <= 0 || !Frame.d_y[0]) {
            qDebug() << "入力データがNULLです(NV12 tile)";
            return;
        }
    } else if (Frame.src_mode == SRC_YUV_PLANE) {
        if (Frame.plane_format == YUVFMT_NONE || !Frame.d_plane[0]) {
            qDebug() << "入力データがNULLです(YUV plane) fmt=" << Frame.plane_format;
            return;
        }
    } else {
        qWarning() << "[VideoEngine] src_mode が未設定です:" << Frame.src_mode;
        return;
    }

    // ★ in-flight クランプ（K フレーム前の GPU 処理が終わるまで投入しない）
    const int fdIdx = frameDoneNo;
    if (frameDoneValid[fdIdx]) {
        cudaEventSynchronize(frameDone[fdIdx]);
    }

    cudaArray_t array;
    cudaError_t err = cudaGraphicsMapResources(1, &cudaResource1, interop_stream);
    if (err != cudaSuccess) {
        qDebug() << "cudaGraphicsMapResources error:" << cudaGetErrorString(err);
        return;
    }
    err = cudaGraphicsSubResourceGetMappedArray(&array, cudaResource1, 0, 0);
    if (err != cudaSuccess) {
        qDebug() << "cudaGraphicsSubResourceGetMappedArray error:" << cudaGetErrorString(err);
        cudaGraphicsUnmapResources(1, &cudaResource1, interop_stream);
        return;
    }

    {
        // ★ surface object は「この map で得た array」から毎回作る
        const int si = inputSurfRetire.acquire();
        cudaSurfaceObject_t surf = 0;
        if (!createInputSurface(array, surf)) {
            cudaGraphicsUnmapResources(1, &cudaResource1, interop_stream);
            return;
        }
        inputSurfRetire.store(si, surf);

        // ★ NVDEC の出力コピー完了を GPU 側で待つ
        if (Frame.decode_ready) {
            cudaStreamWaitEvent(interop_stream, Frame.decode_ready, 0);
        }

        if (Frame.src_mode == SRC_NV12_TILE)
            mergeToSurface(Frame, surf);
        else
            yuvToSurface(Frame, surf);

        inputSurfRetire.retire(si, interop_stream);
    }

    err = cudaGraphicsUnmapResources(1, &cudaResource1, interop_stream);
    if (err != cudaSuccess) {
        qDebug() << "cudaGraphicsUnmapResources error:" << cudaGetErrorString(err);
        return;
    }

    cudaEventRecord(frameDone[fdIdx], interop_stream);
    frameDoneValid[fdIdx] = true;
    frameDoneNo = (fdIdx + 1) % kMaxFramesInFlight;

    FBO_Rendering(Frame);

    fpsCount++;
    if (fpsTimer.elapsed() >= 1000) {
        fps = fpsCount * 1000.0 / fpsTimer.elapsed();
        fpsCount = 0;
        fpsTimer.restart();
    }
}

void VideoEngine::rerender(const VideoFrame& Frame)
{
    if (!initialize_completed_flag || width_ <= 0 || height_ <= 0 || !fbo) return;
    if (!makeCurrent()) return;
    FBO_Rendering(Frame);
}

void VideoEngine::reset()
{
    frame_valid_ = false;
    hist_valid_  = false;
    frame_serial_++;
    emit frameReady();
    emit histogramChanged();
}

// ==============================================================
// FBO レンダリング（GLWidget::FBO_Rendering 相当）
// ==============================================================
void VideoEngine::FBO_Rendering(const VideoFrame& Frame)
{
    // エンコードから通常モード移行の場合はスキップ
    if ((prev_encode_state == STATE_ENCODING) && encode_state == STATE_NOT_ENCODE) {
        prev_encode_state = encode_state;
        return;
    }
    prev_encode_state = encode_state;

    // ★ シーングラフが前のフレームを描き終えてから FBO を上書きする
    waitDisplayDone();

    if (filter_change_flag) {
        setShaderUniformEnable();
        filter_change_flag = false;
    }

    const shader* passes[3];
    int nPass = 0;
    // 適用順は従来どおり Gaussian → Sobel → Averaging
    if (g_AppSettings.gaussianfilterEnabled)  passes[nPass++] = &Gaussian_shader;
    if (g_AppSettings.sobelfilterEnabled)     passes[nPass++] = &Sobel_shader;
    if (g_AppSettings.averagingfilterEnabled) passes[nPass++] = &Averaging_shader;
    // 全部 OFF のときは素通し 1 パス（u_filterEnabled=0）
    if (nPass == 0) passes[nPass++] = &Averaging_shader;

    glBindVertexArray(vao);
    GLuint srcTex = inputTextureID;
    bool   toFbo  = ((nPass % 2) == 1);   // 最終パスが fbo に着地するように開始側を決める
    for (int i = 0; i < nPass; i++) {
        glBindFramebuffer(GL_FRAMEBUFFER, toFbo ? fbo : tempfbo);
        glViewport(0, 0, width_, height_);
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(passes[i]->progId);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, srcTex);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
        srcTex = toFbo ? fboTextureID : tempTextureID;
        toFbo  = !toFbo;
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindVertexArray(0);
    glUseProgram(0);
    glBindTexture(GL_TEXTURE_2D, 0);

    if (screenshot_flag) {
        screenshot_flag = false;
        doScreenshot(Frame);
    }

    if (encode_state == STATE_ENCODING) {
        downloadToGLTexture_and_Encode(Frame);
        return;
    }
    if (encode_state != STATE_NOT_ENCODE) return;

    // ---- 表示モード ----
    if (g_AppSettings.histgram_flag && cudaResource_hist_draw && cudaResource2) {
        histgram_Analysys();
        drawHistogram();
    } else if (hist_valid_) {
        hist_valid_ = false;
        emit histogramChanged();
    }

    frame_valid_ = true;
    frame_serial_++;
    publishFrameFence();
    emit frameReady();
}

void VideoEngine::setShaderUniformEnable()
{
    glUseProgram(Gaussian_shader.progId);
    glUniform1i(Gaussian_shader.loc_filterEnabled, g_AppSettings.gaussianfilterEnabled);
    glUseProgram(Sobel_shader.progId);
    glUniform1i(Sobel_shader.loc_filterEnabled, g_AppSettings.sobelfilterEnabled);
    glUseProgram(Averaging_shader.progId);
    glUniform1i(Averaging_shader.loc_filterEnabled, g_AppSettings.averagingfilterEnabled);
    glUseProgram(0);
}

void VideoEngine::setShaderUniform(int width, int height)
{
    const shader* all[] = { &Gaussian_shader, &Sobel_shader, &Averaging_shader };
    for (const shader* s : all) {
        glUseProgram(s->progId);
        glUniform1i(s->loc_tex, 0);
        glUniform2f(s->loc_texelSize, 1.0f / width, 1.0f / height);
    }
    glUseProgram(0);
}

// ==============================================================
// CUDA ↔ OpenGL のリソース（GLWidget と同じ）
// ==============================================================
void VideoEngine::initCudaTexture(int width, int height)
{
    inputSurfRetire.drain();

    if (cudaResource1) { cudaGraphicsUnregisterResource(cudaResource1); cudaResource1 = nullptr; }
    if (inputTextureID) { glDeleteTextures(1, &inputTextureID); inputTextureID = 0; }

    glGenTextures(1, &inputTextureID);
    glBindTexture(GL_TEXTURE_2D, inputTextureID);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);

    // ★ surface object を作るには SurfaceLoadStore で登録する
    cudaError_t err = cudaGraphicsGLRegisterImage(&cudaResource1, inputTextureID, GL_TEXTURE_2D,
                                                  cudaGraphicsRegisterFlagsSurfaceLoadStore);
    if (err != cudaSuccess) {
        qWarning() << "cudaGraphicsGLRegisterImage error:" << cudaGetErrorString(err);
        cudaResource1 = nullptr;
        glDeleteTextures(1, &inputTextureID);
        inputTextureID = 0;
    }
}

void VideoEngine::initTextureCuda(int width, int height)
{
    fboTexRetire.drain();
    histTexRetire.drain();

    if (cudaResource2) { cudaGraphicsUnregisterResource(cudaResource2); cudaResource2 = nullptr; }
    if (fboTextureID)  { glDeleteTextures(1, &fboTextureID);  fboTextureID = 0; }
    if (fbo)           { glDeleteFramebuffers(1, &fbo);       fbo = 0; }
    if (tempTextureID) { glDeleteTextures(1, &tempTextureID); tempTextureID = 0; }
    if (tempfbo)       { glDeleteFramebuffers(1, &tempfbo);   tempfbo = 0; }

    auto makeTarget = [this](GLuint& f, GLuint& t, int w, int h) {
        glGenFramebuffers(1, &f);
        glBindFramebuffer(GL_FRAMEBUFFER, f);
        glGenTextures(1, &t);
        glBindTexture(GL_TEXTURE_2D, t);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, t, 0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glBindTexture(GL_TEXTURE_2D, 0);
    };
    makeTarget(fbo, fboTextureID, width, height);
    makeTarget(tempfbo, tempTextureID, width, height);

    cudaError_t err = cudaGraphicsGLRegisterImage(&cudaResource2, fboTextureID, GL_TEXTURE_2D,
                                                  cudaGraphicsRegisterFlagsReadOnly);
    if (err != cudaSuccess) {
        qDebug() << "cudaGraphicsGLRegisterImage (fbo) error:" << cudaGetErrorString(err);
        cudaResource2 = nullptr;
    }
}

void VideoEngine::initCudaScreenshot(int width, int height)
{
    waitScreenshotSaved();

    if (d_cudaScreenshot) { cudaFree(d_cudaScreenshot); d_cudaScreenshot = nullptr; }
    if (cudaResource_screenshot) { cudaGraphicsUnregisterResource(cudaResource_screenshot); cudaResource_screenshot = nullptr; }

    cudaError_t err = cudaMallocPitch(&d_cudaScreenshot, &d_cudaScreenshotPitch, width * 4, height);
    if (err != cudaSuccess) {
        qDebug() << "cudaMallocPitch error:" << cudaGetErrorString(err);
        d_cudaScreenshot = nullptr;
        d_cudaScreenshotPitch = 0;
    }

    if (screenshotStream) {
        cudaStreamSynchronize(screenshotStream);
        cudaStreamDestroy(screenshotStream);
        screenshotStream = nullptr;
    }
    err = cudaStreamCreateWithFlags(&screenshotStream, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        qDebug() << "cudaStreamCreateWithFlags error:" << cudaGetErrorString(err);
        screenshotStream = nullptr;
    }

    // ★ GLWidget と同じく fboTextureID をもう一度登録している（読み取り専用）
    err = cudaGraphicsGLRegisterImage(&cudaResource_screenshot, fboTextureID, GL_TEXTURE_2D,
                                      cudaGraphicsRegisterFlagsReadOnly);
    if (err != cudaSuccess) {
        qDebug() << "cudaGraphicsGLRegisterImage (screenshot) error:" << cudaGetErrorString(err);
        cudaResource_screenshot = nullptr;
    }
}

void VideoEngine::initCudaHist()
{
    histTexRetire.drain();

    if (cudaResource_hist_draw) { cudaGraphicsUnregisterResource(cudaResource_hist_draw); cudaResource_hist_draw = nullptr; }
    if (vbo_hist) { glDeleteBuffers(1, &vbo_hist); vbo_hist = 0; }
    if (d_hist_stats) { cudaFree(d_hist_stats); d_hist_stats = nullptr; }
    if (d_hist_data)  { cudaFree(d_hist_data);  d_hist_data = nullptr; }

    glGenBuffers(1, &vbo_hist);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_hist);
    glBufferData(GL_ARRAY_BUFFER, num_bins * 3 * 3 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    cudaError_t err = cudaGraphicsGLRegisterBuffer(&cudaResource_hist_draw, vbo_hist,
                                                   cudaGraphicsRegisterFlagsWriteDiscard);
    if (err != cudaSuccess) {
        qDebug() << "cudaGraphicsGLRegisterBuffer failed:" << cudaGetErrorString(err);
        cudaResource_hist_draw = nullptr;
    }

    cudaMalloc(&d_hist_stats, sizeof(HistStats));
    cudaMalloc(&d_hist_data, sizeof(HistData));
}

void VideoEngine::setHistViewport(const QSize& px, qreal dpr)
{
    // 上限は 4K 全面に出しても足りる程度にしておく
    const QSize s(std::clamp(px.width(), 64, 4096), std::clamp(px.height(), 32, 4096));
    histWanted_ = s;
    histDpr_ = std::max<qreal>(1.0, dpr);
}

// ヒストグラムのプロット先（表示サイズに合わせて作る）
void VideoEngine::initHistTarget(const QSize& size)
{
    if (histTextureID) { glDeleteTextures(1, &histTextureID); histTextureID = 0; }
    if (histFbo)       { glDeleteFramebuffers(1, &histFbo);   histFbo = 0; }
    histSize_ = size;

    glGenFramebuffers(1, &histFbo);
    glBindFramebuffer(GL_FRAMEBUFFER, histFbo);
    glGenTextures(1, &histTextureID);
    glBindTexture(GL_TEXTURE_2D, histTextureID);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, size.width(), size.height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, histTextureID, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
}

bool VideoEngine::mergeToSurface(const VideoFrame& Frame, cudaSurfaceObject_t surf)
{
    const int outW = width_;
    const int outH = height_;
    const int srcW = Frame.tile_w;
    const int srcH = Frame.tile_h;

    uint8_t* y [VideoFrame::MAX_TILES];
    uint8_t* uv[VideoFrame::MAX_TILES];
    size_t   yp[VideoFrame::MAX_TILES];
    size_t   up[VideoFrame::MAX_TILES];
    for (int i = 0; i < VideoFrame::MAX_TILES; i++) {
        y [i] = Frame.d_y[i];
        uv[i] = Frame.d_uv[i];
        yp[i] = Frame.y_pitch[i];
        up[i] = Frame.uv_pitch[i];
    }

    switch (Frame.tile_count) {
    case 1:
        if (VideoInfo.bitdepth == 10)
            return CUDA_IMG_Proc->NV12_to_RGBA_10bit_surf(surf, y[0], yp[0], uv[0], up[0], outW, outH, interop_stream);
        return CUDA_IMG_Proc->NV12_to_RGBA_8bit_surf(surf, y[0], yp[0], uv[0], up[0], outW, outH, interop_stream);
    case 2:
        return CUDA_IMG_Proc->nv12x2_to_rgba_merge_surf(
            surf,
            y[0], yp[0], uv[0], up[0],
            y[1], yp[1], uv[1], up[1],
            outW, outH, srcW, srcH, interop_stream);
    case 4:
        return CUDA_IMG_Proc->nv12x4_to_rgba_merge_surf(
            surf,
            y[0], yp[0], uv[0], up[0],
            y[1], yp[1], uv[1], up[1],
            y[2], yp[2], uv[2], up[2],
            y[3], yp[3], uv[3], up[3],
            outW, outH, srcW, srcH, interop_stream);
    case 8:
        return CUDA_IMG_Proc->nv12x8_to_rgba_merge_surf(
            surf,
            y[0], yp[0], uv[0], up[0],
            y[1], yp[1], uv[1], up[1],
            y[2], yp[2], uv[2], up[2],
            y[3], yp[3], uv[3], up[3],
            y[4], yp[4], uv[4], up[4],
            y[5], yp[5], uv[5], up[5],
            y[6], yp[6], uv[6], up[6],
            y[7], yp[7], uv[7], up[7],
            outW, outH, srcW, srcH, interop_stream);
    default:
        qWarning() << "[VideoEngine] unsupported tile count:" << Frame.tile_count;
        return false;
    }
}

bool VideoEngine::yuvToSurface(const VideoFrame& Frame, cudaSurfaceObject_t surf)
{
    const int W = width_;
    const int H = height_;

    uint8_t* p [3];
    size_t   pp[3];
    for (int i = 0; i < 3; i++) {
        p [i] = Frame.d_plane[i];
        pp[i] = Frame.plane_pitch[i];
    }

    switch (Frame.plane_format) {
    case YUVFMT_YUV420P_8:
        return CUDA_IMG_Proc->yuv420p_to_RGBA_8bit_surf(surf, p[0], pp[0], p[1], pp[1], p[2], pp[2], W, H, interop_stream);
    case YUVFMT_YUV420P_10:
        return CUDA_IMG_Proc->yuv420p_to_RGBA_10bit_surf(surf, p[0], pp[0], p[1], pp[1], p[2], pp[2], W, H,
                                                         Frame.plane_is_be, interop_stream);
    case YUVFMT_YUV422P_8:
        return CUDA_IMG_Proc->yuv422p_to_RGBA_8bit_surf(surf, p[0], pp[0], p[1], pp[1], p[2], pp[2], W, H, interop_stream);
    case YUVFMT_YUV422P_10:
        return CUDA_IMG_Proc->yuv422p_to_RGBA_10bit_surf(surf, p[0], pp[0], p[1], pp[1], p[2], pp[2], W, H,
                                                         Frame.plane_is_be, interop_stream);
    case YUVFMT_UYVY422_8:
        return CUDA_IMG_Proc->uyvy422_to_RGBA_8bit_surf(surf, p[0], pp[0], W, H, interop_stream);
    case YUVFMT_GBRP_10:
        // ★ data[0]=G, data[1]=B, data[2]=R → カーネルは r, g, b の順
        return CUDA_IMG_Proc->rgb_to_RGBA_10bit_surf(surf,
                                                     p[2], pp[2],
                                                     p[0], pp[0],
                                                     p[1], pp[1],
                                                     W, H, interop_stream);
    default:
        qWarning() << "[VideoEngine] unsupported plane format:" << Frame.plane_format;
        return false;
    }
}

bool VideoEngine::createInputSurface(cudaArray_t array, cudaSurfaceObject_t& out)
{
    out = 0;
    cudaResourceDesc resDesc = {};
    resDesc.resType         = cudaResourceTypeArray;
    resDesc.res.array.array = array;

    cudaError_t err = cudaCreateSurfaceObject(&out, &resDesc);
    if (err != cudaSuccess) {
        qDebug() << "cudaCreateSurfaceObject error:" << cudaGetErrorString(err);
        out = 0;
        return false;
    }
    return true;
}

bool VideoEngine::createArrayTexture(cudaArray_t array, cudaTextureObject_t& out, const char* tag)
{
    out = 0;
    cudaResourceDesc resDesc = {};
    resDesc.resType         = cudaResourceTypeArray;
    resDesc.res.array.array = array;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0]   = cudaAddressModeClamp;
    texDesc.addressMode[1]   = cudaAddressModeClamp;
    texDesc.filterMode       = cudaFilterModePoint;
    texDesc.readMode         = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaError_t err = cudaCreateTextureObject(&out, &resDesc, &texDesc, nullptr);
    if (err != cudaSuccess) {
        qDebug() << "cudaCreateTextureObject(" << tag << ") error:" << cudaGetErrorString(err);
        out = 0;
        return false;
    }
    return true;
}

// ==============================================================
// エンコード（GLWidget::downloadToGLTexture_and_Encode と同じ）
// ==============================================================
void VideoEngine::downloadToGLTexture_and_Encode(const VideoFrame& Frame)
{
    if (!cudaResource2) {
        qDebug() << "cudaResource2 is nullptr, can't map";
        return;
    }

    cudaArray_t array;
    cudaError_t err = cudaGraphicsMapResources(1, &cudaResource2, interop_stream);
    if (err != cudaSuccess) {
        qDebug() << "cudaGraphicsMapResources error:" << cudaGetErrorString(err);
        return;
    }
    err = cudaGraphicsSubResourceGetMappedArray(&array, cudaResource2, 0, 0);
    if (err != cudaSuccess) {
        qDebug() << "cudaGraphicsSubResourceGetMappedArray error:" << cudaGetErrorString(err);
        cudaGraphicsUnmapResources(1, &cudaResource2, interop_stream);
        return;
    }

    // ★ GL → CUDA の順序保証をイベントで save_encode の st へ渡す
    cudaEventRecord(encode_gl_event, interop_stream);

    const int ti = fboTexRetire.acquire();
    cudaTextureObject_t tex = 0;
    if (!createArrayTexture(array, tex, "fbo")) {
        cudaGraphicsUnmapResources(1, &cudaResource2, interop_stream);
        return;
    }
    fboTexRetire.store(ti, tex);

    bool finished = false;
    cudaEvent_t convertDone = nullptr;
    if (save_encoder != nullptr && encode_FrameCount <= (MaxFrame - MinFrame)) {
        save_encoder->encode(Frame, tex, encode_gl_event, &convertDone);
        encode_FrameCount++;
    } else {
        finished = true;
    }

    // ★ 変換完了を interop_stream に待たせてから unmap する
    if (convertDone) {
        cudaStreamWaitEvent(interop_stream, convertDone, 0);
    }
    fboTexRetire.retire(ti, interop_stream);

    err = cudaGraphicsUnmapResources(1, &cudaResource2, interop_stream);
    if (err != cudaSuccess) {
        qDebug() << "cudaGraphicsUnmapResources error:" << cudaGetErrorString(err);
        return;
    }

    if (finished) {
        delete save_encoder;
        save_encoder = nullptr;
        encode_FrameCount = 0;
        emit encode_finished();
    }
}

void VideoEngine::encode_mode(int flag)
{
    encode_state = flag;

    if (flag == STATE_ENCODING && save_encoder == nullptr) {
        // ★ save_encode のコンストラクタは VRAM 不足などで例外を投げる
        try {
            save_encoder = new save_encode(height_, width_);
        }
        catch (const std::exception& e) {
            save_encoder = nullptr;
            encode_state = STATE_NOT_ENCODE;
            const QString msg = QString::fromUtf8(e.what());
            qWarning() << "[VideoEngine] save_encode init failed:" << msg;
            emit encode_error(msg);
        }
        catch (...) {
            save_encoder = nullptr;
            encode_state = STATE_NOT_ENCODE;
            qWarning() << "[VideoEngine] save_encode init failed (unknown exception)";
            emit encode_error(QStringLiteral("不明なエラー"));
        }
    }
}

// ==============================================================
// スクリーンショット（GLWidget::FBO_Rendering 内の処理を切り出したもの）
// ==============================================================
void VideoEngine::waitScreenshotSaved()
{
    while (screenshotSaving.load())
        QThread::msleep(1);
}

void VideoEngine::doScreenshot(const VideoFrame& Frame)
{
    bool idle = false;
    if (!screenshotSaving.compare_exchange_strong(idle, true)) {
        qDebug() << "前のキャプチャを保存中のため今回は見送り";
        return;
    }

    // ★ 保存を開始できなかった経路でも必ず screenshot_finished を出す
    auto abortScreenshot = [this](const char* why) {
        qWarning() << "スクリーンショット中止:" << why;
        screenshotSaving.store(false);
        emit screenshot_finished();
    };

    if (!screenshotEncoder) {
        try {
            screenshotEncoder = std::make_unique<gpuimg::Encoder>();
        } catch (const std::exception& e) {
            qWarning() << "gpuimg::Encoder 作成失敗:" << e.what();
        }
    }

    if (!d_cudaScreenshot || !cudaResource_screenshot || !screenshotStream || !screenshotEncoder) {
        abortScreenshot("未初期化");
        return;
    }

    bool ok = false;
    if (cudaGraphicsMapResources(1, &cudaResource_screenshot, screenshotStream) == cudaSuccess) {
        cudaArray_t cudaArray = nullptr;
        ok = cudaGraphicsSubResourceGetMappedArray(&cudaArray, cudaResource_screenshot, 0, 0) == cudaSuccess
          && cudaMemcpy2DFromArrayAsync(d_cudaScreenshot, d_cudaScreenshotPitch, cudaArray, 0, 0,
                                        static_cast<size_t>(width_) * 4, height_,
                                        cudaMemcpyDeviceToDevice, screenshotStream) == cudaSuccess;
        cudaGraphicsUnmapResources(1, &cudaResource_screenshot, screenshotStream);
    }
    if (!ok) {
        abortScreenshot(cudaGetErrorString(cudaGetLastError()));
        return;
    }

    QString saveDir = g_AppSettings.saveScreenshot_folder;
    if (saveDir.isEmpty() || !QDir(saveDir).exists())
        saveDir = QDir::homePath();
    const QString baseName = QFileInfo(g_AppSettings.decode_path).completeBaseName();

    QString ext;
    gpuimg::EncodeParams params = makeScreenshotParams(ext);
    params.stream(screenshotStream)   // 上のコピー完了を GPU 側で待ってから圧縮
          .flipVertical();            // OpenGL は下の行から並ぶので上下反転して保存
    const QString savePath = QDir(saveDir).filePath(
        QString("screenshot_%1_No.%2.%3").arg(baseName).arg(Frame.FrameNo).arg(ext));

    // ★ pitch は cudaMallocPitch が返した値を使う
    const gpuimg::Image img = gpuimg::Image::rgba8(d_cudaScreenshot, width_, height_, d_cudaScreenshotPitch);
    gpuimg::Encoder* enc = screenshotEncoder.get();

    QThreadPool::globalInstance()->start(QRunnable::create(
        [this, enc, img, params, savePath]() {
            try {
                gpuimg_stats st = enc->save(savePath.toUtf8().constData(), img, params);
                qDebug() << "screenshot saved:" << savePath << st.seconds << "s";
            } catch (const std::exception& e) {
                qWarning() << "screenshot:" << e.what();
            }
            // ★ emit を先に（store(false) の瞬間にデストラクタが進みうる）
            emit screenshot_finished();
            screenshotSaving.store(false);
        }));
}

// ==============================================================
// ヒストグラム
// ==============================================================
void VideoEngine::histgram_Analysys()
{
    if (!cudaResource2 || !cudaResource_hist_draw) return;

    // ★ ストリームは 0（calc_histgram / histgram_normalize が既定ストリームで走る）
    cudaGraphicsResource* resources[] = { cudaResource2, cudaResource_hist_draw };
    cudaError_t err = cudaGraphicsMapResources(2, resources, 0);
    if (err != cudaSuccess) {
        qDebug() << "cudaGraphicsMapResources error:" << cudaGetErrorString(err);
        return;
    }

    {
        cudaArray_t cuArray;
        err = cudaGraphicsSubResourceGetMappedArray(&cuArray, cudaResource2, 0, 0);
        if (err != cudaSuccess) {
            qDebug() << "cudaGraphicsSubResourceGetMappedArray error:" << cudaGetErrorString(err);
            cudaGraphicsUnmapResources(2, resources, 0);
            return;
        }
        const int hi = histTexRetire.acquire();
        cudaTextureObject_t htex = 0;
        if (!createArrayTexture(cuArray, htex, "hist")) {
            cudaGraphicsUnmapResources(2, resources, 0);
            return;
        }
        histTexRetire.store(hi, htex);

        cudaMemset(d_hist_data, 0, sizeof(HistData));
        CUDA_IMG_Proc->calc_histgram(d_hist_data, htex, width_, height_);
        histTexRetire.retire(hi, 0);
    }

    {
        CUDA_IMG_Proc->histogram_status(d_hist_data, d_hist_stats);
        cudaMemcpyAsync(&h_hist_stats, d_hist_stats, sizeof(HistStats), cudaMemcpyDeviceToHost, hist_stream);
        cudaStreamSynchronize(hist_stream);   // ★ 直後に h_hist_stats を読むので同期
    }

    {
        float* d_vbo_ptr = nullptr;
        size_t vbo_size = 0;
        err = cudaGraphicsResourceGetMappedPointer((void**)&d_vbo_ptr, &vbo_size, cudaResource_hist_draw);
        if (err != cudaSuccess) {
            qDebug() << "cudaGraphicsResourceGetMappedPointer error:" << cudaGetErrorString(err);
            cudaGraphicsUnmapResources(2, resources, 0);
            return;
        }
        CUDA_IMG_Proc->histgram_normalize(d_vbo_ptr, num_bins, d_hist_data, d_hist_stats);
    }

    err = cudaGraphicsUnmapResources(2, resources, 0);
    if (err != cudaSuccess) {
        qDebug() << "cudaGraphicsUnmapResources error:" << cudaGetErrorString(err);
    }
}

// プロット本体を histFbo へ（GLWidget::paintGL のヒストグラム部分）
//   背景・目盛りの数字は QML 側。ここはグリッド線と RGB の線だけ。
//   ★ 透明背景に描くので、アルファは premultiplied で積む
//     （シーングラフは premultiplied として合成する）。
void VideoEngine::drawHistogram()
{
    hist_labels_ = make_nice_y_labels(h_hist_stats.max_y_axis);
    const bool ok = (h_hist_stats.max_y_axis > 0) && (hist_labels_.size() >= 3);

    if (!ok) {
        if (hist_valid_) { hist_valid_ = false; emit histogramChanged(); }
        return;
    }

    // 表示側のサイズが変わっていたら作り直す（ここは engine のコンテキストがカレント）
    if (histWanted_ != histSize_) initHistTarget(histWanted_);
    glBindFramebuffer(GL_FRAMEBUFFER, histFbo);
    glViewport(0, 0, histSize_.width(), histSize_.height());
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(0);
    glBindVertexArray(0);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glEnable(GL_BLEND);
    // ★ 線の太さを小数で効かせるためにアンチエイリアス線にする
    //   （非 AA の線は整数ピクセルに丸められ、小さい表示で太く見える）
    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

    // ★ 線の太さは表示サイズに比例させる（論理幅 300px で 0.8、600px 以上で 1.6）
    const float plotW = float(histSize_.width() / histDpr_);
    const float lineScale = std::clamp(plotW / 375.0f, 0.8f, 1.6f);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

    const float maxY = (float)h_hist_stats.max_y_axis;
    const float l0 = (float)hist_labels_[0] / maxY;
    const float l1 = (float)hist_labels_[1] / maxY;
    const float l2 = (float)hist_labels_[2] / maxY;

    // --- グリッド ---
    glLineWidth(float(0.6 * lineScale * histDpr_));
    glColor4f(1.0f, 1.0f, 1.0f, 0.14f);
    glBegin(GL_LINES);
    const float xs[] = { 0.25f, 0.5f, 0.75f };
    for (float x : xs) { glVertex2f(x, 0.0f); glVertex2f(x, 1.0f); }
    const float ys[] = { l0, l1, l2 };
    for (float y : ys) { if (y <= 1.0f) { glVertex2f(0.0f, y); glVertex2f(1.0f, y); } }
    glEnd();

    // --- RGB ---
    glBindBuffer(GL_ARRAY_BUFFER, vbo_hist);
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, 0);

    glLineWidth(float(lineScale * histDpr_));
    glColor4f(1.00f, 0.48f, 0.48f, 0.95f);
    glDrawArrays(GL_LINE_STRIP, 0, num_bins);
    glColor4f(0.43f, 0.91f, 0.63f, 0.95f);
    glDrawArrays(GL_LINE_STRIP, num_bins, num_bins);
    glColor4f(0.48f, 0.72f, 1.00f, 0.95f);
    glDrawArrays(GL_LINE_STRIP, num_bins * 2, num_bins);

    glDisableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisable(GL_LINE_SMOOTH);
    glDisable(GL_BLEND);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    hist_valid_ = true;
    emit histogramChanged();
}

QVariantMap VideoEngine::histStats() const
{
    QVariantMap m;
    m["valid"] = hist_valid_;
    m["minR"] = h_hist_stats.min_r; m["maxR"] = h_hist_stats.max_r; m["avgR"] = h_hist_stats.avg_r;
    m["minG"] = h_hist_stats.min_g; m["maxG"] = h_hist_stats.max_g; m["avgG"] = h_hist_stats.avg_g;
    m["minB"] = h_hist_stats.min_b; m["maxB"] = h_hist_stats.max_b; m["avgB"] = h_hist_stats.avg_b;
    m["maxY"] = h_hist_stats.max_y_axis;
    QVariantList labels;
    for (int i = 0; i < (int)hist_labels_.size() && i < 3; i++)
        if (hist_labels_[i] <= h_hist_stats.max_y_axis) labels << hist_labels_[i];
    m["labels"] = labels;
    return m;
}

// 人が見て気持ちいいY軸ラベル（GLWidget と同じ）
std::vector<int> VideoEngine::make_nice_y_labels(int max_value)
{
    const int num_divs = 3;
    std::vector<int> labels;
    if (max_value <= 0) return labels;

    double raw_step = static_cast<double>(max_value) / num_divs;
    double magnitude = pow(10.0, floor(log10(raw_step)));
    double normalized = raw_step / magnitude;

    double nice_factor = 1.0;
    if (normalized < 1.5)      nice_factor = 1.0;
    else if (normalized < 3.0) nice_factor = 2.0;
    else if (normalized < 7.0) nice_factor = 5.0;
    else                       nice_factor = 10.0;

    double nice_step = nice_factor * magnitude;
    for (double v = nice_step; v < max_value; v += nice_step)
        labels.push_back(static_cast<int>(v));
    if (labels.empty() || labels.back() < max_value)
        labels.push_back(static_cast<int>(ceil(max_value / nice_step) * nice_step));
    while (labels.size() < 3)
        labels.push_back(labels.empty() ? max_value : labels.back());
    return labels;
}

// ==============================================================
// GPU 情報（GLWidget と同じ）
// ==============================================================
void VideoEngine::queryCudaGPUs()
{
    std::vector<GPUInfo> savedGPUInfo = g_GPUInfo;
    std::vector<GPUInfo> currentGPUInfo;

    int count = 0;
    cudaGetDeviceCount(&count);
    for (int i = 0; i < count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        GPUInfo info;
        info.deviceID = i;
        info.deviceName = QString::fromUtf8(prop.name);
        info.CC_major = prop.major;
        info.CC_minor = prop.minor;
        info.Max_Memory_Usage = (int)(prop.totalGlobalMem / (1024 * 1024));
        cudaDeviceGetAttribute(&info.pciDomain, cudaDevAttrPciDomainId, i);
        cudaDeviceGetAttribute(&info.pciBus, cudaDevAttrPciBusId, i);
        cudaDeviceGetAttribute(&info.pciDevice, cudaDevAttrPciDeviceId, i);
        info.tile_weight = 10;
        info.openglEnable = false;
        currentGPUInfo.push_back(info);
    }

    auto same = [](const GPUInfo& a, const GPUInfo& b) {
        return a.pciDomain == b.pciDomain && a.pciBus == b.pciBus && a.pciDevice == b.pciDevice &&
               a.deviceName == b.deviceName && a.CC_major == b.CC_major && a.CC_minor == b.CC_minor &&
               a.Max_Memory_Usage == b.Max_Memory_Usage;
    };

    bool allMatch = (savedGPUInfo.size() == currentGPUInfo.size());
    if (allMatch) {
        for (const auto& cur : currentGPUInfo) {
            bool found = false;
            for (const auto& saved : savedGPUInfo) {
                if (same(saved, cur)) { found = true; break; }
            }
            if (!found) { allMatch = false; break; }
        }
    }

    g_GPUInfo.clear();
    for (auto& cur : currentGPUInfo) {
        if (allMatch) {
            for (const auto& saved : savedGPUInfo) {
                if (same(saved, cur)) {
                    cur.tile_weight = saved.tile_weight;
                    cur.openglEnable = saved.openglEnable;
                    break;
                }
            }
        } else {
            cur.tile_weight = 10;
            cur.openglEnable = false;
        }
        g_GPUInfo.push_back(cur);
    }
    qDebug() << (allMatch ? "[GPU MATCH] All GPUs matched -> restored tile_weight"
                          : "[GPU MISMATCH] Partial match -> reset ALL tile_weight=10");
}

void VideoEngine::getCudaDeviceIDFromOpenGL()
{
    unsigned int count = 0;
    int devices[8] = {};
    cudaError_t err = cudaGLGetDevices(&count, devices, 8, cudaGLDeviceListCurrentFrame);
    if (err != cudaSuccess) {
        qDebug() << "cudaGLGetDevices failed:" << cudaGetErrorString(err);
        return;
    }
    if (count == 0 || devices[0] < 0) {
        qDebug() << "No CUDA device associated with OpenGL context.";
        return;
    }

    const int oglDev = devices[0];
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, oglDev);
    GPUInfo ogl;
    ogl.deviceName = QString::fromUtf8(prop.name);
    ogl.CC_major = prop.major;
    ogl.CC_minor = prop.minor;
    ogl.Max_Memory_Usage = (int)(prop.totalGlobalMem / (1024 * 1024));
    cudaDeviceGetAttribute(&ogl.pciDomain, cudaDevAttrPciDomainId, oglDev);
    cudaDeviceGetAttribute(&ogl.pciBus, cudaDevAttrPciBusId, oglDev);
    cudaDeviceGetAttribute(&ogl.pciDevice, cudaDevAttrPciDeviceId, oglDev);

    auto same = [](const GPUInfo& a, const GPUInfo& b) {
        return a.pciDomain == b.pciDomain && a.pciBus == b.pciBus && a.pciDevice == b.pciDevice &&
               a.deviceName == b.deviceName && a.CC_major == b.CC_major && a.CC_minor == b.CC_minor &&
               a.Max_Memory_Usage == b.Max_Memory_Usage;
    };

    // 前回 ini で openglEnable=true だった GPU と比べ、変わっていれば重みを初期化
    bool hasSaved = false, oglChanged = false;
    for (const auto& gpu : g_GPUInfo) {
        if (gpu.openglEnable) { hasSaved = true; oglChanged = !same(gpu, ogl); break; }
    }

    for (auto& gpu : g_GPUInfo) gpu.openglEnable = false;

    bool found = false;
    for (auto& gpu : g_GPUInfo) {
        if (same(gpu, ogl)) {
            found = true;
            gpu.openglEnable = true;
            g_openglDeviceID = gpu.deviceID;
            if (hasSaved && oglChanged) {
                qDebug() << "[INFO] OpenGL GPU changed -> Reset ALL tile_weight = 10";
                for (auto& g : g_GPUInfo) g.tile_weight = 10;
            }
            cudaSetDevice(g_openglDeviceID);
            qDebug() << "Matched OpenGL GPU: deviceID =" << gpu.deviceID << gpu.deviceName;
            break;
        }
    }
    if (!found) {
        qDebug() << "[ERROR] OpenGL GPU match not found in g_GPUInfo:" << ogl.deviceName;
    }
}
