#include "src/main/glwidget.h"
#include <QDebug>
#include <stdexcept>
#include <cmath>
#include <algorithm>

GLWidget::GLWidget(QWindow *parent)
    :  QOpenGLWindow(NoPartialUpdate, parent),
    cudaResource1(nullptr),
    cudaResource2(nullptr),
    inputTextureID(0)
{
    if(CUDA_IMG_Proc==nullptr){
        CUDA_IMG_Proc=new CUDA_ImageProcess();
    }

}


GLWidget::~GLWidget() {
    // ★ 生存中の surface / texture object は、
    //   参照先の cudaArray を手放す（＝リソースを unregister する）前に破棄する
    inputSurfRetire.destroy();
    fboTexRetire.destroy();
    histTexRetire.destroy();

    if (cudaResource1) {
        cudaGraphicsUnregisterResource(cudaResource1);
    }
    if (cudaResource2) {
        cudaGraphicsUnregisterResource(cudaResource2);
    }
    if (inputTextureID) {
        glDeleteTextures(1, &inputTextureID);
    }
    if (fboTextureID) {
        glDeleteTextures(1, &fboTextureID);
        fboTextureID = 0;
    }
    if (fbo) {
        glDeleteFramebuffers(1, &fbo);
        fbo = 0;
    }

    if (interop_stream) {
        cudaStreamSynchronize(interop_stream);
        cudaStreamDestroy(interop_stream);
        interop_stream = nullptr;
    }
    if (encode_gl_event) {
        cudaEventDestroy(encode_gl_event);
        encode_gl_event = nullptr;
    }
    for (int i = 0; i < kMaxFramesInFlight; i++) {
        if (frameDone[i]) {
            cudaEventDestroy(frameDone[i]);
            frameDone[i] = nullptr;
        }
        frameDoneValid[i] = false;
    }
    if (hist_stream) {
        cudaStreamSynchronize(hist_stream);
        cudaStreamDestroy(hist_stream);
        hist_stream = nullptr;
    }
    if (hist_event) {
        cudaEventDestroy(hist_event);
        hist_event = nullptr;
    }

    if (cudaResource_hist_draw) {
        cudaGraphicsUnregisterResource(cudaResource_hist_draw);
        cudaResource_hist_draw = nullptr;
    }
    if (vbo_hist != 0) {
        glDeleteBuffers(1, &vbo_hist);
        vbo_hist = 0;
    }

    //CUDA Mallocされたやつ
    if (d_hist_stats) {
        cudaFree(d_hist_stats);
        d_hist_stats = nullptr;
    }
    if (d_hist_data) {
        cudaFree(d_hist_data);
        d_hist_data = nullptr;
    }

    delete CUDA_IMG_Proc;
    CUDA_IMG_Proc=nullptr;

    qDebug() << "GLWidget: Destructor called";
}

//OpenGL初期化
void GLWidget::initializeGL()
{
    // Qtが作成したGLコンテキストで関数を初期化
    initializeOpenGLFunctions();

    // === 頂点データ設定 ===
    static const float vertices[] = {
        // 位置(x, y), テクスチャ座標(u, v)
        -1.0f, -1.0f, 0.0f, 1.0f,
        1.0f, -1.0f, 1.0f, 1.0f,
        1.0f,  1.0f, 1.0f, 0.0f,
        -1.0f,  1.0f, 0.0f, 0.0f
    };
    static const GLuint indices[] = { 0, 1, 2, 2, 3, 0 };

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    GLuint ebo = 0;
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &ebo);

    // 頂点バッファ
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // インデックスバッファ
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // 頂点属性設定
    constexpr GLint posLoc = 0;
    constexpr GLint texLoc = 1;
    glEnableVertexAttribArray(posLoc);
    glVertexAttribPointer(posLoc, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);

    glEnableVertexAttribArray(texLoc);
    glVertexAttribPointer(texLoc, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));

    glBindVertexArray(0);

    // === シェーダ読み込み ===
    Sobel_program.addShaderFromSourceFile(QOpenGLShader::Vertex, "../../project/shaders/sobel.vert");
    Sobel_program.addShaderFromSourceFile(QOpenGLShader::Fragment, "../../project/shaders/sobel.frag");
    Gaussian_program.addShaderFromSourceFile(QOpenGLShader::Vertex, "../../project/shaders/gaussian.vert");
    Gaussian_program.addShaderFromSourceFile(QOpenGLShader::Fragment, "../../project/shaders/gaussian.frag");
    Averaging_program.addShaderFromSourceFile(QOpenGLShader::Vertex, "../../project/shaders/averaging.vert");
    Averaging_program.addShaderFromSourceFile(QOpenGLShader::Fragment, "../../project/shaders/averaging.frag");

    //シェーダーリンク
    if (!Sobel_program.link()) {
        qDebug() << "Shader link failed:" << Sobel_program.log();
        return;
    }

    if (!Gaussian_program.link()) {
        qDebug() << "Shader compile failed:" << Gaussian_program.log();
        return;
    }

    if (!Averaging_program.link()) {
        qDebug() << "Shader compile failed:" << Averaging_program.log();
        return;
    }

    // === Uniform location 取得 ===
    //ソーベル
    Sobel_shader.progId = Sobel_program.programId();
    Sobel_shader.loc_tex           = glGetUniformLocation(Sobel_shader.progId, "tex");
    Sobel_shader.loc_texelSize     = glGetUniformLocation(Sobel_shader.progId, "texelSize");
    Sobel_shader.loc_filterEnabled = glGetUniformLocation(Sobel_shader.progId, "u_filterEnabled");

    //ガウシアン
    Gaussian_shader.progId = Gaussian_program.programId();
    Gaussian_shader.loc_tex           = glGetUniformLocation(Gaussian_shader.progId, "tex");
    Gaussian_shader.loc_texelSize     = glGetUniformLocation(Gaussian_shader.progId, "texelSize");
    Gaussian_shader.loc_filterEnabled = glGetUniformLocation(Gaussian_shader.progId, "u_filterEnabled");

    //平均化
    Averaging_shader.progId = Averaging_program.programId();
    Averaging_shader.loc_tex           = glGetUniformLocation(Averaging_shader.progId, "tex");
    Averaging_shader.loc_texelSize     = glGetUniformLocation(Averaging_shader.progId, "texelSize");
    Averaging_shader.loc_filterEnabled = glGetUniformLocation(Averaging_shader.progId, "u_filterEnabled");

    //ComputeCapability取得
    queryCudaGPUs();
    getCudaDeviceIDFromOpenGL();

    //stream
    cudaStreamCreate(&interop_stream);
    cudaEventCreateWithFlags(&encode_gl_event, cudaEventDisableTiming);

    // ★ surface / texture object の遅延破棄リング
    inputSurfRetire.init(CudaObjRetireRing::SURFACE);
    fboTexRetire.init(CudaObjRetireRing::TEXTURE);
    histTexRetire.init(CudaObjRetireRing::TEXTURE);

    for (int i = 0; i < kMaxFramesInFlight; i++) {
        cudaEventCreateWithFlags(&frameDone[i], cudaEventDisableTiming);
        frameDoneValid[i] = false;
    }
    cudaStreamCreate(&hist_stream);
    cudaEventCreate(&hist_event);

    // === 初期化完了 ===
    fpsTimer.start();
    initialize_completed_flag = true;
    emit initialized();
}

//FBOレンダリング
void GLWidget::FBO_Rendering(VideoFrame Frame){
    //エンコードから通常モード移行の場合はスキップ
    if((prev_encode_state==STATE_ENCODING)&&encode_state==STATE_NOT_ENCODE){
        prev_encode_state=encode_state;
        return;
    }
    prev_encode_state=encode_state;


    //フィルター適用チェック
    if(filter_change_flag){
        setShaderUniformEnable();
        filter_change_flag=false;
    }

    const shader* passes[3];
    int nPass = 0;

    //適用順は従来どおり Gaussian → Sobel → Averaging
    if (g_AppSettings.gaussianfilterEnabled)  passes[nPass++] = &Gaussian_shader;
    if (g_AppSettings.sobelfilterEnabled)     passes[nPass++] = &Sobel_shader;
    if (g_AppSettings.averagingfilterEnabled) passes[nPass++] = &Averaging_shader;

    //全部OFFのときは素通し1パスだけ（inputTextureID → fbo の転送が必要なため）
    if (nPass == 0) {
        passes[nPass++] = &Averaging_shader;   // u_filterEnabled=0 なので素通し
    }

    //FBO描画開始
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

    // BMP保存
    if (screenshot_flag)
    {
        // =====================================================
        // fbo の内容を CPU に取得
        // =====================================================

        glBindFramebuffer(GL_FRAMEBUFFER, fbo);

        const int width  = width_;
        const int height = height_;

        // RGBA
        std::vector<uint8_t> pixels(
            static_cast<size_t>(width) * height * 4
            );

        glPixelStorei(GL_PACK_ALIGNMENT, 1);

        glReadPixels(
            0,
            0,
            width,
            height,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            pixels.data()
            );

        // OpenGLは下→上なので上下反転
        QImage image(
            pixels.data(),
            width,
            height,
            width * 4,
            QImage::Format_RGBA8888
            );

        QImage flipped = image.mirrored(
            false,  // 左右反転なし
            true    // 上下反転
            );

        // =====================================================
        // 保存先
        // =====================================================

        QString saveDir = g_AppSettings.saveScreenshot_folder;

        if (saveDir.isEmpty() || !QDir(saveDir).exists())
        {
            saveDir = QDir::homePath();
        }

        // 元動画のファイル名（拡張子なし）
        QString baseName =
            QFileInfo(g_AppSettings.decode_path).completeBaseName();

        // screenshot_ファイル名_No.フレーム番号.bmp
        QString screenshotFileName =
            QString("screenshot_%1_No.%2.bmp")
                .arg(baseName)
                .arg(Frame.FrameNo);

        QString savePath =
            QDir(saveDir).filePath(screenshotFileName);

        // BMP保存
        bool ok = flipped.save(savePath, "BMP");

        if (ok)
        {
            qDebug() << "スクリーンショット保存:" << savePath;
        }
        else
        {
            qDebug() << "BMP保存失敗:" << savePath;
        }

        // 元のFramebufferに戻す
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        // フラグを戻す
        screenshot_flag = false;
    }

    if (encode_state==STATE_ENCODING) {
        downloadToGLTexture_and_Encode(Frame);
    } else if(encode_state==STATE_NOT_ENCODE){
        //画面に描画（paintGL を予約。実際の present は Qt が行う）
        update();
    }
}

//画面描画
void GLWidget::paintGL(){
    // ==========================================================
    // ★① 表示可能な状態かを先に確認する
    //    video_open_flag が立つのは initialized_ffmpeg 成功時だが、
    //    width_/height_/fbo が揃うのは最初の uploadToGLTexture なので、
    //    その間に paintGL が走ると不定値で blit してしまう。
    // ==========================================================
    const bool ready =
        VideoInfo.video_open_flag && fbo != 0 && width_ > 0 && height_ > 0;

    if (!ready) {
        //ファイルが閉じている / 未初期化のときは真っ黒に
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        if (viewportWidth > 0 && viewportHeight > 0)
            glViewport(0, 0, viewportWidth, viewportHeight);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        return;
    }

    // ==========================================================
    // ★② フェーズを分離する
    //    QPainter のペイントエンジンは生GL呼び出しの状態を保存/復元しない。
    //    そのため「生OpenGL描画をすべて終えてから QPainter を開く」構成にして、
    //    両者を混在させない（beginNativePainting で囲む必要もなくなる）。
    // ==========================================================

    // ----------------------------------------------------------
    // フェーズ1: 生OpenGL描画
    // ----------------------------------------------------------

    // --- 映像フレームを画面へ ---
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    glViewport(0, 0, viewportWidth, viewportHeight);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // 表示範囲を切り出して blit（FBO は y 上向きなので反転）
    double sx, sy, sw, sh;
    visibleSrcRect(sx, sy, sw, sh);
    const GLint srcX0 = static_cast<GLint>(std::floor(sx));
    const GLint srcX1 = static_cast<GLint>(std::ceil (sx + sw));
    const GLint srcY0 = static_cast<GLint>(std::floor(height_ - (sy + sh)));
    const GLint srcY1 = static_cast<GLint>(std::ceil (height_ - sy));

    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
    glBlitFramebuffer(
        srcX0, srcY0, srcX1, srcY1,
        x0, y0, x1, y1,
        GL_COLOR_BUFFER_BIT,
        GL_LINEAR
        );
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // --- ヒストグラム: CUDA集計（GL描画より先に済ませる）---
    bool histReady = false;
    std::vector<int> labels;
    QRect histRect;

    if (g_AppSettings.histgram_flag &&
        cudaResource_hist_draw && cudaResource2)
    {
        histgram_Analysys();
        labels = make_nice_y_labels(h_hist_stats.max_y_axis);

        // ★ 下で labels[0..2] を触るので、本数が足りているかをここで見る
        histReady = (h_hist_stats.max_y_axis > 0) && (labels.size() >= 3);
    }

    // --- ヒストグラムのグラフ本体（生GL）---
    if (histReady) {
        histRect = QRect(viewportWidth - (480+20)*monitor_scaling,
                         20*monitor_scaling,
                         480*monitor_scaling,
                         360*monitor_scaling);

        glViewport(histRect.x(), histRect.y(), histRect.width(), histRect.height());

        const float maxY = (float)h_hist_stats.max_y_axis;
        const float l0 = (float)labels[0] / maxY;
        const float l1 = (float)labels[1] / maxY;
        const float l2 = (float)labels[2] / maxY;

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        // 背景半透明
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glColor4f(0.0f, 0.0f, 0.0f, 0.5f);
        glBegin(GL_QUADS);
        glVertex2f(0.0f, 0.0f);
        glVertex2f(1.0f, 0.0f);
        glVertex2f(1.0f, 1.0f);
        glVertex2f(0.0f, 1.0f);
        glEnd();

        // --- グリッド ---
        glLineWidth(2.0f);
        glColor4f(1.0f, 1.0f, 1.0f, 0.45f);
        glBegin(GL_LINES);
        //X軸
        glVertex2f(0, 0.0f);      glVertex2f(0, 1.0f);
        glVertex2f(0.25f, 0.0f);  glVertex2f(0.25f, 1.0f);
        glVertex2f(0.5f, 0.0f);   glVertex2f(0.5f, 1.0f);
        glVertex2f(0.75f, 0.0f);  glVertex2f(0.75f, 1.0f);
        glVertex2f(1.0f, 0.0f);   glVertex2f(1.0f, 1.0f);
        //Y軸
        glVertex2f(0.0f, 0);      glVertex2f(1.0f, 0);
        glVertex2f(0.0f, 1.0f);   glVertex2f(1.0f, 1.0f);
        glVertex2f(0.0f, l0);     glVertex2f(1.0f, l0);
        glVertex2f(0.0f, l1);     glVertex2f(1.0f, l1);
        glVertex2f(0.0f, l2);     glVertex2f(1.0f, l2);
        // --- 目盛り ---
        glLineWidth(3.0f);
        glColor4f(1.0f, 1.0f, 1.0f, 0.8f);
        //X軸
        glVertex2f(0, 0.0f);      glVertex2f(0, 0.03f);
        glVertex2f(0.25f, 0.0f);  glVertex2f(0.25f, 0.03f);
        glVertex2f(0.5f, 0.0f);   glVertex2f(0.5f, 0.03f);
        glVertex2f(0.75f, 0.0f);  glVertex2f(0.75f, 0.03f);
        glVertex2f(1.0f, 0.0f);   glVertex2f(1.0f, 0.03f);
        //Y軸
        glVertex2f(0.0f, 0);      glVertex2f(0.03f, 0);
        glVertex2f(0.0f, 1.0f);   glVertex2f(0.03f, 1.0f);
        glVertex2f(0.0f, l0);     glVertex2f(0.03f, l0);
        glVertex2f(0.0f, l1);     glVertex2f(0.03f, l1);
        glVertex2f(0.0f, l2);     glVertex2f(0.03f, l2);
        glEnd();

        // --- ヒストグラム ---
        glBindBuffer(GL_ARRAY_BUFFER, vbo_hist);
        glEnableClientState(GL_VERTEX_ARRAY);
        glVertexPointer(3, GL_FLOAT, 0, 0);

        glLineWidth(1.4f);
        glColor4f(1.0f, 0.0f, 0.0f, 0.9f);
        glDrawArrays(GL_LINE_STRIP, 0, num_bins);
        glColor4f(0.0f, 1.0f, 0.0f, 0.9f);
        glDrawArrays(GL_LINE_STRIP, num_bins, num_bins);
        glColor4f(0.0f, 0.0f, 1.0f, 0.9f);
        glDrawArrays(GL_LINE_STRIP, num_bins * 2, num_bins);

        glDisableClientState(GL_VERTEX_ARRAY);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glDisable(GL_BLEND);

        //★ ビューポートを必ず戻す（戻さないと次の描画が壊れる）
        glViewport(0, 0, viewportWidth, viewportHeight);
    }

    // ----------------------------------------------------------
    // フェーズ2: QPainter（テキストのみ。以降は生GLを呼ばない）
    // ----------------------------------------------------------
    painter.begin(this);

    // ★ QFont の構築はフォント検索を伴うので毎フレーム作らない
    static const QFont kInfoFont("Consolas", 16);

    //ヒストグラムのラベル
    if (histReady) {
        const float maxY = (float)h_hist_stats.max_y_axis;

        //Y座標
        painter.setPen(Qt::white);
        static const QFont kAxisFont("Arial", 10);
        painter.setFont(kAxisFont);
        painter.drawText((histRect.left() - 60)/monitor_scaling,
                         (viewportHeight-25*monitor_scaling)/monitor_scaling,
                         55/monitor_scaling, 20*monitor_scaling,
                         Qt::AlignRight, QString::number(0));
        painter.drawText((histRect.left() - 200)/monitor_scaling,
                         (viewportHeight-(histRect.height())-35*monitor_scaling)/monitor_scaling,
                         195/monitor_scaling, 20*monitor_scaling,
                         Qt::AlignRight, "Max Y:"+QString::number(h_hist_stats.max_y_axis));

        for(int i=0;i<=2;i++){
            if(labels[i] <= h_hist_stats.max_y_axis){
                painter.drawText((histRect.left() - 150)/monitor_scaling,
                                 (viewportHeight-(histRect.height())*((float)labels[i]/maxY)-25*monitor_scaling)/monitor_scaling,
                                 145/monitor_scaling, 20*monitor_scaling,
                                 Qt::AlignRight, QString::number(labels[i]));
            }
        }

        //X座標
        painter.drawText(histRect.left()/monitor_scaling,(viewportHeight-19*monitor_scaling)/monitor_scaling,126, 20*monitor_scaling,Qt::AlignRight,QString::number(64));
        painter.drawText(histRect.left()/monitor_scaling,(viewportHeight-19*monitor_scaling)/monitor_scaling,250, 20*monitor_scaling,Qt::AlignRight,QString::number(128));
        painter.drawText(histRect.left()/monitor_scaling,(viewportHeight-19*monitor_scaling)/monitor_scaling,370, 20*monitor_scaling,Qt::AlignRight,QString::number(192));
        painter.drawText(histRect.left()/monitor_scaling,(viewportHeight-19*monitor_scaling)/monitor_scaling,488, 20*monitor_scaling,Qt::AlignRight,QString::number(256));

        //RGB統計値
        painter.setPen(Qt::white);
        painter.setFont(kInfoFont);
        QRect rect(viewportWidth/monitor_scaling -(325), 0, 350*monitor_scaling, 100*monitor_scaling);
        painter.drawText(rect, Qt::AlignLeft,"R: min:" + QString::number(h_hist_stats.min_r) +" max:" + QString::number(h_hist_stats.max_r) +" avg:" + QString::number(h_hist_stats.avg_r));
        painter.drawText(rect.adjusted(0, 20, 0, 0), Qt::AlignLeft,"G: min:" + QString::number(h_hist_stats.min_g) +" max:" + QString::number(h_hist_stats.max_g) +" avg:" + QString::number(h_hist_stats.avg_g));
        painter.drawText(rect.adjusted(0, 40, 0, 0), Qt::AlignLeft,"B: min:" + QString::number(h_hist_stats.min_b) +" max:" + QString::number(h_hist_stats.max_b) +" avg:" + QString::number(h_hist_stats.avg_b));
    }

    //動画情報描画
    if(g_AppSettings.videoInfo_flag){
        painter.setPen(Qt::white);
        painter.setFont(kInfoFont);
        painter.drawText(2, 20, "Primary Device(Open GL):" + g_GPUInfo[g_openglDeviceID].deviceName + " (CUDA Device " +QString::number(g_openglDeviceID) + ")" +"\n");

        for(int i=0;i<g_GPUInfo.size();i++){
            painter.drawText(2, 40 + 20*i, "CUDA Device " + QString::number(i) + ":" + g_GPUInfo[i].deviceName +"(sm_"+QString::number(g_GPUInfo[i].CC_major)+QString::number(g_GPUInfo[i].CC_minor)+")" + " Usage:" + QString::number(g_GPUInfo[i].GPU_Usage) + "% Memory:" + QString::number(g_GPUInfo[i].Memory_Usage) + "/" + QString::number(g_GPUInfo[i].Max_Memory_Usage) + " MB\n");
        }

        painter.drawText(2, 40 + 20*g_GPUInfo.size(), QString("OpenGL Rendering FPS:%1").arg(fps, 0, 'f', 2));
        painter.drawText(2, 60 + 20*g_GPUInfo.size(), "File Name:" + QString::fromStdString(VideoInfo.Name)+"\n");
        painter.drawText(2, 80 + 20*g_GPUInfo.size(), "Decorder:" + QString::fromStdString(VideoInfo.Codec)+"\n");
        painter.drawText(2, 100 + 20*g_GPUInfo.size(), VideoInfo.decode_mode);
        painter.drawText(2, 120 + 20*g_GPUInfo.size(), "Resolution:" + QString::number(width_)+"×"+QString::number(height_)+"\n");
        painter.drawText(2, 140 + 20*g_GPUInfo.size(), "Video Framerate:" + QString::number(VideoInfo.fps)+"\n");
        painter.drawText(2, 160 + 20*g_GPUInfo.size(), "Max Frame:" + QString::number(VideoInfo.max_framesNo)+"\n");
        painter.drawText(2, 180 + 20*g_GPUInfo.size(), "FrameNo:" + QString::number(FrameNo)+"\n");
        if(VideoInfo.audio)
            painter.drawText(2, 200 + 20*g_GPUInfo.size(), "Audio Channels:" + QString::number(VideoInfo.audio_channels)+"\n");
    }

    if (zoomFactor > 1.0001) {
        drawZoomNavigator(painter);
    }

    painter.end();
}

// ==============================================================
// ズーム
// ==============================================================

// 最大倍率: 表示範囲が 320x240 程度まで
double GLWidget::zoomMax() const
{
    if (width_ <= 0 || height_ <= 0) return 1.0;
    return std::max(1.0, std::min(width_ / 320.0, height_ / 240.0));
}

void GLWidget::resetZoom()
{
    zoomFactor = 1.0;
    viewCx = width_  * 0.5;
    viewCy = height_ * 0.5;
    panning = false;
    panButton = Qt::NoButton;
}

void GLWidget::clampView()
{
    zoomFactor = std::clamp(zoomFactor, 1.0, zoomMax());
    const double sw = width_  / zoomFactor;
    const double sh = height_ / zoomFactor;
    viewCx = std::clamp(viewCx, sw * 0.5, width_  - sw * 0.5);
    viewCy = std::clamp(viewCy, sh * 0.5, height_ - sh * 0.5);
}

void GLWidget::visibleSrcRect(double& sx, double& sy, double& sw, double& sh) const
{
    if (zoomFactor <= 1.0 || width_ <= 0 || height_ <= 0) {
        sx = 0; sy = 0; sw = width_; sh = height_;
        return;
    }
    sw = width_  / zoomFactor;
    sh = height_ / zoomFactor;
    sx = viewCx - sw * 0.5;
    sy = viewCy - sh * 0.5;
}

// ウィンドウ座標 → 表示枠内の 0..1
QPointF GLWidget::windowToDisplayUV(const QPointF& pos) const
{
    const double dispW = static_cast<double>(x1 - x0);
    const double dispH = static_cast<double>(y1 - y0);
    if (dispW <= 0 || dispH <= 0) return QPointF(0.5, 0.5);

    const double px = pos.x() * monitor_scaling;
    const double py = pos.y() * monitor_scaling;
    const double u = std::clamp((px - x0) / dispW, 0.0, 1.0);
    const double v = std::clamp((py - (viewportHeight - y1)) / dispH, 0.0, 1.0);
    return QPointF(u, v);
}

void GLWidget::wheelEvent(QWheelEvent *event)
{
    const bool ready = VideoInfo.video_open_flag && width_ > 0 && height_ > 0;
    if (!ready || encode_state != STATE_NOT_ENCODE) {
        event->ignore();
        return;
    }

    const int delta = event->angleDelta().y();
    if (delta == 0) { event->accept(); return; }

    // カーソル下の画像位置を固定して拡大縮小
    const QPointF uv = windowToDisplayUV(event->position());
    double sx, sy, sw, sh;
    visibleSrcRect(sx, sy, sw, sh);
    const double imgX = sx + uv.x() * sw;
    const double imgY = sy + uv.y() * sh;

    zoomFactor = std::clamp(zoomFactor * std::pow(2.0, delta / 360.0), 1.0, zoomMax());   // 3ノッチで2倍

    const double nsw = width_  / zoomFactor;
    const double nsh = height_ / zoomFactor;
    viewCx = imgX + (0.5 - uv.x()) * nsw;
    viewCy = imgY + (0.5 - uv.y()) * nsh;
    clampView();

    event->accept();
    update();
}

void GLWidget::mousePressEvent(QMouseEvent *event)
{
    const bool panBtn = (event->button() == Qt::LeftButton ||
                         event->button() == Qt::MiddleButton);
    if (panBtn && !panning && zoomFactor > 1.0001) {
        panning = true;
        panButton = event->button();
        panLastPos = event->position();
        setCursor(Qt::ClosedHandCursor);
        event->accept();
        return;
    }
    QOpenGLWindow::mousePressEvent(event);
}

void GLWidget::mouseMoveEvent(QMouseEvent *event)
{
    if (panning) {
        const double dispW = static_cast<double>(x1 - x0);
        const double dispH = static_cast<double>(y1 - y0);
        if (dispW > 0 && dispH > 0) {
            double sx, sy, sw, sh;
            visibleSrcRect(sx, sy, sw, sh);
            const QPointF d = (event->position() - panLastPos) * monitor_scaling;
            viewCx -= d.x() * (sw / dispW);
            viewCy -= d.y() * (sh / dispH);
            clampView();
            update();
        }
        panLastPos = event->position();
        event->accept();
        return;
    }
    QOpenGLWindow::mouseMoveEvent(event);
}

void GLWidget::mouseReleaseEvent(QMouseEvent *event)
{
    if (panning && event->button() == panButton) {
        panning = false;
        panButton = Qt::NoButton;
        unsetCursor();
        event->accept();
        return;
    }
    QOpenGLWindow::mouseReleaseEvent(event);
}

void GLWidget::mouseDoubleClickEvent(QMouseEvent *event)
{
    if (event->button() == Qt::LeftButton && zoomFactor > 1.0001) {
        resetZoom();
        unsetCursor();
        update();
        event->accept();
        return;
    }
    QOpenGLWindow::mouseDoubleClickEvent(event);
}

// 左下ナビ: 外枠=全体, 内枠=表示範囲
void GLWidget::drawZoomNavigator(QPainter& p)
{
    if (width_ <= 0 || height_ <= 0 || x1 <= x0 || y1 <= y0) return;

    const double dispLeft   = x0 / monitor_scaling;
    const double dispBottom = (viewportHeight - y0) / monitor_scaling;
    const double dispW      = (x1 - x0) / monitor_scaling;
    const double dispH      = (y1 - y0) / monitor_scaling;

    const double aspect = static_cast<double>(width_) / height_;
    double navW = 240.0;
    double navH = navW / aspect;
    if (navH > 160.0) { navH = 160.0; navW = navH * aspect; }
    const double fit = std::min(1.0, std::min(dispW * 0.3 / navW, dispH * 0.3 / navH));
    navW *= fit;
    navH *= fit;

    const double margin = 20.0;
    const QRectF outer(dispLeft + margin, dispBottom - margin - navH, navW, navH);

    double sx, sy, sw, sh;
    visibleSrcRect(sx, sy, sw, sh);
    const double k = navW / width_;
    QRectF inner(outer.left() + sx * k, outer.top() + sy * k, sw * k, sh * k);
    if (inner.width() < 4.0) {
        inner.setLeft(inner.center().x() - 2.0);
        inner.setWidth(4.0);
    }
    if (inner.height() < 4.0) {
        inner.setTop(inner.center().y() - 2.0);
        inner.setHeight(4.0);
    }

    p.save();
    p.setRenderHint(QPainter::Antialiasing, false);
    p.setBrush(QColor(0, 0, 0, 60));
    p.setPen(QPen(QColor(230, 30, 40), 2));
    p.drawRect(outer);
    p.setBrush(Qt::NoBrush);
    p.drawRect(inner);

    // 倍率と表示範囲の解像度
    const QString label = QString("x%1  %2x%3")
                              .arg(zoomFactor, 0, 'f', 2)
                              .arg(qRound(sw))
                              .arg(qRound(sh));
    p.setPen(Qt::white);
    static const QFont kOsdFont("Consolas", 11);
    p.setFont(kOsdFont);
    p.drawText(QRectF(outer.left(), outer.top() - 20, std::max(outer.width(), 240.0), 18),
               Qt::AlignLeft | Qt::AlignBottom, label);
    p.restore();
}

//アスペクト比を合わせてリサイズ
void GLWidget::GLresize() {
    //現在のウィンドウのDPIスケールを取得
    monitor_scaling = devicePixelRatio();

    //実ピクセル単位のビューポートサイズを取得
    viewportWidth  = static_cast<GLint>(this->width()  * monitor_scaling);
    viewportHeight = static_cast<GLint>(this->height() * monitor_scaling);

    // ★ 映像サイズが未確定のうちは描画領域を計算できない（0除算になる）
    if (width_ <= 0 || height_ <= 0 || viewportWidth <= 0 || viewportHeight <= 0) {
        x0 = y0 = x1 = y1 = 0;
        return;
    }

    //ソース（動画/FBO）のサイズ
    float srcAspect = static_cast<float>(width_) / static_cast<float>(height_);
    float dstAspect = static_cast<float>(viewportWidth) / static_cast<float>(viewportHeight);

    if (srcAspect > dstAspect) {
        int displayHeight = static_cast<int>(viewportWidth / srcAspect);
        int yOffset = (viewportHeight - displayHeight) / 2;
        x0 = 0;
        y0 = yOffset;
        x1 = viewportWidth;
        y1 = yOffset + displayHeight;
    } else {
        int displayWidth = static_cast<int>(viewportHeight * srcAspect);
        int xOffset = (viewportWidth - displayWidth) / 2;
        x0 = xOffset;
        y0 = 0;
        x1 = xOffset + displayWidth;
        y1 = viewportHeight;
    }
}

//画面をリセット(真っ黒にする)
void GLWidget::GLreset(){
    resetZoom();
    update();
}

//シェーダーUniform、UI設定変更時
void GLWidget::setShaderUniformEnable(){
    glUseProgram(Gaussian_shader.progId);
    glUniform1i(Gaussian_shader.loc_filterEnabled, g_AppSettings.gaussianfilterEnabled);

    glUseProgram(Sobel_shader.progId);
    glUniform1i(Sobel_shader.loc_filterEnabled, g_AppSettings.sobelfilterEnabled);

    glUseProgram(Averaging_shader.progId);
    glUniform1i(Averaging_shader.loc_filterEnabled, g_AppSettings.averagingfilterEnabled);

    glUseProgram(0);
}

//シェーダーUniform、解像度変更時
void GLWidget::setShaderUniform(int width,int height){
    glUseProgram(Gaussian_shader.progId);
    glUniform1i(Gaussian_shader.loc_tex, 0);
    glUniform2f(Gaussian_shader.loc_texelSize, 1.0f / width, 1.0f / height);

    glUseProgram(Sobel_shader.progId);
    glUniform1i(Sobel_shader.loc_tex, 0);
    glUniform2f(Sobel_shader.loc_texelSize, 1.0f / width, 1.0f / height);

    glUseProgram(Averaging_shader.progId);
    glUniform1i(Averaging_shader.loc_tex, 0);
    glUniform2f(Averaging_shader.loc_texelSize, 1.0f / width, 1.0f / height);

    glUseProgram(0);
}

//CUDA→OpenGLの初期化、登録など
void GLWidget::initCudaTexture(int width,int height) {
    //古いリソースを破棄
    // ★ 参照先の cudaArray が無くなるので、先に生存中の object を破棄する
    inputSurfRetire.drain();

    if (cudaResource1) {
        cudaGraphicsUnregisterResource(cudaResource1);
        cudaResource1 = nullptr;
    }
    if (inputTextureID) {
        glDeleteTextures(1, &inputTextureID);
        inputTextureID = 0;
    }

    //新しい解像度で作り直し
    glGenTextures(1, &inputTextureID);
    glBindTexture(GL_TEXTURE_2D, inputTextureID);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // ★ surface object を作るには SurfaceLoadStore で登録する必要がある
    //   （WriteDiscard のままだと cudaCreateSurfaceObject が失敗する）
    cudaError_t err = cudaGraphicsGLRegisterImage(&cudaResource1, inputTextureID, GL_TEXTURE_2D,
                                                  cudaGraphicsRegisterFlagsSurfaceLoadStore);
    if (err != cudaSuccess) {
        qWarning() << "cudaGraphicsGLRegisterImage error:" << cudaGetErrorString(err);
        cudaResource1 = nullptr;
        // ★ ハンドルを捨てる前に GL 側も解放する。16K の RGBA8 は約 1GiB あり、
        //   ここを飛ばすと解像度が変わるたびにそのぶん溜まっていく。
        glDeleteTextures(1, &inputTextureID);
        inputTextureID = 0;
    }
}

// ==============================================================
// ★ NV12タイル → OpenGLテクスチャ（surface object）へ直接 merge
//   中間の linear RGBA バッファを挟まないので、16K でも
//   1 フレームあたり 1GB 超の往復コピーが発生しない。
// ==============================================================
bool GLWidget::mergeToSurface(const VideoFrame& Frame, cudaSurfaceObject_t surf)
{
    const int outW = width_;
    const int outH = height_;
    const int srcW = Frame.tile_w;
    const int srcH = Frame.tile_h;

    // 引数の const を外すだけのローカル（launcher が uint8_t* を取るため）
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
        if (VideoInfo.bitdepth == 10) {
            return CUDA_IMG_Proc->NV12_to_RGBA_10bit_surf(
                surf, y[0], yp[0], uv[0], up[0], outW, outH, interop_stream);
        }
        return CUDA_IMG_Proc->NV12_to_RGBA_8bit_surf(
            surf, y[0], yp[0], uv[0], up[0], outW, outH, interop_stream);

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
        qWarning() << "[GLWidget] unsupported tile count:" << Frame.tile_count;
        return false;
    }
}

// ==============================================================
// ★ ゼロコピー（CPUデコード系の入口）
//   cpudecode / avidecode が上げた YUVプレーンから
//   GLテクスチャ(surface)へ直接 RGBA を書く。
// ==============================================================
bool GLWidget::yuvToSurface(const VideoFrame& Frame, cudaSurfaceObject_t surf)
{
    const int W = width_;
    const int H = height_;

    // 引数の const を外すだけのローカル（launcher が uint8_t* を取るため）
    uint8_t* p [3];
    size_t   pp[3];
    for (int i = 0; i < 3; i++) {
        p [i] = Frame.d_plane[i];
        pp[i] = Frame.plane_pitch[i];
    }

    switch (Frame.plane_format) {
    case YUVFMT_YUV420P_8:
        return CUDA_IMG_Proc->yuv420p_to_RGBA_8bit_surf(
            surf, p[0], pp[0], p[1], pp[1], p[2], pp[2], W, H, interop_stream);

    case YUVFMT_YUV420P_10:
        return CUDA_IMG_Proc->yuv420p_to_RGBA_10bit_surf(
            surf, p[0], pp[0], p[1], pp[1], p[2], pp[2], W, H,
            Frame.plane_is_be, interop_stream);

    case YUVFMT_YUV422P_8:
        return CUDA_IMG_Proc->yuv422p_to_RGBA_8bit_surf(
            surf, p[0], pp[0], p[1], pp[1], p[2], pp[2], W, H, interop_stream);

    case YUVFMT_YUV422P_10:
        return CUDA_IMG_Proc->yuv422p_to_RGBA_10bit_surf(
            surf, p[0], pp[0], p[1], pp[1], p[2], pp[2], W, H,
            Frame.plane_is_be, interop_stream);

    case YUVFMT_UYVY422_8:
        return CUDA_IMG_Proc->uyvy422_to_RGBA_8bit_surf(
            surf, p[0], pp[0], W, H, interop_stream);

    case YUVFMT_GBRP_10:
        // ★ AVFrame の並びは data[0]=G, data[1]=B, data[2]=R
        //   カーネルは r, g, b の順で受け取るので入れ替える
        return CUDA_IMG_Proc->rgb_to_RGBA_10bit_surf(
            surf,
            p[2], pp[2],   // R
            p[0], pp[0],   // G
            p[1], pp[1],   // B
            W, H, interop_stream);

    default:
        qWarning() << "[GLWidget] unsupported plane format:" << Frame.plane_format;
        return false;
    }
}

// ==============================================================
// ★ CUDA object（texture / surface）の遅延破棄リング
//
//   mapped cudaArray の寿命は map 中だけなので、object は map のたびに
//   作り直す。破棄はイベントで遅延させるので CPU は止まらない。
// ==============================================================
void CudaObjRetireRing::init(Kind k)
{
    kind = k;
    no   = 0;

    for (int i = 0; i < SIZE; i++) {
        obj[i]     = 0;
        evValid[i] = false;
        if (!ev[i])
            cudaEventCreateWithFlags(&ev[i], cudaEventDisableTiming);
    }
}

void CudaObjRetireRing::destroySlot(int i)
{
    if (!obj[i]) return;

    if (kind == TEXTURE)
        cudaDestroyTextureObject((cudaTextureObject_t)obj[i]);
    else
        cudaDestroySurfaceObject((cudaSurfaceObject_t)obj[i]);

    obj[i]     = 0;
    evValid[i] = false;
}

// 空きslotを確保する。
//   そのslotに前回の object が残っていれば、それを使っていたカーネルの
//   完了を待ってから破棄する。SIZE が in-flight より十分大きいので、
//   通常この待ちは既に完了していて素通りする。
int CudaObjRetireRing::acquire()
{
    const int i = no;
    no = (no + 1) % SIZE;

    if (obj[i]) {
        if (evValid[i])
            cudaEventSynchronize(ev[i]);
        destroySlot(i);
    }
    return i;
}

// このslotの object を使う処理を stream に積み終えた後で呼ぶ。
void CudaObjRetireRing::retire(int i, cudaStream_t s)
{
    if (i < 0 || i >= SIZE || !ev[i]) return;

    cudaEventRecord(ev[i], s);
    evValid[i] = true;
}

void CudaObjRetireRing::drain()
{
    for (int i = 0; i < SIZE; i++) {
        if (obj[i]) {
            if (evValid[i])
                cudaEventSynchronize(ev[i]);
            destroySlot(i);
        }
    }
    no = 0;
}

void CudaObjRetireRing::destroy()
{
    drain();

    for (int i = 0; i < SIZE; i++) {
        if (ev[i]) {
            cudaEventDestroy(ev[i]);
            ev[i] = nullptr;
        }
    }
}

//入力テクスチャ（cudaResource1）の surface object を作る
bool GLWidget::createInputSurface(cudaArray_t array, cudaSurfaceObject_t& out)
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

//cudaArray から texture object を作る（FBO読み出し / ヒストグラム共通）
//   unnormalized / point sampling。値は 0..255 の生バイトとして読む。
bool GLWidget::createArrayTexture(cudaArray_t array, cudaTextureObject_t& out,
                                  const char* tag)
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
        qDebug() << "cudaCreateTextureObject(" << tag << ") error:"
                 << cudaGetErrorString(err);
        out = 0;
        return false;
    }
    return true;
}

//OpenGL→CUDAの初期化、登録など
void GLWidget::initTextureCuda(int width,int height) {
        // ★ 参照先の cudaArray が無くなるので、先に生存中の object を破棄する。
        //   ヒストグラムも cudaResource2 の array を使うので一緒に drain する。
        fboTexRetire.drain();
        histTexRetire.drain();

        //既存リソースの破棄
        if (cudaResource2) {
            cudaGraphicsUnregisterResource(cudaResource2);
            cudaResource2 = nullptr;
        }
        if (fboTextureID) {
            glDeleteTextures(1, &fboTextureID);
            fboTextureID = 0;
        }
        if (fbo) {
            glDeleteFramebuffers(1, &fbo);
            fbo = 0;
        }
        if (tempTextureID) {
            glDeleteTextures(1, &tempTextureID);
            tempTextureID = 0;
        }
        if (tempfbo) {
            glDeleteFramebuffers(1, &tempfbo);
            tempfbo = 0;
        }

        //新しいサイズで再作成
        //fbo
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glGenTextures(1, &fboTextureID);
        glBindTexture(GL_TEXTURE_2D, fboTextureID);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fboTextureID, 0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        //tempfbo
        glGenFramebuffers(1, &tempfbo);
        glBindFramebuffer(GL_FRAMEBUFFER, tempfbo);
        glGenTextures(1, &tempTextureID);
        glBindTexture(GL_TEXTURE_2D, tempTextureID);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tempTextureID, 0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        //fbo登録
        cudaError_t err = cudaGraphicsGLRegisterImage(&cudaResource2, fboTextureID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly);
        if (err != cudaSuccess) {
            qDebug() << "cudaGraphicsGLRegisterImage (fbo) error:" << cudaGetErrorString(err);
            cudaResource2 = nullptr;
        }
}

//CUDAからOpenGLへ転送
void GLWidget::uploadToGLTexture(VideoFrame Frame) {
    //initialized完了チェック
    if (!initialize_completed_flag) {
        return;
    };

    // エンコード中は paintGL を通らないため、GLコンテキストを明示的にカレントにする
    makeCurrent();

    //解像度の変更に対応
    //   ★ 寸法は 1 度だけ計算して使い回す。式を並べると、どれか 1 つ
    //     書き換え忘れたときにテクスチャの寸法だけ黙ってずれる。
    const int want_w = VideoInfo.width  * VideoInfo.width_scale;
    const int want_h = VideoInfo.height * VideoInfo.height_scale;
    if (want_w != width_ || want_h != height_) {
        initCudaTexture(want_w, want_h);
        initTextureCuda(want_w, want_h);
        setShaderUniform(want_w, want_h);
        initCudaHist();
        width_  = want_w;
        height_ = want_h;
        resetZoom();
        GLresize();
        return;
    }

    // 現在フレーム番号
    FrameNo = Frame.FrameNo;

    //入力データチェック
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
        qWarning() << "[GLWidget] src_mode が未設定です:" << Frame.src_mode;
        return;
    }

    // ==========================================================
    // ★ in-flight クランプ
    //   K フレーム前のGPU処理が終わるまで、このフレームを投入しない。
    //   GPUが間に合っていれば既に完了しているので素通りする。
    //   間に合っていない場合だけ、ここでCPUが待って先行を打ち切る。
    // ==========================================================
    const int fdIdx = frameDoneNo;
    if (frameDoneValid[fdIdx]) {
        cudaEventSynchronize(frameDone[fdIdx]);
    }

    //CUDA→OpenGL転送の準備
    cudaArray_t array;
    cudaError_t err = cudaSuccess;
    // ★ map/unmap を変換カーネルと同じ interop_stream に出す。
    //   map  : これ以前のGL描画が、以降 interop_stream に積む処理より先に完了する
    //   unmap: unmap より前に interop_stream に積んだ処理が、以降のGL描画より先に完了する
    //   この2つで GL↔CUDA の往復が GPU 側だけで閉じるので、CPU同期が要らなくなる。
    err = cudaGraphicsMapResources(1, &cudaResource1, interop_stream);
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

    // ==================================================================
    // ★ ゼロコピー：デコード結果からGLテクスチャへ直接書く
    //   SRC_NV12_TILE : NVDEC の NV12 タイルを merge
    //   SRC_YUV_PLANE : CPUデコードのYUVプレーンを変換
    //   どちらも中間の linear RGBA も cudaMemcpy2DToArray も経由しない。
    // ==================================================================
    {
        // ★ surface object は「この map で得た array」から毎回作る。
        //   mapped array の寿命は map 中だけなので、キャッシュして
        //   使い回すのは仕様違反（古い面を書く事故になる）。
        const int si = inputSurfRetire.acquire();

        cudaSurfaceObject_t surf = 0;
        if (!createInputSurface(array, surf)) {
            cudaGraphicsUnmapResources(1, &cudaResource1, interop_stream);
            return;
        }
        inputSurfRetire.store(si, surf);

        // ★ NVDEC の出力コピー完了を GPU 側で待つ。
        //   cuvid は cuMemcpy2DAsync を積むだけで同期せずに戻るので、
        //   これが無いと結合カーネルが転送途中の面を読み、タイルの
        //   途中に水平の継ぎ目が出る（pts は正しいので検出もされない）。
        if (Frame.decode_ready) {
            cudaStreamWaitEvent(interop_stream, Frame.decode_ready, 0);
        }

        if (Frame.src_mode == SRC_NV12_TILE)
            mergeToSurface(Frame, surf);
        else
            yuvToSurface(Frame, surf);

        // ★ カーネルは interop_stream に積んであるので、
        //   ここで「使い終わり」を記録しておけば、数フレーム後に
        //   完了を確認してから破棄できる（CPUは止めない）。
        inputSurfRetire.retire(si, interop_stream);

        // ★ CPU同期は不要。
        //   ・object の破棄はリングがイベントで遅延させる
        //   ・GLへの順序保証は下の unmap(interop_stream) が与える
        //   GUIスレッドはここで止まらず、そのまま描画発行へ進む。
    }

    err = cudaGraphicsUnmapResources(1, &cudaResource1, interop_stream);
    if (err != cudaSuccess) {
        qDebug() << "cudaGraphicsUnmapResources error:" << cudaGetErrorString(err);
        return;
    }

    // ★ このフレームのCUDA処理完了を記録しておく。
    //   K フレーム後にここを待つことで先行が K に制限される。
    cudaEventRecord(frameDone[fdIdx], interop_stream);
    frameDoneValid[fdIdx] = true;
    frameDoneNo = (fdIdx + 1) % kMaxFramesInFlight;

    FBO_Rendering(Frame);
    fpsCount++;

    // ★ FPS算出はここで行う
    //   paintGL 側に置くと、エンコード中（update() を呼ばない）に
    //   fpsTimer が回らず、エンコード終了直後に異常値が出る
    if (fpsTimer.elapsed() >= 1000) {
        fps = fpsCount * 1000.0 / fpsTimer.elapsed();
        fpsCount = 0;
        fpsTimer.restart();
    }
}

//OpenGL(FBO) → NVENC （★ゼロコピー）
//   NV12変換カーネルが FBO を texture object 経由で直読みする。
//   cudaArray は block-linear なので、いったん linear へ落とすコピーは
//   バイト数から想像するよりずっと重い。それを挟まない。
void GLWidget::downloadToGLTexture_and_Encode(VideoFrame Frame) {
    if (!cudaResource2) {
        qDebug() << "cudaResource2 is nullptr, can't map";
        return;
    }

    //マッピング
    cudaArray_t array;
    cudaError_t err;
    err = cudaGraphicsMapResources(1, &cudaResource2, interop_stream);
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

    // ==========================================================
    // ★ GL → CUDA の順序保証を「イベント」で渡す（CPUは止めない）
    //
    //   cudaGraphicsMapResources() の保証はストリーム限定で、
    //   「map 後に *そのストリームへ* 積んだ CUDA 処理」にしか効かない。
    //   NV12変換は save_encode 側の non-blocking な st で走るため、
    //   何もしないと描画途中のFBOを読んでティアリングする。
    //
    //   そこで map 直後の interop_stream にイベントを記録する。
    //   このイベントは map が挿入したGL待ちの後に発火するので、
    //   st 側で cudaStreamWaitEvent すれば順序が伝わる。
    //   以前はここを cudaStreamSynchronize で殴っていたが、
    //   GUIスレッドが毎フレーム止まるのでイベントに置き換えた。
    // ==========================================================
    cudaEventRecord(encode_gl_event, interop_stream);

    // ---- FBO(cudaArray) の texture object は「この map で得た array」から毎回作る ----
    //   mapped array の寿命は map 中だけなので、キャッシュして使い回すのは
    //   仕様違反。ドライバが実体を差し替えると、変換カーネルが
    //   前のフレームの FBO を読んでしまう（＝エンコード結果が混ざる）。
    const int ti = fboTexRetire.acquire();

    cudaTextureObject_t tex = 0;
    if (!createArrayTexture(array, tex, "fbo")) {
        cudaGraphicsUnmapResources(1, &cudaResource2, interop_stream);
        return;
    }
    fboTexRetire.store(ti, tex);

    // ---- マップ中にエンコード投入（CPUはブロックされない）----
    bool finished = false;
    cudaEvent_t convertDone = nullptr;

    if (save_encoder != nullptr && encode_FrameCount <= (MaxFrame - MinFrame)) {
        save_encoder->encode(Frame, tex, encode_gl_event, &convertDone);
        encode_FrameCount++;
    } else {
        finished = true;
    }

    // ==========================================================
    // ★ 変換完了を待ってから unmap する（これもGPU側で）
    //
    //   unmap の保証は「unmap より前にそのストリームへ積んだ処理が、
    //   以降のGL描画より先に完了する」。変換は st で走っているので、
    //   その完了イベントを interop_stream に待たせてから unmap すると、
    //   次フレームのシェーダがFBOを上書きする前に変換が終わる。
    // ==========================================================
    if (convertDone) {
        cudaStreamWaitEvent(interop_stream, convertDone, 0);
    }

    // ★ texture object の「使い終わり」を記録する。
    //   変換カーネルは save_encode 側の st で走るので、
    //   上の cudaStreamWaitEvent で変換完了を interop_stream に
    //   引き込んだ後に記録することで、このイベントは
    //   「変換が tex を読み終えた」を意味する。
    fboTexRetire.retire(ti, interop_stream);

    //リソースのマップを解除し、制御をOpenGLに戻す
    err = cudaGraphicsUnmapResources(1, &cudaResource2, interop_stream);
    if (err != cudaSuccess) {
        qDebug() << "cudaGraphicsUnmapResources error:" << cudaGetErrorString(err);
        return;
    }

    // ---- 後始末は unmap の外で（encode_finished の先で何が起きても安全に） ----
    if (finished) {
        delete save_encoder;
        save_encoder = nullptr;
        encode_FrameCount = 0;
        emit encode_finished();
    }
}

//エンコード時
void GLWidget::encode_mode(int flag){
    encode_state = flag;

    //エンコードモードの場合はエンコード用インスタンス生成
    if(flag==STATE_ENCODING){
        if(save_encoder==nullptr){
            // ★ save_encode のコンストラクタは
            //     ・hw_frames_ctx の確保失敗（VRAM不足）
            //     ・エンコーダが見つからない / avcodec_open2 失敗
            //     ・出力ファイルが開けない
            //   などで std::runtime_error を投げる。
            //   ここで受けないと誰も catch せず std::terminate でアプリごと落ちる。
            //   16K + B-frame 多め の設定はリングだけで数GB使うので、
            //   VRAM の少ない環境では現実に踏む。
            try {
                save_encoder = new save_encode(height_, width_);
            }
            catch (const std::exception& e) {
                save_encoder = nullptr;
                encode_state = STATE_NOT_ENCODE;   // エンコード経路へ入らせない

                const QString msg = QString::fromUtf8(e.what());
                qWarning() << "[GLWidget] save_encode init failed:" << msg;
                emit encode_error(msg);
                return;
            }
            catch (...) {
                save_encoder = nullptr;
                encode_state = STATE_NOT_ENCODE;

                qWarning() << "[GLWidget] save_encode init failed (unknown exception)";
                emit encode_error(QStringLiteral("不明なエラー"));
                return;
            }
        }
    }
}

//ヒストグラム周り
void GLWidget::initCudaHist() {
    // ★ 参照先の cudaArray が無くなるので、先に生存中の object を破棄する
    histTexRetire.drain();

    // --- 1. 既存 CUDA Graphics Resource を解放 ---
    if (cudaResource_hist_draw) {
        cudaGraphicsUnregisterResource(cudaResource_hist_draw);
        cudaResource_hist_draw = nullptr;
    }

    // --- 2. 既存 VBO を削除 ---
    if (vbo_hist != 0) {
        glDeleteBuffers(1, &vbo_hist);
        vbo_hist = 0;
    }

    // --- 3. 既存 CUDA メモリ を解放 ---
    if (d_hist_stats) {
        cudaFree(d_hist_stats);
        d_hist_stats = nullptr;
    }
    if (d_hist_data) {
        cudaFree(d_hist_data);
        d_hist_data = nullptr;
    }

    // ★ fbo は CUDA へ登録し直さない。
    //   同一 GL オブジェクトの複数回登録は CUDA の未定義動作で、
    //   fboTextureID は initTextureCuda が cudaResource2（ReadOnly）
    //   として登録済み。ヒストグラムはそれをそのまま使う。
    //   再生中は NVENC 経路が、エンコード中はヒストグラムが走らないので
    //   競合しない。
    cudaError_t err = cudaSuccess;

    // --- 4. 新規 VBO 作成 ---
    glGenBuffers(1, &vbo_hist);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_hist);
    glBufferData(GL_ARRAY_BUFFER, num_bins * 3 * 3 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // --- 5. VBO → CUDA Resource 登録 ---
    err = cudaGraphicsGLRegisterBuffer(
        &cudaResource_hist_draw,
        vbo_hist,
        cudaGraphicsRegisterFlagsWriteDiscard
        );
    if (err != cudaSuccess) {
        qDebug() << "cudaGraphicsGLRegisterBuffer failed:" << cudaGetErrorString(err);
    }

    // --- 6. CUDA 側のメモリを確保 ---
    cudaMalloc(&d_hist_stats, sizeof(HistStats));
    cudaMalloc(&d_hist_data, sizeof(HistData));
}

//OpenGLからCUDAへ転送+ヒストグラム解析
void GLWidget::histgram_Analysys(){
    if (!cudaResource2 || !cudaResource_hist_draw) {
        qDebug() << "histgram: graphics resource is nullptr, can't map";
        return;
    }

    // 一括マップ
    //   ★ ストリームは 0 のままにすること。calc_histgram /
    //     histgram_normalize が既定ストリームで走るので、
    //     map/unmap のフェンスも同じストリームに張る必要がある。
    cudaGraphicsResource* resources[] = { cudaResource2, cudaResource_hist_draw };
    cudaError_t err = cudaGraphicsMapResources(2, resources, 0);
    if (err != cudaSuccess) {
        qDebug() << "cudaGraphicsMapResources error:" << cudaGetErrorString(err);
        return;
    }

    //ヒストグラム計算
    {
        //サブリソースの cudaArray を取得
        cudaArray_t cuArray;
        err = cudaGraphicsSubResourceGetMappedArray(&cuArray, cudaResource2, 0, 0);
        if (err != cudaSuccess) {
            qDebug() << "cudaGraphicsSubResourceGetMappedArray error:" << cudaGetErrorString(err);
            cudaGraphicsUnmapResources(2, resources, 0);
            return;
        }

        // ★ 他の2経路と同じ扱い。
        //   texture object は「この map で得た array」から毎回作り、
        //   破棄はリングがイベントで遅延させる。
        const int hi = histTexRetire.acquire();

        cudaTextureObject_t htex = 0;
        if (!createArrayTexture(cuArray, htex, "hist")) {
            cudaGraphicsUnmapResources(2, resources, 0);
            return;
        }
        histTexRetire.store(hi, htex);

        //カーネル呼び出し
        cudaMemset(d_hist_data, 0, sizeof(HistData));
        CUDA_IMG_Proc->calc_histgram(d_hist_data,htex,width_,height_);

        // ★ calc_histgram は既定ストリームで走るので、既定ストリームに
        //   「使い終わり」を記録する。htex を使うのはこのカーネルだけ。
        histTexRetire.retire(hi, 0);
    }

    //ヒストグラム値解析
    {
        CUDA_IMG_Proc->histogram_status(d_hist_data,d_hist_stats);
        cudaMemcpyAsync(&h_hist_stats, d_hist_stats,sizeof(HistStats),cudaMemcpyDeviceToHost,hist_stream);
        cudaStreamSynchronize(hist_stream);   // ★ 直後に h_hist_stats を読むので同期する
    }

    //OpenGL VBO転送
    {
        float* d_vbo_ptr = nullptr;
        size_t vbo_size = 0;
        err = cudaGraphicsResourceGetMappedPointer((void**)&d_vbo_ptr, &vbo_size, cudaResource_hist_draw);
        if (err != cudaSuccess) {
            qDebug() << "cudaGraphicsResourceGetMappedPointer error:" << cudaGetErrorString(err);
            cudaGraphicsUnmapResources(2, resources, 0);
            return;
        }
        CUDA_IMG_Proc->histgram_normalize(d_vbo_ptr,num_bins,d_hist_data,d_hist_stats);
    }

    //一括アンマップ
    err = cudaGraphicsUnmapResources(2, resources, 0);
    if (err != cudaSuccess) {
        qDebug() << "cudaGraphicsUnmapResources error:" << cudaGetErrorString(err);
        return;
    }
}

// 人が見て気持ちいいY軸ラベルを生成
std::vector<int> GLWidget::make_nice_y_labels(int max_value)
{
    int num_divs = 3;
    std::vector<int> labels;
    if (max_value <= 0) return labels;

    // 1. 仮の間隔を計算
    double raw_step = static_cast<double>(max_value) / num_divs;
    double magnitude = pow(10.0, floor(log10(raw_step))); // 10^n の桁
    double normalized = raw_step / magnitude;             // 1〜10の範囲に正規化

    // 2. 正規化した値を 1, 2, 5 のいずれかに丸める
    double nice_factor = 1.0;
    if (normalized < 1.5)
        nice_factor = 1.0;
    else if (normalized < 3.0)
        nice_factor = 2.0;
    else if (normalized < 7.0)
        nice_factor = 5.0;
    else
        nice_factor = 10.0;

    double nice_step = nice_factor * magnitude;

    // 3. ラベル値を作成
    for (double v = nice_step; v < max_value; v += nice_step)
        labels.push_back(static_cast<int>(v));

    // 4. 必要なら最終ラベルを追加
    if (labels.empty() || labels.back() < max_value)
        labels.push_back(static_cast<int>(ceil(max_value / nice_step) * nice_step));

    // ★ 呼び出し側が labels[0..2] を触るので最低3個に揃える
    while (labels.size() < 3)
        labels.push_back(labels.empty() ? max_value : labels.back());

    return labels;
}

//CUDAとopenGLのデバイス設定
void GLWidget::queryCudaGPUs()
{
    // iniから読み込んだ情報を退避
    std::vector<GPUInfo> savedGPUInfo = g_GPUInfo;

    // CUDAで取得したGPU情報を一時的に格納
    std::vector<GPUInfo> currentGPUInfo;

    int count = 0;
    cudaGetDeviceCount(&count);

    for (int i = 0; i < count; i++)
    {
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

        // 初期値
        info.tile_weight = 10;
        info.openglEnable = false;

        currentGPUInfo.push_back(info);
    }

    // ==========================================
    // savedGPUInfo と currentGPUInfo が全一致か確認
    // ==========================================
    bool allMatch = true;

    if (savedGPUInfo.size() != currentGPUInfo.size())
    {
        allMatch = false;
    }
    else
    {
        for (const auto &cur : currentGPUInfo)
        {
            bool found = false;

            for (const auto &saved : savedGPUInfo)
            {
                if (saved.pciDomain == cur.pciDomain &&
                    saved.pciBus == cur.pciBus &&
                    saved.pciDevice == cur.pciDevice &&
                    saved.deviceName == cur.deviceName &&
                    saved.CC_major == cur.CC_major &&
                    saved.CC_minor == cur.CC_minor &&
                    saved.Max_Memory_Usage == cur.Max_Memory_Usage)
                {
                    found = true;
                    break;
                }
            }

            if (!found)
            {
                allMatch = false;
                break;
            }
        }
    }

    // ==========================================
    // g_GPUInfo を確定
    // ==========================================
    g_GPUInfo.clear();

    if (allMatch)
    {
        // 全一致 → iniの設定を復元
        for (auto &cur : currentGPUInfo)
        {
            for (const auto &saved : savedGPUInfo)
            {
                if (saved.pciDomain == cur.pciDomain &&
                    saved.pciBus == cur.pciBus &&
                    saved.pciDevice == cur.pciDevice &&
                    saved.deviceName == cur.deviceName &&
                    saved.CC_major == cur.CC_major &&
                    saved.CC_minor == cur.CC_minor &&
                    saved.Max_Memory_Usage == cur.Max_Memory_Usage)
                {
                    cur.tile_weight = saved.tile_weight;
                    cur.openglEnable = saved.openglEnable;
                    break;
                }
            }

            g_GPUInfo.push_back(cur);
        }

        qDebug() << "[GPU MATCH] All GPUs matched -> restored tile_weight";
    }
    else
    {
        // 一部一致・不一致 → 全GPU初期化
        for (auto &cur : currentGPUInfo)
        {
            cur.tile_weight = 10;
            cur.openglEnable = false;
            g_GPUInfo.push_back(cur);
        }

        qDebug() << "[GPU MISMATCH] Partial match -> reset ALL tile_weight=10";
    }

    // ログ
    for (const auto &gpu : g_GPUInfo)
    {
        qDebug() << "CUDA Device" << gpu.deviceID
                 << ":" << gpu.deviceName
                 << "CC =" << gpu.CC_major << "." << gpu.CC_minor
                 << "VRAM =" << gpu.Max_Memory_Usage << "MB"
                 << "PCI =" << gpu.pciDomain << ":" << gpu.pciBus << ":" << gpu.pciDevice
                 << "tile_weight =" << gpu.tile_weight;
    }
}

void GLWidget::getCudaDeviceIDFromOpenGL()
{
    unsigned int count = 0;
    int devices[8] = {};

    cudaError_t err = cudaGLGetDevices(
        &count,
        devices,
        8,
        cudaGLDeviceListCurrentFrame
        );

    if (err != cudaSuccess)
    {
        qDebug() << "cudaGLGetDevices failed:" << cudaGetErrorString(err);
        return;
    }

    if (count == 0)
    {
        qDebug() << "No CUDA device associated with OpenGL context.";
        return;
    }

    int oglDev = devices[0];
    if (oglDev < 0)
        return;

    // ============================================
    // OpenGL CUDA device の詳細情報を取得
    // ============================================
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, oglDev);

    QString oglName = QString::fromUtf8(prop.name);
    int oglCCMajor = prop.major;
    int oglCCMinor = prop.minor;
    int oglVRAM = (int)(prop.totalGlobalMem / (1024 * 1024));

    int oglDomain = 0;
    int oglBus = 0;
    int oglDevice = 0;

    cudaDeviceGetAttribute(&oglDomain, cudaDevAttrPciDomainId, oglDev);
    cudaDeviceGetAttribute(&oglBus, cudaDevAttrPciBusId, oglDev);
    cudaDeviceGetAttribute(&oglDevice, cudaDevAttrPciDeviceId, oglDev);

    qDebug() << "Current OpenGL CUDA deviceID =" << oglDev
             << "name =" << oglName
             << "CC =" << oglCCMajor << "." << oglCCMinor
             << "VRAM =" << oglVRAM << "MB"
             << "PCI =" << oglDomain << ":" << oglBus << ":" << oglDevice;

    // ============================================
    // 前回iniで openglEnable=true だったGPUの情報を保存
    // ============================================
    int savedDomain = -1;
    int savedBus = -1;
    int savedDevice = -1;
    QString savedName;
    int savedCCMajor = 0;
    int savedCCMinor = 0;
    int savedVRAM = 0;

    for (const auto &gpu : g_GPUInfo)
    {
        if (gpu.openglEnable)
        {
            savedDomain = gpu.pciDomain;
            savedBus = gpu.pciBus;
            savedDevice = gpu.pciDevice;

            savedName = gpu.deviceName;
            savedCCMajor = gpu.CC_major;
            savedCCMinor = gpu.CC_minor;
            savedVRAM = gpu.Max_Memory_Usage;
            break;
        }
    }

    bool hasSavedOpenGL = (savedDomain != -1);

    if (hasSavedOpenGL)
    {
        qDebug() << "Saved OpenGL GPU ="
                 << savedName
                 << "CC =" << savedCCMajor << "." << savedCCMinor
                 << "VRAM =" << savedVRAM << "MB"
                 << "PCI =" << savedDomain << ":" << savedBus << ":" << savedDevice;
    }
    else
    {
        qDebug() << "Saved OpenGL GPU = (none)";
    }

    // ============================================
    // OpenGL GPU が変わったか判定（PCI + Name + CC + VRAM）
    // ============================================
    bool oglChanged = false;

    if (hasSavedOpenGL)
    {
        if (savedDomain != oglDomain ||
            savedBus != oglBus ||
            savedDevice != oglDevice ||
            savedName != oglName ||
            savedCCMajor != oglCCMajor ||
            savedCCMinor != oglCCMinor ||
            savedVRAM != oglVRAM)
        {
            oglChanged = true;
        }
    }

    // ============================================
    // iniのopenglEnableは信用しないので一旦全OFF
    // ============================================
    for (auto &gpu : g_GPUInfo)
        gpu.openglEnable = false;

    // ============================================
    // OpenGL GPU を g_GPUInfo から特定
    // ============================================
    bool found = false;

    for (auto &gpu : g_GPUInfo)
    {
        if (gpu.pciDomain == oglDomain &&
            gpu.pciBus == oglBus &&
            gpu.pciDevice == oglDevice &&
            gpu.deviceName == oglName &&
            gpu.CC_major == oglCCMajor &&
            gpu.CC_minor == oglCCMinor &&
            gpu.Max_Memory_Usage == oglVRAM)
        {
            found = true;
            gpu.openglEnable = true;
            g_openglDeviceID = gpu.deviceID;

            // OpenGL GPU が変わった場合は全GPUを初期値に戻す
            if (oglChanged)
            {
                qDebug() << "[INFO] OpenGL GPU changed -> Reset ALL tile_weight = 10";
                for (auto &g : g_GPUInfo)
                    g.tile_weight = 10;
            }

            cudaSetDevice(g_openglDeviceID);

            qDebug() << "Matched OpenGL GPU:"
                     << "deviceID =" << gpu.deviceID
                     << "name =" << gpu.deviceName
                     << "CC =" << gpu.CC_major << "." << gpu.CC_minor
                     << "VRAM =" << gpu.Max_Memory_Usage << "MB"
                     << "PCI =" << gpu.pciDomain << ":" << gpu.pciBus << ":" << gpu.pciDevice
                     << "tile_weight =" << gpu.tile_weight;

            break;
        }
    }

    if (!found)
    {
        qDebug() << "[ERROR] OpenGL GPU match not found in g_GPUInfo.";
        qDebug() << "OpenGL GPU ="
                 << oglName
                 << "CC =" << oglCCMajor << "." << oglCCMinor
                 << "VRAM =" << oglVRAM << "MB"
                 << "PCI =" << oglDomain << ":" << oglBus << ":" << oglDevice;
    }
}
