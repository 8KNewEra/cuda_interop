#include "glwidget.h"
#include "qdir.h"
#include <QDebug>

GLWidget::GLWidget(QWindow *parent)
    :  QOpenGLWindow(NoPartialUpdate, parent),
    cudaResource1(nullptr),
    cudaResource2(nullptr),
    inputTextureID(0)
{
    if(CUDA_IMG_Proc==nullptr){
        CUDA_IMG_Proc=new CUDA_ImageProcess();
    }

    qDebug() << "GLWidget: Contructor called";
}


GLWidget::~GLWidget() {
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
    }
    if (input_pbo) {
        glDeleteBuffers(1, &input_pbo);
        input_pbo = 0;
    }
    if (output_pbo) {
        glDeleteBuffers(1, &output_pbo);
        output_pbo = 0;
    }
    if (fboTextureID) {
        glDeleteTextures(1, &fboTextureID);
        fboTextureID = 0;
    }
    if (fbo) {
        glDeleteFramebuffers(1, &fbo);
        fbo = 0;
    }
    if (stream) {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
        stream = nullptr;
    }
    if (e) {
        cudaEventDestroy(e);
        e = nullptr;
    }

    if (cudaResource_hist) {
        cudaGraphicsUnregisterResource(cudaResource_hist);
        cudaResource_hist = nullptr;
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
    if (d_y) {
        cudaFree(d_y);
        d_y = nullptr;
    }
    if (d_uv) {
        cudaFree(d_uv);
        d_uv = nullptr;
    }
    if (d_rgba) {
        cudaFree(d_rgba);
        d_rgba = nullptr;
    }
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
    Sobel_program.addShaderFromSourceFile(QOpenGLShader::Vertex, "../../shaders/sobel.vert");
    Sobel_program.addShaderFromSourceFile(QOpenGLShader::Fragment, "../../shaders/sobel.frag");
    Gaussian_program.addShaderFromSourceFile(QOpenGLShader::Vertex, "../../shaders/gaussian.vert");
    Gaussian_program.addShaderFromSourceFile(QOpenGLShader::Fragment, "../../shaders/gaussian.frag");
    Averaging_program.addShaderFromSourceFile(QOpenGLShader::Vertex, "../../shaders/averaging.vert");
    Averaging_program.addShaderFromSourceFile(QOpenGLShader::Fragment, "../../shaders/averaging.frag");

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
    getCudaCapabilityForOpenGLGPU();

    //stream
    cudaStreamCreate(&stream);
    cudaEventCreate(&e);

    // === 初期化完了 ===
    fpsTimer.start();
    initialize_completed_flag = true;
    emit initialized();
}

//FBOレンダリング
void GLWidget::FBO_Rendering(){
    //フィルター適用チェック
    if(filter_change_flag){
        setShaderUniformEnable();
        filter_change_flag=false;
    }

    //FBO描画開始
    glBindVertexArray(vao);

    // Gaussian
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glViewport(0, 0, width_, height_);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(Gaussian_shader.progId);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, inputTextureID);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);

    // Sobel
    glBindFramebuffer(GL_FRAMEBUFFER, tempfbo);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(Sobel_shader.progId);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, fboTextureID);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);

    // Averaging
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(Averaging_shader.progId);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tempTextureID);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);

    glBindVertexArray(0);
    glUseProgram(0);

    if (encode_state==STATE_ENCODING) {
        // GPUエンコード用処理
        downloadToGLTexture_and_Encode();
    } else if(encode_state==STATE_NOT_ENCODE){
        //画面に描画
        Monitor_Rendering();
    }
}

//画面描画
void GLWidget::Monitor_Rendering(){
    // QElapsedTimer timer;
    // timer.start();
    painter.begin(this);

    //動画フレーム描画
    {
        //描画処理
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // FBO → 画面へ転送
        glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
        glBlitFramebuffer(
            0, 0, width_, height_,
            x0, y0, x1, y1,
            GL_COLOR_BUFFER_BIT,
            GL_LINEAR
            );
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    //ヒストグラム描画
    {
        if(histgram_flag){
            //ヒストグラムCUDA集計
            histgram_Analysys();
            auto labels = make_nice_y_labels(h_hist_stats.max_y_axis);
            //int labels[5]={1,2,3,4,5};

            //ヒストグラムグラフ描画
            {
                QRect rect(viewportWidth -(480+20)*monitor_scaling, 20*monitor_scaling, 480*monitor_scaling, 360*monitor_scaling);
                glViewport(rect.x(), rect.y(), rect.width(), rect.height());

                // --- OpenGL描画 ---
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
                glVertex2f(0, 0.0f);
                glVertex2f(0, 1.0f);
                glVertex2f(0.25f, 0.0f);
                glVertex2f(0.25f, 1.0f);
                glVertex2f(0.5f, 0.0f);
                glVertex2f(0.5f, 1.0f);
                glVertex2f(0.75f, 0.0f);
                glVertex2f(0.75f, 1.0f);
                glVertex2f(1.0f, 0.0f);
                glVertex2f(1.0f, 1.0f);
                //Y軸
                glVertex2f(0.0f, 0);
                glVertex2f(1.0f, 0);
                glVertex2f(0.0f, 1.0f);
                glVertex2f(1.0f, 1.0f);
                glVertex2f(0.0f, (float)labels[0]/(float)h_hist_stats.max_y_axis);
                glVertex2f(1.0f, (float)labels[0]/(float)h_hist_stats.max_y_axis);
                glVertex2f(0.0f, (float)labels[1]/(float)h_hist_stats.max_y_axis);
                glVertex2f(1.0f, (float)labels[1]/(float)h_hist_stats.max_y_axis);
                glVertex2f(0.0f, (float)labels[2]/(float)h_hist_stats.max_y_axis);
                glVertex2f(1.0f, (float)labels[2]/(float)h_hist_stats.max_y_axis);
                // --- 目盛り ---
                glLineWidth(3.0f);
                glColor4f(1.0f, 1.0f, 1.0f, 0.8f);
                //X軸
                glVertex2f(0, 0.0f);
                glVertex2f(0, 0.03f);
                glVertex2f(0.25f, 0.0f);
                glVertex2f(0.25f, 0.03f);
                glVertex2f(0.5f, 0.0f);
                glVertex2f(0.5f, 0.03f);
                glVertex2f(0.75f, 0.0f);
                glVertex2f(0.75f, 0.03f);
                glVertex2f(1.0f, 0.0f);
                glVertex2f(1.0f, 0.03f);
                //Y軸
                glVertex2f(0.0f, 0);
                glVertex2f(0.03f, 0);
                glVertex2f(0.0f, 1.0f);
                glVertex2f(0.03f, 1.0f);
                glVertex2f(0.0f, (float)labels[0]/(float)h_hist_stats.max_y_axis);
                glVertex2f(0.03f, (float)labels[0]/(float)h_hist_stats.max_y_axis);
                glVertex2f(0.0f, (float)labels[1]/(float)h_hist_stats.max_y_axis);
                glVertex2f(0.03f, (float)labels[1]/(float)h_hist_stats.max_y_axis);
                glVertex2f(0.0f, (float)labels[2]/(float)h_hist_stats.max_y_axis);
                glVertex2f(0.03f, (float)labels[2]/(float)h_hist_stats.max_y_axis);
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

                // --- Qtでラベルを追加 ---
                // painter.beginNativePainting(); // OpenGLとQtの切り替え開始
                // painter.endNativePainting();

                //ラベル値
                //Y座標
                painter.setPen(Qt::white);
                painter.setFont(QFont("Arial", 10));
                painter.drawText((rect.left() - 60)/monitor_scaling,(viewportHeight-25*monitor_scaling)/monitor_scaling,55/monitor_scaling, 20*monitor_scaling,Qt::AlignRight,QString::number(0));
                painter.drawText((rect.left() - 200)/monitor_scaling,(viewportHeight-(rect.height())-35*monitor_scaling)/monitor_scaling,195/monitor_scaling, 20*monitor_scaling,Qt::AlignRight,"Max Y:"+QString::number(h_hist_stats.max_y_axis));
                for(int i=0;i<=2;i++){
                    if(labels[i]<=h_hist_stats.max_y_axis){
                        painter.drawText((rect.left() - 150)/monitor_scaling,(viewportHeight-(rect.height())*((float)labels[i]/(float)h_hist_stats.max_y_axis)-25*monitor_scaling)/monitor_scaling,145/monitor_scaling, 20*monitor_scaling,Qt::AlignRight,QString::number(labels[i]));
                    }
                }
                //X座標
                painter.drawText(rect.left()/monitor_scaling,(viewportHeight-19*monitor_scaling)/monitor_scaling,126, 20*monitor_scaling,Qt::AlignRight,QString::number(64));
                painter.drawText(rect.left()/monitor_scaling,(viewportHeight-19*monitor_scaling)/monitor_scaling,250, 20*monitor_scaling,Qt::AlignRight,QString::number(128));
                painter.drawText(rect.left()/monitor_scaling,(viewportHeight-19*monitor_scaling)/monitor_scaling,370, 20*monitor_scaling,Qt::AlignRight,QString::number(192));
                painter.drawText(rect.left()/monitor_scaling,(viewportHeight-19*monitor_scaling)/monitor_scaling,488, 20*monitor_scaling,Qt::AlignRight,QString::number(256));
            }

            //ヒストグラム描画
            {
                painter.setPen(Qt::white);
                painter.setFont(QFont("Consolas", 16));
                QRect rect(viewportWidth/monitor_scaling -(316), 0, 350*monitor_scaling, 100*monitor_scaling);  // 右端に幅300のエリア
                painter.drawText(rect, Qt::AlignLeft,"R: min:" + QString::number(h_hist_stats.min_r) +" max:" + QString::number(h_hist_stats.max_r) +" avg:" + QString::number(h_hist_stats.avg_r));
                painter.drawText(rect.adjusted(0, 20, 0, 0), Qt::AlignLeft,"G: min:" + QString::number(h_hist_stats.min_g) +" max:" + QString::number(h_hist_stats.max_g) +" avg:" + QString::number(h_hist_stats.avg_g));
                painter.drawText(rect.adjusted(0, 40, 0, 0), Qt::AlignLeft,"B: min:" + QString::number(h_hist_stats.min_b) +" max:" + QString::number(h_hist_stats.max_b) +" avg:" + QString::number(h_hist_stats.avg_b));
                //painter.drawText(rect.adjusted(0, 60, 0, 0), Qt::AlignLeft,"Axis Y: max:" + QString::number(h_stats.max_y_axis));
            }
        }
    }

    //FPSを算出
    {
        if (fpsTimer.elapsed() >= 1000) {  // 1000ms 経過したら
            fps = fpsCount * 1000.0 / fpsTimer.elapsed(); // FPS計算
            fpsCount = 0;
            fpsTimer.restart();
        }
    }

    //動画情報描画
    {
        if(videoInfo_flag){
            painter.setPen(Qt::white);
            painter.setFont(QFont("Consolas", 16));
            painter.drawText(2, 20, "OpenGL Device:" + QString::fromLatin1((const char*)glGetString(GL_RENDERER))+"\n");
            painter.drawText(2, 40, "CUDA Device:" + QString::fromStdString(g_prop.name)+"(Compute Capability:"+QString::number(g_prop.major)+"."+QString::number(g_prop.minor)+")\n");
            painter.drawText(2, 60, QString("FPS: %1").arg(fps, 0, 'f', 1));
            painter.drawText(2, 80, "GPU Usage:" + QString::number(g_gpu_usage) +"% \n");
            painter.drawText(2, 100, "File Name:" + QString::fromStdString(VideoInfo.Name)+"\n");
            painter.drawText(2, 120, "Decorder:" + QString::fromStdString(VideoInfo.Codec)+"\n");
            painter.drawText(2, 140, "Resolution:" + QString::number(VideoInfo.width)+"×"+QString::number(VideoInfo.height)+"\n");
            painter.drawText(2, 160, "Video Framerate:" + QString::number(VideoInfo.fps)+"\n");
            painter.drawText(2, 180, "Max Frame:" + QString::number(VideoInfo.max_framesNo)+"\n");
            painter.drawText(2, 200, "Current Frame:" + QString::number(VideoInfo.current_frameNo)+"\n");
        }
    }

    painter.end();
    context()->swapBuffers(context()->surface());

    // double seconds = timer.nsecsElapsed() / 1e6; // ナノ秒 →  ミリ秒
    // qDebug()<<seconds;
}

//アスペクト比を合わせてリサイズ
void GLWidget::GLresize() {
    //現在のウィンドウのDPIスケールを取得
    monitor_scaling = devicePixelRatio();

    //実ピクセル単位のビューポートサイズを取得
    viewportWidth  = static_cast<GLint>(this->width()  * monitor_scaling);
    viewportHeight = static_cast<GLint>(this->height() * monitor_scaling);

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
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    context()->swapBuffers(context()->surface());
}

//シェーダーUniform、UI設定変更時
void GLWidget::setShaderUniformEnable(){
    glUseProgram(Gaussian_shader.progId);
    glUniform1i(Gaussian_shader.loc_filterEnabled, gaussianfilterEnabled);

    glUseProgram(Sobel_shader.progId);
    glUniform1i(Sobel_shader.loc_filterEnabled, sobelfilterEnabled);

    glUseProgram(Averaging_shader.progId);
    glUniform1i(Averaging_shader.loc_filterEnabled, averagingfilterEnabled);

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
    if (cudaResource1) {
        cudaGraphicsUnregisterResource(cudaResource1);
        cudaResource1 = nullptr;
    }

    if (input_pbo) {
        glDeleteBuffers(1, &input_pbo);
        input_pbo = 0;
    }

    if (inputTextureID) {
        glDeleteTextures(1, &inputTextureID);
        inputTextureID = 0;
    }

    //新しい解像度で作り直し
    glGenBuffers(1, &input_pbo);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, input_pbo);
    glBufferData(GL_PIXEL_PACK_BUFFER, width * height * 4, NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

    glGenTextures(1, &inputTextureID);
    glBindTexture(GL_TEXTURE_2D, inputTextureID);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    //CUDAリソース再登録
    cudaError_t err = cudaGraphicsGLRegisterBuffer(&cudaResource1, input_pbo, cudaGraphicsRegisterFlagsWriteDiscard);
    if (err != cudaSuccess) {
        qDebug() << "Failed to register PBO: " << cudaGetErrorString(err);
        return;
    }
}

//OpenGL→CUDAの初期化、登録など
void GLWidget::initTextureCuda(int width,int height) {
        //既存リソースの破棄
        if (cudaResource2) {
            cudaGraphicsUnregisterResource(cudaResource2);
            cudaResource2 = nullptr;
        }
        if (output_pbo) {
            glDeleteBuffers(1, &output_pbo);
            output_pbo = 0;
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
        glGenBuffers(1, &output_pbo);
        glBindBuffer(GL_PIXEL_PACK_BUFFER, output_pbo);
        glBufferData(GL_PIXEL_PACK_BUFFER, width * height * 4, NULL, GL_STREAM_READ);
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

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

        //outputpbo登録
        cudaError_t err = cudaGraphicsGLRegisterBuffer(&cudaResource2, output_pbo, cudaGraphicsRegisterFlagsReadOnly);
        if (err != cudaSuccess) {
            qDebug() << "cudaGraphicsGLRegisterBuffer (output_pbo) error:" << cudaGetErrorString(err);
            cudaResource2 = nullptr;
        }
}

//初回、解像度が変わった場合再Malloc
void GLWidget::initCudaMalloc(int width, int height)
{
    //すでに確保済みなら一度解放
    if (d_y) {
        cudaFree(d_y);
        d_y = nullptr;
    }
    if (d_uv) {
        cudaFree(d_uv);
        d_uv = nullptr;
    }
    if (d_rgba) {
        cudaFree(d_rgba);
        d_rgba = nullptr;
    }

    //再確保
    cudaMallocPitch(&d_y, &pitch_y, width, height);
    cudaMallocPitch(&d_uv, &pitch_uv, width, height / 2);
    cudaMallocPitch(&d_rgba, &pitch_rgba, width * 4, height);
}

//CUDAからOpenGLへ転送
void GLWidget::uploadToGLTexture(uint8_t* d_y, size_t pitch_y,uint8_t* d_uv, size_t pitch_uv,int a) {
    // QElapsedTimer timer;
    // timer.start();

    FrameNo = a;
    //initialized完了チェック
    if (!initialize_completed_flag) {
        emit decode_please();
        return;
    };

    //解像度の変更に対応
    if (VideoInfo.width != width_ || VideoInfo.height != height_) {
        initCudaMalloc(VideoInfo.width,VideoInfo.height);
        initCudaTexture(VideoInfo.width,VideoInfo.height);
        initTextureCuda(VideoInfo.width,VideoInfo.height);
        setShaderUniform(VideoInfo.width,VideoInfo.height);
        initCudaHist();
        width_=VideoInfo.width;
        height_=VideoInfo.height;
        GLresize();
        emit decode_please();
        return;
    }

    //入力データチェック
    if (!d_y||!d_uv) {
        qDebug() << "入力データがNULLです";
        emit decode_please();
        return;
    }

    //PBOリソースをCUDAからアクセスできるようにマップする
    cudaError_t err = cudaSuccess;
    err = cudaGraphicsMapResources(1, &cudaResource1, 0);
    if (err != cudaSuccess) {
        qDebug() << "cudaGraphicsMapResources error:" << cudaGetErrorString(err);
        emit decode_please();
        return;
    }

    //マップしたPBOのデバイスポインタを取得する
    uint8_t* pbo_ptr = nullptr;
    size_t pbo_size;
    err = cudaGraphicsResourceGetMappedPointer((void**)&pbo_ptr, &pbo_size, cudaResource1);
    if (err != cudaSuccess) {
        qDebug() << "cudaGraphicsResourceGetMappedPointer error:" << cudaGetErrorString(err);
        cudaGraphicsUnmapResources(1, &cudaResource1, 0);
        emit decode_please();
        return;
    }

    //CUDAカーネルの処理
    size_t pitch_pbo=width_*4;
    CUDA_IMG_Proc->NV12_to_RGBA(pbo_ptr, pitch_pbo, d_y, pitch_y, d_uv, pitch_uv, height_, width_);
    //CUDA_IMG_Proc->Gradation(pbo_ptr,pitch_pbo,d_rgba,pitch_rgba,height,width);

    //PBOリソースのマップを解除し、制御をOpenGLに戻す
    err = cudaGraphicsUnmapResources(1, &cudaResource1, 0);
    if (err != cudaSuccess) {
        qDebug() << "cudaGraphicsUnmapResources error:" << cudaGetErrorString(err);
        emit decode_please();
        return;
    }

    //OpenGLのコマンドで、PBOからテクスチャへデータを転送する
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, input_pbo);
    glBindTexture(GL_TEXTURE_2D, inputTextureID);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_, height_, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // double seconds = timer.nsecsElapsed() / 1e6; // ナノ秒 →  ミリ秒
    // qDebug()<<seconds;

    FBO_Rendering();
    fpsCount++;
}

//OpenGLからCUDAへ転送+エンコード
void GLWidget::downloadToGLTexture_and_Encode() {
    if (!cudaResource2) {
        qDebug() << "cudaResource2 is nullptr, can't map";
        emit decode_please();
        return;
    }

    //FBOからPBOへコピー
    glBindBuffer(GL_PIXEL_PACK_BUFFER, output_pbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glReadPixels(0, 0, width_, height_, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    //マッピング
    cudaError_t err;
    err = cudaGraphicsMapResources(1, &cudaResource2, 0);
    if (err != cudaSuccess) {
        qDebug() << "cudaGraphicsMapResources error:" << cudaGetErrorString(err);
        emit decode_please();
        return;
    }

    //マップしたPBOのデバイスポインタを取得する
    uint8_t* pbo_ptr = nullptr;
    size_t pbo_size;
    err = cudaGraphicsResourceGetMappedPointer((void**)&pbo_ptr, &pbo_size, cudaResource2);
    if (err != cudaSuccess) {
        qDebug() << "cudaGraphicsResourceGetMappedPointer error:" << cudaGetErrorString(err);
        cudaGraphicsUnmapResources(1, &cudaResource2, 0);
        emit decode_please();
        return;
    }

    //CUDAカーネルの処理
    size_t pitch_pbo=width_*4;
    CUDA_IMG_Proc->Flip_RGBA_to_NV12(d_y, pitch_y, d_uv, pitch_uv,pbo_ptr, pitch_pbo,height_, width_);

    //PBOリソースのマップを解除し、制御をOpenGLに戻す
    err = cudaGraphicsUnmapResources(1, &cudaResource2, 0);
    if (err != cudaSuccess) {
        qDebug() << "cudaGraphicsUnmapResources error:" << cudaGetErrorString(err);
        emit decode_please();
        return;
    }

    if(save_encoder!=nullptr&&encode_FrameCount<=MaxFrame){
        save_encoder->encode(d_y,pitch_y,d_uv,pitch_uv);
        encode_FrameCount++;
    }else{
        delete save_encoder;
        save_encoder=nullptr;
        encode_FrameCount=0;

        emit encode_finished();
    }
}

//エンコード時
void GLWidget::encode_mode(int flag){
    if(flag==STATE_ENCODING){
        if(save_encoder==nullptr){
            save_encoder = new save_encode(height_,width_);
        }
        encode_state=flag;
    }else{
        encode_state=flag;
    }
}

//ヒストグラム周り
void GLWidget::initCudaHist() {
    // --- 1. 既存 CUDA Graphics Resource を解放 ---
    if (cudaResource_hist) {
        cudaGraphicsUnregisterResource(cudaResource_hist);
        cudaResource_hist = nullptr;
    }
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
        d_hist_stats = nullptr;
    }

    // --- 4. 新規登録 fbo → cudaResource_hist ---
    cudaError_t err = cudaGraphicsGLRegisterImage(
        &cudaResource_hist,
        fboTextureID,
        GL_TEXTURE_2D,
        cudaGraphicsRegisterFlagsNone
        );
    if (err != cudaSuccess) {
        qDebug() << "cudaGraphicsGLRegisterImage failed:" << cudaGetErrorString(err);
    }

    // --- 5. 新規 VBO 作成 ---
    glGenBuffers(1, &vbo_hist);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_hist);
    glBufferData(GL_ARRAY_BUFFER, num_bins * 3 * 3 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // --- 6. VBO → CUDA Resource 登録 ---
    err = cudaGraphicsGLRegisterBuffer(
        &cudaResource_hist_draw,
        vbo_hist,
        cudaGraphicsRegisterFlagsWriteDiscard
        );
    if (err != cudaSuccess) {
        qDebug() << "cudaGraphicsGLRegisterBuffer failed:" << cudaGetErrorString(err);
    }

    // --- 7. CUDA 側のメモリを確保 ---
    cudaMalloc(&d_hist_stats, sizeof(HistStats));
    cudaMalloc(&d_hist_data, sizeof(HistData));
}

//OpenGLからCUDAへ転送+ヒストグラム解析
void GLWidget::histgram_Analysys(){
    if (!cudaResource_hist) {
        qDebug() << "cudaResource_analysis is nullptr, can't map";
        emit decode_please();
        return;
    }
    if (!cudaResource2) {
        qDebug() << "cudaResource_analysis is nullptr, can't map";
        emit decode_please();
        return;
    }

    //一括マップ
    cudaGraphicsResource* resources[] = { cudaResource_hist, cudaResource_hist_draw };
    cudaError_t err = cudaGraphicsMapResources(2, resources, 0);
    if (err != cudaSuccess) {
        qDebug() << "cudaGraphicsMapResources error:" << cudaGetErrorString(err);
        return;
    }

    //ヒストグラム計算
    {
        //サブリソースの cudaArray を取得
        cudaArray_t cuArray;
        err = cudaGraphicsSubResourceGetMappedArray(&cuArray, cudaResource_hist, 0, 0);
        if (err != cudaSuccess) {
            qDebug() << "cudaGraphicsSubResourceGetMappedArray error:" << cudaGetErrorString(err);
            cudaGraphicsUnmapResources(2, resources, 0);
            return;
        }

        //テクスチャオブジェクトを作る（読み出し用）
        cudaResourceDesc resDesc = {};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray;

        cudaTextureDesc texDesc = {};
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModePoint; // 点サンプリング
        texDesc.readMode = cudaReadModeElementType; // 生の要素（uchar4）を取りたい
        texDesc.normalizedCoords = 0;

        cudaTextureObject_t texObj = 0;
        err = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
        if (err != cudaSuccess) {
            qDebug() << "cudaCreateTextureObject error:" << cudaGetErrorString(err);
            cudaGraphicsUnmapResources(2, resources, 0);
            return;
        }

        //カーネル呼び出し（例: compute_histogram_from_texture）
        cudaMemset(d_hist_data, 0, sizeof(HistData));
        CUDA_IMG_Proc->calc_histgram(d_hist_data,texObj,width_,height_);

        //テクスチャオブジェクト破棄
        cudaDestroyTextureObject(texObj);
    }

    //ヒストグラム値解析
    {
        CUDA_IMG_Proc->histogram_status(d_hist_data,d_hist_stats);
        // cudaEventRecord(e, 0);
        // cudaStreamWaitEvent(stream, e, 0);
        cudaMemcpyAsync(&h_hist_stats, d_hist_stats,sizeof(HistStats),cudaMemcpyDeviceToHost,stream);
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
        cudaGraphicsUnmapResources(2, resources, 0);
        return;
    }

    //Stream同期
    //cudaStreamSynchronize(stream);

    //ヒストグラム描画
    {
        // // 1) マップ
        // cudaError_t err = cudaGraphicsMapResources(1, &cudaResource_hist_draw, 0);
        // if (err != cudaSuccess) {
        //     qDebug() << "cudaGraphicsMapResources error:" << cudaGetErrorString(err);
        //     return;
        // }

        // // 2) サブリソースの cudaArray を取得
        // cudaArray_t cuArray;
        // err = cudaGraphicsSubResourceGetMappedArray(&cuArray, cudaResource_hist_draw, 0, 0);
        // if (err != cudaSuccess) {
        //     qDebug() << "cudaGraphicsSubResourceGetMappedArray error:" << cudaGetErrorString(err);
        //     cudaGraphicsUnmapResources(1, &cudaResource_hist_draw, 0);
        //     return;
        // }

        // // 3) SurfaceObject 作成
        // cudaResourceDesc desc = {};
        // desc.resType = cudaResourceTypeArray;
        // desc.res.array.array = cuArray;

        // cudaSurfaceObject_t surfOut;
        // cudaCreateSurfaceObject(&surfOut, &desc);

        // // 4) CUDA カーネル呼び出し
        // CUDA_IMG_Proc->draw_histgram(surfOut, width_, height_, d_hist_r, d_hist_g, d_hist_b,d_max_r,d_max_g,d_max_b);

        // // 5) surface object 破棄（重要）
        // cudaDestroySurfaceObject(surfOut);

        // // 6) アンマップ
        // err = cudaGraphicsUnmapResources(1, &cudaResource_hist_draw, 0);
        // if (err != cudaSuccess) {
        //     qDebug() << "cudaGraphicsUnmapResources error:" << cudaGetErrorString(err);
        //     emit decode_please();
        //     return;
        // }
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

    return labels;
}

//CUDAとopenGLのデバイス設定
void GLWidget::getCudaCapabilityForOpenGLGPU()
{
    // OpenGL が使用している CUDA デバイスを取得
    unsigned int deviceCount = 0;
    int deviceIDs[8] = {0};

    cudaError_t err = cudaGLGetDevices(
        &deviceCount,
        deviceIDs,
        8,
        cudaGLDeviceListAll
        );

    if (err != cudaSuccess || deviceCount == 0) {
        qDebug() << "cudaGLGetDevices failed:" << cudaGetErrorString(err);
    }

    int glCudaDevice = deviceIDs[0];  // OpenGLが使っているCUDAデバイス

    // CUDA の使用デバイスを設定
    //cudaSetDevice(glCudaDevice);

    // 確認用にプロパティを取得
    cudaGetDeviceProperties(&g_prop, glCudaDevice);

    qDebug() << "OpenGL GPU に CUDA を同期:";
    qDebug() << "  CUDA Device ID =" << glCudaDevice;
    qDebug() << "  GPU Name =" << g_prop.name;
    qDebug() << "  Compute Capability ="
             << g_prop.major << "." << g_prop.minor;
}
