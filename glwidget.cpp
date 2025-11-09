#include "glwidget.h"
#include "qdir.h"
#include <QDebug>
#include "decode_thread.h"

GLWidget::GLWidget(QWindow *parent)
    :  QOpenGLWindow(NoPartialUpdate, parent),
    cudaResource1(nullptr),
    cudaResource2(nullptr),
    inputTextureID(0)
{
    sobelfilterEnabled = 0;

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
    bool ok1 = program.addShaderFromSourceFile(QOpenGLShader::Vertex, "../../shaders/sobel.vert");
    bool ok2 = program.addShaderFromSourceFile(QOpenGLShader::Fragment, "../../shaders/sobel.frag");

    if (!ok1 || !ok2) {
        qDebug() << "Shader compile failed:" << program.log();
        return;
    }

    if (!program.link()) {
        qDebug() << "Shader link failed:" << program.log();
        return;
    }

    // === Uniform location 取得 ===
    shader.progId = program.programId();
    shader.loc_tex           = glGetUniformLocation(shader.progId, "tex");
    shader.loc_texelSize     = glGetUniformLocation(shader.progId, "texelSize");
    shader.loc_filterEnabled = glGetUniformLocation(shader.progId, "u_filterEnabled");

    // === 初期化完了 ===
    fpsTimer.start();
    initialize_completed_flag = true;
    emit initialized();
}

void GLWidget::OpenGL_Rendering(){
    //fboターゲット
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glViewport(0, 0, width_, height_);
    glClear(GL_COLOR_BUFFER_BIT);

    //シェーダの設定
    glUseProgram(shader.progId);

    // uniform 設定
    glUniform1i(shader.loc_tex, 0);
    glUniform2f(shader.loc_texelSize, 1.0f / width_, 1.0f / height_);
    glUniform1i(shader.loc_filterEnabled, sobelfilterEnabled);

    //入力テクスチャ(CUDAから来たもの)を設定
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, inputTextureID);

    //描画エリア、fboTextureIdに描画
    glBindVertexArray(vao);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
    glBindVertexArray(0);
    glUseProgram(0);

    if (encode_state==STATE_ENCODING) {
        // GPUエンコード用処理
        downloadToGLTexture_and_Encode();
    } else if(encode_state==STATE_NOT_ENCODE){
        //画面に描画
        Monitor_Rendering();
        fpsCount++;
    }
}

//一時停止用
void GLWidget::Monitor_Rendering(){
    // QElapsedTimer timer;
    // timer.start();

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

            glViewport(20, 20, 720, 480);

            // 投影・モデルビューを正規化(0..1)
            glMatrixMode(GL_PROJECTION);
            glLoadIdentity();
            glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0);
            glMatrixMode(GL_MODELVIEW);
            glLoadIdentity();

            // 背景（半透明）
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glColor4f(0.0f, 0.0f, 0.0f, 0.5f);
            glBegin(GL_QUADS);
            glVertex2f(0.0f, 0.0f);
            glVertex2f(1.0f, 0.0f);
            glVertex2f(1.0f, 1.0f);
            glVertex2f(0.0f, 1.0f);
            glEnd();

            // --- 4x4 グリッド線を描く ---
            int num_div = 4; // 4x4
            glLineWidth(2.0f);
            glColor4f(1.0f, 1.0f, 1.0f, 0.45f); // 薄い白
            glBegin(GL_LINES);
            for (int i = 0; i <= num_div; ++i) {
                float t = (float)i / (float)num_div; // 0..1
                // 垂直線 (X = t)
                glVertex2f(t, 0.0f);
                glVertex2f(t, 1.0f);
                // 水平線 (Y = t)
                glVertex2f(0.0f, t);
                glVertex2f(1.0f, t);
            }
            glEnd();

            // --- 目盛り（tick）を OpenGL で描く（短い線） ---
            glLineWidth(3.0f);
            glColor4f(1.0f, 1.0f, 1.0f, 0.8f);
            glBegin(GL_LINES);
            // X 軸目盛り（下辺）
            for (int i = 0; i <= num_div; ++i) {
                float t = (float)i / (float)num_div;
                float tick_h = 0.03f; // 目盛りの長さ（正規化Y）
                glVertex2f(t, 0.0f);
                glVertex2f(t, tick_h);
            }
            // Y 軸目盛り（左辺）
            for (int i = 0; i <= num_div; ++i) {
                float t = (float)i / (float)num_div;
                float tick_w = 0.03f;
                glVertex2f(0.0f, t);
                glVertex2f(tick_w, t);
            }
            glEnd();

            // --- ヒストグラム本体 ---
            glBindBuffer(GL_ARRAY_BUFFER, vbo_hist);
            glEnableClientState(GL_VERTEX_ARRAY);
            glVertexPointer(3, GL_FLOAT, 0, 0);

            glLineWidth(1.2f);
            glColor4f(1.0f, 0.0f, 0.0f, 0.9f);
            glDrawArrays(GL_LINE_STRIP, 0, num_bins);       // R
            glColor4f(0.0f, 1.0f, 0.0f, 0.9f);
            glDrawArrays(GL_LINE_STRIP, num_bins, num_bins); // G
            glColor4f(0.0f, 0.0f, 1.0f, 0.9f);
            glDrawArrays(GL_LINE_STRIP, num_bins*2, num_bins); // B

            glDisableClientState(GL_VERTEX_ARRAY);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            glDisable(GL_BLEND);
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
            painter.begin(this);
            painter.setPen(Qt::white);
            painter.setFont(QFont("Consolas", 16));
            painter.drawText(2, 20, "OpenGL Device:" + QString::fromLatin1((const char*)glGetString(GL_RENDERER))+"\n");
            painter.drawText(2, 40, QString("FPS: %1").arg(fps, 0, 'f', 1));
            painter.drawText(2, 60, "GPU Usage:" + QString::number(g_gpu_usage) +"% \n");
            painter.drawText(2, 80, "File Name:" + QString::fromStdString(VideoInfo.Name)+"\n");
            painter.drawText(2, 100, "Decorder:" + QString::fromStdString(VideoInfo.Codec)+"\n");
            painter.drawText(2, 120, "Resolution:" + QString::number(VideoInfo.width)+"×"+QString::number(VideoInfo.height)+"\n");
            painter.drawText(2, 140, "Video Framerate:" + QString::number(VideoInfo.fps)+"\n");
            painter.drawText(2, 160, "Max Frame:" + QString::number(VideoInfo.max_framesNo)+"\n");
            painter.drawText(2, 180, "Current Frame:" + QString::number(VideoInfo.current_frameNo)+"\n");
            // painter.drawText(320, 20, "R: min:" + QString::number(r_stats.minVal)+" max:" + QString::number(r_stats.maxVal)+" avg:"+QString::number(r_stats.average)+"\n");
            // painter.drawText(320, 40, "G: min:" + QString::number(g_stats.minVal)+" max:" + QString::number(g_stats.maxVal)+" avg:"+QString::number(g_stats.average)+"\n");
            // painter.drawText(320, 60, "B: min:" + QString::number(b_stats.minVal)+" max:" + QString::number(b_stats.maxVal)+" avg:"+QString::number(b_stats.average)+"\n");
            painter.end();
        }
    }

    context()->swapBuffers(context()->surface());

    // double seconds = timer.nsecsElapsed() / 1e6; // ナノ秒 →  ミリ秒
    // qDebug()<<seconds;
}

//アスペクト比を合わせてリサイズ
void GLWidget::GLresize() {
    //現在のウィンドウのDPIスケールを取得
    float dpr = devicePixelRatio();

    //実ピクセル単位のビューポートサイズを取得
    viewportWidth  = static_cast<GLint>(this->width()  * dpr);
    viewportHeight = static_cast<GLint>(this->height() * dpr);

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

        //新しいサイズで再作成
        glGenBuffers(1, &output_pbo);
        glBindBuffer(GL_PIXEL_PACK_BUFFER, output_pbo);
        glBufferData(GL_PIXEL_PACK_BUFFER, width * height * 4, NULL, GL_STREAM_READ);
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glGenTextures(1, &fboTextureID);
        glBindTexture(GL_TEXTURE_2D, fboTextureID);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fboTextureID, 0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        //outputpbo登録
        cudaError_t err = cudaGraphicsGLRegisterBuffer(&cudaResource2, output_pbo, cudaGraphicsRegisterFlagsReadOnly);
        if (err != cudaSuccess) {
            qDebug() << "cudaGraphicsGLRegisterBuffer (output_pbo) error:" << cudaGetErrorString(err);
            cudaResource2 = nullptr;
        }
}

//ヒストグラム周り
void GLWidget::initCudaHist(int width,int height){
    if (cudaResource_hist) {
        cudaGraphicsUnregisterResource(cudaResource_hist);
        cudaResource_hist = nullptr;
    }
    if (cudaResource_hist_draw) {
        cudaGraphicsUnregisterResource(cudaResource_hist_draw);
        cudaResource_hist_draw = nullptr;
    }

    //fbo→cuda_histgram
    cudaError_t err = cudaGraphicsGLRegisterImage(
        &cudaResource_hist,
        fboTextureID,
        GL_TEXTURE_2D,
        cudaGraphicsRegisterFlagsNone // 読み取り専用なら ReadOnly を指定可能
        );
    if (err != cudaSuccess) {
        qDebug() << "cudaGraphicsGLRegisterImage failed:" << cudaGetErrorString(err);
    }

    //histgram vbo
    glGenBuffers(1, &vbo_hist);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_hist);
    glBufferData(GL_ARRAY_BUFFER, num_bins * 3 * 3 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    err = cudaGraphicsGLRegisterBuffer(
        &cudaResource_hist_draw,
        vbo_hist,
        cudaGraphicsRegisterFlagsWriteDiscard
        );
    if(err != cudaSuccess) {
        qDebug() << "cudaGraphicsGLRegisterBuffer failed:" << cudaGetErrorString(err);
    }

    //ヒストグラムfbo
    glGenFramebuffers(1, &fbo_hist);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_hist);
    glGenTextures(1, &fboHistTextureID);
    glBindTexture(GL_TEXTURE_2D, fboHistTextureID);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fboHistTextureID, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
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
    // if (d_hist_r) {
    //     cudaFree(d_hist_r);
    //     d_hist_r = nullptr;
    // }
    // if (d_hist_g) {
    //     cudaFree(d_hist_g);
    //     d_hist_g = nullptr;
    // }
    // if (d_hist_b) {
    //     cudaFree(d_hist_b);
    //     d_hist_b = nullptr;
    // }

    //再確保
    cudaMallocPitch(&d_y, &pitch_y, width, height);
    cudaMallocPitch(&d_uv, &pitch_uv, width, height / 2);
    cudaMallocPitch(&d_rgba, &pitch_rgba, width * 4, height);

    //ヒストグラム
    cudaMalloc(&d_hist_r, num_bins * sizeof(unsigned int));
    cudaMalloc(&d_hist_g, num_bins * sizeof(unsigned int));
    cudaMalloc(&d_hist_b, num_bins * sizeof(unsigned int));
    cudaMalloc(&d_max_r, sizeof(unsigned int));
    cudaMalloc(&d_max_g, sizeof(unsigned int));
    cudaMalloc(&d_max_b, sizeof(unsigned int));
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
        initCudaHist(VideoInfo.width,VideoInfo.height);
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

    OpenGL_Rendering();
}

//OpenGLからCUDAへ転送+画像解析
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

    //ヒストグラム計算
    {
        // 1) マップ
        cudaError_t err = cudaGraphicsMapResources(1, &cudaResource_hist, 0);
        if (err != cudaSuccess) {
            qDebug() << "cudaGraphicsMapResources error:" << cudaGetErrorString(err);
            return;
        }

        // 2) サブリソースの cudaArray を取得
        cudaArray_t cuArray;
        err = cudaGraphicsSubResourceGetMappedArray(&cuArray, cudaResource_hist, 0, 0);
        if (err != cudaSuccess) {
            qDebug() << "cudaGraphicsSubResourceGetMappedArray error:" << cudaGetErrorString(err);
            cudaGraphicsUnmapResources(1, &cudaResource_hist, 0);
            return;
        }

        // 3) テクスチャオブジェクトを作る（読み出し用）
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
            cudaGraphicsUnmapResources(1, &cudaResource_hist, 0);
            return;
        }

        // 4) カーネル呼び出し（例: compute_histogram_from_texture）
        cudaMemset(d_hist_r, 0, 256 * sizeof(uint32_t));
        cudaMemset(d_hist_g, 0, 256 * sizeof(uint32_t));
        cudaMemset(d_hist_b, 0, 256 * sizeof(uint32_t));
        cudaMemset(d_max_r, 0, sizeof(unsigned int));
        cudaMemset(d_max_g, 0, sizeof(unsigned int));
        cudaMemset(d_max_b, 0, sizeof(unsigned int));
        CUDA_IMG_Proc->calc_histgram(d_hist_r,d_hist_g,d_hist_b,d_max_r,d_max_g,d_max_b,texObj,width_,height_);
        //CUDA_IMG_Proc->max_histgram(d_max_r,d_max_g,d_max_b,d_hist_r,d_hist_g,d_hist_b);

        //GPU → CPU コピー
        // uint32_t h_hist_r[256];
        // uint32_t h_hist_g[256];
        // uint32_t h_hist_b[256];
        // cudaMemcpy(h_hist_r, d_hist_r, 256 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        // cudaMemcpy(h_hist_g, d_hist_g, 256 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        // cudaMemcpy(h_hist_b, d_hist_b, 256 * sizeof(uint32_t), cudaMemcpyDeviceToHost); // 統計計算

        // // 最初の10個を表示
        // qDebug() << "R histogram (0-9):";
        // for (int i = 0; i < 10; ++i) qDebug() << i << ":" << h_hist_r[i];

        // qDebug() << "G histogram (0-9):";
        // for (int i = 0; i < 10; ++i) qDebug() << i << ":" << h_hist_g[i];

        // qDebug() << "B histogram (0-9):";
        // for (int i = 0; i < 10; ++i) qDebug() << i << ":" << h_hist_b[i];


        //r_stats = computeHistStats(h_hist_r);
        // g_stats = computeHistStats(h_hist_g);
        // b_stats = computeHistStats(h_hist_b);

        // 5) テクスチャオブジェクト破棄
        cudaDestroyTextureObject(texObj);

        //PBOリソースのマップを解除し、制御をOpenGLに戻す
        err = cudaGraphicsUnmapResources(1, &cudaResource_hist, 0);
        if (err != cudaSuccess) {
            qDebug() << "cudaGraphicsUnmapResources error:" << cudaGetErrorString(err);
            emit decode_please();
            return;
        }
    }

    //OpenGL VBO転送
    {
        // 1) マップ
        cudaError_t err = cudaGraphicsMapResources(1, &cudaResource_hist_draw, 0);
        if (err != cudaSuccess) {
            qDebug() << "cudaGraphicsMapResources error:" << cudaGetErrorString(err);
            return;
        }

        float* d_vbo_ptr = nullptr;
        size_t vbo_size = 0;
        err = cudaGraphicsResourceGetMappedPointer((void**)&d_vbo_ptr, &vbo_size, cudaResource_hist_draw);
        if (err != cudaSuccess) {
            qDebug() << "cudaGraphicsResourceGetMappedPointer error:" << cudaGetErrorString(err);
            cudaGraphicsUnmapResources(1, &cudaResource_hist_draw, 0);
            return;
        }

        CUDA_IMG_Proc->copy_histgram_vbo(d_vbo_ptr,num_bins,d_hist_r,d_hist_g,d_hist_b,d_max_r,d_max_g,d_max_b);

        // 6) アンマップ
        err = cudaGraphicsUnmapResources(1, &cudaResource_hist_draw, 0);
        if (err != cudaSuccess) {
            qDebug() << "cudaGraphicsUnmapResources error:" << cudaGetErrorString(err);
            emit decode_please();
            return;
        }

        // // CPU側で受け取る変数
        // unsigned int h_max_r = 0;

        // // デバイス→ホスト転送
        // cudaDeviceSynchronize();
        // err = cudaMemcpy(&h_max_r, d_max_r, sizeof(unsigned int), cudaMemcpyDeviceToHost);

        // if (err != cudaSuccess) {
        //     qDebug() << "cudaMemcpy d_max_r failed:" << cudaGetErrorString(err);
        // } else {
        //     qDebug() << "GPU d_max_r =" << h_max_r;
        // }

        // // VBO を CPU にマップ
        // glBindBuffer(GL_ARRAY_BUFFER, vbo_hist);
        // float* ptr = (float*)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);
        // if(ptr) {
        //     int num_bins = 256;
        //     int r_offset = 0;
        //     int g_offset = num_bins * 3;      // 緑の先頭
        //     int b_offset = num_bins * 3 * 2;  // 青の先頭

        //     for(int i = 0; i < num_bins; i++){
        //         float r = ptr[r_offset + i*3 + 1]; // y 値が 1 つ目の頂点
        //         float g = ptr[g_offset + i*3 + 1];
        //         float b = ptr[b_offset + i*3 + 1];
        //         qDebug() << i << ": R=" << r << " G=" << g << " B=" << b;
        //     }

        //     glUnmapBuffer(GL_ARRAY_BUFFER);
        // } else {
        //     qDebug() << "glMapBuffer failed";
        // }

        // glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

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

//ヒストグラム解析
HistStats GLWidget::computeHistStats(const uint32_t hist[256])
{
    HistStats s;
    s.minVal = -1;
    s.maxVal = -1;

    // 合計値と総カウント数
    uint64_t sum = 0;
    uint64_t totalCount = 0;

    // min
    for (int i = 0; i < 256; i++) {
        if (hist[i] > 0) {
            s.minVal = i;
            break;
        }
    }

    // max
    for (int i = 255; i >= 0; i--) {
        if (hist[i] > 0) {
            s.maxVal = i;
            break;
        }
    }

    // average = (Σ i * hist[i]) / (Σ hist[i])
    for (int i = 0; i < 256; i++) {
        sum += (uint64_t)i * hist[i];
        totalCount += hist[i];
    }

    double avg = (totalCount > 0) ? (double)sum / totalCount : 0.0;

    // 小数点第2位で四捨五入 → 小数点第1位にする
    s.average = std::round(avg * 10.0) / 10.0;

    return s;
}
