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
        downloadToGLTexture();
    } else if(encode_state==STATE_NOT_ENCODE){
        qDebug()<<"aaa";
        //画面に描画
        Monitor_Rendering();
        fpsCount++;
    }
}

//一時停止用
void GLWidget::Monitor_Rendering(){
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

    // --- FPSを算出 ---
    if (fpsTimer.elapsed() >= 1000) {  // 1000ms 経過したら
        fps = fpsCount * 1000.0 / fpsTimer.elapsed(); // FPS計算
        fpsCount = 0;
        fpsTimer.restart();
    }

    if(videoInfo_flag){
        //動画情報描画
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
        painter.end();
    }

    context()->swapBuffers(context()->surface());
}

//アスペクト比を合わせてリサイズ
void GLWidget::GLresize() {
    //現在のウィンドウのDPIスケールを取得
    float dpr = devicePixelRatio();

    //実ピクセル単位のビューポートサイズを取得
    GLint viewportWidth  = static_cast<GLint>(this->width()  * dpr);
    GLint viewportHeight = static_cast<GLint>(this->height() * dpr);

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

        //CUDA登録
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
        width_=VideoInfo.width;
        height_=VideoInfo.height;
        GLresize();
        emit decode_please();
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

//OpenGLからCUDAへ転送
void GLWidget::downloadToGLTexture() {
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

void GLWidget::encode_maxFrame(int maxFrame){
    MaxFrame = maxFrame;
}

void GLWidget::video_info_changed(bool flag){
    videoInfo_flag=flag;
}
