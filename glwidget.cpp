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
    sobelfilterEnabled = 1;

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
void GLWidget::initializeGL() {
    initializeOpenGLFunctions();

    // シェーダ読み込み
    bool ok1=program.addShaderFromSourceFile(QOpenGLShader::Vertex, "../../shaders/sobel.vert");
    bool ok2 =program.addShaderFromSourceFile(QOpenGLShader::Fragment, "../../shaders/sobel.frag");
    program.link();

    if (!ok1 || !ok2) {
        qDebug() << "Shader loading failed:" << program.log();
        return;
    }

    if (!program.link()) {
        qDebug() << "Shader program link failed:" << program.log();
        return;
    }

    // 四角形頂点とテクスチャ座標
    float vertices[] = {
        -1.0f, -1.0f, 0.0f, 1.0f,
        1.0f, -1.0f, 1.0f, 1.0f,
        1.0f,  1.0f, 1.0f, 0.0f,
        -1.0f,  1.0f, 0.0f, 0.0f,
    };
    GLuint indices[] = { 0, 1, 2, 2, 3, 0 };

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    GLuint ebo;
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &ebo);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    int posLoc = 0, texLoc = 1;
    glEnableVertexAttribArray(posLoc);
    glVertexAttribPointer(posLoc, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);

    glEnableVertexAttribArray(texLoc);
    glVertexAttribPointer(texLoc, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));

    glBindVertexArray(0);

    initialize_completed_flag=true;
    emit initialized();
}

//リサイズ
void GLWidget::resizeGL(int w, int h) {
    viewport_height=h;
    viewport_width=w;
    glViewport(0, 0, w, h);
}

//描画
void GLWidget::paintGL() {
    if(encode_flag){
        //fboターゲット
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glViewport(0, 0, width_, height_);
        glClear(GL_COLOR_BUFFER_BIT);

        //シェーダの設定
        program.bind();
        program.setUniformValue("tex", 0);
        program.setUniformValue("texelSize", QVector2D(1.0f / width_, 1.0f / height_));
        program.setUniformValue("u_filterEnabled", sobelfilterEnabled);

        //入力テクスチャ(CUDAから来たもの)を設定
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, inputTextureID);

        //描画エリア、fboTextureIdに描画
        glBindVertexArray(vao);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
        glBindVertexArray(0);
        program.release();

        downloadToGLTexture();
    }else{
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glClear(GL_COLOR_BUFFER_BIT);

        program.bind();
        program.setUniformValue("tex", 0);
        program.setUniformValue("texelSize", QVector2D(1.0f / width_, 1.0f / height_));
        program.setUniformValue("u_filterEnabled", sobelfilterEnabled);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, inputTextureID);

        glBindVertexArray(vao);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
        glBindVertexArray(0);
        program.release();
    }
    g_fps+=1;
    emit decode_please();
}

//CUDA→OpenGLの初期化、登録など
void GLWidget::initCudaTexture(int width,int height) {
    //PBO作成
    glGenBuffers(1, &input_pbo);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, input_pbo);
    glBufferData(GL_PIXEL_PACK_BUFFER, width * height * 4, NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

    glGenTextures(1, &inputTextureID);
    glBindTexture(GL_TEXTURE_2D, inputTextureID);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    //CUDAGraphicResorceとして登録
    cudaError_t err = cudaGraphicsGLRegisterBuffer(&cudaResource1, input_pbo, cudaGraphicsRegisterFlagsWriteDiscard);
    if (err != cudaSuccess) {
        qDebug() << "Failed to register PBO: " << cudaGetErrorString(err);
        return;
    }
}

//OpenGL→CUDAの初期化、登録など
void GLWidget::initTextureCuda(int width,int height) {
    //PBO作成
    glGenBuffers(1, &output_pbo);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, output_pbo);
    glBufferData(GL_PIXEL_PACK_BUFFER, width * height * 4, NULL, GL_STREAM_READ);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

    //FBO登録
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glGenTextures(1, &fboTextureID);
    glBindTexture(GL_TEXTURE_2D, fboTextureID);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fboTextureID, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    //CUDAGraphicResorceとして登録
    cudaError_t err = cudaGraphicsGLRegisterBuffer(&cudaResource2, output_pbo, cudaGraphicsRegisterFlagsReadOnly);
    if (err != cudaSuccess) {
        qDebug() << "cudaGraphicsGLRegisterImage (fbo) error:" << cudaGetErrorString(err);
        cudaResource2 = nullptr;
    }
}

//初回、解像度が変わった場合再Malloc
void GLWidget::initCudaMalloc(int width,int height){
    if (!d_y||!d_uv) {
        cudaMallocPitch(&d_y, &pitch_y, width, height);
        cudaMallocPitch(&d_uv, &pitch_uv, width, height / 2);
        cudaMallocPitch(&d_rgba, &pitch_rgba, width*4, height);
    }
}

//CUDAからOpenGLへ転送
void GLWidget::uploadToGLTexture(uint8_t* d_y, size_t pitch_y,uint8_t* d_uv, size_t pitch_uv,int height, int width,int a) {
    // QElapsedTimer timer;
    // timer.start();

    FrameNo = a;
    //initialized完了チェック
    if (!initialize_completed_flag) {
        emit decode_please();
        return;
    };

    //解像度の変更に対応
    if (width != width_ || height != height_) {
        initCudaMalloc(width,height);
        initCudaTexture(width,height);
        initTextureCuda(width,height);
        width_=width;
        height_=height;
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
    size_t pitch_pbo=width*4;
    CUDA_IMG_Proc->NV12_to_RGBA(d_rgba, pitch_rgba, d_y, pitch_y, d_uv, pitch_uv, height, width);
    CUDA_IMG_Proc->Gradation(pbo_ptr,pitch_pbo,d_rgba,pitch_rgba,height,width);

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
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // double seconds = timer.nsecsElapsed() / 1e6; // ナノ秒 →  ミリ秒
    // qDebug()<<seconds;

    //描画をトリガー
    update();
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

    if(save_encoder!=nullptr){
        save_encoder->encode(d_y,pitch_y,d_uv,pitch_uv);
    }

    //qDebug()<<FrameNo<<":"<<MaxFrame;

    if(FrameNo>=MaxFrame){
        delete save_encoder;
        save_encoder=nullptr;
        encode_flag=false;

        emit encode_finished();
    }
}

void GLWidget::encode_mode(bool flag){
    if(flag==true){
        if(save_encoder==nullptr){
            save_encoder = new save_encode(height_,width_);
        }
        encode_flag=flag;
    }
}

void GLWidget::encode_maxFrame(int maxFrame){
    MaxFrame = maxFrame;
}
