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

    qDebug() << "GLWidget: Destructor called";
}

void GLWidget::initCudaTexture() {
    glGenTextures(1, &inputTextureID);
    glBindTexture(GL_TEXTURE_2D, inputTextureID);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width_, height_, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    cudaError_t err = cudaGraphicsGLRegisterImage(&cudaResource1, inputTextureID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
    if (err != cudaSuccess) {
        qDebug() << "cudaGraphicsGLRegisterImage error:" << cudaGetErrorString(err);
        // エラー発生時の処理を追加 (例えば、nullptrを設定して後続の処理をスキップするなど)
        cudaResource1 = nullptr;
        inputTextureID = 0;
    }
}

// initFBO に統合
void GLWidget::initFBO() {
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    glGenTextures(1, &fboTextureID);
    glBindTexture(GL_TEXTURE_2D, fboTextureID);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width_, height_, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fboTextureID, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // ここでCUDAに登録（1回だけ）
    cudaError_t err = cudaGraphicsGLRegisterImage(&cudaResource2, fboTextureID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly);
    if (err != cudaSuccess) {
        qDebug() << "cudaGraphicsGLRegisterImage (fbo) error:" << cudaGetErrorString(err);
        cudaResource2 = nullptr;
    }
}

void GLWidget::initGPUMat(cv::Size targetSize, cv::cuda::GpuMat& frame)
{
    // 入力フレームに合わせた型とサイズを使う
    if (gpuResized.empty() || gpuResized.size() != targetSize || gpuResized.type() != frame.type()) {
        gpuResized.create(targetSize, frame.type());
    }

    if (gpuRGBA1.empty() || gpuRGBA1.size() != targetSize) {
        gpuRGBA1.create(targetSize, CV_8UC4);
    }

    if (gpuRGBA2.empty() || gpuRGBA2.size() != targetSize) {
        gpuRGBA2.create(targetSize, CV_8UC4);
    }
}

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

void GLWidget::resizeGL(int w, int h) {
    glViewport(0, 0, w, h);
}

void GLWidget::paintGL() {
    if(encode_flag){
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glViewport(0, 0, width_, height_);
        glClear(GL_COLOR_BUFFER_BIT);

        program.bind();
        program.setUniformValue("tex", 0);
        program.setUniformValue("texelSize", QVector2D(1.0f / width_, 1.0f / height_));
        program.setUniformValue("u_filterEnabled", sobelfilterEnabled); // フィルタ無効

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, inputTextureID); // ← FBOの出力結果が入っている

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
        program.setUniformValue("u_filterEnabled", sobelfilterEnabled); // フィルタ無効

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, inputTextureID); // ← FBOの出力結果が入っている

        glBindVertexArray(vao);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
        glBindVertexArray(0);
        program.release();
    }
    g_fps+=1;
    emit decode_please();
}

void GLWidget::uploadToGLTexture(cv::cuda::GpuMat frame,int a) {
    FrameNo=a;
    //initialized完了チェック
    if(!initialize_completed_flag) {
        emit decode_please();
        return;
    };

    //画像サイズに応じてcudaテクスチャのサイズを変更
    width_=frame.cols;
    height_=frame.rows;
    cv::Size targetSize(width_, height_);
    if (gpuResized.size() != targetSize || gpuResized.type() != frame.type()) {
        initCudaTexture();
        initFBO();
        initGPUMat(targetSize,frame);
        emit decode_please();
    }

    // OpenGL用のRGBA形式に変換
    cv::cuda::cvtColor(frame, gpuRGBA1, cv::COLOR_BGR2RGBA, 0);

    if (gpuRGBA1.empty()) {
        qDebug() << "画像の読み込みに失敗しました";
        cudaGraphicsUnmapResources(1, &cudaResource1, 0); // マップ解除
        emit decode_please();
        return;
    }

    //CUDA→OpenGL転送の準備
    cudaArray_t array;
    cudaError_t err = cudaSuccess; // エラーコードを格納する変数、初期値は成功

    err = cudaGraphicsMapResources(1, &cudaResource1, 0);
    if (err != cudaSuccess) {
        qDebug() << "cudaGraphicsMapResources error:" << cudaGetErrorString(err);
        emit decode_please();
        return; // エラーが発生したら処理を中断
    }

    err = cudaGraphicsSubResourceGetMappedArray(&array, cudaResource1, 0, 0);
    if (err != cudaSuccess) {
        qDebug() << "cudaGraphicsSubResourceGetMappedArray error:" << cudaGetErrorString(err);
        cudaGraphicsUnmapResources(1, &cudaResource1, 0); // マップ解除を試みる
        emit decode_please();
        return; // エラーが発生したら処理を中断
    }

    //OpenGL用テクスチャに変換してOpenGL転送
    size_t pitch = gpuRGBA1.step;  // 実際のピッチを取得
    size_t widthBytes = width_ * 4;
    if (widthBytes <= pitch) {
        cudaMemcpy2DToArray(
            array,
            0, 0,
            gpuRGBA1.ptr(),
            pitch,
            widthBytes,
            height_,
            cudaMemcpyDeviceToDevice
            );
    } else {
        qDebug() << "Invalid pitch: widthBytes > pitch!";
        emit decode_please();
    }

    err = cudaGraphicsUnmapResources(1, &cudaResource1, 0);
    if (err != cudaSuccess) {
        qDebug() << "cudaGraphicsUnmapResources error:" << cudaGetErrorString(err);
        emit decode_please();
        return; // エラーが発生したら処理を中断
    }

    update();
}

void GLWidget::downloadToGLTexture() {
    if (!cudaResource2) {
        qDebug() << "cudaResource2 is nullptr, can't map";
        emit decode_please();
        return;
    }

    cudaError_t err;

    // マッピング
    err = cudaGraphicsMapResources(1, &cudaResource2);
    if (err != cudaSuccess) {
        qDebug() << "cudaGraphicsMapResources failed:" << cudaGetErrorString(err);
        emit decode_please();
        return;
    }

    // CUDA array を取得
    cudaArray_t cuArray;
    err = cudaGraphicsSubResourceGetMappedArray(&cuArray, cudaResource2, 0, 0);
    if (err != cudaSuccess || cuArray == nullptr) {
        qDebug() << "cudaGraphicsSubResourceGetMappedArray failed:" << cudaGetErrorString(err);
        cudaGraphicsUnmapResources(1, &cudaResource2); // 解除だけにする
        emit decode_please();
        return;
    }

    // GpuMatにコピー
    err = cudaMemcpy2DFromArray(
        gpuRGBA2.data, gpuRGBA2.step,
        cuArray, 0, 0,
        width_ * 4, height_,
        cudaMemcpyDeviceToDevice
        );
    if (err != cudaSuccess) {
        qDebug() << "cudaMemcpy2DFromArray failed:" << cudaGetErrorString(err);
        cudaGraphicsUnmapResources(1, &cudaResource2);
        emit decode_please();
        return;
    }

    // アンマップ（登録解除しない！）
    cudaGraphicsUnmapResources(1, &cudaResource2);

    if (gpuRGBA2.empty()) {
        qDebug() << "GpuMat is empty, 保存できません";
        emit decode_please();
        return;
    }

    cv::cuda::flip(gpuRGBA2, flipped, 0);  // 1 = 左右反転

    if(save_encoder!=nullptr){
        save_encoder->encode(flipped);
    }

    qDebug()<<FrameNo;

    if(FrameNo>MaxFrame){
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
