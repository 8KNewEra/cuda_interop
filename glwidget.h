#ifndef GLWIDGET_H
#define GLWIDGET_H

#include "save_encode.h"
#include "cuda_imageprocess.h"
#include <QOpenGLFunctions_3_3_Core>
#pragma once
#include <QOpenGLWindow>
#include <QOpenGLShader>
#include <QOpenGLShaderProgram>
#include <QOpenGLFunctions>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <QTimer>

class GLWidget : public QOpenGLWindow, protected QOpenGLFunctions_3_3_Core
{
    Q_OBJECT
signals:
    void decode_please();
    void initialized();
    void encode_finished();

public:
    explicit GLWidget(QWindow *parent = nullptr);
    ~GLWidget();
    void initTextureCuda(int width,int height);
    void initCudaTexture(int width,int height);
    void initCudaMalloc(int width,int height);
    void downloadToGLTexture();
    void uploadToGLTexture(uint8_t* d_y, size_t pitch_y,uint8_t* d_uv, size_t pitch_uv,int height, int width,int a);
    void encode_mode(bool flag);
    void encode_maxFrame(int maxFrame);
    void GLresize();
    void OpenGL_Rendering();
    void GLreset();

protected:
    void initializeGL() override;

private:
    bool initialize_completed_flag=false;
    QOpenGLShaderProgram program;
    int sobelfilterEnabled;

    cudaGraphicsResource* cudaResource1;
    cudaGraphicsResource* cudaResource2;
    GLuint inputTextureID;  // ← 入力用
    GLuint fboTextureID;    // ← 出力先（FBOバインド用）
    GLuint fbo = 0;
    GLuint vao = 0;
    GLuint vbo = 0;
    GLuint input_pbo=0;
    GLuint output_pbo=0;

    int width_, height_;
    // 描画領域を計算
    GLint x0, y0, x1, y1;

    bool encode_flag=false;
    save_encode* save_encoder=nullptr;
    CUDA_ImageProcess* CUDA_IMG_Proc=nullptr;

    int FrameNo=0;
    int MaxFrame=0;

    uint8_t *d_y = nullptr, *d_uv = nullptr,*d_rgba=nullptr;
    size_t pitch_y = 0, pitch_uv = 0,pitch_rgba=0;
};


#endif // GLWIDGET_H
