#ifndef GLWIDGET_H
#define GLWIDGET_H

#include "save_encode.h"
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

public slots:
    void uploadToGLTexture(uint8_t *d_rgba,int a,int width,int height,size_t pitch_rgba);

signals:
    void decode_please();
    void initialized();
    void encode_finished();

public:
    explicit GLWidget(QWindow *parent = nullptr);
    ~GLWidget();
    void downloadToGLTexture();
    void initTextureCuda(int width,int height);
    void initCudaTexture(int width,int height);
    void encode_mode(bool flag);
    void encode_maxFrame(int maxFrame);

protected:
    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;

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
    uint8_t *d_rgba = nullptr;
    size_t pitch_rgba = 0;

    int width_, height_;

    bool encode_flag=false;
    save_encode* save_encoder=nullptr;

    int FrameNo=0;
    int MaxFrame=0;
};


#endif // GLWIDGET_H
