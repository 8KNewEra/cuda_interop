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
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudafilters.hpp>
#include <QTimer>

class GLWidget : public QOpenGLWindow, protected QOpenGLFunctions_3_3_Core
{
    Q_OBJECT

public slots:
    void uploadToGLTexture(cv::cuda::GpuMat frame,int a);

signals:
    void decode_please();
    void initialized();
    void encode_finished();

public:
    explicit GLWidget(QWindow *parent = nullptr);
    ~GLWidget();
    void downloadToGLTexture();
    void initGPUMat(cv::Size targetSize,cv::cuda::GpuMat& frame);
    void initFBO();
    void initCudaTexture();
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

    int width_, height_;
    cv::cuda::GpuMat gpuResized,gpuRGBA1;
    cv::cuda::GpuMat gpuRGBA2,flipped;

    bool encode_flag=false;
    save_encode* save_encoder=nullptr;

    int FrameNo=0;
    int MaxFrame=0;
};


#endif // GLWIDGET_H
