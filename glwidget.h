#ifndef GLWIDGET_H
#define GLWIDGET_H

#include "save_encode.h"
#include "cuda_imageprocess.h"
#include "__global__.h"
#include <QOpenGLFunctions_4_5_Core>
#pragma once
#include <QOpenGLWindow>
#include <QOpenGLShader>
#include <QOpenGLShaderProgram>
#include <QOpenGLFunctions>
#include <QPainter>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <QTimer>

extern int g_gpu_usage;
extern int g_cudaDeviceID;

class GLWidget : public QOpenGLWindow, protected QOpenGLFunctions_4_5_Core
{
    Q_OBJECT
signals:
    void decode_please();
    void initialized();
    void encode_finished();

public:
    explicit GLWidget(QWindow *parent = nullptr);
    ~GLWidget();
    void uploadToGLTexture(AVFrame* rgbaFrame,int a);
    void encode_mode(int flag);
    void GLresize();
    void GLreset();
    void FBO_Rendering();
    void setShaderUniformEnable();

    bool videoInfo_flag=false;
    bool histgram_flag=false;
    int MaxFrame=0;

    //画像処理
    bool filter_change_flag = true;
    int sobelfilterEnabled=0;
    int gaussianfilterEnabled=0;
    int averagingfilterEnabled=0;

protected:
    void initializeGL() override;

private:
    void initTextureCuda(int width,int height);
    void initCudaTexture(int width,int height);
    void initCudaMalloc(int width,int height);
    void setShaderUniform(int width,int height);
    void Monitor_Rendering();
    void initCudaHist();
    void histgram_Analysys();
    void downloadToGLTexture_and_Encode();
    std::vector<int> make_nice_y_labels(int max_value);
    void getCudaCapabilityForOpenGLGPU();

    //CUDA Interop
    cudaGraphicsResource* cudaResource1;
    cudaGraphicsResource* cudaResource2;
    cudaGraphicsResource* cudaResource_hist=nullptr;
    cudaGraphicsResource* cudaResource_hist_draw=nullptr;
    CUDA_ImageProcess* CUDA_IMG_Proc=nullptr;

    //OpenGL周り
    bool initialize_completed_flag=false;
    GLuint inputTextureID;  // ← 入力用
    GLuint fboTextureID;    // ← 出力先（FBOバインド用）
    GLuint tempTextureID;
    GLuint vbo_hist = 0;
    GLuint fbo = 0;
    GLuint vao = 0;
    GLuint vbo = 0;
    GLuint input_pbo=0;
    GLuint output_pbo=0;
    GLuint tempfbo = 0;
    QPainter painter;

    //シェーダ―
    QOpenGLShaderProgram Sobel_program;
    QOpenGLShaderProgram Gaussian_program;
    QOpenGLShaderProgram Averaging_program;
    struct shader{
        GLuint progId=0;
        GLint loc_tex           = 0;
        GLint loc_texelSize     = 0;
        GLint loc_filterEnabled =0;
    };
    shader Sobel_shader;
    shader Gaussian_shader;
    shader Averaging_shader;

    //動画データ
    uint8_t *d_y = nullptr, *d_uv = nullptr,*d_rgba=nullptr;
    size_t pitch_y = 0, pitch_uv = 0,pitch_rgba=0;

    // 描画領域を計算
    float monitor_scaling=1;
    GLint viewportWidth;
    GLint viewportHeight;
    GLint x0, y0, x1, y1;

    //エンコード関連
    int encode_state=STATE_NOT_ENCODE;
    save_encode* save_encoder=nullptr;
    int encode_FrameCount=0;

    //動画情報
    int width_, height_;
    int FrameNo=0;
    const DecodeInfo& VideoInfo = DecodeInfoManager::getInstance().getSettings();

    //fpsタイマー
    QElapsedTimer fpsTimer;
    int fpsCount = 0;
    double fps = 0.0;

    //ヒストグラム
    cudaStream_t stream = nullptr;
    cudaEvent_t e = nullptr;
    HistData* d_hist_data = nullptr;
    HistStats* d_hist_stats = nullptr;
    HistData h_hist_data;
    HistStats h_hist_stats;
    int num_bins = 256;
    int line_y1,line_y2,line_y3,line_y4;
};


#endif // GLWIDGET_H
