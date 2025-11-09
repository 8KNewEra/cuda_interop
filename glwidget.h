#ifndef GLWIDGET_H
#define GLWIDGET_H

#include "save_encode.h"
#include "cuda_imageprocess.h"
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

struct HistStats {
    int minVal;     // 最初に 1 以上が出たbin
    int maxVal;     // 最後に 1 以上が出たbin
    double average; // 平均輝度 (0～255)
};

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
    void uploadToGLTexture(uint8_t* d_y, size_t pitch_y,uint8_t* d_uv, size_t pitch_uv,int a);
    void encode_mode(int flag);
    void GLresize();
    void GLreset();
    void Monitor_Rendering();

    bool videoInfo_flag=false;
    bool histgram_flag=false;
    int MaxFrame=0;

protected:
    void initializeGL() override;

private:
    void initTextureCuda(int width,int height);
    void initCudaTexture(int width,int height);
    void initCudaHist(int width,int height);
    void initCudaMalloc(int width,int height);
    void histgram_Analysys();
    void downloadToGLTexture_and_Encode();
    void OpenGL_Rendering();
    bool initialize_completed_flag=false;
    QOpenGLShaderProgram program;
    int sobelfilterEnabled;

    cudaGraphicsResource* cudaResource1;
    cudaGraphicsResource* cudaResource2;
    cudaGraphicsResource* cudaResource_hist=nullptr;
    cudaGraphicsResource* cudaResource_hist_draw=nullptr;
    CUDA_ImageProcess* CUDA_IMG_Proc=nullptr;
    GLuint inputTextureID;  // ← 入力用
    GLuint fboTextureID;    // ← 出力先（FBOバインド用）
    GLuint fboHistTextureID;
    GLuint fbo_hist;
    GLuint vbo_hist = 0;
    GLuint fbo = 0;
    GLuint vao = 0;
    GLuint vbo = 0;
    GLuint input_pbo=0;
    GLuint output_pbo=0;
    QPainter painter;
    int width_, height_;

    // 描画領域を計算
    GLint x0, y0, x1, y1;

    //エンコード関連
    int encode_state=STATE_NOT_ENCODE;
    save_encode* save_encoder=nullptr;
    int encode_FrameCount=0;

    //動画情報
    int FrameNo=0;
    const DecodeInfo& VideoInfo = DecodeInfoManager::getInstance().getSettings();

    //動画データ
    uint8_t *d_y = nullptr, *d_uv = nullptr,*d_rgba=nullptr;
    size_t pitch_y = 0, pitch_uv = 0,pitch_rgba=0;

    //fpsタイマー
    QElapsedTimer fpsTimer;
    int fpsCount = 0;
    double fps = 0.0;

    //シェーダ―
    struct shader{
        GLuint progId=0;
        GLint loc_tex           = 0;
        GLint loc_texelSize     = 0;
        GLint loc_filterEnabled =0;
    };
    shader shader;

    //ヒストグラム
    uint32_t* d_hist_r;
    uint32_t* d_hist_g;
    uint32_t* d_hist_b;
    unsigned int *d_max_r, *d_max_g, *d_max_b;
    int num_bins = 256;

    HistStats computeHistStats(const uint32_t hist[256]);
    HistStats r_stats;
    HistStats g_stats;
    HistStats b_stats;

    GLint viewportWidth;
    GLint viewportHeight;
};


#endif // GLWIDGET_H
