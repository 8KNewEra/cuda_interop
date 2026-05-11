#ifndef DXWIDGET_H
#define DXWIDGET_H

#include "qpainter.h"
#pragma once

#include "src/videoprocess/save_encode.h"
#include "src/imageprocess/cuda_imageprocess.h"
#include "src/main/__global__.h"

#include <QWidget>
#include <QElapsedTimer>
#include <QVector>
#include <QByteArray>

#include <d3d11.h>
#include <dxgi.h>
#include <dxgi1_2.h>
#include <cuda_runtime.h>

#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include <d3dcompiler.h>
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib,"d3d11.lib")
#pragma comment(lib,"dxgi.lib")

struct Vertex
{
    float x, y;
    float u, v;
};

class DXWidget : public QWidget
{
    Q_OBJECT

signals:
    void decode_please();
    void initialized();
    void encode_finished();

public:
    explicit DXWidget(QWidget* parent = nullptr);
    ~DXWidget();

    // encode制御
    void encode_mode(int flag);
    int MinFrame = 0;
    int MaxFrame = 0;
    int encode_FrameCount = 0;

    //DXレンダリング
    void uploadToDXTexture(VideoFrame Frame);
    void FBO_Rendering(VideoFrame Frame);

    //DX周りリサイズ
    void DXresize();
    void DXreset();

    // 画像処理
    bool filter_change_flag = true;
    double fps = 0.0;

protected:
    void resizeEvent(QResizeEvent* event) override;

private:
    // --------------------------
    // DirectX11 core
    // --------------------------
    bool d3d_initialized = false;
    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* context = nullptr;
    IDXGISwapChain* swapChain = nullptr;
    ID3D11RenderTargetView* rtv = nullptr;
    ID3D11VertexShader* vs = nullptr;
    ID3D11PixelShader*  ps = nullptr;
    ID3D11InputLayout*  inputLayout = nullptr;
    ID3D11Buffer* quadVB = nullptr;
    ID3D11SamplerState* samplerLinear = nullptr;

    //Texture
    ID3D11Texture2D* inputTexture = nullptr;
    ID3D11ShaderResourceView* inputSRV = nullptr;
    ID3D11Texture2D* tempTexture = nullptr;
    ID3D11RenderTargetView* tempRTV = nullptr;
    ID3D11ShaderResourceView* tempSRV = nullptr;
    ID3D11Texture2D* outputTexture = nullptr;
    ID3D11RenderTargetView* outputRTV = nullptr;
    ID3D11ShaderResourceView* outputSRV = nullptr;

    // DirectX 初期化系
    bool initializeD3D();
    bool D3D11_sharder_compile();
    bool createRenderTargetView();
    void releaseRenderTargetView();
    bool createInputLayout(ID3DBlob* vsBlob);
    bool createQuadVB();
    bool createSampler();
    void Monitor_Rendering(VideoFrame Frame);
    void initCudaTexture(int width, int height);
    void initTextureCuda(int width, int height);
    void downloadToDXTexture_and_Encode(VideoFrame Frame);
    void queryCudaGPUs();
    void getCudaDeviceIDFromD3D11();

    const char* g_VSCode = R"(
        struct VS_IN {
            float2 pos : POSITION;
            float2 uv  : TEXCOORD0;
        };

        struct VS_OUT {
            float4 pos : SV_POSITION;
            float2 uv  : TEXCOORD0;
        };

        VS_OUT main(VS_IN input)
        {
            VS_OUT o;
            o.pos = float4(input.pos, 0, 1);
            o.uv  = input.uv;
            return o;
        }
    )";
    const char* g_PSCode = R"(
        Texture2D tex0 : register(t0);
        SamplerState samp0 : register(s0);

        float4 main(float4 pos : SV_POSITION, float2 uv : TEXCOORD0) : SV_TARGET
        {
            return tex0.Sample(samp0, uv);
        }
    )";

    // --------------------------
    // CUDA Interop resources
    // --------------------------
    cudaGraphicsResource* cudaInputRes = nullptr;
    cudaGraphicsResource* cudaOutputRes = nullptr;
    cudaStream_t interop_stream = nullptr;
    cudaEvent_t  interop_event  = nullptr;
    CUDA_ImageProcess* CUDA_IMG_Proc = nullptr;
    ID3D11Buffer* histVB = nullptr;
    cudaGraphicsResource* cudaResource_hist = nullptr;      // fboTexture相当 (D3D11 texture)
    cudaGraphicsResource* cudaResource_hist_draw = nullptr; // histVB相当 (D3D11 buffer)

    // --------------------------
    // 状態管理
    // --------------------------
    QPainter painter;
    int width_ = 0;
    int height_ = 0;
    int FrameNo = 0;
    const DecodeInfo& VideoInfo = DecodeInfoManager::getInstance().getSettings();

    // --------------------------
    // Encode
    // --------------------------
    int encode_state = STATE_NOT_ENCODE;
    int prev_encode_state = STATE_NOT_ENCODE;
    save_encode* save_encoder = nullptr;

    // --------------------------
    // fps
    // --------------------------
    QElapsedTimer fpsTimer;
    int fpsCount = 0;

    // --------------------------
    // Histogram
    // --------------------------
    cudaStream_t hist_stream = nullptr;
    cudaEvent_t  hist_event  = nullptr;
    HistData*  d_hist_data = nullptr;
    HistStats* d_hist_stats = nullptr;
    HistData  h_hist_data;
    HistStats h_hist_stats;
    int num_bins = 256;
    int line_y1, line_y2, line_y3, line_y4;
    void initCudaHist();
    void histgram_Analysys();
    std::vector<int> make_nice_y_labels(int max_value);

    // --------------------------
    // viewport
    // --------------------------
    float monitor_scaling = 1.0f;
    int viewportWidth = 0;
    int viewportHeight = 0;
    int x0 = 0, y0 = 0, x1 = 0, y1 = 0;

    // --------------------------
    // Audio
    // --------------------------
    bool audio_mode = false;
    QVector<QByteArray> audio_pcm{};
};

#endif // DXWIDGET_H
