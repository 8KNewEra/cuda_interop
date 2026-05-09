#include "src/main/glwidget.h"
#include "qdir.h"
#include "qtimer.h"
#include <QDebug>

#define Rerease 0

DXWidget::DXWidget(QWidget* parent)
    : QWidget(parent)
{
    setAttribute(Qt::WA_NativeWindow);
    setAttribute(Qt::WA_PaintOnScreen);
    setAttribute(Qt::WA_NoSystemBackground);

    if (CUDA_IMG_Proc == nullptr) {
        CUDA_IMG_Proc = new CUDA_ImageProcess();
    }

    cudaStreamCreate(&interop_stream);
    cudaEventCreate(&interop_event);

    cudaStreamCreate(&hist_stream);
    cudaEventCreate(&hist_event);

    if (!initializeD3D()) {
        qDebug() << "initializeD3D failed";
        return;
    }

    // ★ connect後に必ず届く
    QTimer::singleShot(0, this, [this]() {
        emit initialized();
    });

    qDebug() << "DXWidget: Constructor called";
}


DXWidget::~DXWidget()
{
    // CUDA interop unregister
    if (cudaInputRes) {
        cudaGraphicsUnregisterResource(cudaInputRes);
        cudaInputRes = nullptr;
    }

    if (cudaOutputRes) {
        cudaGraphicsUnregisterResource(cudaOutputRes);
        cudaOutputRes = nullptr;
    }

    // CUDA stream/event
    if (interop_stream) {
        cudaStreamSynchronize(interop_stream);
        cudaStreamDestroy(interop_stream);
        interop_stream = nullptr;
    }

    if (interop_event) {
        cudaEventDestroy(interop_event);
        interop_event = nullptr;
    }

    if (hist_stream) {
        cudaStreamSynchronize(hist_stream);
        cudaStreamDestroy(hist_stream);
        hist_stream = nullptr;
    }

    if (hist_event) {
        cudaEventDestroy(hist_event);
        hist_event = nullptr;
    }

    // CUDA malloc
    if (d_hist_stats) {
        cudaFree(d_hist_stats);
        d_hist_stats = nullptr;
    }

    if (d_hist_data) {
        cudaFree(d_hist_data);
        d_hist_data = nullptr;
    }

    // CUDA image process
    if (CUDA_IMG_Proc) {
        delete CUDA_IMG_Proc;
        CUDA_IMG_Proc = nullptr;
    }

    // DirectX textures
    if (inputSRV) {
        inputSRV->Release();
        inputSRV = nullptr;
    }
    if (inputTexture) {
        inputTexture->Release();
        inputTexture = nullptr;
    }

    if (outputSRV) {
        outputSRV->Release();
        outputSRV = nullptr;
    }
    if (outputTexture) {
        outputTexture->Release();
        outputTexture = nullptr;
    }

    // RenderTarget
    if (rtv) {
        rtv->Release();
        rtv = nullptr;
    }

    // SwapChain / Context / Device
    if (swapChain) {
        swapChain->Release();
        swapChain = nullptr;
    }

    if (context) {
        context->Release();
        context = nullptr;
    }

    if (device) {
        device->Release();
        device = nullptr;
    }

    qDebug() << "DXWidget: Destructor called";
}






bool DXWidget::initializeD3D()
{
    HWND hwnd = (HWND)winId();
    if (!hwnd) {
        qDebug() << "HWND is null";
        return false;
    }

    DXGI_SWAP_CHAIN_DESC sd{};
    sd.BufferCount = 2;
    sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow = hwnd;
    sd.SampleDesc.Count = 1;
    sd.Windowed = TRUE;
    sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;

    UINT createFlags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;

    IDXGIFactory1* factory = nullptr;
    HRESULT hr = CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&factory);
    if (FAILED(hr) || !factory) {
        qDebug() << "CreateDXGIFactory1 failed";
        return false;
    }

    // CUDA deviceID から PCI情報を取る
    int cudaDev = g_openglDeviceID;
    int pciBus = 0;
    int pciDevice = 0;
    int pciDomain = 0;

    cudaDeviceGetAttribute(&pciBus, cudaDevAttrPciBusId, cudaDev);
    cudaDeviceGetAttribute(&pciDevice, cudaDevAttrPciDeviceId, cudaDev);
    cudaDeviceGetAttribute(&pciDomain, cudaDevAttrPciDomainId, cudaDev);

    IDXGIAdapter1* adapter = nullptr;

    // factoryから全adapterを列挙して PCI一致のものを探す
    for (UINT i = 0; ; i++) {
        IDXGIAdapter1* tmp = nullptr;
        if (factory->EnumAdapters1(i, &tmp) != S_OK)
            break;

        DXGI_ADAPTER_DESC1 desc{};
        tmp->GetDesc1(&desc);

        // LUIDは古いCUDAだと使えないので名前比較+PCI比較が必要だが
        // DXGI側はPCIを直接出せないため、ここではとりあえず最初のadapterを使う例
        // （本当は SetupAPI を使う必要あり）

        adapter = tmp;
        break;
    }

    factory->Release();

    if (!adapter) {
        qDebug() << "No DXGI adapter found";
        return false;
    }

    D3D_FEATURE_LEVEL featureLevel;
    hr = D3D11CreateDeviceAndSwapChain(
        adapter,
        D3D_DRIVER_TYPE_UNKNOWN,
        nullptr,
        createFlags,
        nullptr,
        0,
        D3D11_SDK_VERSION,
        &sd,
        &swapChain,
        &device,
        &featureLevel,
        &context
        );

    adapter->Release();

    if (FAILED(hr)) {
        qDebug() << "D3D11CreateDeviceAndSwapChain failed";
        return false;
    }

    if (!createRenderTarget()) {
        qDebug() << "createRenderTarget failed";
        return false;
    }

    queryCudaGPUs();
    getCudaDeviceIDFromD3D11();

    cudaStreamCreate(&interop_stream);
    cudaEventCreate(&interop_event);
    cudaStreamCreate(&hist_stream);
    cudaEventCreate(&hist_event);

    fpsTimer.start();
    d3d_initialized = true;

    return true;
}

bool DXWidget::createRenderTarget()
{
    if (!swapChain || !device) return false;

    ID3D11Texture2D* backBuffer = nullptr;

    HRESULT hr = swapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)&backBuffer);
    if (FAILED(hr)) {
        qDebug() << "swapChain->GetBuffer failed";
        return false;
    }

    hr = device->CreateRenderTargetView(backBuffer, nullptr, &rtv);
    backBuffer->Release();

    if (FAILED(hr)) {
        qDebug() << "CreateRenderTargetView failed";
        return false;
    }

    return true;
}

void DXWidget::releaseRenderTarget()
{
    if (rtv) {
        rtv->Release();
        rtv = nullptr;
    }
}

void DXWidget::cleanup()
{
    releaseRenderTarget();

    if (swapChain) {
        swapChain->Release();
        swapChain = nullptr;
    }

    if (context) {
        context->Release();
        context = nullptr;
    }

    if (device) {
        device->Release();
        device = nullptr;
    }
}

void DXWidget::paintEvent(QPaintEvent*)
{
    if (!d3d_initialized) return;

    float clearColor[4] = {0.0f, 0.0f, 0.0f, 1.0f};

    context->OMSetRenderTargets(1, &rtv, nullptr);
    context->ClearRenderTargetView(rtv, clearColor);

    swapChain->Present(1, 0);
}

void DXWidget::resizeEvent(QResizeEvent* event)
{
    QWidget::resizeEvent(event);

    if (!d3d_initialized) return;

    context->OMSetRenderTargets(0, nullptr, nullptr);

    releaseRenderTarget();

    swapChain->ResizeBuffers(
        0,
        width(),
        height(),
        DXGI_FORMAT_UNKNOWN,
        0
        );

    createRenderTarget();
    update();
}



//FBOレンダリング
void DXWidget::FBO_Rendering(VideoFrame Frame)
{
    if ((prev_encode_state == STATE_ENCODING) && encode_state == STATE_NOT_ENCODE) {
        prev_encode_state = encode_state;
        return;
    }
    prev_encode_state = encode_state;

    if (!d3d_initialized) return;

    if (!inputTexture || !fboTexture) {
        qDebug() << "inputTexture or fboTexture is null";
        return;
    }

    // inputTexture → fboTexture コピー
    context->CopyResource(fboTexture, inputTexture);

    if (encode_state == STATE_ENCODING) {
        downloadToDXTexture_and_Encode(Frame);
        Monitor_Rendering(Frame);
    } else {
        Monitor_Rendering(Frame);
    }
}

//画面描画
void DXWidget::Monitor_Rendering(VideoFrame Frame){
    // QElapsedTimer timer;
    // timer.start();
    //painter.begin(this);

    //動画フレーム描画
    {
        if (!d3d_initialized) return;
        if (!swapChain || !context || !fboTexture) return;

        ID3D11Texture2D* backBuffer = nullptr;
        HRESULT hr = swapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)&backBuffer);
        if (FAILED(hr) || !backBuffer) return;

        // backbufferのサイズ取得
        D3D11_TEXTURE2D_DESC bbDesc{};
        backBuffer->GetDesc(&bbDesc);

        // fboTextureのサイズ取得
        D3D11_TEXTURE2D_DESC fboDesc{};
        fboTexture->GetDesc(&fboDesc);

        // コピー範囲（backbufferサイズにクリップ）
        D3D11_BOX srcBox{};
        srcBox.left   = 0;
        srcBox.top    = 0;
        srcBox.front  = 0;
        srcBox.right  = min(bbDesc.Width,  fboDesc.Width);
        srcBox.bottom = min(bbDesc.Height, fboDesc.Height);
        srcBox.back   = 1;

        // fboTexture → backbuffer コピー（左上部分だけ）
        context->CopySubresourceRegion(
            backBuffer,
            0,
            0, 0, 0,
            fboTexture,
            0,
            &srcBox
            );

        swapChain->Present(1, 0);

        backBuffer->Release();
    }

     qDebug()<<"aaaaaaaaa";

    //ヒストグラム描画
    // {
    //     if(g_AppSettings.histgram_flag){
    //         //ヒストグラムCUDA集計
    //         histgram_Analysys();
    //         auto labels = make_nice_y_labels(h_hist_stats.max_y_axis);
    //         //int labels[5]={1,2,3,4,5};

    //         //ヒストグラムグラフ描画
    //         {
    //             QRect rect(viewportWidth -(480+20)*monitor_scaling, 20*monitor_scaling, 480*monitor_scaling, 360*monitor_scaling);
    //             glViewport(rect.x(), rect.y(), rect.width(), rect.height());

    //             // --- OpenGL描画 ---
    //             glMatrixMode(GL_PROJECTION);
    //             glLoadIdentity();
    //             glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0);
    //             glMatrixMode(GL_MODELVIEW);
    //             glLoadIdentity();

    //             // 背景半透明
    //             glEnable(GL_BLEND);
    //             glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    //             glColor4f(0.0f, 0.0f, 0.0f, 0.5f);
    //             glBegin(GL_QUADS);
    //             glVertex2f(0.0f, 0.0f);
    //             glVertex2f(1.0f, 0.0f);
    //             glVertex2f(1.0f, 1.0f);
    //             glVertex2f(0.0f, 1.0f);
    //             glEnd();

    //             // --- グリッド ---
    //             glLineWidth(2.0f);
    //             glColor4f(1.0f, 1.0f, 1.0f, 0.45f);
    //             glBegin(GL_LINES);
    //             //X軸
    //             glVertex2f(0, 0.0f);
    //             glVertex2f(0, 1.0f);
    //             glVertex2f(0.25f, 0.0f);
    //             glVertex2f(0.25f, 1.0f);
    //             glVertex2f(0.5f, 0.0f);
    //             glVertex2f(0.5f, 1.0f);
    //             glVertex2f(0.75f, 0.0f);
    //             glVertex2f(0.75f, 1.0f);
    //             glVertex2f(1.0f, 0.0f);
    //             glVertex2f(1.0f, 1.0f);
    //             //Y軸
    //             glVertex2f(0.0f, 0);
    //             glVertex2f(1.0f, 0);
    //             glVertex2f(0.0f, 1.0f);
    //             glVertex2f(1.0f, 1.0f);
    //             glVertex2f(0.0f, (float)labels[0]/(float)h_hist_stats.max_y_axis);
    //             glVertex2f(1.0f, (float)labels[0]/(float)h_hist_stats.max_y_axis);
    //             glVertex2f(0.0f, (float)labels[1]/(float)h_hist_stats.max_y_axis);
    //             glVertex2f(1.0f, (float)labels[1]/(float)h_hist_stats.max_y_axis);
    //             glVertex2f(0.0f, (float)labels[2]/(float)h_hist_stats.max_y_axis);
    //             glVertex2f(1.0f, (float)labels[2]/(float)h_hist_stats.max_y_axis);
    //             // --- 目盛り ---
    //             glLineWidth(3.0f);
    //             glColor4f(1.0f, 1.0f, 1.0f, 0.8f);
    //             //X軸
    //             glVertex2f(0, 0.0f);
    //             glVertex2f(0, 0.03f);
    //             glVertex2f(0.25f, 0.0f);
    //             glVertex2f(0.25f, 0.03f);
    //             glVertex2f(0.5f, 0.0f);
    //             glVertex2f(0.5f, 0.03f);
    //             glVertex2f(0.75f, 0.0f);
    //             glVertex2f(0.75f, 0.03f);
    //             glVertex2f(1.0f, 0.0f);
    //             glVertex2f(1.0f, 0.03f);
    //             //Y軸
    //             glVertex2f(0.0f, 0);
    //             glVertex2f(0.03f, 0);
    //             glVertex2f(0.0f, 1.0f);
    //             glVertex2f(0.03f, 1.0f);
    //             glVertex2f(0.0f, (float)labels[0]/(float)h_hist_stats.max_y_axis);
    //             glVertex2f(0.03f, (float)labels[0]/(float)h_hist_stats.max_y_axis);
    //             glVertex2f(0.0f, (float)labels[1]/(float)h_hist_stats.max_y_axis);
    //             glVertex2f(0.03f, (float)labels[1]/(float)h_hist_stats.max_y_axis);
    //             glVertex2f(0.0f, (float)labels[2]/(float)h_hist_stats.max_y_axis);
    //             glVertex2f(0.03f, (float)labels[2]/(float)h_hist_stats.max_y_axis);
    //             glEnd();

    //             // --- ヒストグラム ---
    //             glBindBuffer(GL_ARRAY_BUFFER, vbo_hist);
    //             glEnableClientState(GL_VERTEX_ARRAY);
    //             glVertexPointer(3, GL_FLOAT, 0, 0);

    //             glLineWidth(1.4f);
    //             glColor4f(1.0f, 0.0f, 0.0f, 0.9f);
    //             glDrawArrays(GL_LINE_STRIP, 0, num_bins);
    //             glColor4f(0.0f, 1.0f, 0.0f, 0.9f);
    //             glDrawArrays(GL_LINE_STRIP, num_bins, num_bins);
    //             glColor4f(0.0f, 0.0f, 1.0f, 0.9f);
    //             glDrawArrays(GL_LINE_STRIP, num_bins * 2, num_bins);

    //             glDisableClientState(GL_VERTEX_ARRAY);
    //             glBindBuffer(GL_ARRAY_BUFFER, 0);
    //             glDisable(GL_BLEND);

    //             // --- Qtでラベルを追加 ---
    //             // painter.beginNativePainting(); // OpenGLとQtの切り替え開始
    //             // painter.endNativePainting();

    //             //ラベル値
    //             //Y座標
    //             painter.setPen(Qt::white);
    //             painter.setFont(QFont("Arial", 10));
    //             painter.drawText((rect.left() - 60)/monitor_scaling,(viewportHeight-25*monitor_scaling)/monitor_scaling,55/monitor_scaling, 20*monitor_scaling,Qt::AlignRight,QString::number(0));
    //             painter.drawText((rect.left() - 200)/monitor_scaling,(viewportHeight-(rect.height())-35*monitor_scaling)/monitor_scaling,195/monitor_scaling, 20*monitor_scaling,Qt::AlignRight,"Max Y:"+QString::number(h_hist_stats.max_y_axis));
    //             for(int i=0;i<=2;i++){
    //                 if(labels[i]<=h_hist_stats.max_y_axis){
    //                     painter.drawText((rect.left() - 150)/monitor_scaling,(viewportHeight-(rect.height())*((float)labels[i]/(float)h_hist_stats.max_y_axis)-25*monitor_scaling)/monitor_scaling,145/monitor_scaling, 20*monitor_scaling,Qt::AlignRight,QString::number(labels[i]));
    //                 }
    //             }
    //             //X座標
    //             painter.drawText(rect.left()/monitor_scaling,(viewportHeight-19*monitor_scaling)/monitor_scaling,126, 20*monitor_scaling,Qt::AlignRight,QString::number(64));
    //             painter.drawText(rect.left()/monitor_scaling,(viewportHeight-19*monitor_scaling)/monitor_scaling,250, 20*monitor_scaling,Qt::AlignRight,QString::number(128));
    //             painter.drawText(rect.left()/monitor_scaling,(viewportHeight-19*monitor_scaling)/monitor_scaling,370, 20*monitor_scaling,Qt::AlignRight,QString::number(192));
    //             painter.drawText(rect.left()/monitor_scaling,(viewportHeight-19*monitor_scaling)/monitor_scaling,488, 20*monitor_scaling,Qt::AlignRight,QString::number(256));
    //         }

    //         //ヒストグラム描画
    //         {
    //             painter.setPen(Qt::white);
    //             painter.setFont(QFont("Consolas", 16));
    //             QRect rect(viewportWidth/monitor_scaling -(325), 0, 350*monitor_scaling, 100*monitor_scaling);  // 右端に幅300のエリア
    //             painter.drawText(rect, Qt::AlignLeft,"R: min:" + QString::number(h_hist_stats.min_r) +" max:" + QString::number(h_hist_stats.max_r) +" avg:" + QString::number(h_hist_stats.avg_r));
    //             painter.drawText(rect.adjusted(0, 20, 0, 0), Qt::AlignLeft,"G: min:" + QString::number(h_hist_stats.min_g) +" max:" + QString::number(h_hist_stats.max_g) +" avg:" + QString::number(h_hist_stats.avg_g));
    //             painter.drawText(rect.adjusted(0, 40, 0, 0), Qt::AlignLeft,"B: min:" + QString::number(h_hist_stats.min_b) +" max:" + QString::number(h_hist_stats.max_b) +" avg:" + QString::number(h_hist_stats.avg_b));
    //             //painter.drawText(rect.adjusted(0, 60, 0, 0), Qt::AlignLeft,"Axis Y: max:" + QString::number(h_stats.max_y_axis));
    //         }
    //     }
    // }

    //FPSを算出
    // {
    //     if (fpsTimer.elapsed() >= 1000) {  // 1000ms 経過したら
    //         fps = fpsCount * 1000.0 / fpsTimer.elapsed(); // FPS計算
    //         fpsCount = 0;
    //         fpsTimer.restart();
    //     }
    // }

    // //動画情報描画
    // {
    //     if(g_AppSettings.videoInfo_flag){
    //         painter.setPen(Qt::white);
    //         painter.setFont(QFont("Consolas", 16));
    //         painter.drawText(2, 20, "Primary Device(Open GL):" + g_GPUInfo[g_openglDeviceID].deviceName + " (CUDA Device " +QString::number(g_openglDeviceID) + ")" +"\n");

    //         for(int i=0;i<g_GPUInfo.size();i++){
    //             painter.drawText(2, 40 + 20*i, "CUDA Device " + QString::number(i) + ":" + g_GPUInfo[i].deviceName +"(sm_"+QString::number(g_GPUInfo[i].CC_major)+QString::number(g_GPUInfo[i].CC_minor)+")" + " Usage:" + QString::number(g_GPUInfo[i].GPU_Usage) + "% Memory:" + QString::number(g_GPUInfo[i].Memory_Usage) + "/" + QString::number(g_GPUInfo[i].Max_Memory_Usage) + " MB\n");
    //         }

    //         painter.drawText(2, 40 + 20*g_GPUInfo.size(), QString("OpenGL Rendering FPS:%1").arg(fps, 0, 'f', 2));
    //         painter.drawText(2, 60 + 20*g_GPUInfo.size(), "File Name:" + QString::fromStdString(VideoInfo.Name)+"\n");
    //         painter.drawText(2, 80 + 20*g_GPUInfo.size(), "Decorder:" + QString::fromStdString(VideoInfo.Codec)+"\n");
    //         painter.drawText(2, 100 + 20*g_GPUInfo.size(), VideoInfo.decode_mode);
    //         painter.drawText(2, 120 + 20*g_GPUInfo.size(), "Resolution:" + QString::number(width_)+"×"+QString::number(height_)+"\n");
    //         painter.drawText(2, 140 + 20*g_GPUInfo.size(), "Video Framerate:" + QString::number(VideoInfo.fps)+"\n");
    //         painter.drawText(2, 160 + 20*g_GPUInfo.size(), "Max Frame:" + QString::number(VideoInfo.max_framesNo)+"\n");
    //         painter.drawText(2, 180 + 20*g_GPUInfo.size(), "Current Frame:" + QString::number(Frame.FrameNo)+"\n");
    //         if(VideoInfo.audio)
    //             painter.drawText(2, 200 + 20*g_GPUInfo.size(), "Audio Channels:" + QString::number(VideoInfo.audio_channels)+"\n");
    //     }
    // }

    // painter.end();

    // double seconds = timer.nsecsElapsed() / 1e6; // ナノ秒 →  ミリ秒
    // qDebug()<<seconds;
}

//アスペクト比を合わせてリサイズ
void DXWidget::DXresize() {
    // //現在のウィンドウのDPIスケールを取得
    // monitor_scaling = devicePixelRatio();

    // //実ピクセル単位のビューポートサイズを取得
    // viewportWidth  = static_cast<GLint>(this->width()  * monitor_scaling);
    // viewportHeight = static_cast<GLint>(this->height() * monitor_scaling);

    // //ソース（動画/FBO）のサイズ
    // float srcAspect = static_cast<float>(width_) / static_cast<float>(height_);
    // float dstAspect = static_cast<float>(viewportWidth) / static_cast<float>(viewportHeight);

    // if (srcAspect > dstAspect) {
    //     int displayHeight = static_cast<int>(viewportWidth / srcAspect);
    //     int yOffset = (viewportHeight - displayHeight) / 2;
    //     x0 = 0;
    //     y0 = yOffset;
    //     x1 = viewportWidth;
    //     y1 = yOffset + displayHeight;
    // } else {
    //     int displayWidth = static_cast<int>(viewportHeight * srcAspect);
    //     int xOffset = (viewportWidth - displayWidth) / 2;
    //     x0 = xOffset;
    //     y0 = 0;
    //     x1 = xOffset + displayWidth;
    //     y1 = viewportHeight;
    // }
}

//画面をリセット(真っ黒にする)
void DXWidget::DXreset(){
    // glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    // glClear(GL_COLOR_BUFFER_BIT);
    // context()->swapBuffers(context()->surface());
}

//シェーダーUniform、UI設定変更時
// void DXWidget::setShaderUniformEnable(){
//     glUseProgram(Gaussian_shader.progId);
//     glUniform1i(Gaussian_shader.loc_filterEnabled, g_AppSettings.gaussianfilterEnabled);

//     glUseProgram(Sobel_shader.progId);
//     glUniform1i(Sobel_shader.loc_filterEnabled, g_AppSettings.sobelfilterEnabled);

//     glUseProgram(Averaging_shader.progId);
//     glUniform1i(Averaging_shader.loc_filterEnabled, g_AppSettings.averagingfilterEnabled);

//     glUseProgram(0);
// }

//シェーダーUniform、解像度変更時
// void DXWidget::setShaderUniform(int width,int height){
//     glUseProgram(Gaussian_shader.progId);
//     glUniform1i(Gaussian_shader.loc_tex, 0);
//     glUniform2f(Gaussian_shader.loc_texelSize, 1.0f / width, 1.0f / height);

//     glUseProgram(Sobel_shader.progId);
//     glUniform1i(Sobel_shader.loc_tex, 0);
//     glUniform2f(Sobel_shader.loc_texelSize, 1.0f / width, 1.0f / height);

//     glUseProgram(Averaging_shader.progId);
//     glUniform1i(Averaging_shader.loc_tex, 0);
//     glUniform2f(Averaging_shader.loc_texelSize, 1.0f / width, 1.0f / height);

//     glUseProgram(0);
// }

//CUDA→OpenGLの初期化、登録など
void DXWidget::initCudaTexture(int width, int height)
{
    if (cudaInputRes) {
        cudaGraphicsUnregisterResource(cudaInputRes);
        cudaInputRes = nullptr;
    }

    if (inputTexture) {
        inputTexture->Release();
        inputTexture = nullptr;
    }

    // D3D11 Texture 作成
    D3D11_TEXTURE2D_DESC desc{};
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
    desc.CPUAccessFlags = 0;
    desc.MiscFlags = 0;

    HRESULT hr = device->CreateTexture2D(&desc, nullptr, &inputTexture);
    if (FAILED(hr) || !inputTexture) {
        qDebug() << "CreateTexture2D(inputTexture) failed";
        return;
    }

    IDXGIDevice* dxgiDevice = nullptr;
    device->QueryInterface(__uuidof(IDXGIDevice), (void**)&dxgiDevice);

    IDXGIAdapter* adapter = nullptr;
    dxgiDevice->GetAdapter(&adapter);

    DXGI_ADAPTER_DESC desc2{};
    adapter->GetDesc(&desc2);

    qDebug() << "D3D Device Adapter =" << QString::fromWCharArray(desc2.Description);

    adapter->Release();
    dxgiDevice->Release();

    // CUDA登録
    cudaError_t err = cudaGraphicsD3D11RegisterResource(
        &cudaInputRes,
        inputTexture,
        cudaGraphicsRegisterFlagsNone
        );

    if (err != cudaSuccess) {
        qDebug() << "cudaGraphicsD3D11RegisterResource failed:" << cudaGetErrorString(err);
        cudaInputRes = nullptr;
        return;
    }

    // SRV作成（描画用）
    if (inputSRV) {
        inputSRV->Release();
        inputSRV = nullptr;
    }

    hr = device->CreateShaderResourceView(inputTexture, nullptr, &inputSRV);
    if (FAILED(hr)) {
        qDebug() << "CreateShaderResourceView(inputSRV) failed";
    }

    int curDev = -1;
    cudaGetDevice(&curDev);
    qDebug() << "CUDA current device =" << curDev;

}

//OpenGL→CUDAの初期化、登録など
void DXWidget::initTextureCuda(int width, int height)
{
    // ==========================
    // 既存CUDA登録解除
    // ==========================
    if (cudaResource2) {
        cudaGraphicsUnregisterResource(cudaResource2);
        cudaResource2 = nullptr;
    }
    if (cudaTempRes) {
        cudaGraphicsUnregisterResource(cudaTempRes);
        cudaTempRes = nullptr;
    }

    // ==========================
    // 既存DirectXリソース破棄
    // ==========================
    if (fboSRV) { fboSRV->Release(); fboSRV = nullptr; }
    if (fboRTV) { fboRTV->Release(); fboRTV = nullptr; }
    if (fboTexture) { fboTexture->Release(); fboTexture = nullptr; }

    if (tempSRV) { tempSRV->Release(); tempSRV = nullptr; }
    if (tempRTV) { tempRTV->Release(); tempRTV = nullptr; }
    if (tempTexture) { tempTexture->Release(); tempTexture = nullptr; }

    // ==========================
    // Texture2D 作成 (FBO相当)
    // ==========================
    D3D11_TEXTURE2D_DESC desc{};
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
    desc.CPUAccessFlags = 0;
    desc.MiscFlags = 0;

    HRESULT hr = device->CreateTexture2D(&desc, nullptr, &fboTexture);
    if (FAILED(hr)) {
        qDebug() << "CreateTexture2D (fboTexture) failed";
        return;
    }

    hr = device->CreateRenderTargetView(fboTexture, nullptr, &fboRTV);
    if (FAILED(hr)) {
        qDebug() << "CreateRenderTargetView (fboRTV) failed";
        return;
    }

    hr = device->CreateShaderResourceView(fboTexture, nullptr, &fboSRV);
    if (FAILED(hr)) {
        qDebug() << "CreateShaderResourceView (fboSRV) failed";
        return;
    }

    // ==========================
    // Texture2D 作成 (tempFBO相当)
    // ==========================
    hr = device->CreateTexture2D(&desc, nullptr, &tempTexture);
    if (FAILED(hr)) {
        qDebug() << "CreateTexture2D (tempTexture) failed";
        return;
    }

    hr = device->CreateRenderTargetView(tempTexture, nullptr, &tempRTV);
    if (FAILED(hr)) {
        qDebug() << "CreateRenderTargetView (tempRTV) failed";
        return;
    }

    hr = device->CreateShaderResourceView(tempTexture, nullptr, &tempSRV);
    if (FAILED(hr)) {
        qDebug() << "CreateShaderResourceView (tempSRV) failed";
        return;
    }

    // ==========================
    // CUDA登録（ReadOnly）
    // ==========================
    cudaError_t err = cudaGraphicsD3D11RegisterResource(
        &cudaResource2,
        fboTexture,
        cudaGraphicsRegisterFlagsNone
        );

    if (err != cudaSuccess) {
        qDebug() << "cudaGraphicsD3D11RegisterResource (fboTexture) error:"
                 << cudaGetErrorString(err);
        cudaResource2 = nullptr;
        return;
    }

    err = cudaGraphicsD3D11RegisterResource(
        &cudaTempRes,
        tempTexture,
        cudaGraphicsRegisterFlagsNone
        );

    if (err != cudaSuccess) {
        qDebug() << "cudaGraphicsD3D11RegisterResource (tempTexture) error:"
                 << cudaGetErrorString(err);
        cudaTempRes = nullptr;
        return;
    }

    qDebug() << "D3D11 textures registered to CUDA:" << width << "x" << height;
}

//初回、解像度が変わった場合再Malloc
// void DXWidget::initCudaMalloc(int width, int height)
// {

// }

//CUDAからDirectX11へ転送
void DXWidget::uploadToDXTexture(VideoFrame Frame)
{
    if (!d3d_initialized) {
        return;
    }

    // 解像度変更対応
    int newW = VideoInfo.width  * VideoInfo.width_scale;
    int newH = VideoInfo.height * VideoInfo.height_scale;

    if (newW != width_ || newH != height_) {
cudaSetDevice(g_openglDeviceID);   // ← D3D11と同じCUDA device
        // initCudaMalloc(newW, newH);

        // CUDA→D3D11 (inputTexture)
        initCudaTexture(newW, newH);

        // D3D11→CUDA (outputTextureなど)
        initTextureCuda(newW, newH);

        initCudaHist();

        width_ = newW;
        height_ = newH;

        DXresize();
        return;
    }

    // 入力データチェック
    if (!Frame.d_decode_rgba) {
        qDebug() << "入力データがNULLです";
        return;
    }

    if (!cudaInputRes) {
        qDebug() << "cudaInputRes is null";
        return;
    }

    // CUDA → D3D11 Texture 転送
    cudaError_t err;
    cudaArray_t array;

    err = cudaGraphicsMapResources(1, &cudaInputRes, interop_stream);
    if (err != cudaSuccess) {
        qDebug() << "cudaGraphicsMapResources error:" << cudaGetErrorString(err);
        return;
    }

    err = cudaGraphicsSubResourceGetMappedArray(&array, cudaInputRes, 0, 0);
    if (err != cudaSuccess) {
        qDebug() << "cudaGraphicsSubResourceGetMappedArray error:" << cudaGetErrorString(err);
        cudaGraphicsUnmapResources(1, &cudaInputRes, interop_stream);
        return;
    }

    size_t widthBytes = width_ * 4;

    if (widthBytes <= Frame.decode_pitch) {
        err = cudaMemcpy2DToArrayAsync(
            array,
            0, 0,
            Frame.d_decode_rgba,
            Frame.decode_pitch,
            widthBytes,
            height_,
            cudaMemcpyDeviceToDevice,
            interop_stream
            );

        if (err != cudaSuccess) {
            qDebug() << "cudaMemcpy2DToArrayAsync error:" << cudaGetErrorString(err);
        }

        cudaEventRecord(interop_event, interop_stream);
        cudaEventSynchronize(interop_event);

    } else {
        qDebug() << "Invalid pitch: widthBytes > pitch!";
        cudaGraphicsUnmapResources(1, &cudaInputRes, interop_stream);
        return;
    }

    err = cudaGraphicsUnmapResources(1, &cudaInputRes, interop_stream);
    if (err != cudaSuccess) {
        qDebug() << "cudaGraphicsUnmapResources error:" << cudaGetErrorString(err);
        return;
    }

    // DirectX11 描画へ
    FBO_Rendering(Frame);

    fpsCount++;
}

//DirectX11からCUDAへ転送+エンコード
void DXWidget::downloadToDXTexture_and_Encode(VideoFrame Frame)
{
    if (!cudaResource2) {
        qDebug() << "cudaResource2 is nullptr, can't map";
        return;
    }

    if (!Frame.d_encode_rgba) {
        qDebug() << "Frame.d_encode_rgba is nullptr";
        return;
    }

    cudaArray_t array;
    cudaError_t err;

    // Map (D3D11 Texture -> CUDA)
    err = cudaGraphicsMapResources(1, &cudaResource2, interop_stream);
    if (err != cudaSuccess) {
        qDebug() << "cudaGraphicsMapResources error:" << cudaGetErrorString(err);
        return;
    }

    err = cudaGraphicsSubResourceGetMappedArray(&array, cudaResource2, 0, 0);
    if (err != cudaSuccess) {
        qDebug() << "cudaGraphicsSubResourceGetMappedArray error:" << cudaGetErrorString(err);
        cudaGraphicsUnmapResources(1, &cudaResource2, interop_stream);
        return;
    }

    // Copy D3D11 texture -> Frame.d_encode_rgba
    size_t widthBytes = width_ * 4;

    if (widthBytes <= Frame.encode_pitch) {

        err = cudaMemcpy2DFromArrayAsync(
            Frame.d_encode_rgba,
            Frame.encode_pitch,
            array,
            0, 0,
            widthBytes,
            height_,
            cudaMemcpyDeviceToDevice,
            interop_stream
            );

        if (err != cudaSuccess) {
            qDebug() << "cudaMemcpy2DFromArrayAsync error:" << cudaGetErrorString(err);
        }

        cudaEventRecord(interop_event, interop_stream);
        cudaEventSynchronize(interop_event);

    } else {
        qDebug() << "Invalid pitch: widthBytes > pitch!";
        cudaGraphicsUnmapResources(1, &cudaResource2, interop_stream);
        return;
    }

    // Unmap
    err = cudaGraphicsUnmapResources(1, &cudaResource2, interop_stream);
    if (err != cudaSuccess) {
        qDebug() << "cudaGraphicsUnmapResources error:" << cudaGetErrorString(err);
        return;
    }

    // Encode
    if (save_encoder != nullptr && encode_FrameCount <= (MaxFrame - MinFrame)) {
        save_encoder->encode(Frame);
        encode_FrameCount++;
    }
    else {
        delete save_encoder;
        save_encoder = nullptr;
        encode_FrameCount = 0;
        emit encode_finished();
    }
}

//エンコード時
void DXWidget::encode_mode(int flag){
    encode_state = flag;

    //エンコードモードの場合はエンコード用インスタンス生成
    if(flag==STATE_ENCODING){
        if(save_encoder==nullptr){
            save_encoder = new save_encode(height_,width_);
        }
    }
}

//ヒストグラム周り
void DXWidget::initCudaHist()
{
    // --- 1. 既存 CUDA Graphics Resource を解放 ---
    if (cudaResource_hist) {
        cudaGraphicsUnregisterResource(cudaResource_hist);
        cudaResource_hist = nullptr;
    }

    if (cudaResource_hist_draw) {
        cudaGraphicsUnregisterResource(cudaResource_hist_draw);
        cudaResource_hist_draw = nullptr;
    }

    // --- 2. 既存 D3D11 Buffer を削除 ---
    if (histVB) {
        histVB->Release();
        histVB = nullptr;
    }

    // --- 3. 既存 CUDA メモリ を解放 ---
    if (d_hist_stats) {
        cudaFree(d_hist_stats);
        d_hist_stats = nullptr;
    }

    if (d_hist_data) {
        cudaFree(d_hist_data);
        d_hist_data = nullptr;
    }

    // --- 4. D3D11 Texture (fboTexture) → CUDA Resource 登録 ---
    if (!fboTexture) {
        qDebug() << "fboTexture is nullptr";
        return;
    }

    cudaError_t err = cudaGraphicsD3D11RegisterResource(
        &cudaResource_hist,
        fboTexture,
        cudaGraphicsRegisterFlagsNone
        );

    if (err != cudaSuccess) {
        qDebug() << "cudaGraphicsD3D11RegisterResource (hist texture) failed:"
                 << cudaGetErrorString(err);
        cudaResource_hist = nullptr;
        return;
    }

    // --- 5. D3D11 VertexBuffer 作成（OpenGL VBOの代替） ---
    D3D11_BUFFER_DESC vbDesc{};
    vbDesc.ByteWidth = num_bins * 3 * 3 * sizeof(float);
    vbDesc.Usage = D3D11_USAGE_DEFAULT;
    vbDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    vbDesc.CPUAccessFlags = 0;
    vbDesc.MiscFlags = 0;

    HRESULT hr = device->CreateBuffer(&vbDesc, nullptr, &histVB);
    if (FAILED(hr)) {
        qDebug() << "CreateBuffer(histVB) failed";
        return;
    }

    // --- 6. D3D11 Buffer → CUDA Resource 登録 ---
    err = cudaGraphicsD3D11RegisterResource(
        &cudaResource_hist_draw,
        histVB,
        cudaGraphicsRegisterFlagsWriteDiscard
        );

    if (err != cudaSuccess) {
        qDebug() << "cudaGraphicsD3D11RegisterResource (histVB) failed:"
                 << cudaGetErrorString(err);
        cudaResource_hist_draw = nullptr;
        return;
    }

    // --- 7. CUDA 側のメモリを確保 ---
    cudaMalloc(&d_hist_stats, sizeof(HistStats));
    cudaMalloc(&d_hist_data, sizeof(HistData));
}

//DirectX11からCUDAへ転送+ヒストグラム解析
void DXWidget::histgram_Analysys()
{
    if (!cudaResource_hist) {
        qDebug() << "cudaResource_hist is nullptr, can't map";
        return;
    }
    if (!cudaResource_hist_draw) {
        qDebug() << "cudaResource_hist_draw is nullptr, can't map";
        return;
    }

    // 一括マップ
    cudaGraphicsResource* resources[] = { cudaResource_hist, cudaResource_hist_draw };
    cudaError_t err = cudaGraphicsMapResources(2, resources, interop_stream);
    if (err != cudaSuccess) {
        qDebug() << "cudaGraphicsMapResources error:" << cudaGetErrorString(err);
        return;
    }

    // ==========================
    // ヒストグラム計算
    // ==========================
    cudaArray_t cuArray;
    err = cudaGraphicsSubResourceGetMappedArray(&cuArray, cudaResource_hist, 0, 0);
    if (err != cudaSuccess) {
        qDebug() << "cudaGraphicsSubResourceGetMappedArray error:" << cudaGetErrorString(err);
        cudaGraphicsUnmapResources(2, resources, interop_stream);
        return;
    }

    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    cudaTextureDesc texDesc{};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaTextureObject_t texObj = 0;
    err = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    if (err != cudaSuccess) {
        qDebug() << "cudaCreateTextureObject error:" << cudaGetErrorString(err);
        cudaGraphicsUnmapResources(2, resources, interop_stream);
        return;
    }

    cudaMemsetAsync(d_hist_data, 0, sizeof(HistData), interop_stream);
    CUDA_IMG_Proc->calc_histgram(d_hist_data, texObj, width_, height_);

    cudaDestroyTextureObject(texObj);

    // ==========================
    // ヒストグラム統計解析
    // ==========================
    CUDA_IMG_Proc->histogram_status(d_hist_data, d_hist_stats);

    cudaMemcpyAsync(&h_hist_stats,
                    d_hist_stats,
                    sizeof(HistStats),
                    cudaMemcpyDeviceToHost,
                    hist_stream);

    // ==========================
    // D3D11 VertexBuffer に書き込み（VBOの代替）
    // ==========================
    float* d_vb_ptr = nullptr;
    size_t vb_size = 0;

    err = cudaGraphicsResourceGetMappedPointer((void**)&d_vb_ptr, &vb_size, cudaResource_hist_draw);
    if (err != cudaSuccess) {
        qDebug() << "cudaGraphicsResourceGetMappedPointer error:" << cudaGetErrorString(err);
        cudaGraphicsUnmapResources(2, resources, interop_stream);
        return;
    }

    CUDA_IMG_Proc->histgram_normalize(d_vb_ptr, num_bins, d_hist_data, d_hist_stats);

    // ==========================
    // 一括アンマップ
    // ==========================
    err = cudaGraphicsUnmapResources(2, resources, interop_stream);
    if (err != cudaSuccess) {
        qDebug() << "cudaGraphicsUnmapResources error:" << cudaGetErrorString(err);
        return;
    }

    // CPU側で h_hist_stats を使うなら同期が必要
    cudaStreamSynchronize(hist_stream);
}

// 人が見て気持ちいいY軸ラベルを生成
std::vector<int> DXWidget::make_nice_y_labels(int max_value)
{
    int num_divs = 3;
    std::vector<int> labels;
    if (max_value <= 0) return labels;

    // 1. 仮の間隔を計算
    double raw_step = static_cast<double>(max_value) / num_divs;
    double magnitude = pow(10.0, floor(log10(raw_step))); // 10^n の桁
    double normalized = raw_step / magnitude;             // 1〜10の範囲に正規化

    // 2. 正規化した値を 1, 2, 5 のいずれかに丸める
    double nice_factor = 1.0;
    if (normalized < 1.5)
        nice_factor = 1.0;
    else if (normalized < 3.0)
        nice_factor = 2.0;
    else if (normalized < 7.0)
        nice_factor = 5.0;
    else
        nice_factor = 10.0;

    double nice_step = nice_factor * magnitude;

    // 3. ラベル値を作成
    for (double v = nice_step; v < max_value; v += nice_step)
        labels.push_back(static_cast<int>(v));

    // 4. 必要なら最終ラベルを追加
    if (labels.empty() || labels.back() < max_value)
        labels.push_back(static_cast<int>(ceil(max_value / nice_step) * nice_step));

    return labels;
}

//CUDAとopenGLのデバイス設定
void DXWidget::queryCudaGPUs()
{
    // iniから読み込んだ情報を退避
    std::vector<GPUInfo> savedGPUInfo = g_GPUInfo;

    // CUDAで取得したGPU情報を一時的に格納
    std::vector<GPUInfo> currentGPUInfo;

    int count = 0;
    cudaGetDeviceCount(&count);

    for (int i = 0; i < count; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        GPUInfo info;
        info.deviceID = i;
        info.deviceName = QString::fromUtf8(prop.name);
        info.CC_major = prop.major;
        info.CC_minor = prop.minor;
        info.Max_Memory_Usage = (int)(prop.totalGlobalMem / (1024 * 1024));

        cudaDeviceGetAttribute(&info.pciDomain, cudaDevAttrPciDomainId, i);
        cudaDeviceGetAttribute(&info.pciBus, cudaDevAttrPciBusId, i);
        cudaDeviceGetAttribute(&info.pciDevice, cudaDevAttrPciDeviceId, i);

        // 初期値
        info.tile_weight = 10;
        info.openglEnable = false;

        currentGPUInfo.push_back(info);
    }

    // ==========================================
    // savedGPUInfo と currentGPUInfo が全一致か確認
    // ==========================================
    bool allMatch = true;

    if (savedGPUInfo.size() != currentGPUInfo.size())
    {
        allMatch = false;
    }
    else
    {
        for (const auto &cur : currentGPUInfo)
        {
            bool found = false;

            for (const auto &saved : savedGPUInfo)
            {
                if (saved.pciDomain == cur.pciDomain &&
                    saved.pciBus == cur.pciBus &&
                    saved.pciDevice == cur.pciDevice &&
                    saved.deviceName == cur.deviceName &&
                    saved.CC_major == cur.CC_major &&
                    saved.CC_minor == cur.CC_minor &&
                    saved.Max_Memory_Usage == cur.Max_Memory_Usage)
                {
                    found = true;
                    break;
                }
            }

            if (!found)
            {
                allMatch = false;
                break;
            }
        }
    }

    // ==========================================
    // g_GPUInfo を確定
    // ==========================================
    g_GPUInfo.clear();

    if (allMatch)
    {
        // 全一致 → iniの設定を復元
        for (auto &cur : currentGPUInfo)
        {
            for (const auto &saved : savedGPUInfo)
            {
                if (saved.pciDomain == cur.pciDomain &&
                    saved.pciBus == cur.pciBus &&
                    saved.pciDevice == cur.pciDevice &&
                    saved.deviceName == cur.deviceName &&
                    saved.CC_major == cur.CC_major &&
                    saved.CC_minor == cur.CC_minor &&
                    saved.Max_Memory_Usage == cur.Max_Memory_Usage)
                {
                    cur.tile_weight = saved.tile_weight;
                    cur.openglEnable = saved.openglEnable;
                    break;
                }
            }

            g_GPUInfo.push_back(cur);
        }

        qDebug() << "[GPU MATCH] All GPUs matched -> restored tile_weight";
    }
    else
    {
        // 一部一致・不一致 → 全GPU初期化
        for (auto &cur : currentGPUInfo)
        {
            cur.tile_weight = 10;
            cur.openglEnable = false;
            g_GPUInfo.push_back(cur);
        }

        qDebug() << "[GPU MISMATCH] Partial match -> reset ALL tile_weight=10";
    }

    // ログ
    for (const auto &gpu : g_GPUInfo)
    {
        qDebug() << "CUDA Device" << gpu.deviceID
                 << ":" << gpu.deviceName
                 << "CC =" << gpu.CC_major << "." << gpu.CC_minor
                 << "VRAM =" << gpu.Max_Memory_Usage << "MB"
                 << "PCI =" << gpu.pciDomain << ":" << gpu.pciBus << ":" << gpu.pciDevice
                 << "tile_weight =" << gpu.tile_weight;
    }
}

void DXWidget::getCudaDeviceIDFromD3D11()
{
    if (!device) {
        qDebug() << "D3D11 device is null";
        return;
    }

    // D3D11 device -> DXGI adapter
    IDXGIDevice* dxgiDevice = nullptr;
    HRESULT hr = device->QueryInterface(__uuidof(IDXGIDevice), (void**)&dxgiDevice);
    if (FAILED(hr) || !dxgiDevice) {
        qDebug() << "QueryInterface(IDXGIDevice) failed";
        return;
    }

    IDXGIAdapter* adapter = nullptr;
    hr = dxgiDevice->GetAdapter(&adapter);
    dxgiDevice->Release();

    if (FAILED(hr) || !adapter) {
        qDebug() << "GetAdapter failed";
        return;
    }

    DXGI_ADAPTER_DESC desc{};
    adapter->GetDesc(&desc);

    QString dxName = QString::fromWCharArray(desc.Description);

    // AdapterのPCI情報を取得 (IDXGIAdapter1必須)
    IDXGIAdapter1* adapter1 = nullptr;
    hr = adapter->QueryInterface(__uuidof(IDXGIAdapter1), (void**)&adapter1);
    adapter->Release();

    if (FAILED(hr) || !adapter1) {
        qDebug() << "QueryInterface(IDXGIAdapter1) failed";
        return;
    }

    DXGI_ADAPTER_DESC1 desc1{};
    adapter1->GetDesc1(&desc1);
    adapter1->Release();

    // 注意: DXGI_DESC1 には busId は直接無い
    // → Windows API SetupDi で取る必要があるため、ここでは名前一致方式にする

    qDebug() << "D3D11 Adapter =" << dxName;

    // ============================================
    // CUDA device を名前で一致させる（簡易）
    // ============================================
    int cudaCount = 0;
    cudaGetDeviceCount(&cudaCount);

    int matched = -1;

    for (int i = 0; i < cudaCount; i++) {
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, i);

        QString cudaName = QString::fromUtf8(prop.name);

        if (cudaName.contains(dxName, Qt::CaseInsensitive) ||
            dxName.contains(cudaName, Qt::CaseInsensitive))
        {
            matched = i;
            break;
        }
    }

    if (matched < 0) {
        qDebug() << "[ERROR] No CUDA device matched D3D11 adapter name";
        return;
    }

    cudaSetDevice(matched);
    g_openglDeviceID = matched;

    qDebug() << "Matched CUDA DeviceID =" << matched;
}
