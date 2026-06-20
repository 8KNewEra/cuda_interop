#include "ai_imageprocess.h"
#include "qdebug.h"

AI_ImageProcess::AI_ImageProcess(QObject* parent)
    : QThread(parent) {
    //Build_RIFE_TensorRT_Engine();
}

//RIFE ONNXビルド用
void AI_ImageProcess::Build_RIFE_TensorRT_Engine() {
    // 💡 ONNXが格納されているフォルダパス
    QString modelsFolder = "E:/cuda_interop/project/models/";

    QDir dir(modelsFolder);
    if (!dir.exists()) {
        qCritical() << "指定されたモデルフォルダが存在しません:" << modelsFolder;
        return;
    }

    // 💡 .onnx ファイルだけをフィルタリングして全取得
    QStringList filters;
    filters << "*.onnx";
    QFileInfoList onnxFiles = dir.entryInfoList(filters, QDir::Files, QDir::Name);

    if (onnxFiles.isEmpty()) {
        qWarning() << "フォルダ内に .onnx ファイルが見つかりませんでした。";
        return;
    }

    qDebug() << "==================================================";
    qDebug() << "  TensorRT Batch Engine Build Started (" << onnxFiles.size() << " models)";
    qDebug() << "==================================================";

    QElapsedTimer batchTimer;
    batchTimer.start();

    // 💡 見つかったONNXファイルを1枚ずつ順番にビルド
    for (const QFileInfo& fileInfo : onnxFiles) {
        QString onnxPath = fileInfo.absoluteFilePath();
        // 出力先は、同じフォルダの「ファイル名.engine」にする
        QString enginePath = QCoreApplication::applicationDirPath() + "/engines/" + fileInfo.baseName() + ".engine";

        qDebug() << "\n---> [Processing]:" << fileInfo.fileName();

        QElapsedTimer singleTimer;
        singleTimer.start();

        // 1. Builderの作成
        nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
        if (!builder) {
            qCritical() << "Builderの作成に失敗しました。";
            continue;
        }

        // 💡【スッキリ修正！】
        // 今回書き出したONNXの型（FP32）を、最新TensorRTにそのまま厳格に認識させるため
        // kSTRONGLY_TYPED フラグを有効化（1本化）します。これでお使いの最新環境でも確実に通ります。
        uint32_t flags = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
        nvinfer1::INetworkDefinition* network = builder->createNetworkV2(flags);

        // 2. Parserを使ってONNXを読み込む
        nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
        if (!parser->parseFromFile(onnxPath.toStdString().c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
            qCritical() << "ONNXのパースに失敗しました:" << fileInfo.fileName();
            delete parser; delete network; delete builder;
            continue;
        }

        // 3. 最最適化設定（Config）の作成
        nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();

        // ワークスペース（探索用メモリプール）を4GB許可
        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 4ULL << 30);

        // 4. ビルド実行
        qDebug() << "   Building Engine... (This may take a few minutes)...";
        nvinfer1::IHostMemory* serializedModel = builder->buildSerializedNetwork(*network, *config);

        if (!serializedModel) {
            qCritical() << "   Engineのビルドに失敗しました。";
            delete config; delete parser; delete network; delete builder;
            continue;
        }

        // 5. 完成したEngineをファイルに保存
        QFile file(enginePath);
        if (file.open(QIODevice::WriteOnly)) {
            file.write(static_cast<const char*>(serializedModel->data()), serializedModel->size());
            file.close();
            qDebug() << "   ✨ SUCCESS! Engine saved to:" << fileInfo.baseName() + ".engine";
            qDebug() << "   ⏱️ Build Time for this model:" << singleTimer.elapsed() / 1000.0 << "seconds.";
        } else {
            qCritical() << "   ファイル保存に失敗しました:" << enginePath;
        }

        // 後片付け（次のループのために毎回綺麗にする）
        delete serializedModel;
        delete config;
        delete parser;
        delete network;
        delete builder;
    }

    qDebug() << "\n==================================================";
    qDebug() << "  🎉 ALL ENGINES COMPLETED!";
    qDebug() << "  Total Batch Time:" << batchTimer.elapsed() / 1000.0 / 60.0 << "minutes.";
    qDebug() << "==================================================";
}

// RIFE初期化
bool AI_ImageProcess::loadRifeTensorRT(int targetRatio) {
    QString enginepath = QCoreApplication::applicationDirPath()+ "/engines/rife_" +QString::number(targetRatio)+ "x_1k.engine";

    QFile file(enginepath);
    if (!file.open(QIODevice::ReadOnly)) {
        qCritical() << "[RIFE] ファイルが開けません:" << enginepath;
        return false;
    }
    QByteArray engineData = file.readAll();
    file.close();

    // 共通コンテキスト・ランタイムの生成
    m_rife_instances.runtime = nvinfer1::createInferRuntime(gLogger);
    if (!m_rife_instances.runtime) {
        qCritical() << "[RIFE] InferRuntimeの生成に失敗しました。";
        return false;
    }

    m_rife_instances.engine = m_rife_instances.runtime->deserializeCudaEngine(engineData.constData(), engineData.size());
    if (!m_rife_instances.engine) {
        qCritical() << "[RIFE] Engineのデシリアライズに失敗しました。";
        delete m_rife_instances.runtime;
        return false;
    }

    m_rife_instances.context = m_rife_instances.engine->createExecutionContext();
    if (!m_rife_instances.context) {
        qCritical() << "[RIFE] ExecutionContextの生成に失敗しました。";
        delete m_rife_instances.engine; delete m_rife_instances.runtime;
        return false;;
    }

    // テンソルの自動走査とバインド
    std::map<int, void*> ordered_output_ptrs;
    int32_t numTensors = m_rife_instances.engine->getNbIOTensors();
    for (int32_t i = 0; i < numTensors; ++i) {
        const char* tensorName = m_rife_instances.engine->getIOTensorName(i);
        nvinfer1::TensorIOMode mode = m_rife_instances.engine->getTensorIOMode(tensorName);
        nvinfer1::Dims dims = m_rife_instances.engine->getTensorShape(tensorName);

        int64_t elementCount = 1;
        for (int32_t d = 0; d < dims.nbDims; ++d) {
            elementCount *= dims.d[d];
        }

        // RIFEの入力テンソルの形状は [1, 3, Height, Width] の4次元 (NCHW)
        if (mode == nvinfer1::TensorIOMode::kINPUT && strcmp(tensorName, "img0") == 0) {
            if (dims.nbDims == 4) {
                m_rife_instances.modelHeight = dims.d[2];
                m_rife_instances.modelWidth  = dims.d[3];
                qDebug() << "[RIFE Auto Shape] Model Native Resolution detected:"
                         << m_rife_instances.modelWidth << "x" << m_rife_instances.modelHeight;
            }
        }

        nvinfer1::DataType dataType = m_rife_instances.engine->getTensorDataType(tensorName);
        size_t typeSize = (dataType == nvinfer1::DataType::kHALF) ? 2 : 4;
        size_t byteSize = elementCount * typeSize;
        void* d_ptr = nullptr;
        cudaError_t err = cudaMalloc(&d_ptr, byteSize);
        if (err != cudaSuccess) {
            qCritical() << "[RIFE] cudaMallocエラー:" << tensorName;
            return false;
        }

        // このインスタンスのコンテキストにアドレスをバインド
        m_rife_instances.context->setTensorAddress(tensorName, d_ptr);
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            if (strcmp(tensorName, "img0") == 0) m_rife_instances.d_img0 = d_ptr;
            if (strcmp(tensorName, "img1") == 0) m_rife_instances.d_img1 = d_ptr;
        }
        else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
            int outIdx = -1;
            if (strcmp(tensorName, "output_frame") == 0) {
                outIdx = 1;
            } else if (sscanf(tensorName, "out_%d", &outIdx) == 1) {
                // 通常のマルチ出力
            }

            if (outIdx != -1) {
                ordered_output_ptrs[outIdx] = d_ptr;
            }
        }
    }

    // 整合性チェック
    if (!m_rife_instances.d_img0 || !m_rife_instances.d_img1 || ordered_output_ptrs.empty()) {
        qCritical() << "[RIFE] テンソルマッピングに失敗しました。";
        delete m_rife_instances.context; delete m_rife_instances.engine; delete m_rife_instances.runtime;
        return false;
    }

    // 時系列順に出力ポインタを配列へ展開
    for (const auto& [idx, ptr] : ordered_output_ptrs) {
        m_rife_instances.d_outputs.push_back(ptr);
    }

    // Frame構造体の初期化（インスタンス内部にカプセル化）
    m_rife_instances.gpu_float_img0.data     = static_cast<uint8_t*>(m_rife_instances.d_img0);
    m_rife_instances.gpu_float_img0.width    = m_rife_instances.modelWidth;
    m_rife_instances.gpu_float_img0.height   = m_rife_instances.modelHeight;
    m_rife_instances.gpu_float_img0.pitch    = 0;
    m_rife_instances.gpu_float_img0.channels = 3;

    m_rife_instances.gpu_float_img1.data     = static_cast<uint8_t*>(m_rife_instances.d_img1);
    m_rife_instances.gpu_float_img1.width    = m_rife_instances.modelWidth;
    m_rife_instances.gpu_float_img1.height   = m_rife_instances.modelHeight;
    m_rife_instances.gpu_float_img1.pitch    = 0;
    m_rife_instances.gpu_float_img1.channels = 3;

    for (size_t i = 0; i < m_rife_instances.d_outputs.size(); ++i) {
        gpuFrame outFrame;
        outFrame.data     = static_cast<uint8_t*>(m_rife_instances.d_outputs[i]);
        outFrame.width    = m_rife_instances.modelWidth;
        outFrame.height   = m_rife_instances.modelHeight;
        outFrame.pitch    = 0;
        outFrame.channels = 3;
        m_rife_instances.gpu_float_outputs.push_back(outFrame);
    }

    qDebug() << "\n==================================================";
    qDebug() << "     ENGINE INSTANCES LOADED AND SORTED!";
    qDebug() << "  Load instance target ratio " << targetRatio;
    qDebug() << "==================================================";

    return true;
}

// モデルアンロード
void AI_ImageProcess::unloadRifeEngine() {
    // CUDAメモリ解放
    if (m_rife_instances.d_img0) cudaFree(m_rife_instances.d_img0);
    if (m_rife_instances.d_img1) cudaFree(m_rife_instances.d_img1);
    for (void* ptr : m_rife_instances.d_outputs) {
        if (ptr) cudaFree(ptr);
    }
    m_rife_instances.d_outputs.clear();
    m_rife_instances.gpu_float_outputs.clear();

    // TensorRTオブジェクト破棄
    if (m_rife_instances.context) {
        delete m_rife_instances.context;
        m_rife_instances.context = nullptr;
    }
    if (m_rife_instances.engine)  {
        delete m_rife_instances.engine;
        m_rife_instances.engine = nullptr;
    }
    if (m_rife_instances.runtime) {
        delete m_rife_instances.runtime;
        m_rife_instances.runtime = nullptr;
    }
}

// フレーム補完
void AI_ImageProcess::rife_interpolate(const gpuFrame& frame0, const gpuFrame& frame1,
                                       std::vector<gpuFrame>& out_frames,
                                       cudaStream_t stream, CUDA_ImageProcess *CUDA_Img_Proc)
{
    if (!frame0.data || !frame1.data || out_frames.empty()) return;

    // 要求された倍率（例：7枚なら 8倍補間）
    int targetRatio = out_frames.size() + 1;

    // 対応する倍率のエンジンがロードされていない場合は再ロード
    {
        std::lock_guard<std::mutex> lock(m_engine_mutex);
        if (m_rife_instances.engine == nullptr || targetRatio != m_rife_instances.d_outputs.size() + 1) {
            // ロード中は推論が走らないように、必要に応じて lock_guard を使用
            unloadRifeEngine();
            if(!loadRifeTensorRT(targetRatio)) return;
        }
    }

    // 前処理 RGBA→CHW float
    {
        std::lock_guard<std::mutex> lock(m_engine_mutex);
        CUDA_Img_Proc->RGBA_to_CHW_Float(frame0, m_rife_instances.gpu_float_img0, stream);
        CUDA_Img_Proc->RGBA_to_CHW_Float(frame1, m_rife_instances.gpu_float_img1, stream);

        // 推論
        m_rife_instances.context->enqueueV3(stream);

        // 後処理 CHW float→RGBA
        for (size_t i = 0; i < out_frames.size(); ++i) {
            CUDA_Img_Proc->CHW_Float_to_RGBA(m_rife_instances.gpu_float_outputs[i], out_frames[i], stream);
        }

        cudaStreamSynchronize(stream);
    }
}

//TensorRTの初期化
void AI_ImageProcess::initYoloTensorRT() {
    // // ====================================================
    // // 1. Engineファイルの読み込みと Context の生成
    // // ====================================================
    // const QString enginePath = "D:/cuda_interop/glwidget_1/engine/yolo26x_4080S.engine";
    // QFile file(enginePath);
    // if (!file.open(QIODevice::ReadOnly)) {
    //     qCritical() << "Engineファイルが開けません:" << enginePath;
    //     return;
    // }
    // QByteArray engineData = file.readAll();
    // file.close();



    // // ランタイム、エンジン、コンテキストの生成
    // m_runtime = nvinfer1::createInferRuntime(gLogger);
    // m_engine = m_runtime->deserializeCudaEngine(engineData.constData(), engineData.size());
    // if (!m_engine) {
    //     qCritical() << "Engineのデシリアライズに失敗しました。";
    //     return;
    // }

    // // ★ここでついに m_context が初期化されます！
    // m_context = m_engine->createExecutionContext();
    // if (!m_context) {
    //     qCritical() << "Contextの生成に失敗しました。";
    //     return;
    // }

    // qDebug() << "TensorRT Engine loaded and Context created successfully!";

    // // ====================================================
    // // 2. GPUメモリとCPUメモリの確保
    // // ====================================================
    // cudaMalloc(&m_d_input, 1 * 3 * 640 * 640 * sizeof(float));
    // cudaMalloc(&m_d_output, 1 * 300 * 6 * sizeof(float));

    // m_h_output.resize(1 * 300 * 6);

    // // ====================================================
    // // 3. OpenCVのゼロコピー用 GpuMat の準備
    // // ====================================================
    // float* d_ptr = static_cast<float*>(m_d_input);
    // m_input_channels.clear();
    // m_input_channels.push_back(cv::cuda::GpuMat(640, 640, CV_32FC1, d_ptr));
    // m_input_channels.push_back(cv::cuda::GpuMat(640, 640, CV_32FC1, d_ptr + 640 * 640));
    // m_input_channels.push_back(cv::cuda::GpuMat(640, 640, CV_32FC1, d_ptr + 2 * 640 * 640));
}

//Real ESRGAN ONNXビルド用
void AI_ImageProcess::Build_SuperRes_TensorRT_Engine()
{
    QString onnxPath =
        "E:/cuda_interop/project/models/FSRCNN_x2.onnx";

    QString enginePath =
        "E:/cuda_interop/project/engines/FSRCNN_x2.engine";

    auto builder =
        std::unique_ptr<nvinfer1::IBuilder>(
            nvinfer1::createInferBuilder(gLogger));

    if (!builder)
    {
        qDebug() << "createInferBuilder failed";
        return;
    }

    auto network =
        std::unique_ptr<nvinfer1::INetworkDefinition>(
            builder->createNetworkV2(0));

    if (!network)
    {
        qDebug() << "createNetworkV2 failed";
        return;
    }

    auto parser =
        std::unique_ptr<nvonnxparser::IParser>(
            nvonnxparser::createParser(
                *network,
                gLogger));

    if (!parser)
    {
        qDebug() << "createParser failed";
        return;
    }

    bool ok =
        parser->parseFromFile(
            onnxPath.toStdString().c_str(),
            static_cast<int>(
                nvinfer1::ILogger::Severity::kWARNING));

    if (!ok)
    {
        qDebug() << "ONNX parse failed";

        for (int i = 0; i < parser->getNbErrors(); i++)
        {
            qDebug()
            << parser->getError(i)->desc();
        }

        return;
    }

    auto config =
        std::unique_ptr<nvinfer1::IBuilderConfig>(
            builder->createBuilderConfig());

    if (!config)
    {
        qDebug() << "createBuilderConfig failed";
        return;
    }

    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 4ULL << 30);


    auto profile =
        builder->createOptimizationProfile();

    if (!profile)
    {
        qDebug() << "createOptimizationProfile failed";
        return;
    }

    profile->setDimensions(
        "input",
        nvinfer1::OptProfileSelector::kMIN,
        nvinfer1::Dims4(
            1,
            3,
            1024,
            2048));

    profile->setDimensions(
        "input",
        nvinfer1::OptProfileSelector::kOPT,
        nvinfer1::Dims4(
            1,
            3,
            1024,
            2048));

    profile->setDimensions(
        "input",
        nvinfer1::OptProfileSelector::kMAX,
        nvinfer1::Dims4(
            1,
            3,
            1024,
            2048));

    config->addOptimizationProfile(
        profile);

    auto serializedEngine =
        std::unique_ptr<nvinfer1::IHostMemory>(
            builder->buildSerializedNetwork(
                *network,
                *config));

    if (!serializedEngine)
    {
        qDebug() << "buildSerializedNetwork failed";
        return;
    }

    QFile file(enginePath);

    if (!file.open(QIODevice::WriteOnly))
    {
        qDebug() << "cannot open engine file";
        return;
    }

    file.write(
        static_cast<const char*>(
            serializedEngine->data()),
        serializedEngine->size());

    file.close();

    qDebug()
        << "TensorRT Engine Saved:"
        << enginePath;
}

void AI_ImageProcess::init_SuperRes_TensorRT(int width, int height)
{
    QString enginePath =
        "E:/cuda_interop/project/engines/FSRCNN_x2.engine";

    QFile file(enginePath);

    if (!file.open(QIODevice::ReadOnly))
    {
        qDebug() << "engine open failed";
        return;
    }

    QByteArray engineData =
        file.readAll();

    file.close();

    m_superres_instances.runtime = nvinfer1::createInferRuntime(gLogger);;
    if (!m_superres_instances.runtime)
    {
        qDebug() << "createInferRuntime failed";
        return;
    }
    m_superres_instances.engine = m_superres_instances.runtime->deserializeCudaEngine(engineData.constData(), engineData.size());;
    if (!m_superres_instances.engine)
    {
        qDebug() << "deserializeCudaEngine failed";
        return;
    }

    m_superres_instances.context = m_superres_instances.engine->createExecutionContext();

    if (!m_superres_instances.context)
    {
        qDebug() << "createExecutionContext failed";
        return;
    }

    for(int i=0;i<m_superres_instances.engine->getNbIOTensors();i++)
    {
        auto name = m_superres_instances.engine->getIOTensorName(i);

        qDebug()
            << name
            << (int)m_superres_instances.engine->getTensorDataType(name);
    }


    // 💡 4. テンソルの自動走査とバインド
    int32_t numTensors = m_superres_instances.engine->getNbIOTensors();

    for (int32_t i = 0; i < numTensors; ++i) {
        const char* tensorName = m_superres_instances.engine->getIOTensorName(i);
        nvinfer1::TensorIOMode mode = m_superres_instances.engine->getTensorIOMode(tensorName);
        nvinfer1::Dims dims = m_superres_instances.engine->getTensorShape(tensorName);

        int64_t elementCount = 1;
        for (int32_t d = 0; d < dims.nbDims; ++d) {
            elementCount *= dims.d[d];
        }

        nvinfer1::DataType dataType = m_superres_instances.engine->getTensorDataType(tensorName);
        size_t typeSize = (dataType == nvinfer1::DataType::kHALF) ? 2 : 4;

        size_t byteSize = elementCount * typeSize;
        void* d_ptr = nullptr;
        cudaError_t err = cudaMalloc(&d_ptr, byteSize);
        if (err != cudaSuccess) {
            qCritical() << "[RIFE] cudaMallocエラー:" << tensorName;
            return;
        }

        qDebug()
            << i
            << tensorName
            << dims.nbDims
            << byteSize;

        // このインスタンスのコンテキストにアドレスをバインド
        m_superres_instances.context->setTensorAddress(tensorName, d_ptr);

        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            if (strcmp(tensorName, "input") == 0) m_superres_instances.d_input = d_ptr;
        }
        else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
            if (strcmp(tensorName, "output") == 0) m_superres_instances.d_output = d_ptr;
        }
    }

    m_superres_instances.gpu_float_input.data     = static_cast<uint8_t*>(m_superres_instances.d_input);
    m_superres_instances.gpu_float_input.width    = width;
    m_superres_instances.gpu_float_input.height   = height;
    m_superres_instances.gpu_float_input.pitch    = 0;
    m_superres_instances.gpu_float_input.channels = 3;

    m_superres_instances.gpu_float_output.data     = static_cast<uint8_t*>(m_superres_instances.d_output);
    m_superres_instances.gpu_float_output.width    = width*2;
    m_superres_instances.gpu_float_output.height   = height*2;
    m_superres_instances.gpu_float_output.pitch    = 0;
    m_superres_instances.gpu_float_output.channels = 3;

    qDebug()
        << (int)m_superres_instances.engine
               ->getTensorDataType("input");

    qDebug()
        << (int)m_superres_instances.engine
               ->getTensorDataType("output");

    qDebug() << "TensorRT Ready";
}


void AI_ImageProcess::run_SuperRes(const gpuFrame& in_frame,gpuFrame& out_frames,
                                   cudaStream_t stream, CUDA_ImageProcess *CUDA_Img_Proc)
{
    if (!in_frame.data || !out_frames.data) return;

    CUDA_Img_Proc->RGBA_to_CHW_Float(in_frame, m_superres_instances.gpu_float_input, stream);
    m_superres_instances.context->enqueueV3(stream);
    CUDA_Img_Proc->CHW_Float_to_RGBA(m_superres_instances.gpu_float_output, out_frames, stream);

    cudaStreamSynchronize(stream);
}

//画像認識
void AI_ImageProcess::yolo_analysis(gpuFrame img) { // ★値渡しではなく参照渡し(&)推奨
    // if (img.empty() || !m_context) return;

    // // ====================================================
    // // 1. 画像の前処理（すべてGPU上で完結）
    // // ====================================================
    // // ① 640x640にリサイズ
    // cv::cuda::resize(img, m_gpu_resized, cv::Size(640, 640), 0, 0, cv::INTER_LINEAR, m_stream);

    // // ② RGBA から RGB に変換 ← ★ここを修正！
    // cv::cuda::cvtColor(m_gpu_resized, m_gpu_rgb, cv::COLOR_RGBA2RGB, 0, m_stream);

    // // ③ 0〜255の整数を、0.0〜1.0の少数（float）に変換
    // m_gpu_rgb.convertTo(m_gpu_float, CV_32FC3, 1.0 / 255.0, m_stream);

    // // ④ HWC配列(RGBRGB...) を CHW配列(RRR...GGG...BBB...)に分割
    // cv::cuda::split(m_gpu_float, m_input_channels, m_stream);

    // // ====================================================
    // // 2. 推論の実行
    // // ====================================================
    // m_context->setTensorAddress("images", m_d_input);
    // m_context->setTensorAddress("output0", m_d_output);

    // cudaStream_t raw_stream = cv::cuda::StreamAccessor::getStream(m_stream);
    // m_context->enqueueV3(raw_stream);

    // // ====================================================
    // // 3. 結果の取得（GPU -> CPU）
    // // ====================================================
    // size_t outputSize = 1 * 300 * 6 * sizeof(float);
    // cudaMemcpyAsync(m_h_output.data(), m_d_output, outputSize, cudaMemcpyDeviceToHost, raw_stream);

    // //qDebug()<<m_h_output;

    // // GPUの全ての作業が終わるまで待機
    // m_stream.waitForCompletion();
}


