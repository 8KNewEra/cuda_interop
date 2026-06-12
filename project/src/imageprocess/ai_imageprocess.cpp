#include "ai_imageprocess.h"
#include "qdebug.h"

// TensorRT用のロガー（エラーや警告をQtのコンソールに出力します）
class MyTRTLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        // INFOレベル以上を出力（ビルドの進捗が見えるようにします）
        if (severity <= Severity::kINFO) {
            qDebug() << "[TensorRT]" << msg;
        }
    }
} gLogger;

AI_ImageProcess::AI_ImageProcess(QObject* parent)
    : QThread(parent) {
    //testBuildTensorRTEngine();
}

//ONNXビルド用
void AI_ImageProcess::testBuildTensorRTEngine() {
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
        QString enginePath = "E:/cuda_interop/project/engines/" + fileInfo.baseName() + ".engine";

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

// RIFE初期化 (任意の n 倍補間対応版)
void AI_ImageProcess::initRifeTensorRT(int width, int height) {
    QString engineFolder = "E:/cuda_interop/project/engines/";

    QDir dir(engineFolder);
    if (!dir.exists()) {
        qCritical() << "[RIFE] エンジンフォルダが存在しません:" << engineFolder;
        return;
    }

    // 💡 既存のインスタンス配列を完全にクリア（再初期化対応）
    m_rife_instances.clear();

    // 💡 1. 命名規則 「rife_*x_1k.engine」 に一致するファイルを全検索
    QStringList filters;
    filters << "rife_*x_1k.engine";
    QFileInfoList engineFiles = dir.entryInfoList(filters, QDir::Files, QDir::Name);

    if (engineFiles.isEmpty()) {
        qWarning() << "[RIFE] フォルダ内に符合する .engine ファイルが見つかりませんでした。";
        return;
    }

    // 💡 2. 【核心】ファイル名から倍率(n)を抽出し、mapを使って自動的に「低い順」にソート
    std::map<int, QString> sortedEnginePaths;
    for (const QFileInfo& fileInfo : engineFiles) {
        int ratio = 2;
        // ファイル名（例: rife_3x_1k.engine）から数値を抽出
        if (sscanf(fileInfo.fileName().toStdString().c_str(), "rife_%dx_1k.engine", &ratio) == 1) {
            sortedEnginePaths[ratio] = fileInfo.absoluteFilePath();
        }
    }

    qDebug() << "==================================================";
    qDebug() << "  [RIFE] Multi-Engine Serialization Started";
    qDebug() << "  Detected models count:" << sortedEnginePaths.size();
    qDebug() << "==================================================";

    // 💡 3. 低い順（2x -> 3x -> 4x）にソートされたmapをループして一気に初期化
    for (const auto& [ratio, path] : sortedEnginePaths) {
        qDebug() << "\n---> [Loading Engine]:" << QFileInfo(path).fileName() << "(Ratio:" << ratio << "X)";

        QFile file(path);
        if (!file.open(QIODevice::ReadOnly)) {
            qCritical() << "[RIFE] ファイルが開けません:" << path;
            continue;
        }
        QByteArray engineData = file.readAll();
        file.close();

        // 新しい構造体インスタンスを作成
        RifeEngineInstance instance;
        instance.interpolateRatio = ratio;

        // 共通コンテキスト・ランタイムの生成
        instance.runtime = nvinfer1::createInferRuntime(gLogger);
        if (!instance.runtime) {
            qCritical() << "[RIFE] InferRuntimeの生成に失敗しました。";
            continue;
        }

        instance.engine = instance.runtime->deserializeCudaEngine(engineData.constData(), engineData.size());
        if (!instance.engine) {
            qCritical() << "[RIFE] Engineのデシリアライズに失敗しました。";
            delete instance.runtime;
            continue;
        }

        instance.context = instance.engine->createExecutionContext();
        if (!instance.context) {
            qCritical() << "[RIFE] ExecutionContextの生成に失敗しました。";
            delete instance.engine; delete instance.runtime;
            continue;
        }

        // 💡 4. テンソルの自動走査とバインド
        std::map<int, void*> ordered_output_ptrs;
        int32_t numTensors = instance.engine->getNbIOTensors();

        for (int32_t i = 0; i < numTensors; ++i) {
            const char* tensorName = instance.engine->getIOTensorName(i);
            nvinfer1::TensorIOMode mode = instance.engine->getTensorIOMode(tensorName);
            nvinfer1::Dims dims = instance.engine->getTensorShape(tensorName);

            int64_t elementCount = 1;
            for (int32_t d = 0; d < dims.nbDims; ++d) {
                elementCount *= dims.d[d];
            }

            nvinfer1::DataType dataType = instance.engine->getTensorDataType(tensorName);
            size_t typeSize = (dataType == nvinfer1::DataType::kHALF) ? 2 : 4;

            size_t byteSize = elementCount * typeSize;
            void* d_ptr = nullptr;
            cudaError_t err = cudaMalloc(&d_ptr, byteSize);
            if (err != cudaSuccess) {
                qCritical() << "[RIFE] cudaMallocエラー:" << tensorName;
                return;
            }

            // このインスタンスのコンテキストにアドレスをバインド
            instance.context->setTensorAddress(tensorName, d_ptr);

            if (mode == nvinfer1::TensorIOMode::kINPUT) {
                if (strcmp(tensorName, "img0") == 0) instance.d_img0 = d_ptr;
                if (strcmp(tensorName, "img1") == 0) instance.d_img1 = d_ptr;
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
        if (!instance.d_img0 || !instance.d_img1 || ordered_output_ptrs.empty()) {
            qCritical() << "[RIFE] テンソルマッピングに失敗しました。";
            delete instance.context; delete instance.engine; delete instance.runtime;
            continue;
        }

        // 時系列順に出力ポインタを配列へ展開
        for (const auto& [idx, ptr] : ordered_output_ptrs) {
            instance.d_outputs.push_back(ptr);
        }

        // 💡 5. Frame構造体の初期化（インスタンス内部にカプセル化）
        instance.gpu_float_img0.data     = static_cast<uint8_t*>(instance.d_img0);
        instance.gpu_float_img0.width    = width;
        instance.gpu_float_img0.height   = height;
        instance.gpu_float_img0.pitch    = 0;
        instance.gpu_float_img0.channels = 3;

        instance.gpu_float_img1.data     = static_cast<uint8_t*>(instance.d_img1);
        instance.gpu_float_img1.width    = width;
        instance.gpu_float_img1.height   = height;
        instance.gpu_float_img1.pitch    = 0;
        instance.gpu_float_img1.channels = 3;

        for (size_t i = 0; i < instance.d_outputs.size(); ++i) {
            gpuFrame outFrame;
            outFrame.data     = static_cast<uint8_t*>(instance.d_outputs[i]);
            outFrame.width    = width;
            outFrame.height   = height;
            outFrame.pitch    = 0;
            outFrame.channels = 3;
            instance.gpu_float_outputs.push_back(outFrame);
        }

        // 💡 完成した完璧なインスタンスを、メイン配列に格納！
        m_rife_instances[ratio] = instance;

        qDebug() << "   [Initialization Successful]:" << ratio << "X model loaded seamlessly.";
    }

    qDebug() << "\n==================================================";
    qDebug() << "  🎉 ALL ENGINE INSTANCES LOADED AND SORTED!";
    qDebug() << "  Total instances in vector:" << m_rife_instances.size();
    qDebug() << "==================================================";
}

//フレーム補完
void AI_ImageProcess::rife_interpolate(const gpuFrame& frame0, const gpuFrame& frame1,
                                       std::vector<gpuFrame>& out_frames,
                                       cudaStream_t stream, CUDA_ImageProcess *CUDA_Img_Proc)
{
    if (!frame0.data || !frame1.data || out_frames.empty()) return;

    // 要求された倍率（例：7枚なら 8倍補間）
    int targetRatio = out_frames.size() + 1;

    // 💡【新・ループ撲滅チート技】
    // マップから指定倍率（8など）のエンジンをダイレクトに検索！
    auto it = m_rife_instances.find(targetRatio);

    // 💡 安全チェック（フォルダの中に本当に 8x.engine が入っていなかった時だけ弾く）
    if (it == m_rife_instances.end()) {
        qCritical() << "[RIFE Inference] 要求された倍率" << targetRatio
                    << "X に対応するエンジンがフォルダ内に存在しません（ロードされていません）！";
        return;
    }

    // 💡 発見したエンジンインスタンスのポインタを確定（速度はvectorの時と変わりません）
    RifeEngineInstance* targetInstance = &(it->second);

    // 💡 以降の前処理・推論・後処理コードは、1文字も変更せず完全にそのままでOK！
    CUDA_Img_Proc->RGBA_to_CHW_Float(frame0, targetInstance->gpu_float_img0, stream);
    CUDA_Img_Proc->RGBA_to_CHW_Float(frame1, targetInstance->gpu_float_img1, stream);

    targetInstance->context->enqueueV3(stream);

    for (size_t i = 0; i < out_frames.size(); ++i) {
        CUDA_Img_Proc->CHW_Float_to_RGBA(targetInstance->gpu_float_outputs[i], out_frames[i], stream);
    }

    cudaStreamSynchronize(stream);
}

