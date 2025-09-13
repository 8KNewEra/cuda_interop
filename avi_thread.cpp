#include "avi_thread.h"
#include "dstorage.h"
#include <dstorage.h>
#include "opencv2/imgcodecs.hpp"
#include <QDebug>
#include <QOpenGLContext>
#include <QOpenGLFunctions>
#include <QTimer>
#include <npp.h>
#include <wrl/client.h>
#include <fstream>
#include <iostream>
#include <cuda_runtime.h>

using Microsoft::WRL::ComPtr;
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
#include <libavutil/hwcontext.h>
#include <libavutil/frame.h>
}

avi_thread::avi_thread(QObject *parent)
    : QObject(parent), g_video_play_flag(true), timer(new QTimer(this)) {
}

avi_thread::~avi_thread() {
    stopProcessing();
    qDebug() << "AVI Thread: Destructor called";
}

void avi_thread::startProcessing() {
    frameList = ParseAviIndex("D:/test.avi"); // 最初だけ呼び出す
    qDebug() << "AVI Thread: Start Processing";
}

void avi_thread::stopProcessing() {
    qDebug() << "AVI Thread: Stop Processing";
    timer->stop();
}

void avi_thread::resumePlayback() {
    QMutexLocker locker(&mutex);
    g_video_play_flag = true;
}

void avi_thread::pausePlayback() {
    QMutexLocker locker(&mutex);
    g_video_play_flag = false;
}

void avi_thread::processFrame() {
    QMutexLocker locker(&mutex);

    if (!g_video_play_flag&&slider_No == Get_Frame_No) {
        return;
    }

    get_decode_image();
}

// フレーム取得
void avi_thread::get_decode_image() {
    cv::cuda::GpuMat gpuFrame;
    loadFrameToGpu(100, gpuFrame);
}

//ここからdxstorage
bool avi_thread::initialize() {
    frameList = ParseAviIndex(aviFilePath);
    if (frameList.empty()) return false;

    return initDirectStorage();
}

std::vector<FrameIndex> avi_thread::ParseAviIndex(const std::string& aviPath) {
    std::ifstream ifs(aviPath, std::ios::binary);
    if (!ifs) return {};

    ifs.seekg(0, std::ios::end);
    size_t filesize = (size_t)ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    std::vector<uint8_t> filebuf(filesize);
    ifs.read((char*)filebuf.data(), filesize);

    const char* idx1_str = "idx1";
    size_t idx1_pos = std::string((char*)filebuf.data(), filesize).find(idx1_str);
    if (idx1_pos == std::string::npos) return {};
    qDebug()<<"aaa";

    uint32_t idx1_size = *reinterpret_cast<uint32_t*>(&filebuf[idx1_pos + 4]);
    size_t entries = idx1_size / 16;

    std::vector<FrameIndex> frames;
    size_t entry_start = idx1_pos + 8;
    size_t baseOffset = 12 + 4 + 4;

    for (size_t i = 0; i < entries; i++) {
        size_t pos = entry_start + i * 16;
        char chunk_id[5] = {0};
        memcpy(chunk_id, &filebuf[pos], 4);

        if (strncmp(chunk_id, "00dc", 4) != 0) continue;

        uint32_t offset = *reinterpret_cast<uint32_t*>(&filebuf[pos + 8]);
        uint32_t size = *reinterpret_cast<uint32_t*>(&filebuf[pos + 12]);

        frames.push_back(FrameIndex{ offset + baseOffset, size, 1920, 1080, 0 }); // 例：解像度は仮定
    }

    //フレームレートを元にタイマー設定
    double framerate = 30;
    qDebug() << "Framerate:" << framerate;
    interval_ms = static_cast<double>(1000.0 / framerate);
    elapsedTimer.start();
    timer->start(interval_ms);

    connect(timer, &QTimer::timeout, this, &avi_thread::processFrame);

    //再生
    g_video_play_flag = true;

    return frames;
}

bool avi_thread::initDirectStorage() {
    HANDLE fileHandle = CreateFileA(aviFilePath.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (fileHandle == INVALID_HANDLE_VALUE) return false;
    gpuFileHandle = fileHandle;

    ComPtr<IDStorageFactory> factory;
    if (FAILED(DStorageGetFactory(IID_PPV_ARGS(&factory)))) return false;
    dstorageFactory = factory.Get();

    DSTORAGE_QUEUE_DESC desc = {};
    desc.Capacity = 1;
    desc.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
    desc.Device = nullptr;

    ComPtr<IDStorageQueue> queue;
    if (FAILED(factory->CreateQueue(&desc, IID_PPV_ARGS(&queue)))) return false;
    dstorageQueue = queue.Get();

    return true;
}

bool avi_thread::readFrameToGpu(const FrameIndex& frame, cv::cuda::GpuMat& outGpuMat) {
    // CUDAバッファ確保
    void* gpuBuf = nullptr;
    cudaMalloc(&gpuBuf, frame.size);

    // DirectStorageリクエスト構築
    DSTORAGE_REQUEST request = {};
    request.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
    request.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_MEMORY;

    request.Source.File.Source = gpuFile;  // IDStorageFile*
    request.Source.File.Offset = frame.offset;
    request.Source.File.Size = frame.size;

    // 宛先をGPUメモリに指定
    request.Destination.Memory.Buffer = gpuBuf;
    request.Destination.Memory.Size = frame.size;
    request.UncompressedSize = frame.size;

    // フェンス作成（同期用）
    ComPtr<ID3D12Fence> fence;
    if (FAILED(g_d3d12Device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)))) {
        std::cerr << "Failed to create fence\n";
        cudaFree(gpuBuf);
        return false;
    }

    HANDLE fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    UINT64 fenceValue = 1;

    // DirectStorage送信
    queue->EnqueueRequest(&request);
    queue->EnqueueSignal(nullptr, 1);
    queue->Submit();

    // 完了待ち
    if (fence->GetCompletedValue() < fenceValue) {
        fence->SetEventOnCompletion(fenceValue, fenceEvent);
        WaitForSingleObject(fenceEvent, INFINITE);
    }
    CloseHandle(fenceEvent);

    // GpuMatにコピー（RGBとして仮定）
    outGpuMat.create(frame.height, frame.width, CV_8UC3);
    cudaMemcpy(outGpuMat.data, gpuBuf, frame.size, cudaMemcpyDeviceToDevice);

    // メモリ解放
    cudaFree(gpuBuf);
    return true;
}


bool avi_thread::loadFrameToGpu(int frameNumber, cv::cuda::GpuMat& outGpuMat) {
    if (frameNumber < 0 || frameNumber >= static_cast<int>(frameList.size())) return false;
    return readFrameToGpu(frameList[frameNumber], outGpuMat);
}

