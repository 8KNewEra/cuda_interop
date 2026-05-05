#include "src/info/info_thread.h"
#include "qtimer.h"
#include <nvml.h>
#include <QDebug>

info_thread::info_thread(QObject *parent)
    : QThread(parent) {
}

info_thread::~info_thread() {
    nvmlShutdown();
}

void info_thread::run() {
    // NVML 初期化
    nvmlReturn_t result = nvmlInit();
    if (result != NVML_SUCCESS) {
        qDebug() << "Failed to initialize NVML:" << nvmlErrorString(result);
        return;
    }

    QTimer timer;
    connect(&timer, &QTimer::timeout, this, &info_thread::check_gpu_usage, Qt::DirectConnection);
    timer.setInterval(1000);
    timer.start();
    fflush(stdout);
    exec();
    fflush(stdout);
    timer.stop();
    timer.disconnect();
}

void info_thread::check_gpu_usage() {
    nvmlDevice_t device;
    nvmlUtilization_t utilization;
    nvmlMemory_t memInfo;
    unsigned int encUtil = 0, encSampling = 0;
    unsigned int decUtil = 0, decSampling = 0;

    for(int i=0;i<g_GPUInfo.size();i++){
        // 最初の GPU の使用率を取得
        nvmlReturn_t result = nvmlDeviceGetHandleByIndex(g_GPUInfo[i].deviceID, &device);
        if (result != NVML_SUCCESS) {
            qDebug() << "Failed to get device handle:" << nvmlErrorString(result);
            return;
        }

        result = nvmlDeviceGetUtilizationRates(device, &utilization);
        if (result != NVML_SUCCESS) {
            qDebug() << "Failed to get utilization rates:" << nvmlErrorString(result);
            return;
        }

        result = nvmlDeviceGetEncoderUtilization(device, &encUtil, &encSampling);
        if (result != NVML_SUCCESS) {
            qDebug() << "Failed to get encoder utilization:" << nvmlErrorString(result);
            return;
        }

        result = nvmlDeviceGetDecoderUtilization(device, &decUtil, &decSampling);
        if (result != NVML_SUCCESS) {
            qDebug() << "Failed to get decoder utilization:" << nvmlErrorString(result);
            return;
        }
        unsigned int totalLike = std::max({ utilization.gpu, encUtil, decUtil });

        result = nvmlDeviceGetMemoryInfo(device, &memInfo);
        if (result != NVML_SUCCESS) {
            qDebug() << "Failed to get decoder utilization:" << nvmlErrorString(result);
            return;
        }

        g_GPUInfo[i].GPU_Usage = totalLike;
        g_GPUInfo[i].Memory_Usage = (int)(memInfo.used / (1024 * 1024)); // MB
    }
}
