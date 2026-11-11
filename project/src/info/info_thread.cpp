#include "src/info/info_thread.h"
#include "qtimer.h"
#include <nvml.h>
#include <QDebug>
#include <algorithm>
#include <cstdio>

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

// ★ CUDA のデバイス番号と NVML の番号は並び順が違うことがある
//   （CUDA は既定で速い順、NVML は PCI 順）。PCI バス ID で同じ GPU を引く。
static bool nvmlHandleFor(const GPUInfo& g, nvmlDevice_t* out)
{
    char busId[32];
    std::snprintf(busId, sizeof(busId), "%08x:%02x:%02x.0", g.pciDomain, g.pciBus, g.pciDevice);
    if (nvmlDeviceGetHandleByPciBusId_v2(busId, out) == NVML_SUCCESS) return true;
    return nvmlDeviceGetHandleByIndex(g.deviceID, out) == NVML_SUCCESS;   // 予備
}

void info_thread::check_gpu_usage() {
    for (size_t i = 0; i < g_GPUInfo.size(); i++) {
        nvmlDevice_t device;
        if (!nvmlHandleFor(g_GPUInfo[i], &device)) {
            qDebug() << "Failed to get device handle for" << g_GPUInfo[i].deviceName;
            continue;   // ★ 1 枚失敗しても他の GPU は更新する
        }

        // 取れなかった項目は 0 扱い（古い GPU / ドライバでも止まらないように）
        nvmlUtilization_t utilization{};
        unsigned int encUtil = 0, encSampling = 0;
        unsigned int decUtil = 0, decSampling = 0;
        nvmlMemory_t memInfo{};
        nvmlDeviceGetUtilizationRates(device, &utilization);
        nvmlDeviceGetEncoderUtilization(device, &encUtil, &encSampling);
        nvmlDeviceGetDecoderUtilization(device, &decUtil, &decSampling);
        const bool memOk = nvmlDeviceGetMemoryInfo(device, &memInfo) == NVML_SUCCESS;

        g_GPUInfo[i].GPU_Usage     = (int)std::max({ utilization.gpu, encUtil, decUtil });
        g_GPUInfo[i].Encoder_Usage = (int)encUtil;
        g_GPUInfo[i].Decoder_Usage = (int)decUtil;
        if (memOk)
            g_GPUInfo[i].Memory_Usage = (int)(memInfo.used / (1024 * 1024)); // MB
    }
}
