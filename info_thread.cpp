#include "info_thread.h"
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

    // シグナルとスロットの接続
        connect(&timer, &QTimer::timeout, this, &info_thread::check_gpu_usage, Qt::DirectConnection);

    // 50msごとにシグナルを発行
    timer.setInterval(1000);
    timer.start();

    printf("debug : info Thread Starts\n");
    fflush(stdout);

    exec();  // イベントループを開始し、ブロックされる

    fflush(stdout);

    // タイマーを停止して接続を解除
    timer.stop();
    timer.disconnect();
}

void info_thread::check_gpu_usage() {
    g_gpu_usage = get_gpu_usage();
}

int info_thread::get_gpu_usage() {
    nvmlDevice_t device;
    nvmlUtilization_t utilization;

    // 最初の GPU の使用率を取得
    nvmlReturn_t result = nvmlDeviceGetHandleByIndex(0, &device);
    if (result != NVML_SUCCESS) {
        qDebug() << "Failed to get device handle:" << nvmlErrorString(result);
        return -1;
    }

    result = nvmlDeviceGetUtilizationRates(device, &utilization);
    if (result != NVML_SUCCESS) {
        qDebug() << "Failed to get utilization rates:" << nvmlErrorString(result);
        return -1;
    }

    return utilization.gpu;
}
