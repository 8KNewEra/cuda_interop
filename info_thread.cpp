#include "info_thread.h"
#include "qtimer.h"
#include <QDebug>

info_thread::info_thread(QObject *parent)
    : QThread(parent) {
}

info_thread::~info_thread() {

}

void info_thread::run() {
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
    g_gpu_usage = get_gpu_usage();
}

int info_thread::get_gpu_usage() {
    return 0;
}
