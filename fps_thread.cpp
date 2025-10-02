#include "fps_thread.h"
#include "qtimer.h"

fps_thread::fps_thread(QObject *parent) : QThread(parent), timer(nullptr) {}

fps_thread::~fps_thread() {
    timer->stop();
    delete timer;
    timer = nullptr;
}

void fps_thread::run() {
    QTimer timer;
    connect(&timer, &QTimer::timeout, this, &fps_thread::timer_hit, Qt::DirectConnection);
    timer.setInterval(1000);
    timer.start();
    fflush(stdout);
    exec();
    fflush(stdout);
    timer.stop();
    timer.disconnect();
}

void fps_thread::timer_hit() {
    emit fps_signal();
}
