#include "fps_thread.h"
#include "qdebug.h"
#include "qtimer.h"

fps_thread::fps_thread(QObject *parent) : QThread(parent){}

fps_thread::~fps_thread() {
}

void fps_thread::run() {
    const int fps = 60;
    const double interval = 1.0 / fps;

    using clock = std::chrono::steady_clock;

    auto start = clock::now();
    int processedFrames = 0;

    while (running) {
        auto now = clock::now();
        double elapsed = std::chrono::duration<double>(now - start).count();

        int targetFrames = static_cast<int>(elapsed / interval);

        while (processedFrames < targetFrames) {
            emit fps_signal();
            processedFrames++;
        }

        // 精度目的ではなく CPU 負荷軽減用
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
}

void fps_thread::stop() {
    running = false;
}
