#include "fps_thread.h"
#include "qdebug.h"
#include "qtimer.h"

fps_thread::fps_thread(double fps, QObject *parent) : QThread(parent){
    target_fps = fps;
}

fps_thread::~fps_thread() {
}

void fps_thread::run() {
    const double fps = target_fps; // 正式 29.97
    const double interval = 1.0 / fps;

    using clock = std::chrono::steady_clock;
    auto start = clock::now();

    int frames = 0;

    while (running) {
        auto now = clock::now();
        double elapsed = std::chrono::duration<double>(now - start).count();

        int targetFrames = (int)(elapsed / interval);

        if (frames < targetFrames) {
            emit fps_signal();
            frames++;
        }

        std::this_thread::yield();
    }
}


void fps_thread::stop() {
    running = false;
}
