#include "src/main/fps_thread.h"
#include "src/main/__global__.h"
#include "qdebug.h"
#include "qtimer.h"

fps_thread::fps_thread(double fps, QObject *parent) : QThread(parent){
    target_fps = fps;
}

fps_thread::~fps_thread() {
}

void fps_thread::run() {
    using clock = std::chrono::steady_clock;
    start_ = clock::now();
    frames = 0;

    while (running) {
        const double interval = 1.0 / target_fps;

        auto now_ = clock::now();
        double elapsed = std::chrono::duration<double>(now_ - start_).count();

        int targetFrames = (int)(elapsed / interval);

        if (frames < targetFrames) {
            g_last_tick_ns.store(mono_now_ns(), std::memory_order_relaxed);   // 表示の計測（刻みの時刻）
            emit fps_signal();
            frames++;
        }

        std::this_thread::yield();
    }
}

void fps_thread::change_speed(double fps) {
    target_fps = fps;
    start_ = std::chrono::steady_clock::now();
    frames = 0;
}

void fps_thread::stop() {
    running = false;
}
