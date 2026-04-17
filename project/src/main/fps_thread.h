#ifndef FPS_THREAD_H
#define FPS_THREAD_H

#include <QThread>
#include <QObject> // 明示的にインクルード推奨

class fps_thread : public QThread {
    Q_OBJECT // <--- これが重要

public:
    explicit fps_thread(double fps,QObject *parent = nullptr);
    ~fps_thread() override;
    void stop();
    void change_speed(double fps);

signals:
    void fps_signal();

protected:
    void run() override;

private:
    std::atomic<bool> running{true};
    double target_fps = 0.0;
    int frames = 0;
    std::chrono::steady_clock::time_point start_;
};

#endif // FPS_THREAD_H
