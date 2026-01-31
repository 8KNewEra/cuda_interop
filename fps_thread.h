#ifndef FPS_THREAD_H
#define FPS_THREAD_H

#include <QThread>
#include <QObject> // 明示的にインクルード推奨

class fps_thread : public QThread {
    Q_OBJECT // <--- これが重要

public:
    explicit fps_thread(QObject *parent = nullptr);
    ~fps_thread() override;
    void stop();

signals:
    void fps_signal();

protected:
    void run() override;

private:
    std::atomic<bool> running{true};
    double fps = 0.0;
};

#endif // FPS_THREAD_H
