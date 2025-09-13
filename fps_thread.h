#ifndef FPS_THREAD_H
#define FPS_THREAD_H

#include <QThread>
#include <QObject> // 明示的にインクルード推奨
#include <opencv2/core/mat.hpp>

class fps_thread : public QThread {
    Q_OBJECT // <--- これが重要

public:
    explicit fps_thread(QObject *parent = nullptr);
    ~fps_thread() override;

signals:
    void fps_signal();

public slots:
    void timer_hit();

protected:
    void run() override;

private:
    QTimer* timer;
};

#endif // FPS_THREAD_H
