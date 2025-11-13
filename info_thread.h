#ifndef INFO_THREAD_H
#define INFO_THREAD_H

#include <QThread>
#include <QTimer>
#include "__global__.h"

extern int g_gpu_usage;

class info_thread : public QThread {
    Q_OBJECT

public:
    explicit info_thread(QObject *parent = nullptr);
    ~info_thread();

protected:
    void run() override;

private slots:
    void check_gpu_usage();

private:
    int get_gpu_usage();
};

#endif // INFO_THREAD_H
