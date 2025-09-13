#include "fps_thread.h"
#include "qtimer.h"

//コンストラクタ
fps_thread::fps_thread(QObject *parent) : QThread(parent), timer(nullptr) {

}

//デストラクタ
fps_thread::~fps_thread() {
    timer->stop();
    delete timer;
    timer = nullptr;
}

void fps_thread::run() {
    QTimer timer;

    // シグナルとスロットの接続
    connect(&timer, &QTimer::timeout, this, &fps_thread::timer_hit, Qt::DirectConnection);

    // 50msごとにシグナルを発行
    timer.setInterval(1000);
    timer.start();

    printf("debug : fps Thread Starts\n");
    fflush(stdout);

    exec();  // イベントループを開始し、ブロックされる

    printf("debug : fps Thread Stops\n");
    fflush(stdout);

    // タイマーを停止して接続を解除
    timer.stop();
    timer.disconnect();
}

void fps_thread::timer_hit() {
    emit fps_signal();
}
