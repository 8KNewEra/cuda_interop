#include "mainwindow.h"
#include "decode_thread.h"
#include "ui_mainwindow.h"
#include "glwidget.h"
#include <dstorage.h>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    g_fps=0;
    //qDebug() <<cv::getBuildInformation();
    ui->setupUi(this);

    IDStorageFactory* factory = nullptr;
    HRESULT hr = DStorageGetFactory(IID_PPV_ARGS(&factory));
    if (FAILED(hr)) {
        qDebug() << "Failed to create DirectStorage factory!";
    }

    cv::cuda::setDevice(0);

    GLwidgetInitialized();

    // 起動時または最初の読み込み時
    //std::vector<FrameIndex> frameIndex = ParseAviIndex("video.avi");

    //スライダー
    ui->Live_horizontalSlider->setFixedHeight(20);
    ui->play_pushButton->setFixedWidth(30);
    ui->play_pushButton->setFixedHeight(26);
    ui->pause_pushButton->setFixedWidth(30);
    ui->pause_pushButton->setFixedHeight(26);
    ui->reverse_pushButton->setFixedWidth(30);
    ui->reverse_pushButton->setFixedHeight(26);
    ui->save_pushButton->setFixedWidth(30);
    ui->save_pushButton->setFixedHeight(26);

    QObject::connect(ui->save_pushButton, &QPushButton::clicked, this, &MainWindow::gpu_encode, Qt::QueuedConnection);

    start_decode_thread();
    //start_avi_thread();
    start_fps_thread();
    start_info_thread();
}

MainWindow::~MainWindow()
{
    delete ui;
    delete glWidget;
    delete decodestream;
    ui = nullptr;
    glWidget = nullptr;
    decodestream = nullptr;
}

void MainWindow::GLwidgetInitialized(){
    glWidget = new GLWidget();
    QWidget* container = QWidget::createWindowContainer(glWidget);
    container->setMinimumSize(320, 240);
    container->setFocusPolicy(Qt::StrongFocus);

    // openGLContainerにコンテナを追加
    QLayout* layout = ui->openGLContainer->layout();
    if (!layout) {
        layout = new QVBoxLayout(ui->openGLContainer);
        ui->openGLContainer->setLayout(layout);
    }
    layout->addWidget(container);

    // 初期化完了後の処理
    connect(glWidget, &GLWidget::initialized, this, [=]() {
        qDebug() << "GLWidget 初期化完了";
        // CUDA-OpenGL連携リソース作成などここで行う
    });

    glWidget->show();
}

void MainWindow::resizeEvent(QResizeEvent *event)
{
    QMainWindow::resizeEvent(event); // デフォルトの resizeEvent 処理を呼び出す

    QSize newSize = event->size();
    window_width = newSize.width();
    window_height = newSize.height();

    ui->Live_horizontalSlider->setGeometry(102, window_height-25,window_width-160,window_height-5);
    ui->play_pushButton->setGeometry(36, window_height-28,66,window_height-10);
    ui->pause_pushButton->setGeometry(69, window_height-28,99,window_height-10);
    ui->reverse_pushButton->setGeometry(3, window_height-28,33,window_height-10);
    ui->save_pushButton->setGeometry(window_width-50, window_height-28,window_width-120,window_height-10);
    ui->openGLContainer->setGeometry(0, 0, window_width, window_height-30); // 位置とサイズを指定

    setMinimumSize(QSize(480, 320));
}

void MainWindow::changeEvent(QEvent *event)
{
    QMainWindow::changeEvent(event);

    if (event->type() == QEvent::WindowStateChange) {
        if (isFullScreen()) {
            // レイアウトを使用している場合は、glWidget がレイアウト全体を占めるように
            // (GridLayout の場合、通常は自動的にそうなります)
            ui->openGLContainer->setGeometry(0, 0, width(), height());
        } else {
            ui->openGLContainer->setGeometry(0, 0, window_width, window_height - 28);
        }
    }
}

void MainWindow::decode_view(uint8_t *d_rgba,int width,int height,size_t pitch_rgba){
    QObject::connect(glWidget, &GLWidget::decode_please, decodestream, &decode_thread::receve_decode_flag,Qt::SingleShotConnection);
    if(d_rgba){
        //OpenGLへ画像を渡して描画
        glWidget->uploadToGLTexture(d_rgba,slider_No, width, height,pitch_rgba);
    }
}

//fps表示
void MainWindow::fps_view(){
    qDebug()<<g_fps<<" fps";
    g_fps=0;
}

//動画の進捗に合わせてスライダーを動かす
void MainWindow::slider_control(int Frame_No){
    if (!ui->Live_horizontalSlider->isSliderDown()&&!encode_flag) {
        ui->Live_horizontalSlider->setValue(Frame_No);
    }
    slider_No=Frame_No;

    //最終フレームまで行ったら再度0フレームへ遷移
    // qDebug()<<slider_No<<":"<<slider_max;
    if(slider_No==slider_max&&!encode_flag){
        emit send_manual_slider(0);
        emit send_manual_resumeplayback();
    }
}

//動画の範囲に合わせてスライダーの範囲を変更
void MainWindow::slider_set_range(int pts,int maxframe,int frame_rate){
    qDebug() << "Framerate:" << framerate;
    qDebug()<<"MaxFrames:" <<maxframe;
    framerate=frame_rate;
    slider_max=maxframe;
    frame_pts=pts;
    ui->Live_horizontalSlider->setRange(0, maxframe);
}

//ライブスレッド開始
void MainWindow::start_decode_thread() {
    if (run_decode_thread == 1) {
        stop_decode_thread();
    }

    if (run_decode_thread == 0) {
        //const char* input_filename = "D:/4K.mp4";
        //const char* input_filename = "D:/test2.mp4";
        const char* input_filename = "D:/ph8K120fps.mp4";
        decodestream = new decode_thread(input_filename);
        decode__thread = new QThread;

        decodestream->moveToThread(decode__thread);
        QObject::connect(decodestream, &decode_thread::send_decode_image, this, &MainWindow::decode_view);
        QObject::connect(this, &MainWindow::send_decode_speed, decodestream, &decode_thread::set_decode_speed);

        QObject::connect(decodestream, &decode_thread::send_slider, this, &MainWindow::slider_control);
        QObject::connect(decodestream, &decode_thread::send_video_info, this, &MainWindow::slider_set_range);
        QObject::connect(decode__thread, &QThread::started, decodestream, &decode_thread::startProcessing);
        QObject::connect(decode__thread, &QThread::finished, decodestream, &decode_thread::stopProcessing);

        QObject::connect(ui->play_pushButton, &QPushButton::clicked, decodestream, &decode_thread::resumePlayback, Qt::QueuedConnection);
        QObject::connect(ui->pause_pushButton, &QPushButton::clicked, decodestream, &decode_thread::pausePlayback, Qt::QueuedConnection);
        QObject::connect(ui->Live_horizontalSlider, &QSlider::sliderMoved, decodestream, &decode_thread::sliderPlayback, Qt::QueuedConnection);
        QObject::connect(ui->reverse_pushButton, &QPushButton::clicked, decodestream, &decode_thread::reversePlayback, Qt::QueuedConnection);
        QObject::connect(this, &MainWindow::send_manual_pause, decodestream, &decode_thread::pausePlayback);
        QObject::connect(this, &::MainWindow::send_manual_resumeplayback, decodestream, &decode_thread::resumePlayback);
        QObject::connect(this, &MainWindow::send_manual_slider, decodestream, &decode_thread::sliderPlayback);

        QObject::connect(decode__thread, &QThread::finished, decodestream, &QObject::deleteLater);
        QObject::connect(decode__thread, &QThread::finished, decode__thread, &QObject::deleteLater);

        decode__thread->start();
        run_decode_thread = 1;
    }
}

//ライブスレッド開始
void MainWindow::start_avi_thread() {
    if (run_avi_thread == 1) {
        stop_decode_thread();
    }

    if (run_avi_thread == 0) {
        avistream = new avi_thread;
        avi_img_thread = new QThread;

        avistream->moveToThread(avi_img_thread);

        QObject::connect(avi_img_thread, &QThread::started, avistream, &avi_thread::startProcessing);
        QObject::connect(avi_img_thread, &QThread::finished, avistream, &avi_thread::stopProcessing);
        QObject::connect(avistream, &avi_thread::send_slider, this, &MainWindow::slider_control);
        QObject::connect(avistream, &avi_thread::slider_max_min, this, &MainWindow::slider_set_range);
        QObject::connect(avistream, &avi_thread::send_decode_image, this, &MainWindow::decode_view);


        QObject::connect(ui->play_pushButton, &QPushButton::clicked, avistream, &avi_thread::resumePlayback, Qt::QueuedConnection);
        QObject::connect(ui->pause_pushButton, &QPushButton::clicked, avistream, &avi_thread::pausePlayback, Qt::QueuedConnection);

        QObject::connect(avi_img_thread, &QThread::finished, avistream, &QObject::deleteLater);
        QObject::connect(avi_img_thread, &QThread::finished, avi_img_thread, &QObject::deleteLater);

        avi_img_thread->start();
        run_avi_thread = 1;
    }
}


//ライブスレッド停止
void MainWindow::stop_decode_thread(){
}

//fpsスレッド開始
void MainWindow::start_fps_thread(){
    //fpsthread
    fpsstream = new fps_thread;
    fps_view_thread = new QThread;
    fpsstream->moveToThread(fps_view_thread);

    QObject::connect(fpsstream, &fps_thread::fps_signal,
                     this, &MainWindow::fps_view);

    fpsstream->start();
}

//infoスレッド
void MainWindow::start_info_thread(){
    //fpsthread
    infostream = new info_thread;
    info_view_thread = new QThread;
    infostream->moveToThread(info_view_thread);

    infostream->start();
}

void MainWindow::gpu_encode(){
    qDebug()<<"エンコード開始";

    //パフォーマンス評価用
    QElapsedTimer timer;
    timer.start();

    //現在のフレーム位置を記憶
    int Now_Frame=slider_No;

    //停止→フレームを0番にシーク
    encode_flag=true;
    emit send_manual_pause();
    emit send_manual_slider(0);

    //FrameNoが0なことを確認
    while (slider_No != 0) {
        QThread::msleep(1); // CPU負荷を抑える
        QCoreApplication::processEvents(); // UI更新
    }

    // 進捗ダイアログを作成
    progress = new QProgressDialog("エンコード中...", "キャンセル",1, slider_max, this);
    progress->setWindowModality(Qt::WindowModal);
    progress->setMinimumDuration(0); // すぐ表示
    progress->setValue(0);

    //エンコード開始
    glWidget->encode_mode(encode_flag);
    glWidget->encode_maxFrame(slider_max);
    emit send_manual_resumeplayback();

    // 処理ループ内で更新
    while(true) {
        // プログレス更新
        //qDebug()<<slider_No<<":"<<slider_max;
        progress->setValue(slider_No);

        // ユーザーがキャンセルを押したか確認
        if (progress->wasCanceled()){
            glWidget->encode_maxFrame(slider_No);
            break;
        }

        if(slider_No>=slider_max-1){
            glWidget->encode_maxFrame(slider_No);
            break;
        }

        QCoreApplication::processEvents();  // UI応答性の確保
    }

    connect(glWidget, &GLWidget::encode_finished, this, [=]() {
        encode_flag=false;
        double seconds = timer.nsecsElapsed() / 1e9; // ナノ秒 → 秒
        qDebug() << "エンコード終了"
                 << QString::number(seconds, 'f', 3) // 小数点以下3桁
                 << "秒";

        emit send_manual_slider(Now_Frame);
        emit send_manual_resumeplayback();
        progress->setValue(slider_max); // 完了
    }, Qt::SingleShotConnection);
}
