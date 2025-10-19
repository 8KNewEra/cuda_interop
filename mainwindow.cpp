#include "mainwindow.h"
#include "decode_thread.h"
#include "ui_mainwindow.h"
#include "glwidget.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    g_fps=0;

    //qDebug() <<cv::getBuildInformation();

    ui->setupUi(this);

    GLwidgetInitialized();

    //スライダー
    ui->Live_horizontalSlider->setFixedHeight(20);
    ui->play_pushButton->setFixedWidth(30);
    ui->play_pushButton->setFixedHeight(26);
    ui->pause_pushButton->setFixedWidth(30);
    ui->pause_pushButton->setFixedHeight(26);
    ui->reverse_pushButton->setFixedWidth(30);
    ui->reverse_pushButton->setFixedHeight(26);
    ui->comboBox_speed->setFixedWidth(55);
    ui->comboBox_speed->setFixedHeight(26);
    ui->label_speed->setFixedWidth(85);
    ui->label_speed->setFixedHeight(26);

    //再生速度combobox
    QStringList items;
    items << "1" << "2" << "5" << "10" << "15" << "20" << "30" << "60" << "125" << "250" << "500" << "1000";
    ui->comboBox_speed->addItems(items);

    //ESCショートカット
    QShortcut *escShortcut = new QShortcut(QKeySequence(Qt::Key_Escape), container);
    escShortcut->setContext(Qt::ApplicationShortcut);
    connect(escShortcut, &QShortcut::activated, this, [this] {
        if (isFullScreenMode) {
            qDebug() << "ESC pressed, exiting fullscreen";
            toggleFullScreen();
        }
    });

    QObject::connect(ui->comboBox_speed, &QComboBox::currentTextChanged, this, &MainWindow::set_preview_speed, Qt::QueuedConnection);
    QObject::connect(ui->actionOpenFile, &QAction::triggered, this, &MainWindow::Open_Video_File,Qt::QueuedConnection);
    QObject::connect(ui->actionCloseFile, &QAction::triggered, this, &MainWindow::Close_Video_File,Qt::QueuedConnection);
    QObject::connect(ui->actionFileSave, &QAction::triggered, this, &MainWindow::gpu_encode,Qt::QueuedConnection);
}

MainWindow::~MainWindow()
{
    stop_decode_thread();

    delete ui;
    delete glWidget;
    ui = nullptr;
    glWidget = nullptr;
}

void MainWindow::GLwidgetInitialized(){
    glWidget = new GLWidget();

    container = QWidget::createWindowContainer(glWidget);;
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
        start_fps_thread();
        start_info_thread();
    });

    glWidget->show();
}

void MainWindow::resizeEvent(QResizeEvent *event)
{
    QMainWindow::resizeEvent(event);

    QSize newSize = event->size();
    window_width = newSize.width();
    window_height = newSize.height();

    ui->Live_horizontalSlider->setGeometry(107, window_height-49,window_width-267,window_height-5);
    ui->play_pushButton->setGeometry(41, window_height-53,66,window_height-5);
    ui->pause_pushButton->setGeometry(74, window_height-53,99,window_height-5);
    ui->reverse_pushButton->setGeometry(8, window_height-53,33,window_height-5);
    ui->label_speed->setGeometry(window_width-154, window_height-53,window_width-230,window_height-5);
    ui->comboBox_speed->setGeometry(window_width-65, window_height-53,window_width-120,window_height-5);
    ui->openGLContainer->setGeometry(0, 0, window_width, window_height-48); // 位置とサイズを指定

    setMinimumSize(QSize(480, 320));

    glWidget->GLresize();
}

void MainWindow::changeEvent(QEvent *event)
{
    QMainWindow::changeEvent(event);

    if (event->type() == QEvent::WindowStateChange) {
        if (windowState() & Qt::WindowMaximized) {
            toggleFullScreen();
        }
    }
}

// MainWindow.cpp
void MainWindow::toggleFullScreen()
{
    if (!isFullScreenMode) {
        //フルスクリーン化
        container->setParent(nullptr);
        container->setWindowFlags(Qt::FramelessWindowHint);
        container->showFullScreen();
        container->raise();
        container->activateWindow();

        //UIを隠す
        ui->Live_horizontalSlider->hide();
        ui->play_pushButton->hide();
        ui->pause_pushButton->hide();
        ui->reverse_pushButton->hide();
        ui->comboBox_speed->hide();

        glWidget->GLresize();

        isFullScreenMode = true;

        //元ウィンドウは隠す
        this->hide();

    } else {
        //元ウィンドウを再表示
        this->showNormal();
        this->raise();
        this->activateWindow();

        //元に戻す
        container->setWindowFlags(Qt::Widget);
        container->setParent(ui->openGLContainer);

        QLayout* layout = ui->openGLContainer->layout();
        if (!layout) {
            layout = new QVBoxLayout(ui->openGLContainer);
            ui->openGLContainer->setLayout(layout);
        }
        layout->addWidget(container);

        container->showNormal();

        //UIを再表示
        ui->Live_horizontalSlider->show();
        ui->play_pushButton->show();
        ui->pause_pushButton->show();
        ui->reverse_pushButton->show();
        ui->comboBox_speed->show();

        glWidget->GLresize();

        isFullScreenMode = false;
    }
}

//ファイルを開く
void MainWindow::Open_Video_File()
{
    // ファイルダイアログを開く
    QString filePath = QFileDialog::getOpenFileName(
        this,
        tr("動画ファイルを開く"), // ダイアログのタイトル
        QDir::homePath(),       // 初期ディレクトリ
        // フィルタをMP4のみに限定する
        tr("MP4動画ファイル (*.mp4)")
        );

    // ファイルが選択されたかどうかを確認
    if (!filePath.isEmpty()) {
        qDebug() << "選択されたファイル:" << filePath;

        input_filename = filePath;

        if(run_decode_thread){
            stop_decode_thread();
        }

        start_decode_thread();

    } else {
        qDebug() << "ファイル選択がキャンセルされました";
    }
}

void MainWindow::Close_Video_File()
{
    if(run_decode_thread){
        stop_decode_thread();
    }

    glWidget->makeCurrent();
    glWidget->GLreset();
    glWidget->doneCurrent();
}

//動画表示
void MainWindow::decode_view(uint8_t* d_y, size_t pitch_y,uint8_t* d_uv, size_t pitch_uv,int height, int width){
    if(run_decode_thread){
        QObject::connect(this, &MainWindow::decode_please, decodestream, &decode_thread::receve_decode_flag,Qt::SingleShotConnection);

        //コンテキストを作成
        glWidget->makeCurrent();

        //OpenGLへ画像を渡して描画
        glWidget->uploadToGLTexture(d_y,pitch_y,d_uv,pitch_uv,width,height,slider_No);

        //コンテキストを破棄
        glWidget->doneCurrent();

        g_fps+=1;
        emit decode_please();
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

void MainWindow::set_preview_speed(const QString &text){
    preview_speed=text.toInt();

    //30fpsの場合、timerの精度の関係上33の方が良い
    if(preview_speed==15){
        preview_speed=16;
    }else if(preview_speed==20){
        preview_speed=22;
    }else if(preview_speed==30){
        preview_speed=33;
    }
    emit send_decode_speed(preview_speed);
}

//動画の範囲に合わせてスライダーの範囲を変更
void MainWindow::slider_set_range(int pts,int maxframe,int frame_rate){
    qDebug() << "Framerate:" << framerate;
    qDebug()<<"MaxFrames:" <<maxframe;
    framerate=frame_rate;
    slider_max=maxframe;
    frame_pts=pts;
    ui->Live_horizontalSlider->setRange(0, maxframe);
    glWidget->GLresize();
}

//ライブスレッド開始
void MainWindow::start_decode_thread() {
    if (!run_decode_thread) {
        decodestream = new decode_thread(input_filename);
        decode__thread = new QThread;

        decodestream->moveToThread(decode__thread);
        QObject::connect(decodestream, &decode_thread::send_decode_image, this, &MainWindow::decode_view);
        QObject::connect(this, &MainWindow::send_decode_speed, decodestream, &decode_thread::set_decode_speed);

        QObject::connect(decodestream, &decode_thread::send_slider, this, &MainWindow::slider_control);
        QObject::connect(decodestream, &decode_thread::send_video_info, this, &MainWindow::slider_set_range);

        QObject::connect(ui->play_pushButton, &QPushButton::clicked, decodestream, &decode_thread::resumePlayback, Qt::QueuedConnection);
        QObject::connect(ui->pause_pushButton, &QPushButton::clicked, decodestream, &decode_thread::pausePlayback, Qt::QueuedConnection);
        QObject::connect(ui->Live_horizontalSlider, &QSlider::sliderMoved, decodestream, &decode_thread::sliderPlayback, Qt::QueuedConnection);
        QObject::connect(ui->reverse_pushButton, &QPushButton::clicked, decodestream, &decode_thread::reversePlayback, Qt::QueuedConnection);
        QObject::connect(this, &MainWindow::send_manual_pause, decodestream, &decode_thread::pausePlayback);
        QObject::connect(this, &::MainWindow::send_manual_resumeplayback, decodestream, &decode_thread::resumePlayback);
        QObject::connect(this, &MainWindow::send_manual_slider, decodestream, &decode_thread::sliderPlayback);

        QObject::connect(decodestream, &decode_thread::finished,decode__thread, &QThread::quit,Qt::SingleShotConnection);
        QObject::connect(decode__thread, &QThread::finished,decodestream, &QObject::deleteLater,Qt::SingleShotConnection);
        QObject::connect(decode__thread, &QThread::finished,decode__thread, &QObject::deleteLater,Qt::SingleShotConnection);

        QObject::connect(decode__thread, &QThread::started, this, [this]() {
            this->run_decode_thread = true;
            QMetaObject::invokeMethod(decodestream, "startProcessing", Qt::QueuedConnection);

            ui->comboBox_speed->setCurrentIndex(6);

        }, Qt::AutoConnection);

        decode__thread->start();
    }
}

//ライブスレッド停止
void MainWindow::stop_decode_thread(){
    if (run_decode_thread) {
        run_decode_thread=false;
        decodestream->stopProcessing();
        decode__thread->quit();
        decode__thread->wait();
    }
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
    if(run_decode_thread){
        qDebug()<<"エンコード開始";

        //パフォーマンス評価用
        QElapsedTimer timer;
        timer.start();

        //現在のフレーム位置を記憶
        int Now_Frame=slider_No;

        //停止→フレームを0番にシーク
        encode_flag=true;
        emit send_manual_pause();
        emit send_decode_speed(1000);
        emit send_manual_slider(0);

        //FrameNoが0なことを確認
        while (slider_No != 0) {
            QThread::msleep(1);
            QCoreApplication::processEvents();
        }

        // 進捗ダイアログを作成
        progress = new QProgressDialog("エンコード中...", "キャンセル",1, slider_max, this);
        progress->setWindowModality(Qt::WindowModal);
        progress->setMinimumDuration(0);
        progress->setValue(0);

        //エンコード開始
        glWidget->encode_mode(encode_flag);
        glWidget->encode_maxFrame(slider_max);
        emit send_manual_resumeplayback();

        // 処理ループ内で更新
        while(true) {
            //qDebug()<<slider_No<<":"<<slider_max;
            progress->setValue(slider_No);

            if (progress->wasCanceled()){
                glWidget->encode_maxFrame(slider_No);
                break;
            }

            if(slider_No>=slider_max-1){
                glWidget->encode_maxFrame(slider_No);
                break;
            }

            QCoreApplication::processEvents();
        }

        connect(glWidget, &GLWidget::encode_finished, this, [=]() {
            encode_flag=false;
            double seconds = timer.nsecsElapsed() / 1e9;
            qDebug() << "エンコード終了"
                     << QString::number(seconds, 'f', 3)
                     << "秒";

            emit send_decode_speed(preview_speed);
            emit send_manual_slider(Now_Frame);
            emit send_manual_resumeplayback();
            progress->setValue(slider_max);
        }, Qt::SingleShotConnection);
    }
}
