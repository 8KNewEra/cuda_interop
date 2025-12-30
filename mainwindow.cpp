#include "mainwindow.h"
#include "cpudecode.h"
#include "decode_thread.h"
#include "nvgpudecode.h"
#include "ui_mainwindow.h"
#include "glwidget.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    //qDebug() <<cv::getBuildInformation();

    //MainWindow
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
            toggleFullScreen();
        }
    });

    //動画情報
    QObject::connect(ui->action_videoinfo, &QAction::triggered,this, [&](bool flag) {
        glWidget->videoInfo_flag=flag;
    }, Qt::QueuedConnection);
    QObject::connect(ui->action_histgram, &QAction::triggered,this, [&](bool flag) {
        glWidget->histgram_flag=flag;
    }, Qt::QueuedConnection);

    QObject::connect(ui->comboBox_speed, &QComboBox::currentTextChanged, this, &MainWindow::set_preview_speed, Qt::QueuedConnection);
    QObject::connect(ui->actionOpenFile, &QAction::triggered, this, &MainWindow::Open_Video_File,Qt::QueuedConnection);
    QObject::connect(ui->actionCloseFile, &QAction::triggered, this, &MainWindow::Close_Video_File,Qt::QueuedConnection);
    QObject::connect(ui->actionFileSave, &QAction::triggered, this, &MainWindow::encode_set,Qt::QueuedConnection);

    //画像処理
    QObject::connect(ui->action_filter_sobel, &QAction::triggered,this, [&](bool flag) {
        if(flag){
            glWidget->sobelfilterEnabled=1;
        }else{
            glWidget->sobelfilterEnabled=0;
        }
        glWidget->filter_change_flag=true;
    }, Qt::QueuedConnection);

    QObject::connect(ui->action_filter_gausian, &QAction::triggered,this, [&](bool flag) {
        if(flag){
            glWidget->gaussianfilterEnabled=1;
        }else{
            glWidget->gaussianfilterEnabled=0;
        }
        glWidget->filter_change_flag=true;
    }, Qt::QueuedConnection);

    QObject::connect(ui->action_filter_averaging, &QAction::triggered,this, [&](bool flag) {
        if(flag){
            glWidget->averagingfilterEnabled=1;
        }else{
            glWidget->averagingfilterEnabled=0;
        }
        glWidget->filter_change_flag=true;
    }, Qt::QueuedConnection);

    QObject::connect(ui->action_audio_low_laytency, &QAction::triggered,this, [&](bool flag) {
        if(run_decode_thread){
            if(flag){
                decodestream->audio_mode=true;
            }else{
                decodestream->audio_mode=false;
            }
        }
    }, Qt::QueuedConnection);

    // ---------- QAudioSink ----------
    QAudioFormat fmt;
    fmt.setSampleRate(out_sample_rate);
    fmt.setChannelCount(2);
    fmt.setSampleFormat(QAudioFormat::Int16);

    audioSink = new QAudioSink(fmt);
    audioSink->setBufferSize(200 * 1024);  // ← 200KB (約200ms)
    audioOutput = audioSink->start();
}

MainWindow::~MainWindow()
{
    stop_decode_thread();

    delete ui;
    delete glWidget;
    ui = nullptr;
    glWidget = nullptr;
}

//OpenGL初期化
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
        ui->actionOpenFile->setEnabled(true);
        ui->info->setEnabled(true);

        //エンコード設定用
        encodeSetting = new encode_setting();
        encodeSetting->setWindowModality(Qt::ApplicationModal);
        encodeSetting->setFixedSize(515, 464);
        encodeSetting->hide();
        QObject::connect(encodeSetting, &encode_setting::signal_encode_start,this, &MainWindow::start_encode,Qt::QueuedConnection);
        QObject::connect(encodeSetting, &encode_setting::signal_encode_finished,this, &MainWindow::finished_encode,Qt::QueuedConnection);
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

    setMinimumSize(QSize(640, 480));

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

//ファイルを閉じる
void MainWindow::Close_Video_File()
{
    if(run_decode_thread){
        stop_decode_thread();
    }

    glWidget->makeCurrent();
    glWidget->GLreset();
    glWidget->GLreset();
    glWidget->doneCurrent();
}

//動画表示
void MainWindow::decode_view(uint8_t* d_rgba, size_t pitch_rgba,int slider){
    if(run_decode_thread){
        //シグナルセット
        QObject::connect(this, &MainWindow::decode_please, decodestream, &decode_thread::receve_decode_flag,Qt::SingleShotConnection);
        if(encode_state==STATE_ENCODING){
            QObject::connect(this, &MainWindow::decode_please, decodestream, &decode_thread::processFrame,Qt::SingleShotConnection);
        }

        //UIの制御
        if (!ui->Live_horizontalSlider->isSliderDown()&&encode_state==STATE_NOT_ENCODE) {
            ui->Live_horizontalSlider->setValue(slider);
        }
        slider_No=slider;

        //描画を開始
        //コンテキストを作成
        glWidget->makeCurrent();

        //OpenGLへ画像を渡して描画、一時停止の場合は情報描画のみ
        if(d_rgba&&pitch_rgba>0&&encode_state!=STATE_ENCODE_READY){
            glWidget->uploadToGLTexture(d_rgba,pitch_rgba,slider);
        }else if(encode_state==STATE_NOT_ENCODE){
            glWidget->FBO_Rendering();
        }

        //コンテキストを破棄
        glWidget->doneCurrent();

        //デコードスレッドシグナル
        if(encode_state!=STATE_ENCODE_READY)
            emit decode_please();

    }
}

void MainWindow::play_audio(QByteArray pcm){
    // if (audioSink->bytesFree() > 0)
    //     qDebug() << "AUDIO UNDERFLOW !!" << audioSink->bytesFree();

    if (audioOutput)
        audioOutput->write(pcm);
}

//fps表示
void MainWindow::fps_view(){

}

//再生速度コンボボックス
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
void MainWindow::slider_set_range(){
    if(VideoInfo.audio)
        ui->action_audio_low_laytency->setEnabled(true);

    qDebug() << "Framerate:" << VideoInfo.fps;
    qDebug()<<"MaxFrames:" <<VideoInfo.max_framesNo;
    ui->Live_horizontalSlider->setRange(0, VideoInfo.max_framesNo);
    glWidget->GLresize();
}

//デコードスレッド開始
void MainWindow::start_decode_thread() {
    if (!run_decode_thread) {
        decodestream = new cpudecode(input_filename,audio_mode);
        decode__thread = new QThread;

        decodestream->moveToThread(decode__thread);
        QObject::connect(decodestream, &decode_thread::send_decode_image, this, &MainWindow::decode_view);
        QObject::connect(decodestream, &decode_thread::send_audio, this, &MainWindow::play_audio);

        QObject::connect(this, &MainWindow::send_decode_speed, decodestream, &decode_thread::set_decode_speed);
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

        QObject::connect(decodestream, &decode_thread::decode_error, this, [this](const QString &error) {
            Close_Video_File();

            // Qtのメインスレッドで警告ポップアップを表示
            QMetaObject::invokeMethod(this, [this, error]() {
                QMessageBox::warning(this,
                                     tr("デコードエラー"),
                                     tr("デコード中にエラーが発生しました:\n%1").arg(error),
                                     QMessageBox::Ok);
            }, Qt::QueuedConnection);
        });

        QObject::connect(decode__thread, &QThread::started, this, [this]() {
            run_decode_thread = true;
            QMetaObject::invokeMethod(decodestream, "startProcessing", Qt::QueuedConnection);

            ui->comboBox_speed->setCurrentIndex(6);

            ui->play_pushButton->setEnabled(true);
            ui->pause_pushButton->setEnabled(true);
            ui->reverse_pushButton->setEnabled(true);
            ui->Live_horizontalSlider->setEnabled(true);
            ui->label_speed->setEnabled(true);
            ui->comboBox_speed->setEnabled(true);
            ui->actionFileSave->setEnabled(true);
            ui->actionCloseFile->setEnabled(true);
            ui->action_videoinfo->setEnabled(true);
            ui->action_histgram->setEnabled(true);
            ui->action_filter_sobel->setEnabled(true);
            ui->action_filter_gausian->setEnabled(true);
            ui->action_filter_averaging->setEnabled(true);

        }, Qt::AutoConnection);

        decode__thread->start();
    }
}

//デコードスレッド停止
void MainWindow::stop_decode_thread(){
    if (run_decode_thread) {
        audio_mode=decodestream->audio_mode;
        run_decode_thread=false;
        ui->play_pushButton->setEnabled(false);
        ui->pause_pushButton->setEnabled(false);
        ui->reverse_pushButton->setEnabled(false);
        ui->Live_horizontalSlider->setEnabled(false);
        ui->label_speed->setEnabled(false);
        ui->comboBox_speed->setEnabled(false);
        ui->actionFileSave->setEnabled(false);
        ui->actionCloseFile->setEnabled(false);
        ui->action_videoinfo->setEnabled(false);
        ui->action_histgram->setEnabled(false);
        ui->action_filter_sobel->setEnabled(false);
        ui->action_filter_gausian->setEnabled(false);
        ui->action_filter_averaging->setEnabled(false);
        ui->action_audio_low_laytency->setEnabled(false);

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

//エンコードウィンドウ起動時
void MainWindow::encode_set(){
    //ウィンドウ起動フラグを立てる
    encode_state=STATE_ENCODE_READY;
    decodestream->encode_flag=true;
    glWidget->encode_mode(encode_state);

    //現在のフレーム位置を記憶
    Now_Frame=slider_No;

    //停止/再生速度を最大に、エンコードは最速でやるため
    emit send_manual_pause();

    encodeSetting->slider(0,VideoInfo.max_framesNo);
    encodeSetting->show();
}

//エンコード開始
void MainWindow::start_encode(){
    if(run_decode_thread){
        qDebug()<<"エンコード開始";

        //パフォーマンス評価用
        QElapsedTimer timer;
        timer.start();

        //終了検知
        bool wasCanceled=false;
        connect(encodeSetting, &encode_setting::signal_encode_stop, this, [&]() {
            wasCanceled=true;
        }, Qt::SingleShotConnection);
        connect(decodestream, &decode_thread::decode_end, this, [&]() {
            wasCanceled=true;
        }, Qt::SingleShotConnection);
        QObject::connect(this, &MainWindow::decode_please, decodestream, &decode_thread::receve_decode_flag,Qt::SingleShotConnection);

        //エンコード開始
        encode_state=STATE_ENCODING;
        emit decode_please();
        glWidget->encode_mode(encode_state);
        glWidget->MaxFrame=VideoInfo.max_framesNo;

        //FrameNoが0なことを確認
        emit send_manual_slider(0);
        while (slider_No != 0) {
            QThread::msleep(1);
            QCoreApplication::processEvents();
        }
        emit send_manual_resumeplayback();

        // 処理ループ内で更新
        while(true) {
            //qDebug()<<slider_No<<":"<<VideoInfo.max_framesNo;
            encodeSetting->progress_bar(slider_No);

            if (wasCanceled){
                glWidget->MaxFrame=slider_No;
                break;
            }

            // if(slider_No>=VideoInfo.max_framesNo){
            //     glWidget->encode_maxFrame(slider_No);
            //     break;
            // }

            QCoreApplication::processEvents();
        }

        connect(glWidget, &GLWidget::encode_finished, this, [=]() {
            emit send_manual_pause();

            encode_state=STATE_ENCODE_READY;
            glWidget->encode_mode(encode_state);

            //エンコード設定uiに終了通知と処理時間
            double seconds = timer.nsecsElapsed() / 1e9;
            QString encode_time = QString::number(seconds, 'f', 3)+"秒";
            encodeSetting->encode_end(encode_time);

        }, Qt::SingleShotConnection);

        QObject::connect(this, &MainWindow::decode_please, decodestream, &decode_thread::receve_decode_flag,Qt::SingleShotConnection);
    }
}

//エンコードウィンドウを閉じる
void MainWindow::finished_encode(){
    encode_state=STATE_NOT_ENCODE;
    decodestream->encode_flag=false;
    slider_No=Now_Frame;
    emit decode_please();
    emit send_manual_slider(Now_Frame);
    emit send_manual_resumeplayback();
    glWidget->encode_mode(encode_state);
}
