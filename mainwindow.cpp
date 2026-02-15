#include "mainwindow.h"
#include "avidecode.h"
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

    CSS_Design();

    GLwidgetInitialized();

    //スライダー
    ui->Live_horizontalSlider->setFixedHeight(20);
    ui->back10s_pushButton->setFixedWidth(30);
    ui->back10s_pushButton->setFixedHeight(26);
    ui->back1frame_pushButton->setFixedWidth(30);
    ui->back1frame_pushButton->setFixedHeight(26);
    ui->play_pushButton->setFixedWidth(30);
    ui->play_pushButton->setFixedHeight(26);
    ui->go1frame_pushButton->setFixedWidth(30);
    ui->go1frame_pushButton->setFixedHeight(26);
    ui->stop_pushButton->setFixedWidth(30);
    ui->stop_pushButton->setFixedHeight(26);
    ui->reverse_pushButton->setFixedWidth(30);
    ui->reverse_pushButton->setFixedHeight(26);
    ui->go10s_pushButton->setFixedWidth(30);
    ui->go10s_pushButton->setFixedHeight(26);
    ui->label_time->setFixedWidth(132);
    ui->label_time->setFixedHeight(26);
    ui->pushButton_speed->setFixedWidth(30);
    ui->pushButton_speed->setFixedHeight(26);
    ui->pushButton_volume->setFixedWidth(30);
    ui->pushButton_volume->setFixedHeight(26);
    // ui->comboBox_speed->setFixedWidth(55);
    // ui->comboBox_speed->setFixedHeight(26);
    // ui->label_speed->setFixedWidth(85);
    // ui->label_speed->setFixedHeight(26);

    //再生速度combobox
    // QStringList items;
    // items << "1" << "2" << "5" << "10" << "15" << "20" << "30" << "60" << "125" << "250" << "500" << "1000";
    // ui->comboBox_speed->addItems(items);

    //ESCショートカット
    QShortcut *escShortcut = new QShortcut(QKeySequence(Qt::Key_Escape), container);
    escShortcut->setContext(Qt::ApplicationShortcut);
    connect(escShortcut, &QShortcut::activated, this, [this] {
        if (isFullScreenMode) {
            toggleFullScreen();
        }
    });

    //ESCショートカット
    QShortcut *spaceShortcut = new QShortcut(QKeySequence(Qt::Key_Space), container);
    spaceShortcut->setContext(Qt::ApplicationShortcut);
    connect(spaceShortcut, &QShortcut::activated, this, [this] {
        if(pause_flag){
            emit send_manual_resumeplayback();
        }else{
            emit send_manual_pause();
        }
    });

    //動画情報
    QObject::connect(ui->action_videoinfo, &QAction::triggered,this, [&](bool flag) {
        glWidget->videoInfo_flag=flag;
    }, Qt::QueuedConnection);
    QObject::connect(ui->action_histgram, &QAction::triggered,this, [&](bool flag) {
        glWidget->histgram_flag=flag;
    }, Qt::QueuedConnection);

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
                audio_mode=true;
                decodestream->audio_mode=true;
            }else{
                audio_mode=false;
                decodestream->audio_mode=false;
            }
        }
    }, Qt::QueuedConnection);

    //音量ボタンアイコン設定
    ui->pushButton_volume->setIcon(style()->standardIcon( QStyle::SP_MediaVolume));
    ui->actionOpenFile->setIcon(makeWhiteIcon(QIcon::fromTheme("document-open")));
    ui->info->setIcon(makeWhiteIcon(QIcon::fromTheme("dialog-information")));
    ui->actionCloseFile->setIcon(makeWhiteIcon(QIcon::fromTheme("edit-clear")));
    ui->actionFileSave->setIcon(makeWhiteIcon(QIcon::fromTheme("document-print")));

    //ダークUI
    HWND hwnd = (HWND)winId();
    BOOL dark = TRUE;
    DwmSetWindowAttribute(hwnd, 20, &dark, sizeof(dark)); // Windows 11
    DwmSetWindowAttribute(hwnd, 19, &dark, sizeof(dark)); // Windows 10 fallback
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
        auto* vlayout = new QVBoxLayout(ui->openGLContainer);
        vlayout->setContentsMargins(0, 0, 0, 12);   // ★ 超重要
        vlayout->setSpacing(0);                    // ★ 超重要
        ui->openGLContainer->setLayout(vlayout);
        layout = vlayout;
    }
    layout->setContentsMargins(0, 0, 0, 12);
    layout->setSpacing(0);
    layout->addWidget(container);

    // 初期化完了後の処理
    connect(glWidget, &GLWidget::initialized, this, [=]() {
        qDebug() << "GLWidget 初期化完了";
        start_info_thread();

        ui->actionOpenFile->setEnabled(true);
        ui->info->setEnabled(true);

        //エンコード設定用
        encodeSetting = new encode_setting();
        encodeSetting->setWindowModality(Qt::ApplicationModal);
        encodeSetting->hide();
        QObject::connect(encodeSetting, &encode_setting::signal_encode_start,this, &MainWindow::start_encode,Qt::QueuedConnection);
        QObject::connect(encodeSetting, &encode_setting::signal_encode_finished,this, &MainWindow::finished_encode,Qt::QueuedConnection);

        //オーディオ音量調整
        audioVolume = new audio_volume(this);
        audioVolume->setAnchorButton(ui->pushButton_volume);
        ui->pushButton_volume->installEventFilter(this);
        //音量調節ボタン
        connect(ui->pushButton_volume, &QPushButton::clicked, this, [this]() {
            if (!audioVolume)
                return;

            // すでに表示中なら閉じる
            if (audioVolume->isVisible()) {
                audioVolume->hide();
                audioVolume->hideTimer.stop();
            }
            // 非表示なら表示
            else {
                audioVolume->showPopup();  // ← 既存関数を活用
            }
        });
        //スライダー制御シグナル受信
        connect(audioVolume, &audio_volume::volumeChanged,this, [&](int value) {
            g_audio_vol = value;

            QStyle::StandardPixmap icon;

            if (value == 0)
                icon = QStyle::SP_MediaVolumeMuted;   // ミュート
            else if (value < 40)
                icon = QStyle::SP_MediaVolume;         // 小
            else
                icon = QStyle::SP_MediaVolume;         // 中〜大（Qtは統一）

            ui->pushButton_volume->setIcon(style()->standardIcon(icon));
        },Qt::QueuedConnection);

        //再生速度
        videoSpeed = new video_speed(this);
        videoSpeed->setAnchorButton(ui->pushButton_speed);
        ui->pushButton_speed->installEventFilter(this);
        //再生速度ボタン
        connect(ui->pushButton_speed, &QPushButton::clicked, this, [this]() {
            if (!videoSpeed)
                return;

            // すでに表示中なら閉じる
            if (videoSpeed->isVisible()) {
                videoSpeed->hide();
                videoSpeed->hideTimer.stop();
            }
            // 非表示なら表示
            else {
                videoSpeed->setCurrentSpeed(video_speed_ratio);
                videoSpeed->showPopup();
            }
        });

        //ボタンシグナル受信
        connect(videoSpeed, &video_speed::speedChanged, this,[this](double value) {
            video_speed_ratio = value;

            if(fpsstream){
                fpsstream->change_speed(video_speed_ratio*VideoInfo.fps);
            }

            QString text;
            // 整数 (1.0, 2.0) は小数1桁
            if (qFuzzyCompare(video_speed_ratio, qRound(video_speed_ratio))) {
                text = QString("x %1").arg(video_speed_ratio, 0, 'f', 1);
                //フォントサイズをちょっと大きくする
                QFont font = ui->pushButton_speed->font();
                font.setPointSize(7);
            } else {
                // 最大2桁、不要な0削除
                QString s = QString::number(video_speed_ratio, 'f', 2);
                s.remove(QRegularExpression("0+$"));
                s.remove(QRegularExpression("\\.$"));
                text = "x " + s;
                //フォントサイズを小さくする
                QFont font = ui->pushButton_speed->font();
                font.setPointSize(5);
            }
            ui->pushButton_speed->setText(text);
            videoSpeed->hide();
            videoSpeed->hideTimer.stop();
        },Qt::QueuedConnection);
    });

    glWidget->show();
}

//アイコンを白くする
QIcon MainWindow::makeWhiteIcon(const QIcon &icon) {
    QPixmap pix = icon.pixmap(32, 32);
    QImage img = pix.toImage().convertToFormat(QImage::Format_ARGB32);

    for (int y = 0; y < img.height(); y++) {
        for (int x = 0; x < img.width(); x++) {
            QColor c = img.pixelColor(x, y);
            img.setPixelColor(x, y, QColor(255, 255, 255, c.alpha()));
        }
    }
    return QIcon(QPixmap::fromImage(img));
}

//UIデザイン
void MainWindow::CSS_Design(){
    //MainWindow
    this->setStyleSheet(R"(
        QMainWindow {
            background-color: #202020;
        }

        /* ===== Menu Bar ===== */
        QMenuBar {
            background-color: #444444;
            color: white;
        }

        QMenuBar::item {
            color: white;
            background: transparent;
        }

        QMenuBar::item:selected {
            background-color: #6a6a6a;
        }

        QMenuBar::item:disabled {
            color: #888888;
        }

        /* ===== QMenu ===== */
        QMenu {
            background-color: #262626;
            color: #dcdcdc;
            border: 1px solid #3a3a3a;
            padding: 4px;
        }

        /* メニュー項目 */
        QMenu::item {
            padding: 3px 12px 3px 6px; /* ← 左を詰める */
            border-radius: 4px;
        }

        /* アイコンサイズを統一 */
        QMenu::icon {
            width: 16px;
            height: 16px;
        }

        /* 選択時 */
        QMenu::item:selected {
            background-color: #555555;
            color: white;
        }

        /* 無効項目 */
        QMenu::item:disabled {
            color: #777777;
        }

        /* セパレーター */
        QMenu::separator {
            height: 1px;
            background-color: #3a3a3a;
            margin: 5px 6px;
        }

        /* サブメニュー矢印余白縮小 */
        QMenu::right-arrow {
            padding-right: 6px;
        }
    )");

    //再生、停止、逆再生
    QString transportStyle = R"(
        QPushButton {
            background: qlineargradient(
                x1:0, y1:0, x2:0, y2:1,
                stop:0 rgba(80,80,80,230),
                stop:1 rgba(35,35,35,230)
            );
            border: 1px solid rgba(255,255,255,70);
            border-radius: 8px;
            padding: 2px;
            color: rgb(235,235,235); /* ← 通常テキスト 白 */
            font-weight: 600;
        }

        /* Hover = 発光ブルー */
        QPushButton:hover {
            background: qlineargradient(
                x1:0, y1:0, x2:0, y2:1,
                stop:0 rgba(130,190,255,220),
                stop:1 rgba(60,110,220,220)
            );
            border: 1px solid rgba(160,210,255,220);
            color: white; /* ← Hover 時は完全白 */
        }

        /* Pressed = 押し込み */
        QPushButton:pressed {
            background: qlineargradient(
                x1:0, y1:0, x2:0, y2:1,
                stop:0 rgba(50,50,50,230),
                stop:1 rgba(20,20,20,230)
            );
            padding-left: 7px;
            padding-top: 7px;
            color: rgb(200,200,200); /* ← 押下時 少し暗く */
        }

        /* Disabled */
        QPushButton:disabled {
            color: rgba(150,150,150,140);
            background: rgba(40,40,40,200);
            border: 1px solid rgba(255,255,255,30);
        }

        /* フォーカス枠 完全無効 */
        QPushButton:focus {
            border: 1px solid rgba(255,255,255,70);
            outline: none;
        }
    )";

    ui->back10s_pushButton->setStyleSheet(transportStyle);
    ui->back1frame_pushButton->setStyleSheet(transportStyle);
    ui->reverse_pushButton->setStyleSheet(transportStyle);
    ui->play_pushButton->setStyleSheet(transportStyle);
    ui->go1frame_pushButton->setStyleSheet(transportStyle);
    ui->stop_pushButton->setStyleSheet(transportStyle);
    ui->go10s_pushButton->setStyleSheet(transportStyle);
    ui->pushButton_speed->setStyleSheet(transportStyle);
    ui->pushButton_volume->setStyleSheet(transportStyle);
    ui->back10s_pushButton->setFocusPolicy(Qt::NoFocus);
    ui->back1frame_pushButton->setFocusPolicy(Qt::NoFocus);
    ui->reverse_pushButton->setFocusPolicy(Qt::NoFocus);
    ui->play_pushButton->setFocusPolicy(Qt::NoFocus);
    ui->go1frame_pushButton->setFocusPolicy(Qt::NoFocus);
    ui->stop_pushButton->setFocusPolicy(Qt::NoFocus);
    ui->go10s_pushButton->setFocusPolicy(Qt::NoFocus);
    ui->pushButton_volume->setFocusPolicy(Qt::NoFocus);
    ui->pushButton_volume->setIcon(
        style()->standardIcon(QStyle::SP_MediaVolume)
        );
    ui->pushButton_volume->setIconSize(QSize(20, 20));


    //スライダー
    ui->Live_horizontalSlider->setStyleSheet(R"(
        QSlider::groove:horizontal {
            height: 6px;
            background: rgba(255,255,255,40);
            border-radius: 3px;
        }

        QSlider::sub-page:horizontal {
            background: rgb(90,170,255);
            border-radius: 3px;
        }

        QSlider::handle:horizontal {
            width: 14px;
            height: 14px;
            background: white;
            border-radius: 7px;
            margin: -5px 0;
        }
        )");

    //再生時間表示
    ui->label_time->setStyleSheet(R"(
        QLabel {
            color: rgba(220,220,220,200);
            font-family: Consolas;
        }
        )");
}

//マウスカーソルホバー処理
bool MainWindow::eventFilter(QObject *obj, QEvent *event)
{
    //音量調整
    if (obj == ui->pushButton_volume) {
        if (event->type() == QEvent::Leave) {
            audioVolume->hideTimer.start(); // ← 1秒後に消すか判断
        }
    }

    //再生速度
    if (obj == ui->pushButton_speed) {
        if (event->type() == QEvent::Leave) {
            videoSpeed->hideTimer.start(); // ← 1秒後に消すか判断
        }
    }

    return QMainWindow::eventFilter(obj, event);
}

//ウィンドウサイズ変更
void MainWindow::resizeEvent(QResizeEvent *event)
{
    QMainWindow::resizeEvent(event);

    QSize newSize = event->size();
    window_width = newSize.width();
    window_height = newSize.height();

    ui->Live_horizontalSlider->setFixedWidth(window_width-421);
    ui->Live_horizontalSlider->setGeometry(241, window_height-49,window_width-286,window_height-5);
    ui->back10s_pushButton->setGeometry(8, window_height-53,33,window_height-5);
    ui->back1frame_pushButton->setGeometry(41, window_height-53,66,window_height-5);
    ui->reverse_pushButton->setGeometry(74, window_height-53,99,window_height-5);
    ui->play_pushButton->setGeometry(107, window_height-53,132,window_height-5);
    ui->go1frame_pushButton->setGeometry(140, window_height-53,165,window_height-5);
    ui->stop_pushButton->setGeometry(173, window_height-53,198,window_height-5);
    ui->go10s_pushButton->setGeometry(206, window_height-53,231,window_height-5);
    ui->label_time->setGeometry(window_width-191, window_height-52,window_width-261,window_height-5);
    ui->pushButton_speed->setGeometry(window_width-71, window_height-53,window_width-41,window_height-5);
    ui->pushButton_volume->setGeometry(window_width-38, window_height-53,window_width-60,window_height-5);
    ui->openGLContainer->setGeometry(0, 0, window_width, window_height-48); // 位置とサイズを指定

    setMinimumSize(QSize(640, 480));

    glWidget->GLresize();
}

//最大スクリーン検知
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
        ui->back1frame_pushButton->hide();
        ui->back10s_pushButton->hide();
        ui->reverse_pushButton->hide();
        ui->play_pushButton->hide();
        ui->go1frame_pushButton->hide();
        ui->stop_pushButton->hide();
        ui->go10s_pushButton->hide();
        ui->label_time->hide();
        // ui->comboBox_speed->hide();

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
        ui->back1frame_pushButton->show();
        ui->back10s_pushButton->show();
        ui->reverse_pushButton->show();
        ui->play_pushButton->show();
        ui->go1frame_pushButton->show();
        ui->stop_pushButton->show();
        ui->go10s_pushButton->show();
        ui->label_time->show();
        // ui->comboBox_speed->show();

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
        tr("動画ファイルを開く"),
        QDir::homePath(),
        tr("動画ファイル (*.mp4 *.avi)")
        );

    // ファイルが選択されたかどうかを確認
    if (!filePath.isEmpty()) {
        qDebug() << "選択されたファイル:" << filePath;

        Close_Video_File();
        ui->info->setEnabled(false);
        ui->actionOpenFile->setEnabled(false);

        start_decode_thread(filePath);

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

//GPU利用可否の判定
bool MainWindow::canUseGpuDecode(QString filename)
{
    QByteArray File_byteArray = filename.toUtf8();
    const char* input_filename = File_byteArray.constData();
    AVFormatContext* fmt = nullptr;

    if (avformat_open_input(&fmt, input_filename, nullptr, nullptr) < 0)
        return false;

    if (avformat_find_stream_info(fmt, nullptr) < 0) {
        avformat_close_input(&fmt);
        return false;
    }

    int video_stream = -1;
    for (unsigned i = 0; i < fmt->nb_streams; i++) {
        if (fmt->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream = i;
            break;
        }
    }

    if (video_stream < 0) {
        avformat_close_input(&fmt);
        return false;
    }

    AVCodecParameters* par = fmt->streams[video_stream]->codecpar;
    if (par->codec_id == AV_CODEC_ID_H264)
        return (par->width <= 4096 && par->height <= 4096);

    if (par->codec_id == AV_CODEC_ID_HEVC || par->codec_id == AV_CODEC_ID_AV1)
        return (par->width <= 8192 && par->height <= 8192);

    avformat_close_input(&fmt);

    return false;
}

//動画の範囲に合わせてスライダーの範囲を変更
void MainWindow::init_decodethread_complete(){
    if(VideoInfo.audio)
        ui->action_audio_low_laytency->setEnabled(true);

    //一通りUIのセットを行う
    ui->actionOpenFile->setEnabled(true);
    ui->info->setEnabled(true);
    ui->back1frame_pushButton->setEnabled(true);
    ui->back10s_pushButton->setEnabled(true);
    ui->reverse_pushButton->setEnabled(true);
    ui->play_pushButton->setEnabled(true);
    ui->go1frame_pushButton->setEnabled(true);
    ui->stop_pushButton->setEnabled(true);
    ui->go10s_pushButton->setEnabled(true);
    ui->Live_horizontalSlider->setEnabled(true);
    ui->actionFileSave->setEnabled(true);
    ui->actionCloseFile->setEnabled(true);
    ui->action_videoinfo->setEnabled(true);
    ui->action_histgram->setEnabled(true);
    ui->action_filter_sobel->setEnabled(true);
    ui->action_filter_gausian->setEnabled(true);
    ui->action_filter_averaging->setEnabled(true);
    ui->Live_horizontalSlider->setRange(0, VideoInfo.max_framesNo);
    ui->play_pushButton->setText("||");

    //時間の桁数に応じてフォント調整
    if (VideoInfo.max_hour > 0) {
        QFont font = ui->label_time->font();
        font.setPointSize(8);   // 文字サイズ
        ui->label_time->setFont(font);
        ui->label_time->setText(QString::asprintf("00:00:00/%02d:%02d:%02d", VideoInfo.max_hour, VideoInfo.max_minute, VideoInfo.max_second));
    }else {
        QFont font = ui->label_time->font();
        font.setPointSize(12);   // 文字サイズ
        ui->label_time->setFont(font);
        ui->label_time->setText(QString::asprintf("00:00/%02d:%02d", VideoInfo.max_minute, VideoInfo.max_second));
    }

    qDebug() << "Framerate:" << VideoInfo.fps;
    qDebug()<<"MaxFrames:" <<VideoInfo.max_framesNo;
    start_fps_thread(VideoInfo.fps*video_speed_ratio);
    init_async_audio();
    glWidget->GLresize();
}

//動画表示
void MainWindow::decode_view(VideoFrame Frame,bool pause,bool reverse){
    //停止フラグ更新
    pause_flag = pause;
    reverse_flag = reverse;
    FrameNo = Frame.FrameNo;

    if(run_decode_thread){
        //シグナルセット
        QObject::connect(this, &MainWindow::decode_please, decodestream, &decode_thread::receve_decode_flag,Qt::SingleShotConnection);
        if(encode_state==STATE_ENCODING){
            QObject::connect(this, &MainWindow::decode_please, decodestream, &decode_thread::processFrame,Qt::SingleShotConnection);
        }

        //再生時間表示
        if (VideoInfo.max_hour > 0) {
            ui->label_time->setText(QString::asprintf("%02d:%02d:%02d/%02d:%02d:%02d", Frame.hour, Frame.minute, Frame.second, VideoInfo.max_hour, VideoInfo.max_minute, VideoInfo.max_second));
        }else {
            ui->label_time->setText(QString::asprintf("%02d:%02d/%02d:%02d", Frame.minute, Frame.second, VideoInfo.max_minute, VideoInfo.max_second));
        }

        //UIの制御
        if (!ui->Live_horizontalSlider->isSliderDown()&&encode_state==STATE_NOT_ENCODE) {
            ui->Live_horizontalSlider->setValue(Frame.FrameNo);
        }
        slider_No=Frame.FrameNo;

        //描画を開始
        //コンテキストを作成
        glWidget->makeCurrent();

        //OpenGLへ画像を渡して描画、一時停止の場合は情報描画のみ
        if(!pause_flag&&encode_state!=STATE_ENCODE_READY){
            glWidget->uploadToGLTexture(Frame);
        }else if(encode_state==STATE_NOT_ENCODE){
            glWidget->FBO_Rendering(Frame);
        }

        //コンテキストを破棄
        glWidget->doneCurrent();

        //デコードスレッドシグナル
        if(encode_state!=STATE_ENCODE_READY)
            emit decode_please();
    }
}

//非同期オーディオ再生
void MainWindow::play_audio(QByteArray pcm)
{
    if (encode_state == STATE_NOT_ENCODE && !audio_mode) {
        if (audioOutput && audioSink) {
            if (audioSink->bytesFree() >= pcm.size()) {

                float volume = g_audio_vol / 100.0f;

                int16_t* samples = reinterpret_cast<int16_t*>(pcm.data());
                int sampleCount = pcm.size() / sizeof(int16_t);

                for (int i = 0; i < sampleCount; i++) {
                    int32_t v = samples[i] * volume;

                    // クリップ防止
                    if (v > 32767) v = 32767;
                    if (v < -32768) v = -32768;

                    samples[i] = static_cast<int16_t>(v);
                }

                audioOutput->write(pcm);
            }
        }
    }
}

//10秒戻しボタン制御
void MainWindow::back10s_pushbutton_control(){
    int seek = FrameNo-VideoInfo.fps*10;
    if(seek<0) seek = 0;
    emit send_manual_slider(seek);
    emit send_manual_resumeplayback();
}

//1フレーム戻しボタン制御
void MainWindow::back1frame_pushbutton_control(){
    ui->play_pushButton->setText("▶");
    emit send_manual_back1frame();
}

//逆再生
void MainWindow::reverse_pushbutton_control(){
    emit send_manual_reverse();
    ui->play_pushButton->setText("▶");
}

//再生・一時停止の制御
void MainWindow::switch_resume_pause(){
    if(pause_flag){
        emit send_manual_resumeplayback();
        ui->play_pushButton->setText("||");
    }else{
        if(reverse_flag){
            emit send_manual_resumeplayback();
            ui->play_pushButton->setText("||");
        }else{
            emit send_manual_pause();
            ui->play_pushButton->setText("▶");
        }
    }
}

//1フレーム送りボタン制御
void MainWindow::go1frame_pushbutton_control(){
    ui->play_pushButton->setText("▶");
    emit send_manual_go1frame();
}

//停止ボタン制御
void MainWindow::stop_pushbutton_control(){
    emit send_manual_slider(0);
    ui->play_pushButton->setText("||");
    emit send_manual_resumeplayback();
}

//10秒送りボタン制御
void MainWindow::go10s_pushbutton_control(){
    int seek = FrameNo+VideoInfo.fps*10;
    if(seek>VideoInfo.max_framesNo) seek = VideoInfo.max_framesNo;
    emit send_manual_slider(seek);
    emit send_manual_resumeplayback();
}

//スライダー操作
void MainWindow::slider_control(int value){
    emit send_manual_slider(value);
    ui->play_pushButton->setText("▶");
}

//デコードスレッド開始
void MainWindow::start_decode_thread(QString filePath) {
    if (!run_decode_thread) {
        if(canUseGpuDecode(filePath)){
            decodestream = new nvgpudecode(filePath,audio_mode);
        }else{
            QString ext = QFileInfo(filePath).suffix().toLower();
            if (ext == "mp4") {
                decodestream = new cpudecode(filePath, audio_mode);
            }
            else if (ext == "avi") {
                decodestream = new avidecode(filePath, audio_mode);
            }else{
                QMessageBox::warning(this,
                                     tr("ファイルを開けません"),
                                     tr("非対応の動画です"),
                                     QMessageBox::Ok);
                return;
            }
        }

        decode__thread = new QThread;

        decodestream->moveToThread(decode__thread);
        QObject::connect(decodestream, &decode_thread::send_decode_image, this, &MainWindow::decode_view,Qt::QueuedConnection);
        QObject::connect(decodestream, &decode_thread::send_audio, this, &MainWindow::play_audio, Qt::QueuedConnection);

        QObject::connect(decodestream, &decode_thread::send_slider_info, this, &MainWindow::init_decodethread_complete);

        QObject::connect(ui->back1frame_pushButton, &QPushButton::clicked, this, &MainWindow::back1frame_pushbutton_control, Qt::QueuedConnection);
        QObject::connect(ui->back10s_pushButton, &QPushButton::clicked, this, &MainWindow::back10s_pushbutton_control, Qt::QueuedConnection);
        QObject::connect(ui->reverse_pushButton, &QPushButton::clicked, this, &MainWindow::reverse_pushbutton_control, Qt::QueuedConnection);
        QObject::connect(ui->play_pushButton, &QPushButton::clicked, this, &MainWindow::switch_resume_pause, Qt::QueuedConnection);
        QObject::connect(ui->go1frame_pushButton, &QPushButton::clicked, this, &MainWindow::go1frame_pushbutton_control, Qt::QueuedConnection);
        QObject::connect(ui->stop_pushButton, &QPushButton::clicked, this, &MainWindow::stop_pushbutton_control, Qt::QueuedConnection);
        QObject::connect(ui->go10s_pushButton, &QPushButton::clicked, this, &MainWindow::go10s_pushbutton_control, Qt::QueuedConnection);
        QObject::connect(ui->Live_horizontalSlider, &QSlider::sliderMoved, this, &MainWindow::slider_control, Qt::QueuedConnection);
        QObject::connect(this, &MainWindow::send_manual_resumeplayback, decodestream, &decode_thread::resumePlayback);
        QObject::connect(this, &MainWindow::send_manual_pause, decodestream, &decode_thread::pausePlayback);
        QObject::connect(this, &MainWindow::send_manual_reverse, decodestream, &decode_thread::reversePlayback);
        QObject::connect(this, &MainWindow::send_manual_back1frame, decodestream, &decode_thread::back1frame, Qt::QueuedConnection);
        QObject::connect(this, &MainWindow::send_manual_go1frame, decodestream, &decode_thread::go1frame, Qt::QueuedConnection);
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

                ui->actionOpenFile->setEnabled(true);
                ui->info->setEnabled(true);
            }, Qt::QueuedConnection);
        },Qt::SingleShotConnection);

        //スレッドが開始してから処理を開始
        QObject::connect(decode__thread, &QThread::started, this, [this]() {
            run_decode_thread = true;
            QMetaObject::invokeMethod(decodestream, "startProcessing", Qt::QueuedConnection);
        }, Qt::SingleShotConnection);

        decode__thread->start();
    }
}

//デコードスレッド停止
void MainWindow::stop_decode_thread(){
    if (run_decode_thread) {
        audio_mode=decodestream->audio_mode;
        run_decode_thread=false;

        //UI設定
        ui->actionOpenFile->setEnabled(true);
        ui->info->setEnabled(true);
        ui->back1frame_pushButton->setEnabled(false);
        ui->back10s_pushButton->setEnabled(false);
        ui->reverse_pushButton->setEnabled(false);
        ui->play_pushButton->setEnabled(false);
        ui->go1frame_pushButton->setEnabled(false);
        ui->stop_pushButton->setEnabled(false);
        ui->go10s_pushButton->setEnabled(false);
        ui->Live_horizontalSlider->setEnabled(false);
        // ui->label_speed->setEnabled(false);
        // ui->comboBox_speed->setEnabled(false);
        ui->actionFileSave->setEnabled(false);
        ui->actionCloseFile->setEnabled(false);
        ui->action_videoinfo->setEnabled(false);
        ui->action_histgram->setEnabled(false);
        ui->action_filter_sobel->setEnabled(false);
        ui->action_filter_gausian->setEnabled(false);
        ui->action_filter_averaging->setEnabled(false);
        ui->action_audio_low_laytency->setEnabled(false);
        ui->play_pushButton->setText("▶");
        ui->label_time->setText(QString::asprintf("00:00:00/00:00:00"));
        QFont font = ui->label_time->font();
        font.setPointSize(8);   // 文字サイズ
        ui->label_time->setFont(font);

        decodestream->stopProcessing();
        decode__thread->quit();
        decode__thread->wait();

        stop_fps_thread();

        //
        delete audioSink;
        audioSink=nullptr;
    }
}

//fpsスレッド開始
void MainWindow::start_fps_thread(double target_fps){
    if(!fpsstream){
        //fpsthread
        fpsstream = new fps_thread(target_fps);
        fps_view_thread = new QThread;
        fpsstream->moveToThread(fps_view_thread);

        QObject::connect(fpsstream, &fps_thread::fps_signal,
                         decodestream, &decode_thread::processFrame);

        fpsstream->start();
    }
}

//fpsスレッド停止
void MainWindow::stop_fps_thread()
{
    if (!fps_view_thread || !fpsstream)
        return;

    fpsstream->stop();          // worker に停止指示
    fps_view_thread->quit();    // event loop 停止
    fps_view_thread->wait();    // 完全停止待ち

    QObject::disconnect(fpsstream, &fps_thread::fps_signal,
                     decodestream, &decode_thread::processFrame);

    delete fps_view_thread;
    fps_view_thread = nullptr;
    fpsstream = nullptr;
}

//infoスレッド
void MainWindow::start_info_thread(){
    //fpsthread
    infostream = new info_thread;
    info_view_thread = new QThread;
    infostream->moveToThread(info_view_thread);
    infostream->start();
}

//非同期オーディオ初期化
void MainWindow::init_async_audio(){
    // ---------- QAudioSink ----------
    QAudioFormat fmt;
    fmt.setSampleRate(VideoInfo.out_sample_rate);
    fmt.setChannelCount(2);
    fmt.setSampleFormat(QAudioFormat::Int16);

    audioSink = new QAudioSink(fmt);
    audioSink->setBufferSize(200 * 1024);  // ← 200KB (約200ms)
    audioOutput = audioSink->start();
}

//エンコードウィンドウ起動時
void MainWindow::encode_set(){
    //ウィンドウ起動フラグを立てる
    encode_state = STATE_ENCODE_READY;
    glWidget->encode_mode(encode_state);
    decodestream->encode_state = encode_state;

    //現在のフレーム位置を記憶
    Now_Frame = slider_No;

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
        encode_state = STATE_ENCODING;
        decodestream->encode_state = encode_state;
        glWidget->encode_mode(encode_state);
        glWidget->MaxFrame = VideoInfo.max_framesNo;
        emit decode_please();

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
            decodestream->encode_state = encode_state;
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
    encode_state = STATE_NOT_ENCODE;
    glWidget->encode_mode(encode_state);
    decodestream->encode_state = encode_state;
    slider_No = Now_Frame;
    emit decode_please();
    emit send_manual_slider(Now_Frame);
    emit send_manual_resumeplayback();
}
