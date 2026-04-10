#include "mainwindow.h"
#include "avidecode.h"
#include "cpudecode.h"
#include "decode_thread.h"
#include "nvgpudecode.h"
#include "ui_mainwindow.h"
#include "rangeslider.h"
#include "glwidget.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    //qDebug() <<cv::getBuildInformation();

    //MainWindow
    ui->setupUi(this);

    //スライダー
    rangeSlider = new RangeSlider(this);
    ui->horizontalLayout_slider->addWidget(rangeSlider);
    rangeSlider->setParent(ui->centralwidget);
    rangeSlider->setRange(0, 1000);
    rangeSlider->setValues(0, 1000);
    rangeSlider->setPlayValue(0);
    rangeSlider->setEnabled(false);

    CSS_Design();
    GLwidgetInitialized();

    //スライダー
    rangeSlider->setFixedHeight(38);
    ui->cutstart_pushButton->setFixedWidth(30);
    ui->cutstart_pushButton->setFixedHeight(26);
    ui->backstartframe_pushButton->setFixedWidth(30);
    ui->backstartframe_pushButton->setFixedHeight(26);
    ui->back1frame_pushButton->setFixedWidth(30);
    ui->back1frame_pushButton->setFixedHeight(26);
    ui->play_pushButton->setFixedWidth(30);
    ui->play_pushButton->setFixedHeight(26);
    ui->go1frame_pushButton->setFixedWidth(30);
    ui->go1frame_pushButton->setFixedHeight(26);
    ui->goendframe_pushButton->setFixedWidth(30);
    ui->goendframe_pushButton->setFixedHeight(26);
    ui->stop_pushButton->setFixedWidth(30);
    ui->stop_pushButton->setFixedHeight(26);
    ui->reverse_pushButton->setFixedWidth(30);
    ui->reverse_pushButton->setFixedHeight(26);
    ui->cutend_pushButton->setFixedWidth(30);
    ui->cutend_pushButton->setFixedHeight(26);

    ui->label_play_time->setFixedWidth(240);
    ui->label_play_time->setFixedHeight(26);
    ui->label_start_time->setFixedWidth(240);
    ui->label_start_time->setFixedHeight(26);
    ui->label_end_time->setFixedWidth(240);
    ui->label_end_time->setFixedHeight(26);
    ui->label_range_time->setFixedWidth(240);
    ui->label_range_time->setFixedHeight(26);
    ui->label_range_time->hide();
    ui->pushButton_speed->setFixedWidth(30);
    ui->pushButton_speed->setFixedHeight(26);
    ui->pushButton_volume->setFixedWidth(30);
    ui->pushButton_volume->setFixedHeight(26);
    ui->jumpmode_pushButton->setFixedWidth(72);
    ui->jumpmode_pushButton->setFixedHeight(22);
    ui->jumpvalue_spinbox->setFixedWidth(65);
    ui->jumpvalue_spinbox->setFixedHeight(21);
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

    //ジャンプ機能、lineedit
    ui->jumpvalue_spinbox->setValue(1);           // 初期値

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
        vlayout->setContentsMargins(0, 0, 0, 0);   // ★ 超重要
        vlayout->setSpacing(0);                    // ★ 超重要
        ui->openGLContainer->setLayout(vlayout);
        layout = vlayout;
    }
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);
    layout->addWidget(container);

    // 初期化完了後の処理
    connect(glWidget, &GLWidget::initialized, this, [=]() {
        qDebug() << "GLWidget 初期化完了";
        start_info_thread();

        ui->actionOpenFile->setEnabled(true);

        //エンコード設定用
        encodeSetting = new encode_setting();
        encodeSetting->setWindowModality(Qt::ApplicationModal);
        encodeSetting->hide();
        QObject::connect(encodeSetting, &encode_setting::signal_encode_start,this, &MainWindow::start_encode,Qt::QueuedConnection);
        QObject::connect(encodeSetting, &encode_setting::signal_encode_finished,this, &MainWindow::finished_encode,Qt::QueuedConnection);


        /* 音量調整ボタン */
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

        /* 再生速度 */
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

            ui->label_range_time->hide();
        },Qt::QueuedConnection);

        /* Frameジャンプ、モード選択 */
        jumpMode = new jump_mode(this);
        jumpMode->setAnchorButton(ui->jumpmode_pushButton);
        ui->jumpmode_pushButton->installEventFilter(this);
        //Frameジャンプ、モード選択ボタン
        connect(ui->jumpmode_pushButton, &QPushButton::clicked, this, [this]() {
            if (!jumpMode)
                return;

            // すでに表示中なら閉じる
            if (jumpMode->isVisible()) {
                jumpMode->hide();
                jumpMode->hideTimer.stop();
            }
            // 非表示なら表示
            else {
                jumpMode->setCurrentMode(frame_jump_mode);
                jumpMode->showPopup();
            }
        });

        //ボタンシグナル受信
        connect(jumpMode, &jump_mode::modeChanged, this,[this](int value) {
            frame_jump_mode = value;

            QString text;
            // 整数 (1.0, 2.0) は小数1桁
            if (value == JUMP_MODE_SECOND) {
                e_jumpmode = JUMP_MODE_SECOND;
                ui->jumpvalue_spinbox->setValue(jumpValueSecond);
                text = "秒";
                QFont font = ui->jumpmode_pushButton->font();
                font.setPointSize(7);
            }else if(value == JUMP_MODE_FRAME) {
                e_jumpmode = JUMP_MODE_FRAME;
                ui->jumpvalue_spinbox->setValue(jumpValueFrame);
                text = "フレーム";
                QFont font = ui->jumpmode_pushButton->font();
                font.setPointSize(5);
            }else if(value == JUMP_MODE_TARGETFRAME) {
                e_jumpmode = JUMP_MODE_TARGETFRAME;
                ui->jumpvalue_spinbox->setValue(jumpValueFrameNo);
                text = "任意フレーム";
                QFont font = ui->jumpmode_pushButton->font();
                font.setPointSize(5);
            }
            ui->jumpmode_pushButton->setText(text);
            jumpMode->hide();
            jumpMode->hideTimer.stop();
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

    ui->cutstart_pushButton->setStyleSheet(transportStyle);
    ui->backstartframe_pushButton->setStyleSheet(transportStyle);
    ui->back1frame_pushButton->setStyleSheet(transportStyle);
    ui->reverse_pushButton->setStyleSheet(transportStyle);
    ui->play_pushButton->setStyleSheet(transportStyle);
    ui->go1frame_pushButton->setStyleSheet(transportStyle);
    ui->goendframe_pushButton->setStyleSheet(transportStyle);
    ui->stop_pushButton->setStyleSheet(transportStyle);
    ui->cutend_pushButton->setStyleSheet(transportStyle);
    ui->pushButton_speed->setStyleSheet(transportStyle);
    ui->pushButton_volume->setStyleSheet(transportStyle);
    ui->jumpmode_pushButton->setStyleSheet(transportStyle);
    ui->cutstart_pushButton->setFocusPolicy(Qt::NoFocus);
    ui->backstartframe_pushButton->setFocusPolicy(Qt::NoFocus);
    ui->back1frame_pushButton->setFocusPolicy(Qt::NoFocus);
    ui->reverse_pushButton->setFocusPolicy(Qt::NoFocus);
    ui->play_pushButton->setFocusPolicy(Qt::NoFocus);
    ui->go1frame_pushButton->setFocusPolicy(Qt::NoFocus);
    ui->goendframe_pushButton->setFocusPolicy(Qt::NoFocus);
    ui->stop_pushButton->setFocusPolicy(Qt::NoFocus);
    ui->cutend_pushButton->setFocusPolicy(Qt::NoFocus);
    ui->jumpmode_pushButton->setFocusPolicy(Qt::NoFocus);

    ui->pushButton_volume->setFocusPolicy(Qt::NoFocus);
    ui->pushButton_volume->setIcon(
        style()->standardIcon(QStyle::SP_MediaVolume)
        );
    ui->pushButton_volume->setIconSize(QSize(20, 20));

    //スピンボックス制御
    QString inputStyle = R"(
    /* ===== QSpinBox ===== */
        QSpinBox {
            background: qlineargradient(
                x1:0, y1:0, x2:0, y2:1,
                stop:0 rgba(30,30,30,230),
                stop:1 rgba(15,15,15,230)
            );

            border: 1px solid rgba(0,0,0,200);
            border-radius: 3px;

            /* 内側の凹み表現 */
            padding: 3px 6px;
            padding-right: 2px;   /* ★ ボタン領域分広げる */

            color: rgb(200,210,220);
            font-family: Consolas;
            font-size: 13px;
        }

        /* 内側ハイライト（上）と影（下） */
        QSpinBox {
            border: 1px solid rgba(255,255,255,100);
        }

        /* Hover */
        QSpinBox:hover {
            border: 1px solid rgba(120,150,180,255);
        }

        /* Focus */
        QSpinBox:focus {
            border: 1px solid rgba(130,190,255,240);
            background: qlineargradient(
                x1:0, y1:0, x2:0, y2:1,
                stop:0 rgba(40,60,90,220),
                stop:1 rgba(15,25,40,220)
            );
            color: white;
        }

        /* 選択範囲 */
        QSpinBox::selection {
            background: rgba(100,170,255,200);
            color: white;
        }

        /* Disabled */
        QSpinBox:disabled {
            background: rgba(30,30,30,200);
            border: 1px solid rgba(120,120,120,120);
            color: rgba(140,140,140,150);
        }
    )";
    ui->jumpvalue_spinbox->setStyleSheet(inputStyle);
    ui->jumpvalue_spinbox->setFocusPolicy(Qt::StrongFocus);

    //スライダー
    rangeSlider->setStyleSheet(R"(
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
    ui->label_start_time->setStyleSheet(R"(
        QLabel {
            color: rgba(220,220,220,200);
            font-family: Consolas;
        }
        )");
    ui->label_play_time->setStyleSheet(R"(
        QLabel {
            color: rgba(255,100,100,200);
            font-family: Consolas;
        }
        )");
    ui->label_end_time->setStyleSheet(R"(
        QLabel {
            color: rgba(220,220,220,200);
            font-family: Consolas;
        }
        )");
    ui->label_range_time->setStyleSheet(R"(
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

    //フレームジャンプモード
    if (obj == ui->jumpmode_pushButton) {
        if (event->type() == QEvent::Leave) {
            jumpMode->hideTimer.start(); // ← 1秒後に消すか判断
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

    ui->openGLContainer->setGeometry(0, 0, window_width, window_height-89); // 位置とサイズを指定

    int slider_length = window_width-325;
    rangeSlider->setFixedWidth(slider_length);
    rangeSlider->setGeometry(245, window_height-62,window_width-286,window_height-5);

    ui->cutstart_pushButton->setGeometry(8, window_height-83,198,window_height-5);
    ui->backstartframe_pushButton->setGeometry(41, window_height-83,165,window_height-5);
    ui->reverse_pushButton->setGeometry(74, window_height-83,99,window_height-5);
    ui->play_pushButton->setGeometry(107, window_height-83,132,window_height-5);
    ui->goendframe_pushButton->setGeometry(140, window_height-83,165,window_height-5);
    ui->stop_pushButton->setGeometry(173, window_height-83,198,window_height-5);
    ui->cutend_pushButton->setGeometry(206, window_height-83,198,window_height-5);

    ui->back1frame_pushButton->setGeometry(18, window_height-53,66,window_height-5);
    ui->jumpvalue_spinbox->setGeometry(53, window_height-51,231,window_height-5);
    ui->jumpmode_pushButton->setGeometry(118, window_height-51,231,window_height-5);
    ui->go1frame_pushButton->setGeometry(194, window_height-53,165,window_height-5);

    ui->pushButton_speed->setGeometry(window_width-71, window_height-83,41,window_height-5);
    ui->pushButton_volume->setGeometry(window_width-38, window_height-83,41,window_height-5);
    ui->label_start_time->setGeometry(slider_length*0.5-100, window_height-89,41,window_height-5);
    ui->label_play_time->setGeometry(slider_length*0.5+125, window_height-89,41,window_height-5);
    ui->label_end_time->setGeometry(slider_length*0.5+350, window_height-89,41,window_height-5);
    ui->label_range_time->setGeometry(window_width-340, window_height-89,41,window_height-5);

    setMinimumSize(QSize(990, 540));
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
        rangeSlider->hide();
        ui->cutstart_pushButton->hide();
        ui->backstartframe_pushButton->hide();
        ui->back1frame_pushButton->hide();
        ui->reverse_pushButton->hide();
        ui->play_pushButton->hide();
        ui->go1frame_pushButton->hide();
        ui->goendframe_pushButton->hide();
        ui->stop_pushButton->hide();
        ui->cutend_pushButton->hide();
        ui->label_start_time->hide();
        ui->label_play_time->hide();
        ui->label_end_time->hide();
        ui->label_range_time->hide();
        ui->pushButton_volume->hide();
        ui->pushButton_speed->hide();
        ui->jumpvalue_spinbox->hide();
        ui->jumpmode_pushButton->hide();
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
        rangeSlider->show();
        ui->cutstart_pushButton->show();
        ui->backstartframe_pushButton->show();
        ui->back1frame_pushButton->show();
        ui->reverse_pushButton->show();
        ui->play_pushButton->show();
        ui->go1frame_pushButton->show();
        ui->goendframe_pushButton->show();
        ui->stop_pushButton->show();
        ui->cutend_pushButton->show();
        ui->label_start_time->show();
        ui->label_play_time->show();
        ui->label_end_time->show();
        //ui->label_range_time->show();
        ui->pushButton_volume->show();
        ui->pushButton_speed->show();
        ui->jumpvalue_spinbox->show();
        ui->jumpmode_pushButton->show();

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

//現在のフレームを始端
void MainWindow::cutstart_pushbutton_control(){
    emit send_manual_pause();
    ui->play_pushButton->setText("▶");
    rangeSlider->setStartValue(FrameNo);
}

//始端へジャンプするボタン制御
void MainWindow::backstartframe_pushbutton_control(){
    emit send_manual_pause();
    ui->play_pushButton->setText("▶");
    emit send_manual_high_res_slider(VideoInfo.start_range_framesNo);
}

//1フレーム戻しボタン制御
void MainWindow::back1frame_pushbutton_control(){
    ui->play_pushButton->setText("▶");
    emit send_manual_pause();

    if(ui->jumpvalue_spinbox->text().isEmpty()){
        if(e_jumpmode == JUMP_MODE_SECOND){
            ui->jumpvalue_spinbox->setValue(jumpValueSecond);
        }else if(e_jumpmode == JUMP_MODE_FRAME){
            ui->jumpvalue_spinbox->setValue(jumpValueFrame);
        }else if(e_jumpmode == JUMP_MODE_TARGETFRAME){
            ui->jumpvalue_spinbox->setValue(FrameNo);
            jumpValueFrameNo = FrameNo;
        }
    }

    int seek{};
    if(e_jumpmode == JUMP_MODE_SECOND){
        seek = FrameNo-VideoInfo.fps*jumpValueSecond;
        emit send_manual_high_res_slider(seek);
    }else if(e_jumpmode == JUMP_MODE_FRAME){
        if(jumpValueFrame == 1){
            emit send_manual_back1frame();
        }else{
            seek = FrameNo-jumpValueFrame;
            emit send_manual_high_res_slider(seek);
        }
    }else if(e_jumpmode == JUMP_MODE_TARGETFRAME){
        seek = jumpValueFrameNo;
        emit send_manual_high_res_slider(seek);
    }
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
    emit send_manual_pause();

    if(ui->jumpvalue_spinbox->text().isEmpty()){
        if(e_jumpmode == JUMP_MODE_SECOND){
            ui->jumpvalue_spinbox->setValue(jumpValueSecond);
        }else if(e_jumpmode == JUMP_MODE_FRAME){
            ui->jumpvalue_spinbox->setValue(jumpValueFrame);
        }else if(e_jumpmode == JUMP_MODE_TARGETFRAME){
            ui->jumpvalue_spinbox->setValue(FrameNo);
            jumpValueFrameNo = FrameNo;
        }
    }

    int seek{};
    if(e_jumpmode == JUMP_MODE_SECOND){
        seek = FrameNo+VideoInfo.fps*jumpValueSecond;
        emit send_manual_high_res_slider(seek);
    }else if(e_jumpmode == JUMP_MODE_FRAME){
        if(jumpValueFrame == 1){
            emit send_manual_go1frame();
        }else{
            seek = FrameNo+jumpValueFrame;
            emit send_manual_high_res_slider(seek);
        }
    }else if(e_jumpmode == JUMP_MODE_TARGETFRAME){
        seek = jumpValueFrameNo;
        emit send_manual_high_res_slider(seek);
    }
}

//終端へジャンプするボタン制御
void MainWindow::goendframe_pushbutton_control(){
    emit send_manual_pause();
    ui->play_pushButton->setText("▶");
    emit send_manual_high_res_slider(VideoInfo.end_range_framesNo);
}

//停止ボタン制御
void MainWindow::stop_pushbutton_control(){
    rangeSlider->setStartValue(0);
    rangeSlider->setEndValue(VideoInfo.max_framesNo);
    emit send_manual_slider(0);
    ui->play_pushButton->setText("▶");
    range_label_control(VideoInfo.max_framesNo/VideoInfo.fps,VideoInfo.max_framesNo);
}

//現在のフレームを終端
void MainWindow::cutend_pushbutton_control(){
    emit send_manual_pause();
    ui->play_pushButton->setText("▶");
    rangeSlider->setEndValue(FrameNo);
}

//スライダー操作
void MainWindow::slider_control(int value){
    emit send_manual_slider(value);
    ui->play_pushButton->setText("▶");
}

void MainWindow::slider_start_control(int value){
    //時間を計算 時:分:秒
    double start_time = value/VideoInfo.fps;
    int start_hour = start_time / 3600;
    int start_minute = (int(start_time) % 3600) / 60;
    int start_second = fmod(start_time, 60.0);

    //時間を計算 時:分:秒
    range_label_control(VideoInfo.end_range_framesNo/VideoInfo.fps-value/VideoInfo.fps,VideoInfo.end_range_framesNo-value);

    if (start_hour > 0) {
        ui->label_start_time->setText(QString::asprintf("開始:%02d:%02d:%02d(%d Frames)", start_hour, start_minute, start_second, value));
    }else{
        ui->label_start_time->setText(QString::asprintf("開始:%02d:%02d(%d Frames)", start_minute, start_second, value));
    }

    emit send_manual_range_start_slider(value);
}

void MainWindow::slider_end_control(int value){
    //時間を計算 時:分:秒
    double end_time = value/VideoInfo.fps;
    int end_hour = end_time / 3600;
    int end_minute = (int(end_time) % 3600) / 60;
    int end_second = fmod(end_time, 60.0);

    //時間を計算 時:分:秒
    range_label_control(value/VideoInfo.fps-VideoInfo.start_range_framesNo/VideoInfo.fps,value-VideoInfo.start_range_framesNo);

    if (end_hour > 0) {
        ui->label_end_time->setText(QString::asprintf("終了:%02d:%02d:%02d(%d Frames)", end_hour, end_minute, end_second, value));
    }else{
        ui->label_end_time->setText(QString::asprintf("終了:%02d:%02d(%d Frames)", end_minute, end_second, value));
    }

    emit send_manual_range_end_slider(value);
}

void MainWindow::get_jump_value(){
    if(ui->jumpvalue_spinbox->text().isEmpty()) return;

    int value = ui->jumpvalue_spinbox->text().toInt();
    if(e_jumpmode == JUMP_MODE_SECOND){
        if (value < 1) value = 1;
        if (value > VideoInfo.max_framesNo/VideoInfo.fps) value = VideoInfo.max_framesNo/VideoInfo.fps;
        jumpValueSecond = value;
    }else if(e_jumpmode == JUMP_MODE_FRAME){
        if (value < 1) value = 1;
        if (value > VideoInfo.max_framesNo) value = VideoInfo.max_framesNo;
        jumpValueFrame = value;
    }
    else if(e_jumpmode == JUMP_MODE_TARGETFRAME){
        if (value < 0) value = 0;
        if (value > VideoInfo.max_framesNo) value = VideoInfo.max_framesNo;
        jumpValueFrameNo = value;
    }

    ui->jumpvalue_spinbox->setValue(value);
}

void MainWindow::range_label_control(int range_time,int FrameNo){
    //qDebug()<<range_time<<":"<<FrameNo;
    int range_hour = range_time / 3600;
    int range_minute = (int(range_time) % 3600) / 60;
    int range_second = fmod(range_time, 60.0);

    if (range_hour > 0) {
        ui->label_range_time->setText(QString::asprintf("再生範囲:%02d:%02d:%02d(%d Frames)", range_hour, range_minute, range_second, FrameNo));
    }else{
        ui->label_range_time->setText(QString::asprintf("再生範囲:%02d:%02d(%d Frames)", range_minute, range_second, FrameNo));
    }
}

//UIの有効無効制御
void MainWindow::heavy_process_UI_control(bool flag){
    ui->cutstart_pushButton->setEnabled(flag);
    ui->backstartframe_pushButton->setEnabled(flag);
    ui->back1frame_pushButton->setEnabled(flag);
    ui->reverse_pushButton->setEnabled(flag);
    ui->play_pushButton->setEnabled(flag);
    ui->go1frame_pushButton->setEnabled(flag);
    ui->goendframe_pushButton->setEnabled(flag);
    ui->stop_pushButton->setEnabled(flag);
    ui->cutend_pushButton->setEnabled(flag);
    ui->jumpvalue_spinbox->setEnabled(flag);
    ui->jumpmode_pushButton->setEnabled(flag);
    ui->actionFileSave->setEnabled(flag);
    ui->action_filter_sobel->setEnabled(flag);
    ui->action_filter_gausian->setEnabled(flag);
    ui->action_filter_averaging->setEnabled(flag);
    ui->action_audio_low_laytency->setEnabled(flag);
    rangeSlider->setEnabled(flag);
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
    ui->actionCloseFile->setEnabled(true);
    heavy_process_UI_control(true);

    //時間の桁数に応じてフォント調整
    if (VideoInfo.max_hour > 0) {
        QFont font_start = ui->label_start_time->font();
        QFont font_play = ui->label_play_time->font();
        QFont font_end = ui->label_end_time->font();
        QFont font_range = ui->label_range_time->font();
        font_start.setPointSize(8);   // 文字サイズ
        font_play.setPointSize(8);   // 文字サイズ
        font_end.setPointSize(8);   // 文字サイズ
        font_range.setPointSize(8);   // 文字サイズ
        ui->label_start_time->setFont(font_start);
        ui->label_play_time->setFont(font_play);
        ui->label_end_time->setFont(font_end);
        ui->label_range_time->setFont(font_range);
        ui->label_start_time->setText(QString::asprintf("開始:%02d:%02d:%02d(%d Frames)", 0, 0, 0, 0));
        ui->label_play_time->setText(QString::asprintf("再生:%02d:%02d:%02d(%d Frames)", 0, 0, 0, 0));
        ui->label_end_time->setText(QString::asprintf("終了:%02d:%02d:%02d(%d Frames)", VideoInfo.max_hour, VideoInfo.max_minute, VideoInfo.max_second, VideoInfo.max_framesNo));
        ui->label_range_time->setText(QString::asprintf("再生範囲:%02d:%02d:%02d(%d Frames)", VideoInfo.max_hour, VideoInfo.max_minute, VideoInfo.max_second, VideoInfo.max_framesNo));
    }else {
        QFont font_start = ui->label_start_time->font();
        QFont font_play = ui->label_play_time->font();
        QFont font_end = ui->label_end_time->font();
        QFont font_range = ui->label_range_time->font();
        font_start.setPointSize(11);   // 文字サイズ
        font_play.setPointSize(11);   // 文字サイズ
        font_end.setPointSize(11);   // 文字サイズ
        font_range.setPointSize(11);   // 文字サイズ
        ui->label_start_time->setFont(font_start);
        ui->label_play_time->setFont(font_play);
        ui->label_end_time->setFont(font_end);
        ui->label_range_time->setFont(font_range);
        ui->label_start_time->setText(QString::asprintf("開始:%02d:%02d:%02d(%d Frames)", 0, 0, 0, 0));
        ui->label_play_time->setText(QString::asprintf("再生:%02d:%02d:%02d(%d Frames)", 0, 0, 0, 0));
        ui->label_end_time->setText(QString::asprintf("終了:%02d:%02d:%02d(%d Frames)", VideoInfo.max_hour, VideoInfo.max_minute, VideoInfo.max_second, VideoInfo.max_framesNo));
        ui->label_range_time->setText(QString::asprintf("再生範囲:%02d:%02d:%02d(%d Frames)", VideoInfo.max_hour, VideoInfo.max_minute, VideoInfo.max_second, VideoInfo.max_framesNo));
    }

    ui->play_pushButton->setText("||");
    ui->label_range_time->hide();
    rangeSlider->setRange(0, VideoInfo.max_framesNo);
    rangeSlider->setValues(0, VideoInfo.max_framesNo);
    rangeSlider->setPlayValue(0);

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
            ui->label_play_time->setText(QString::asprintf("再生:%02d:%02d:%02d(%d Frames)", Frame.hour, Frame.minute, Frame.second, Frame.FrameNo));
        }else {
            ui->label_play_time->setText(QString::asprintf("再生:%02d:%02d(%d Frames)", Frame.minute, Frame.second,Frame.FrameNo));
        }

        //1フレーム戻しが入っている場合はUI有効に
        if(!decodestream->back1frame_flag){
            heavy_process_UI_control(true);
        }

        //UIの制御
        if (encode_state==STATE_NOT_ENCODE) {
            //ui->Live_horizontalSlider->setValue(Frame.FrameNo);
            rangeSlider->setPlayValue(Frame.FrameNo);
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
        QObject::connect(decodestream, &decode_thread::heavy_process_signal, this, &MainWindow::heavy_process_UI_control);

        QObject::connect(ui->cutstart_pushButton, &QPushButton::clicked, this, &MainWindow::cutstart_pushbutton_control, Qt::QueuedConnection);
        QObject::connect(ui->backstartframe_pushButton, &QPushButton::clicked, this, &MainWindow::backstartframe_pushbutton_control, Qt::QueuedConnection);
        QObject::connect(ui->back1frame_pushButton, &QPushButton::clicked, this, &MainWindow::back1frame_pushbutton_control, Qt::QueuedConnection);
        QObject::connect(ui->reverse_pushButton, &QPushButton::clicked, this, &MainWindow::reverse_pushbutton_control, Qt::QueuedConnection);
        QObject::connect(ui->play_pushButton, &QPushButton::clicked, this, &MainWindow::switch_resume_pause, Qt::QueuedConnection);
        QObject::connect(ui->go1frame_pushButton, &QPushButton::clicked, this, &MainWindow::go1frame_pushbutton_control, Qt::QueuedConnection);
        QObject::connect(ui->goendframe_pushButton, &QPushButton::clicked, this, &MainWindow::goendframe_pushbutton_control, Qt::QueuedConnection);
        QObject::connect(ui->stop_pushButton, &QPushButton::clicked, this, &MainWindow::stop_pushbutton_control, Qt::QueuedConnection);
        QObject::connect(ui->cutend_pushButton, &QPushButton::clicked, this, &MainWindow::cutend_pushbutton_control, Qt::QueuedConnection);
        QObject::connect(ui->jumpvalue_spinbox, &QSpinBox::textChanged, this, &MainWindow::get_jump_value, Qt::QueuedConnection);
        QObject::connect(rangeSlider, &RangeSlider::playValueChanged, this, &MainWindow::slider_control, Qt::QueuedConnection);
        QObject::connect(rangeSlider, &RangeSlider::rangeEndChanged, this, &MainWindow::slider_end_control, Qt::QueuedConnection);
        QObject::connect(rangeSlider, &RangeSlider::rangeStartChanged, this, &MainWindow::slider_start_control, Qt::QueuedConnection);
        QObject::connect(rangeSlider, &RangeSlider::playValueReleaseChanged, decodestream, &decode_thread::high_res_sliderPlayback, Qt::QueuedConnection);
        //QObject::connect(ui->Live_horizontalSlider, &QSlider::sliderMoved, this, &MainWindow::slider_control, Qt::QueuedConnection);
        QObject::connect(this, &MainWindow::send_manual_resumeplayback, decodestream, &decode_thread::resumePlayback);
        QObject::connect(this, &MainWindow::send_manual_pause, decodestream, &decode_thread::pausePlayback);
        QObject::connect(this, &MainWindow::send_manual_reverse, decodestream, &decode_thread::reversePlayback);
        QObject::connect(this, &MainWindow::send_manual_back1frame, decodestream, &decode_thread::back1frame, Qt::QueuedConnection);
        QObject::connect(this, &MainWindow::send_manual_go1frame, decodestream, &decode_thread::go1frame, Qt::QueuedConnection);
        QObject::connect(this, &MainWindow::send_manual_slider, decodestream, &decode_thread::sliderPlayback);
        QObject::connect(this, &MainWindow::send_manual_range_end_slider, decodestream, &decode_thread::slider_range_end);
        QObject::connect(this, &MainWindow::send_manual_range_start_slider, decodestream, &decode_thread::slider_range_start);
        QObject::connect(this, &MainWindow::send_manual_high_res_slider, decodestream, &decode_thread::high_res_sliderPlayback);

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
        ui->actionCloseFile->setEnabled(false);
        ui->info->setEnabled(false);
        heavy_process_UI_control(false);
        ui->play_pushButton->setText("▶");
        ui->label_start_time->setText(QString::asprintf("開始:00:00:00(0 Frames)"));
        ui->label_play_time->setText(QString::asprintf("再生:00:00:00(0 Frames)"));
        ui->label_end_time->setText(QString::asprintf("終了:00:00:00(0 Frames)"));
        ui->label_range_time->setText(QString::asprintf("再生範囲:00:00:00(0 Frames)"));
        QFont font_start = ui->label_start_time->font();
        QFont font_play = ui->label_play_time->font();
        QFont font_end = ui->label_end_time->font();
        QFont font_range = ui->label_range_time->font();
        font_start.setPointSize(8);   // 文字サイズ
        font_play.setPointSize(8);   // 文字サイズ
        font_end.setPointSize(8);   // 文字サイズ
        font_range.setPointSize(8);   // 文字サイズ
        ui->label_start_time->setFont(font_start);
        ui->label_play_time->setFont(font_play);
        ui->label_end_time->setFont(font_end);
        ui->label_range_time->setFont(font_range);

        //Disconnect
        QObject::disconnect(decodestream, &decode_thread::send_decode_image, this, &MainWindow::decode_view);
        QObject::disconnect(decodestream, &decode_thread::send_audio, this, &MainWindow::play_audio);

        QObject::disconnect(decodestream, &decode_thread::send_slider_info, this, &MainWindow::init_decodethread_complete);
        QObject::disconnect(decodestream, &decode_thread::heavy_process_signal, this, &MainWindow::heavy_process_UI_control);

        QObject::disconnect(ui->cutstart_pushButton, &QPushButton::clicked, this, &MainWindow::cutstart_pushbutton_control);
        QObject::disconnect(ui->backstartframe_pushButton, &QPushButton::clicked, this, &MainWindow::backstartframe_pushbutton_control);
        QObject::disconnect(ui->back1frame_pushButton, &QPushButton::clicked, this, &MainWindow::back1frame_pushbutton_control);
        QObject::disconnect(ui->reverse_pushButton, &QPushButton::clicked, this, &MainWindow::reverse_pushbutton_control);
        QObject::disconnect(ui->play_pushButton, &QPushButton::clicked, this, &MainWindow::switch_resume_pause);
        QObject::disconnect(ui->go1frame_pushButton, &QPushButton::clicked, this, &MainWindow::go1frame_pushbutton_control);
        QObject::disconnect(ui->goendframe_pushButton, &QPushButton::clicked, this, &MainWindow::goendframe_pushbutton_control);
        QObject::disconnect(ui->stop_pushButton, &QPushButton::clicked, this, &MainWindow::stop_pushbutton_control);
        QObject::disconnect(ui->cutend_pushButton, &QPushButton::clicked, this, &MainWindow::cutend_pushbutton_control);
        QObject::disconnect(ui->jumpvalue_spinbox, &QSpinBox::textChanged, this, &MainWindow::get_jump_value);
        QObject::disconnect(rangeSlider, &RangeSlider::playValueChanged, this, &MainWindow::slider_control);
        QObject::disconnect(rangeSlider, &RangeSlider::rangeStartChanged, this, &MainWindow::slider_start_control);
        QObject::disconnect(rangeSlider, &RangeSlider::rangeEndChanged, this, &MainWindow::slider_end_control);
        QObject::disconnect(rangeSlider, &RangeSlider::playValueReleaseChanged, decodestream, &decode_thread::high_res_sliderPlayback);
        QObject::disconnect(this, &MainWindow::send_manual_resumeplayback, decodestream, &decode_thread::resumePlayback);
        QObject::disconnect(this, &MainWindow::send_manual_pause, decodestream, &decode_thread::pausePlayback);
        QObject::disconnect(this, &MainWindow::send_manual_reverse, decodestream, &decode_thread::reversePlayback);
        QObject::disconnect(this, &MainWindow::send_manual_back1frame, decodestream, &decode_thread::back1frame);
        QObject::disconnect(this, &MainWindow::send_manual_go1frame, decodestream, &decode_thread::go1frame);
        QObject::disconnect(this, &MainWindow::send_manual_slider, decodestream, &decode_thread::sliderPlayback);
        QObject::disconnect(this, &MainWindow::send_manual_range_end_slider, decodestream, &decode_thread::slider_range_end);
        QObject::disconnect(this, &MainWindow::send_manual_range_start_slider, decodestream, &decode_thread::slider_range_start);
        QObject::disconnect(this, &MainWindow::send_manual_high_res_slider, decodestream, &decode_thread::high_res_sliderPlayback);

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

    //停止/再生速度を最大に、エンコードは最速でやるため
    emit send_manual_pause();

    encodeSetting->slider(0,VideoInfo.end_range_framesNo-VideoInfo.start_range_framesNo);
    encodeSetting->show();
    qDebug()<<VideoInfo.end_range_framesNo-VideoInfo.start_range_framesNo;
}

//エンコード開始
void MainWindow::start_encode(){
    if(run_decode_thread){
        qDebug()<<"エンコード開始";

        //パフォーマンス評価用
        QElapsedTimer timer;
        timer.start();

        //終了検知
        wasCanceled=false;
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
        glWidget->MinFrame = VideoInfo.start_range_framesNo;
        glWidget->MaxFrame = VideoInfo.end_range_framesNo;
        emit decode_please();

        //FrameNoが0なことを確認
        emit send_manual_high_res_slider(VideoInfo.start_range_framesNo);
        emit send_manual_resumeplayback();

        //修了処理
        connect(glWidget, &GLWidget::encode_finished, this, [=]() {
            wasCanceled=true;
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

        // 処理ループ内で更新
        while(true) {
            encodeSetting->progress_bar(glWidget->encode_FrameCount);

            if (wasCanceled){
                glWidget->MaxFrame=slider_No;
                break;
            }

            if(glWidget->encode_FrameCount>(glWidget->MaxFrame-glWidget->MinFrame)){
                break;
            }

            QCoreApplication::processEvents();
        }
    }
}

//エンコードウィンドウを閉じる
void MainWindow::finished_encode(){
    encode_state = STATE_NOT_ENCODE;
    glWidget->encode_mode(encode_state);
    decodestream->encode_state = encode_state;
    emit decode_please();
    emit send_manual_resumeplayback();
}
