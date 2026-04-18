#include "src/ui_control/video_speed.h"
#include "ui_video_speed.h"

video_speed::video_speed(QWidget *parent)
    : QWidget(parent),
    ui(new Ui::video_speed)
{
    ui->setupUi(this);

    // フレームなし & 常に前面
    setWindowFlags(Qt::FramelessWindowHint | Qt::Tool | Qt::WindowStaysOnTopHint);

    // 透過は paintEvent に任せる
    setAttribute(Qt::WA_TranslucentBackground);

    //待機タイマー
    hideTimer.setInterval(125); // 500ms 待機
    hideTimer.setSingleShot(true);

    fadeAnim = new QPropertyAnimation(this, "windowOpacity");
    fadeAnim->setDuration(400); // フェード時間 500ms
    fadeAnim->setStartValue(1.0);
    fadeAnim->setEndValue(0.0);

    connect(fadeAnim, &QPropertyAnimation::finished, this, [this]() {
        hide();
        setWindowOpacity(1.0); // 次回用に戻す
    });

    connect(&hideTimer, &QTimer::timeout, this, [this]() {

        // 両方にマウスが無ければフェード開始
        if (!mouseInside &&
            (!anchorButton || !anchorButton->underMouse()))
        {
            fadeAnim->stop();
            fadeAnim->start();
        }
    });

    //ボタン初期設定
    setCurrentSpeed(g_AppSettings.video_speed_ratio);

    //ボタン
    connect(ui->pushButton_025, &QPushButton::clicked, this, [&]() {
        emit speedChanged(0.25);
    });
    connect(ui->pushButton_05, &QPushButton::clicked, this, [&]() {
        emit speedChanged(0.5);
    });
    connect(ui->pushButton_075, &QPushButton::clicked, this, [&]() {
        emit speedChanged(0.75);
    });
    connect(ui->pushButton_10, &QPushButton::clicked, this, [&]() {
        emit speedChanged(1.0);
    });
    connect(ui->pushButton_15, &QPushButton::clicked, this, [&]() {
        emit speedChanged(1.5);
    });
    connect(ui->pushButton_20, &QPushButton::clicked, this, [&]() {
        emit speedChanged(2.0);
    });
    connect(ui->pushButton_40, &QPushButton::clicked, this, [&]() {
        emit speedChanged(4.0);
    });
}

video_speed::~video_speed()
{
    delete ui;
}

void video_speed::paintEvent(QPaintEvent *)
{
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);

    QColor bg(60, 60, 60, 150);
    p.setBrush(bg);
    p.setPen(Qt::NoPen);

    QRect r = rect();
    QPainterPath path;
    path.addRoundedRect(r, 12, 12);

    p.drawPath(path);
}

void video_speed::setAnchorButton(QPushButton* button)
{
    anchorButton = button;
}

void video_speed::showEvent(QShowEvent *event)
{
    QWidget::showEvent(event);

    if (!anchorButton)
        return;

    QPoint btnPos = anchorButton->mapToGlobal(QPoint(0, 0));

    int x = btnPos.x() + (anchorButton->width() - width()) / 2;
    int y = btnPos.y() - height(); // ← ピッタリ接触

    move(x, y);
}


void video_speed::enterEvent(QEnterEvent *)
{
    mouseInside = true;

    hideTimer.stop();
    fadeAnim->stop();

    setWindowOpacity(1.0); // 即表示
}

void video_speed::leaveEvent(QEvent *)
{
    mouseInside = false;
    hideTimer.start(); // 500ms後にフェード開始
}

void video_speed::showPopup()
{
    fadeAnim->stop();
    hideTimer.stop();

    setCurrentSpeed(currentSpeed);

    setWindowOpacity(1.0);
    show();
    raise();
}

//既に選択されている場合はその個所の色を変える
void video_speed::setCurrentSpeed(double speed)
{
    currentSpeed = speed;

    // 通常スタイル
    const QString normalStyle = R"(
        QPushButton {
            background: qlineargradient(
                x1:0, y1:0, x2:0, y2:1,
                stop:0 rgba(80,80,80,230),
                stop:1 rgba(35,35,35,230)
            );
            border-radius: 8px;
            padding: 4px;
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
            border-radius: 8px;
            padding: 4px;
            color: rgb(235,235,235); /* ← 通常テキスト 白 */
            font-weight: 600;
        }
    )";

    // 選択スタイル
    const QString highlightStyle = R"(
        QPushButton {
            background: qlineargradient(
                x1:0, y1:0, x2:0, y2:1,
                stop:0 rgba(130,190,255,255),
                stop:1 rgba(60,110,220,255)
            );
            border-radius: 8px;
            padding: 4px;
            color: rgb(235,235,235); /* ← 通常テキスト 白 */
            font-weight: 600;
        }
    )";

    // 全ボタンをリセット
    QList<QPushButton*> buttons = {
        ui->pushButton_025,
        ui->pushButton_05,
        ui->pushButton_075,
        ui->pushButton_10,
        ui->pushButton_15,
        ui->pushButton_20,
        ui->pushButton_40
    };

    for (auto* btn : buttons)
        btn->setStyleSheet(normalStyle);

    // 選択ボタンのみハイライト
    auto highlight = [&](QPushButton* btn, double value) {
        if (qAbs(currentSpeed - value) < 0.0001)
            btn->setStyleSheet(highlightStyle);
    };

    highlight(ui->pushButton_025, 0.25);
    highlight(ui->pushButton_05,  0.5);
    highlight(ui->pushButton_075, 0.75);
    highlight(ui->pushButton_10,  1.0);
    highlight(ui->pushButton_15,  1.5);
    highlight(ui->pushButton_20,  2.0);
    highlight(ui->pushButton_40,  4.0);
}




