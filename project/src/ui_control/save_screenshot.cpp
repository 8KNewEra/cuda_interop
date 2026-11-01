#include "src/ui_control/save_screenshot.h"
#include "qpainter.h"
#include "ui_save_screenshot.h"

save_screenshot::save_screenshot(QWidget *parent)
    : QWidget(parent),
    ui(new Ui::save_screenshot)
{
    ui->setupUi(this);

    // 通常スタイル
    const QString style = R"(
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

    ui->pushButton_savescreenshotpath->setStyleSheet(style);

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

    connect(ui->pushButton_savescreenshotpath, &QPushButton::clicked, this, [&]() {
        emit modeChanged();
    });
}

save_screenshot::~save_screenshot()
{
    delete ui;
}

void save_screenshot::paintEvent(QPaintEvent *)
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

void save_screenshot::setAnchorButton(QPushButton* button)
{
    anchorButton = button;
}

void save_screenshot::showEvent(QShowEvent *event)
{
    QWidget::showEvent(event);

    if (!anchorButton)
        return;

    QPoint btnPos = anchorButton->mapToGlobal(QPoint(0, 0));

    int x = btnPos.x() + (anchorButton->width() - width()) / 2;
    int y = btnPos.y() + 22; // ← ピッタリ接触

    move(x, y);
}


void save_screenshot::enterEvent(QEnterEvent *)
{
    mouseInside = true;

    hideTimer.stop();
    fadeAnim->stop();

    setWindowOpacity(1.0); // 即表示
}

void save_screenshot::leaveEvent(QEvent *)
{
    mouseInside = false;
    hideTimer.start(); // 500ms後にフェード開始
}

void save_screenshot::showPopup()
{
    fadeAnim->stop();
    hideTimer.stop();

    setWindowOpacity(1.0);
    show();
    raise();
}
