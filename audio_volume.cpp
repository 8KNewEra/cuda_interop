#include "audio_volume.h"
#include "qpainter.h"
#include "ui_audio_volume.h"

audio_volume::audio_volume(QWidget *parent)
    : QWidget(parent),
    ui(new Ui::audio_volume)
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

    //スライダー
    ui->verticalSlider_volume->setStyleSheet(R"(
        QSlider::groove:vertical {
            background: #2A2A2A;
            width: 6px;
            border-radius: 3px;
        }

        QSlider::sub-page:vertical {
            background: #444444;   /* ← 下側（つまみより下）水色 */
            border-radius: 3px;
        }

        QSlider::add-page:vertical {
            background: #4FC3F7;   /* ← 上側（未到達部分） */
            border-radius: 3px;
        }

        QSlider::handle:vertical {
            background: white;
            height: 12px;
            width: 12px;
            margin: 0 -6px;
            border-radius: 4px;
        }
    )");

    //スライダーの値をmainWindowへ
    connect(ui->verticalSlider_volume, &QSlider::valueChanged,
            this, &audio_volume::volumeChanged);
}

audio_volume::~audio_volume()
{
    delete ui;
}

void audio_volume::paintEvent(QPaintEvent *)
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

void audio_volume::setAnchorButton(QPushButton* button)
{
    anchorButton = button;
}

void audio_volume::showEvent(QShowEvent *event)
{
    QWidget::showEvent(event);

    if (!anchorButton)
        return;

    QPoint btnPos = anchorButton->mapToGlobal(QPoint(0, 0));

    int x = btnPos.x() + (anchorButton->width() - width()) / 2;
    int y = btnPos.y() - height(); // ← ピッタリ接触

    move(x, y);
}


void audio_volume::enterEvent(QEnterEvent *)
{
    mouseInside = true;

    hideTimer.stop();
    fadeAnim->stop();

    setWindowOpacity(1.0); // 即表示
}

void audio_volume::leaveEvent(QEvent *)
{
    mouseInside = false;
    hideTimer.start(); // 500ms後にフェード開始
}

void audio_volume::showPopup()
{
    fadeAnim->stop();
    hideTimer.stop();

    setWindowOpacity(1.0);
    show();
    raise();
}

