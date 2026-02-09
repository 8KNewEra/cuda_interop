#include "audio_volume.h"
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

    //浮かぶ数値バブル
    valueLabel = new QLabel(this);
    valueLabel->setAlignment(Qt::AlignCenter);
    valueLabel->setStyleSheet(
        "QLabel {"
        "background: rgba(30,30,30,230);"
        "color: white;"
        "border-radius: 3px;"
        "padding: 2px 4px;"
        "font-size: 11px;"
        "}"
        );
    valueLabel->hide();

    //ドラッグ位置で表示非表示
    connect(ui->verticalSlider_volume, &QSlider::sliderPressed, this, [&]() {
        valueLabel->show();
    });
    connect(ui->verticalSlider_volume, &QSlider::sliderReleased, this, [&]() {
        valueLabel->hide();
    });

    //スライダーの値をmainWindowへ
    connect(ui->verticalSlider_volume, &QSlider::valueChanged, this, [&](int value) {
        updateValuePopup(value);
        valueLabel->show();
        emit volumeChanged(value);
    });

    updateValuePopup(ui->verticalSlider_volume->value());
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

//スライダーの制御に応じて数値ラベルを表示
void audio_volume::updateValuePopup(int value)
{
    valueLabel->setText(QString("%1%").arg(value));

    QStyleOptionSlider opt;
    opt.initFrom(ui->verticalSlider_volume);
    opt.orientation = Qt::Vertical;
    opt.minimum = ui->verticalSlider_volume->minimum();
    opt.maximum = ui->verticalSlider_volume->maximum();
    opt.sliderPosition = value;
    opt.sliderValue = value;

    QRect handleRect = ui->verticalSlider_volume->style()->subControlRect(
        QStyle::CC_Slider, &opt, QStyle::SC_SliderHandle, ui->verticalSlider_volume
        );

    // スライダー内ローカル → 親座標へ変換
    QPoint sliderPos = ui->verticalSlider_volume->mapToParent(handleRect.center());

    // ハンドルの上に表示
    valueLabel->adjustSize();
    if(value>75){
        valueLabel->move(sliderPos.x() - valueLabel->width() / 2 + 1,valueLabel->height() - sliderPos.y() + 107);
    }else{
        valueLabel->move(sliderPos.x() - valueLabel->width() / 2 + 1,valueLabel->height() - sliderPos.y() + 75);
    }
}


