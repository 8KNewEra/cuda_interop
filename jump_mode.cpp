#include "jump_mode.h"
#include "ui_jump_mode.h"

jump_mode::jump_mode(QWidget *parent)
    : QWidget(parent),
    ui(new Ui::jump_mode)
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

    //ボタン
    setCurrentMode(JUMP_MODE_SECOND);

    //ボタン
    connect(ui->pushButton_second, &QPushButton::clicked, this, [&]() {
        emit modeChanged(JUMP_MODE_SECOND);
    });
    connect(ui->pushButton_Frame, &QPushButton::clicked, this, [&]() {
        emit modeChanged(JUMP_MODE_FRAME);
    });
    connect(ui->pushButton_targetFrame, &QPushButton::clicked, this, [&]() {
        emit modeChanged(JUMP_MODE_TARGETFRAME);
    });
}

jump_mode::~jump_mode()
{
    delete ui;
}

void jump_mode::paintEvent(QPaintEvent *)
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

void jump_mode::setAnchorButton(QPushButton* button)
{
    anchorButton = button;
}

void jump_mode::showEvent(QShowEvent *event)
{
    QWidget::showEvent(event);

    if (!anchorButton)
        return;

    QPoint btnPos = anchorButton->mapToGlobal(QPoint(0, 0));

    int x = btnPos.x() + (anchorButton->width() - width()) / 2;
    int y = btnPos.y() + 22; // ← ピッタリ接触

    move(x, y);
}


void jump_mode::enterEvent(QEnterEvent *)
{
    mouseInside = true;

    hideTimer.stop();
    fadeAnim->stop();

    setWindowOpacity(1.0); // 即表示
}

void jump_mode::leaveEvent(QEvent *)
{
    mouseInside = false;
    hideTimer.start(); // 500ms後にフェード開始
}

void jump_mode::showPopup()
{
    fadeAnim->stop();
    hideTimer.stop();

    setCurrentMode(currentMode);

    setWindowOpacity(1.0);
    show();
    raise();
}

//既に選択されている場合はその個所の色を変える
void jump_mode::setCurrentMode(int mode)
{
    currentMode = mode;

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
        ui->pushButton_second,
        ui->pushButton_Frame,
        ui->pushButton_targetFrame,
    };

    for (auto* btn : buttons)
        btn->setStyleSheet(normalStyle);

    // 選択ボタンのみハイライト
    auto highlight = [&](QPushButton* btn, int value) {
        if (currentMode == value){
            btn->setStyleSheet(highlightStyle);
        }
    };

    highlight(ui->pushButton_second, JUMP_MODE_SECOND);
    highlight(ui->pushButton_Frame,  JUMP_MODE_FRAME);
    highlight(ui->pushButton_targetFrame, JUMP_MODE_TARGETFRAME);
}
