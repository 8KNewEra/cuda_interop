#ifndef VIDEO_SPEED_H
#define VIDEO_SPEED_H

#include "qlabel.h"
#pragma once
#include <QWidget>
#include <QPushButton>
#include <QTimer>
#include <QPropertyAnimation>
#include <QPainterPath>
#include "qpainter.h"
#include "qstyleoption.h"

namespace Ui {
class video_speed;
}

class video_speed : public QWidget
{
    Q_OBJECT

signals:
    void speedChanged(double value);

public:
    explicit video_speed(QWidget *parent = nullptr);
    ~video_speed();

    void setAnchorButton(QPushButton* button);
    void setCurrentSpeed(double speed);
    QTimer hideTimer;
    void showPopup();

protected:
    void enterEvent(QEnterEvent *event) override;
    void leaveEvent(QEvent *event) override;
    void showEvent(QShowEvent *event) override;
    void paintEvent(QPaintEvent *event) override;

private:
    Ui::video_speed *ui;
    QPushButton* anchorButton = nullptr;
    bool mouseInside = false;
    QPropertyAnimation* fadeAnim = nullptr;
    double currentSpeed = 1.0;
};

#endif // VIDEO_SPEED_H
