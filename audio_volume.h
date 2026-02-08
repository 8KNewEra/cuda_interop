#ifndef AUDIO_VOLUME_H
#define AUDIO_VOLUME_H

#pragma once
#include <QWidget>
#include <QPushButton>
#include <QTimer>
#include <QPropertyAnimation>
#include <QPainterPath>


namespace Ui {
class audio_volume;
}

class audio_volume : public QWidget
{
    Q_OBJECT

signals:
    void volumeChanged(int value);

public:
    explicit audio_volume(QWidget *parent = nullptr);
    ~audio_volume();

    void setAnchorButton(QPushButton* button);
    QTimer hideTimer;
    void showPopup();

protected:
    void enterEvent(QEnterEvent *event) override;
    void leaveEvent(QEvent *event) override;
    void showEvent(QShowEvent *event) override;
    void paintEvent(QPaintEvent *event) override;

private:
    Ui::audio_volume *ui;
    QPushButton* anchorButton = nullptr;
    bool mouseInside = false;
    QPropertyAnimation* fadeAnim = nullptr;
};

#endif // AUDIO_VOLUME_H
