#ifndef SAVE_SCREENSHOT_H
#define SAVE_SCREENSHOT_H

#pragma once
#include <QWidget>
#include <QPushButton>
#include <QTimer>
#include <QPropertyAnimation>
#include <QPainterPath>

namespace Ui {
class save_screenshot;
}

class save_screenshot : public QWidget
{
    Q_OBJECT

signals:
    void modeChanged();

public:
    explicit save_screenshot(QWidget *parent = nullptr);
    ~save_screenshot();

    void setAnchorButton(QPushButton* button);
    QTimer hideTimer;
    void showPopup();

protected:
    void enterEvent(QEnterEvent *event) override;
    void leaveEvent(QEvent *event) override;
    void showEvent(QShowEvent *event) override;
    void paintEvent(QPaintEvent *event) override;

private:
    Ui::save_screenshot *ui;
    QPushButton* anchorButton = nullptr;
    bool mouseInside = false;
    QPropertyAnimation* fadeAnim = nullptr;
};

#endif // SAVE_SCREENSHOT_H
