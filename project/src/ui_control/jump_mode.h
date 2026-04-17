#ifndef JUMP_MODE_H
#define JUMP_MODE_H

#include "src/main/__global__.h"
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
class jump_mode;
}

class jump_mode : public QWidget
{
    Q_OBJECT

signals:
    void modeChanged(double value);

public:
    explicit jump_mode(QWidget *parent = nullptr);
    ~jump_mode();

    void setAnchorButton(QPushButton* button);
    void setCurrentMode(int mode);
    QTimer hideTimer;
    void showPopup();

protected:
    void enterEvent(QEnterEvent *event) override;
    void leaveEvent(QEvent *event) override;
    void showEvent(QShowEvent *event) override;
    void paintEvent(QPaintEvent *event) override;

private:
    Ui::jump_mode *ui;
    QPushButton* anchorButton = nullptr;
    bool mouseInside = false;
    QPropertyAnimation* fadeAnim = nullptr;
    int currentMode = JUMP_MODE_SECOND;
};

#endif // JUMP_EDIT_H
