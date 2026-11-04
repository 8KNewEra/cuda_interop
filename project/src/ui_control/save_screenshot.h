#ifndef SAVE_SCREENSHOT_H
#define SAVE_SCREENSHOT_H

#pragma once
#include <QWidget>
#include <QPushButton>
#include <QTimer>

namespace Ui {
class save_screenshot;
}

class save_screenshot : public QWidget
{
    Q_OBJECT

public:
    explicit save_screenshot(QWidget *parent = nullptr);
    ~save_screenshot();

private:
    Ui::save_screenshot *ui;
    void screenshot_path_dialog();
};

#endif // SAVE_SCREENSHOT_H
