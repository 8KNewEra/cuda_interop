#ifndef RANGESLIDER_H
#define RANGESLIDER_H

#pragma once
#include <QWidget>

class RangeSlider : public QWidget
{
    Q_OBJECT

public:
    explicit RangeSlider(QWidget* parent = nullptr);

    void setRange(int min, int max);
    void setMinimum(int min);
    void setMaximum(int max);

    void setStartValue(int value);
    void setEndValue(int value);
    void setPlayValue(int value);

    void setValues(int start, int end);

signals:
    void rangeStartChanged(int start);
    void rangeEndChanged(int end);
    void playValueChanged(int value);
    void playValueReleaseChanged(int value);

protected:
    void paintEvent(QPaintEvent*) override;
    void mousePressEvent(QMouseEvent*) override;
    void mouseMoveEvent(QMouseEvent*) override;
    void mouseReleaseEvent(QMouseEvent*) override;

private:
    enum Handle { NoHandle, StartHandle, EndHandle, PlayHandle };

    int m_min = 0;
    int m_max = 100;

    int m_start = 20;
    int m_end   = 80;
    int m_play  = 50;

    Handle m_activeHandle = NoHandle;

    int pixelPosFromValue(int value) const;
    int valueFromPixelPos(int pos) const;
    QRect handleRect(int value) const;

    int m_minSpacing = 5;   // 値ベースの最小間隔
    int m_minPixelSpacing = 14;  // ハンドル直径より少し大きめ

    bool m_userInteraction = false;
};

#endif // RANGESLIDER_H
